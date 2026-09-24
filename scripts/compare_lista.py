"""Train supervised ConvLISTA and compare with ISTA/FISTA on the same sample.

python scripts/lista.py --data-root . --output runs/lista_z3 --device cuda --gpus 2
All three component images are TRAINING samples. There is no held-out score.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn

from deblur import learn_kernel_joint
from lista import ConvLISTA
from torch_inverse import PaddedFFT, evaluate, solve_torch, synchronize

COMPONENTS = ("Bx", "By", "Bz")


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_model(model, b, truth, *, epochs, lr, device_ids=None, grad_clip=1.,
                callback=None):
    """Full-batch supervised MSE, with model selection on training loss only.

    Epoch 0 is the untrained network. Save best weights by re-evaluating AFTER
    updates, avoiding the common error of pairing pre-update loss with weights.
    DataParallel splits the 3-image batch; MSE is evaluated after gathering all
    predictions, so uneven 2+1 GPU chunks still have correct sample weights.
    """
    if epochs < 0 or lr <= 0 or not np.isfinite(lr) or grad_clip <= 0:
        raise ValueError("invalid training settings")
    parallel = nn.DataParallel(model, device_ids=device_ids) if device_ids else model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows, best_loss, best_epoch, best_state = [], float('inf'), 0, None
    synchronize(b.device)
    start = perf_counter()
    for epoch in range(epochs+1):
        grad_norm = None
        if epoch:
            parallel.train()
            optimizer.zero_grad(set_to_none=True)
            pred = parallel(b)
            loss = (pred-truth).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at epoch {epoch}")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=True))
            optimizer.step()
            model.project_parameters()
        parallel.eval()
        with torch.no_grad():
            current = float((parallel(b)-truth).square().mean())
        if not np.isfinite(current):
            raise FloatingPointError(f"nonfinite post-update loss at epoch {epoch}")
        if current < best_loss:
            best_loss, best_epoch, best_state = current, epoch, cpu_state(model)
        synchronize(b.device)
        row = {"epoch": epoch, "training_mse_normalized": current,
               "training_seconds": perf_counter()-start,
               "gradient_norm_before_clipping": grad_norm,
               "threshold_normalized": float(model.threshold.detach())}
        rows.append(row)
        if callback is not None:
            callback(row)
    elapsed = perf_counter()-start
    model.load_state_dict(best_state)
    model.eval()
    return rows, best_epoch, best_loss, elapsed


@torch.no_grad()
def benchmark(fn, device, repeats=3):
    """Warm-up then synchronized median inference time, no metrics or file I/O."""
    fn()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = perf_counter()
        out = fn()
        synchronize(device)
        samples.append(perf_counter()-start)
        del out
    return {"median_seconds": float(np.median(samples)), "samples_seconds": samples}


@torch.no_grad()
def fixed_steps(b, operator, lam, method, n_iter, initial):
    """Untimed diagnostic-free baseline for the latency benchmark."""
    x = b.clone() if initial == 'blurred' else torch.zeros_like(b)
    y, t = x.clone(), 1.
    step = .99/operator.bound
    from torch_inverse import shrink
    for _ in range(n_iter):
        point = x if method == 'ista' else y
        nx = shrink(point-step*operator.adjoint(operator(point)-b), step*lam)
        if method == 'fista':
            nt = (1+np.sqrt(1+4*t*t))/2
            y = nx+(t-1)/nt*(nx-x)
            t = nt
        x = nx
    return x


def load_problem(root, z, pad):
    cache = root / 'cache'
    meta = json.loads((cache/'grid.json').read_text())
    if z <= .5 or z not in meta['z_values'] or .5 not in meta['z_values']:
        raise ValueError('need reference 0.5 and requested target >0.5 in grid.json')
    if meta.get('field_unit') != 'T' or meta.get('length_unit') != 'um':
        raise ValueError('cache units must be T and um')
    shape = (meta['ny'], meta['nx'])
    arrays, paths = [], [cache/'grid.json']
    for height in (.5, z):
        planes = []
        for c in COMPONENTS:
            p = cache/f'z{height:04.1f}_{c}.npy'
            a = np.load(p, allow_pickle=False)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f'invalid cache array {p}')
            # Match fields.load default scaling before learning the kernel.
            planes.append(a * 1e6)
            paths.append(p)
        arrays.append(np.stack(planes))
    sharp, blur = arrays
    start = perf_counter()
    h = learn_kernel_joint(list(sharp), list(blur), pad=pad)
    kernel_seconds = perf_counter()-start
    return sharp, blur, h, meta, paths, kernel_seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--z', type=float, default=3.)
    parser.add_argument('--pad', type=int, default=32)
    parser.add_argument('--lambda-ista', type=float, default=1e-4)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--kernel-size', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--baseline-iterations', type=int, default=200)
    parser.add_argument('--initial', choices=('blurred', 'zero'), default='blurred')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--record-every', type=int, default=10)
    parser.add_argument('--checkpoint-layers', action='store_true')
    parser.add_argument('--benchmark-repeats', type=int, default=3)
    args = parser.parse_args(argv)
    if (args.pad < 0 or args.epochs < 0 or args.layers < 1 or args.kernel_size < 1
            or args.kernel_size % 2 == 0 or args.baseline_iterations < args.layers
            or args.record_every < 1 or args.benchmark_repeats < 1 or args.gpus < 1):
        parser.error('invalid integer settings; baseline-iterations must be >= layers')
    if (not np.isfinite(args.lambda_ista) or args.lambda_ista < 0
            or not np.isfinite(args.lr) or args.lr <= 0):
        parser.error('lambda must be nonnegative and learning rate positive, both finite')
    if args.device == 'cuda' and (not torch.cuda.is_available()
                                  or torch.cuda.device_count() < args.gpus):
        parser.error(f'requested {args.gpus} CUDA GPUs are not available')
    if args.device == 'cpu' and args.gpus != 1:
        parser.error('CPU mode requires --gpus 1')
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    sharp, blur, h, grid, inputs, kernel_seconds = load_problem(args.data_root, args.z, args.pad)
    scale = float(max(np.max(np.abs(sharp)), np.max(np.abs(blur))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('cannot normalize zero/nonfinite fields')
    lam = args.lambda_ista / scale
    truth = torch.as_tensor(sharp.astype(np.float64)/scale, dtype=torch.float32,
                            device=device).unsqueeze(1)
    b = torch.as_tensor(blur.astype(np.float64)/scale, dtype=torch.float32,
                        device=device).unsqueeze(1)
    model = ConvLISTA(h, sharp.shape[-2:], args.pad, lam, layers=args.layers,
                      kernel_size=args.kernel_size, initial=args.initial,
                      checkpoint_layers=args.checkpoint_layers).to(device)
    operator = model.operator
    model.eval()
    with torch.no_grad():
        initial_prediction = model(b)
        equivalent = fixed_steps(b, operator, lam, 'ista', args.layers, args.initial)
        torch.testing.assert_close(initial_prediction, equivalent, rtol=2e-5, atol=2e-6)
        initialization_error = float((initial_prediction-equivalent).abs().max())
    state_before = cpu_state(model)
    predictions, histories, baselines = {}, {}, {}
    for method in ('ista', 'fista'):
        print(f'Running {method.upper()} on {device}', flush=True)
        x, history, wall_seconds = solve_torch(
            b, operator, lam, method=method, n_iter=args.baseline_iterations,
            initial=args.initial, truth=truth, scale=scale, record_every=args.record_every)
        predictions[method] = x.cpu().numpy()[:, 0]*scale
        histories[method] = history
        baselines[method] = {'iterations': args.baseline_iterations,
                             'wall_seconds_with_metrics': wall_seconds}
    print('Training LISTA on all three component images (no holdout).', flush=True)

    def log(row):
        if row['epoch'] % 10 == 0 or row['epoch'] == args.epochs:
            print(f"epoch {row['epoch']:4d}: train MSE={row['training_mse_normalized']:.7g}", flush=True)

    ids = list(range(args.gpus)) if args.device == 'cuda' and args.gpus > 1 else None
    training, best_epoch, best_loss, training_seconds = train_model(
        model, b, truth, epochs=args.epochs, lr=args.lr, device_ids=ids, callback=log)
    # All inference comparisons run on the SAME single device. The second GPU
    # is used for training only, so method latency is not a hardware comparison.
    model.eval()
    path_rows = []
    with torch.no_grad():
        for k, x in enumerate(model.iterates(b)):
            path_rows.append({'iteration': k,
                              'components': evaluate(x, b, truth, operator, lam, scale)})
        predictions['lista'] = x.cpu().numpy()[:, 0]*scale
        predictions['lista_initial'] = initial_prediction.cpu().numpy()[:, 0]*scale
    histories['lista'] = path_rows
    histories['lista_initial_final'] = evaluate(initial_prediction, b, truth, operator, lam, scale)
    comparisons = {'lista': {'layers': args.layers,
                             'metrics': path_rows[-1]['components'],
                             'inference': benchmark(lambda: model(b), device, args.benchmark_repeats)}}
    for method in ('ista', 'fista'):
        for count in sorted(set((args.layers, args.baseline_iterations))):
            fn = lambda method=method, count=count: fixed_steps(
                b, operator, lam, method, count, args.initial)
            with torch.no_grad():
                prediction = fn()
                scores = evaluate(prediction, b, truth, operator, lam, scale)
            comparisons[f'{method}_{count}'] = {
                'iterations': count, 'metrics': scores,
                'inference': benchmark(fn, device, args.benchmark_repeats)}
    learned_changes = {name: float((param.detach().cpu()-state_before[name]).norm())
                       for name, param in model.named_parameters()}
    checkpoint_data = {
        'state_dict': cpu_state(model), 'layers': args.layers, 'kernel_size': args.kernel_size,
        'pad': args.pad, 'shape': list(sharp.shape[-2:]), 'initial': args.initial,
        'lambda_uT': args.lambda_ista, 'lambda_normalized': lam, 'scale_uT': scale,
        'norm_bound': operator.bound, 'best_training_epoch': best_epoch,
        'best_training_mse': best_loss, 'dtype': 'float32',
        'architecture': 'tied physics-preserving ConvLISTA corrections',
        'training_target': 'sharp COMSOL fields; same sample',
    }
    torch.save(checkpoint_data, args.output/'lista_best_training.pt')
    np.save(args.output/'kernel.npy', h)
    np.savez_compressed(args.output/'reconstructions.npz', sharp=sharp, blurred=blur, **predictions)
    source_paths = sorted(Path(__file__).parent.glob('*.py'))
    results = {
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'evaluation': 'SAME-SAMPLE TRAINING RECONSTRUCTION; no validation or test split',
        'components': COMPONENTS, 'grid': grid,
        'architecture': checkpoint_data['architecture'], 'parameter_count': sum(p.numel() for p in model.parameters()),
        'scale_uT': scale, 'lambda_normalized': lam, 'norm_bound': operator.bound,
        'step': model.step, 'initialization_max_abs_error_normalized': initialization_error,
        'training_seconds_including_post_update_evaluation': training_seconds,
        'kernel_seconds': kernel_seconds, 'best_training_epoch': best_epoch,
        'best_training_mse_normalized': best_loss, 'learned_parameter_change_l2': learned_changes,
        'training': training, 'histories': histories, 'baselines': baselines,
        'comparisons': comparisons,
        'timing': 'Training may use two GPUs; all inference uses one identical device/dtype/batch. '
                  'Median warmed synchronized latency excludes I/O and metrics. Iteration histories exclude metrics from solver_seconds.',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
                        'device': str(device), 'gpu_names': [torch.cuda.get_device_name(i) for i in range(args.gpus)]
                        if device.type == 'cuda' else [], 'deterministic_algorithms': True,
                        'amp': False, 'tf32': False},
        'input_sha256': {str(p.relative_to(args.data_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    }
    (args.output/'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    from plot_comparison import plot_results
    plot_results(args.output)
    print(f'Saved checkpoints, arrays, metrics and figures in {args.output}', flush=True)


if __name__ == '__main__':
    main()
