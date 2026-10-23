"""Train shared ConvLISTA and compare ISTA/FISTA on geometry-safe pairs.

Default: joint fitted operators, all pairs used for training and evaluated as
same-sample reconstructions. Geometry holdouts require analytic operators.
Run on one GPU, with bounded component batches and epoch-boundary resume.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
import csv
import io
import json
import math
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from combined_data import COMPONENTS, read_manifest
from kernel_crops import sha256
from combined_lista import SharedConvLISTA
from compare_lista import cpu_state, fixed_steps, benchmark
from deblur import learn_kernel_joint
from propagator import k_grid, analytic_transfer
from torch_inverse import PaddedFFT, evaluate, solve_torch, synchronize


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def split_pairs(manifest, validation, test, operator):
    validation, test = set(validation), set(test)
    geometries = set(manifest['geometries'])
    if validation & test or not (validation | test) <= geometries:
        raise ValueError('Unknown or overlapping validation/test geometries')
    if (validation or test) and operator != 'analytic':
        raise ValueError('Held-out geometries require --operator analytic to avoid reference-assisted kernel fitting')
    result = {'train': [], 'validation': [], 'test': []}
    for pair in manifest['pairs']:
        split = 'test' if pair['geometry'] in test else (
            'validation' if pair['geometry'] in validation else 'train')
        result[split].append(pair)
    if not result['train']:
        raise ValueError('At least one training geometry is required')
    return result


def training_scale(root, pairs):
    paths = {p[k] for p in pairs for k in ('sharp', 'blurred')}
    scale = max(float(np.abs(np.load(root/p, mmap_mode='r', allow_pickle=False)).max()) for p in paths)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Cannot normalize zero/nonfinite training data')
    return scale


def build_physics(manifest, root, pairs, directory, pad, method):
    directory.mkdir()
    entries = {}
    for pair in pairs:
        grid = manifest['geometries'][pair['geometry']]['grid']
        if method == 'joint':
            sharp = np.load(root/pair['sharp'], allow_pickle=False)
            blurred = np.load(root/pair['blurred'], allow_pickle=False)
            h = learn_kernel_joint(list(sharp), list(blurred), pad=pad)
        else:
            shape = (grid['ny']+2*pad, grid['nx']+2*pad)
            h = analytic_transfer(k_grid(shape, grid['dx'], grid['dy']), pair['dz_um'])
        if not np.isfinite(h).all() or np.max(np.abs(h)) <= 0:
            raise ValueError(f"Invalid operator: {pair['id']}")
        path = directory/f"{pair['id']}.npy"
        np.save(path, h, allow_pickle=False)
        entries[pair['id']] = dict(path=str(path.relative_to(directory.parent)), sha256=sha256(path))
        print(f"Prepared operator {pair['id']}", flush=True)
    return entries


def load_pair(pair, manifest, root, output, physics, pad, device):
    sharp = np.load(root/pair['sharp'], mmap_mode='r', allow_pickle=False)
    blurred = np.load(root/pair['blurred'], mmap_mode='r', allow_pickle=False)
    h = np.load(output/physics[pair['id']]['path'], allow_pickle=False)
    operator = PaddedFFT(h, sharp.shape[-2:], pad).to(device)
    return sharp, blurred, operator


def chunks(sharp, blurred, scale, size, device):
    for start in range(0, 3, size):
        end = min(start+size, 3)
        truth = torch.tensor(np.asarray(sharp[start:end], dtype=np.float64)/scale,
                             dtype=torch.float32, device=device).unsqueeze(1)
        b = torch.tensor(np.asarray(blurred[start:end], dtype=np.float64)/scale,
                         dtype=torch.float32, device=device).unsqueeze(1)
        yield start, b, truth


def dataset_loss(model, pairs, ctx, backward=False):
    if not pairs:
        return None
    total = 0.
    model.train(backward)
    with torch.set_grad_enabled(backward):
        for pair in pairs:
            sharp, blur, operator = load_pair(pair, **ctx['load'])
            for _, b, truth in chunks(sharp, blur, ctx['scale'], ctx['batch_size'], ctx['device']):
                loss = (model(b, operator)-truth).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss: {pair['id']}")
                weight = len(b)/(3*len(pairs))
                total += float(loss.detach())*weight
                if backward:
                    (loss*weight).backward()
    return total


def write_csv(path, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(stream.getvalue())
    tmp.replace(path)


def compare(model, splits, ctx, args):
    """Same single device and component batch for every method; per-batch stop."""
    model.eval()
    rows, nonconverged = [], []
    directory = args.output/'comparison'
    directory.mkdir(exist_ok=True)
    for split, pairs in splits.items():
        for pair in pairs:
            print(f"Comparing {pair['id']} ({split})", flush=True)
            sharp, blur, operator = load_pair(pair, **ctx['load'])
            predictions, details = {}, []
            for start, b, truth in chunks(sharp, blur, ctx['scale'], args.batch_size, ctx['device']):
                functions = {'lista': lambda: model(b, operator)}
                for method in ('ista', 'fista'):
                    functions[f'{method}_{args.layers}'] = lambda method=method: fixed_steps(
                        b, operator, ctx['lam'], method, args.layers, args.initial)
                    functions[f'{method}_to_epsilon'] = lambda method=method: solve_torch(
                        b, operator, ctx['lam'], method=method, initial=args.initial,
                        epsilon=args.epsilon, max_iterations=args.max_iterations,
                        scale=ctx['scale'], record_every=args.record_every, diagnostics=False)
                for name, fn in functions.items():
                    with torch.no_grad():
                        result = fn()
                        history = None
                        if isinstance(result, tuple):
                            x, history, _ = result
                            stop = history[-1]
                            if not stop['converged']:
                                nonconverged.append(dict(pair=pair['id'], method=name, batch_start=start,
                                                         reason=stop['stop_reason']))
                            details.append(dict(method=name, batch_start=start, history=history))
                        else:
                            x, stop = result, None
                        metrics = evaluate(x, b, truth, operator, ctx['lam'], ctx['scale'])
                        if any(not math.isfinite(v) for metric in metrics for v in metric.values() if v is not None):
                            raise FloatingPointError(f'Nonfinite metrics: {pair["id"]} {name}')
                        timing = benchmark(fn, ctx['device'], args.benchmark_repeats)
                        if args.save_reconstructions:
                            predictions.setdefault(name, []).append(x.cpu().numpy()[:, 0]*ctx['scale'])
                    for offset, metric in enumerate(metrics):
                        rows.append(dict(pair_id=pair['id'], geometry=pair['geometry'], split=split,
                                         sharp_z_um=pair['sharp_z_um'], blur_z_um=pair['blur_z_um'],
                                         component=COMPONENTS[start+offset], method=name,
                                         iterations=stop['iteration'] if stop else args.layers,
                                         converged=stop['converged'] if stop else None,
                                         stop_reason=stop['stop_reason'] if stop else 'fixed_depth',
                                         relative_pg=stop['components'][offset]['relative_pg'] if stop else None,
                                         batch_start=start, batch_size=len(b),
                                         median_batch_seconds=timing['median_seconds'], **metric))
            write_json(directory/f"{pair['id']}.json", dict(pair=pair, split=split, histories=details,
                        rows=[r for r in rows if r['pair_id'] == pair['id']]))
            if args.save_reconstructions:
                np.savez_compressed(directory/f"{pair['id']}.npz", sharp=sharp, blurred=blur,
                                    **{name: np.concatenate(xs) for name, xs in predictions.items()})
            write_csv(args.output/'comparison.csv', rows)
    summary = []
    for split in splits:
        for method in sorted({r['method'] for r in rows}):
            selected = [r for r in rows if r['split'] == split and r['method'] == method]
            if not selected:
                continue
            nrmse = [r['nrmse_range'] for r in selected if r['nrmse_range'] is not None]
            timings = {(r['pair_id'], r['batch_start']): r['median_batch_seconds'] for r in selected}
            summary.append(dict(split=split, method=method, component_pairs=len(selected),
                                mean_range_nrmse_percent=100*float(np.mean(nrmse)) if nrmse else None,
                                mean_rmse_uT=float(np.mean([r['rmse_uT'] for r in selected])),
                                sum_median_batch_seconds=sum(timings.values()),
                                nonconverged_component_rows=sum(r['converged'] is False for r in selected)))
    write_csv(args.output/'summary.csv', summary)
    write_json(args.output/'metrics.json', dict(
        evaluation=ctx['evaluation'], scale_uT=ctx['scale'], lambda_normalized=ctx['lam'],
        parameter_count=sum(p.numel() for p in model.parameters()),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(ctx['device']) if ctx['device'].type == 'cuda' else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(ctx['device']) if ctx['device'].type == 'cuda' else None,
        summary=summary, nonconverged_batches=nonconverged,
        timing='Same single device, float32 and component batches; warmed median; adaptive checks included. '
               'Sum of batch medians is not end-to-end dataset latency.',
        aggregation='Unweighted mean of per-component, per-pair metrics; range NRMSE fractions in comparison.csv.'))
    return nonconverged


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--operator', choices=('joint', 'analytic'), default='joint')
    p.add_argument('--validation-geometry', action='append', default=[])
    p.add_argument('--test-geometry', action='append', default=[])
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--layers', type=int, default=16)
    p.add_argument('--kernel-size', type=int, default=9)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lambda-ista', type=float, default=1e-4)
    p.add_argument('--pad', type=int, default=32)
    p.add_argument('--epsilon', type=float, default=1e-3)
    p.add_argument('--max-iterations', type=int, default=100000)
    p.add_argument('--record-every', type=int, default=100)
    p.add_argument('--benchmark-repeats', type=int, default=3)
    p.add_argument('--batch-size', type=int, choices=(1, 2, 3), default=1)
    p.add_argument('--initial', choices=('blurred', 'zero'), default='blurred')
    p.add_argument('--checkpoint-layers', action='store_true')
    p.add_argument('--save-reconstructions', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    if (args.epochs < 0 or args.layers < 1 or args.pad < 0 or args.kernel_size < 1
            or args.kernel_size % 2 != 1 or args.max_iterations < 1 or args.record_every < 1
            or args.benchmark_repeats < 1 or not 0 < args.epsilon < 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.lambda_ista) or args.lambda_ista < 0):
        p.error('Invalid numerical settings')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable: run in a GPU allocation with CUDA-enabled PyTorch')
    args.manifest, args.output = args.manifest.resolve(), args.output.resolve()
    manifest = read_manifest(args.manifest)
    splits = split_pairs(manifest, args.validation_geometry, args.test_geometry, args.operator)
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.manifest.parent
    config = {k: v for k, v in vars(args).items() if k not in ('manifest', 'output', 'resume', 'epochs')}
    sources = {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))}
    contract = dict(config=config, manifest_sha256=sha256(args.manifest), source_sha256=sources)
    scale = training_scale(root, splits['train'])
    lam = args.lambda_ista/scale
    model_config = dict(lam=lam, layers=args.layers, kernel_size=args.kernel_size,
                        initial=args.initial, checkpoint_layers=args.checkpoint_layers)
    model = SharedConvLISTA(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    evaluation = ('Geometry holdout with analytic operators; training-only normalization; '
                  'selection on validation MSE when present, otherwise training MSE.'
                  if splits['validation'] or splits['test'] else
                  'SAME-SAMPLE combined training reconstruction; no independent validation/test split.')
    if args.resume:
        saved = torch.load(args.output/'last.pt', map_location='cpu', weights_only=True)
        if saved['contract'] != contract or saved['scale_uT'] != scale:
            raise ValueError('Resume requires identical data, code and configuration (epochs may increase)')
        if args.epochs < saved['epoch']:
            raise ValueError('Requested epoch count precedes checkpoint')
        setup = json.loads((args.output/'setup.json').read_text())
        physics = setup['physics']
        for entry in physics.values():
            if sha256(args.output/entry['path']) != entry['sha256']:
                raise ValueError('Saved operator checksum mismatch')
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        best_state, best_loss, best_epoch = saved['best_state'], saved['best_loss'], saved['best_epoch']
        training, start_epoch, previous_seconds = saved['training'], saved['epoch']+1, saved['training_seconds']
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        physics = build_physics(manifest, root, manifest['pairs'], args.output/'physics', args.pad, args.operator)
        write_json(args.output/'setup.json', dict(contract=contract, model_config=model_config,
            scale_uT=scale, physics=physics, manifest=manifest, evaluation=evaluation,
            splits={name: [pair['id'] for pair in pairs] for name, pairs in splits.items()},
            environment=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                             cuda=torch.version.cuda, device=str(device),
                             gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                             slurm_job_id=os.environ.get('SLURM_JOB_ID'))))
        best_state, best_loss, best_epoch = cpu_state(model), float('inf'), 0
        training, start_epoch, previous_seconds = [], 0, 0.
    ctx = dict(load=dict(manifest=manifest, root=root, output=args.output,
                        physics=physics, pad=args.pad, device=device),
               scale=scale, lam=lam, batch_size=args.batch_size, device=device, evaluation=evaluation)
    write_json(args.output/'status.json', dict(status='training', requested_epochs=args.epochs,
               resume=args.resume, slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    synchronize(device)
    began = perf_counter()
    for epoch in range(start_epoch, args.epochs+1):
        if epoch:
            optimizer.zero_grad(set_to_none=True)
            dataset_loss(model, splits['train'], ctx, backward=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            model.project_parameters()
        train_loss = dataset_loss(model, splits['train'], ctx)
        val_loss = dataset_loss(model, splits['validation'], ctx)
        selected_loss = val_loss if val_loss is not None else train_loss
        if selected_loss < best_loss:
            best_loss, best_epoch, best_state = selected_loss, epoch, cpu_state(model)
        synchronize(device)
        seconds = previous_seconds+perf_counter()-began
        training.append(dict(epoch=epoch, train_mse_normalized=train_loss,
                             validation_mse_normalized=val_loss, training_seconds=seconds))
        saved = dict(contract=contract, model_config=model_config, scale_uT=scale,
                     model_state=cpu_state(model), optimizer_state=optimizer.state_dict(),
                     epoch=epoch, best_state=best_state, best_loss=best_loss, best_epoch=best_epoch,
                     training=training, training_seconds=seconds)
        save_checkpoint(args.output/'last.pt', saved)
        write_json(args.output/'training.json', training)
        print(f'Epoch {epoch}/{args.epochs}: train={train_loss:.8g}, validation={val_loss}, best={best_epoch}', flush=True)
    model.load_state_dict(best_state)
    save_checkpoint(args.output/'best.pt', dict(model_config=model_config, model_state=best_state,
                    scale_uT=scale, best_epoch=best_epoch, best_loss=best_loss, contract=contract))
    write_json(args.output/'status.json', dict(status='comparing', completed_epochs=args.epochs,
               best_epoch=best_epoch, slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    failed = compare(model, splits, ctx, args)
    from plot_combined import plot_results
    plot_results(args.output)
    write_json(args.output/'status.json', dict(status='baselines_nonconverged' if failed else 'complete',
                    best_epoch=best_epoch, completed_epochs=args.epochs, nonconverged_batches=failed))
    if failed:
        raise SystemExit('Comparison saved, but some classical baselines did not converge; inspect status.json')
    print(f'Complete: {args.output}', flush=True)


if __name__ == '__main__':
    main()
