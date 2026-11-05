"""Run trained 16-layer ConvLISTA checkpoints at greater depth, without retraining.

The 163 parameters are tied across layers, so a deeper network repeats the same
learned update. This is an inference-only depth extrapolation, NOT a model
trained at 32/64 layers. Matched-depth ISTA/FISTA use the same operator,
normalization, blurred initialization and lambda.

Example (GPU server):
    python scripts/depth_extension.py --model combined --device cuda --output runs/depth_extension_v1/combined
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np

# Experiment copies of this script read data from the main project via QDM_PROJECT_ROOT.
ROOT = Path(os.environ.get('QDM_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
COMPONENTS = ('Bx', 'By', 'Bz')
RUNS = dict(
    combined=dict(run=ROOT/'runs/combined_full_200', checkpoint='best.pt'),
    synthetic=dict(run=ROOT/'runs/synthetic_data_hypothesis_fast_v2/augmented_1080', checkpoint='final.pt'),
)
ACTUAL = ROOT/'runs/combined_full_200'
SAVE_PAIRS = ('L1_C1_z06.0_to_z00.5', 'L1_C1_z10.0_to_z00.5')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def field_metrics(prediction, reference):
    if not np.isfinite(prediction).all():
        return dict(rmse_uT=math.nan, nrmse_range=math.nan, relative_l2=math.nan, amplitude_slope=math.nan, finite=False)
    error = prediction.astype(np.float64)-reference
    rmse = float(np.sqrt(np.mean(error**2)))
    return dict(rmse_uT=rmse, nrmse_range=rmse/float(reference.max()-reference.min()),
                relative_l2=float(np.linalg.norm(error)/np.linalg.norm(reference)),
                amplitude_slope=float(np.sum(prediction*reference)/np.sum(reference*reference)), finite=True)


def operators(kind, actual_setup, pair, run_setup):
    """Yield the operator used by each completed experiment for this pair."""
    from torch_inverse import PaddedFFT
    shape = (actual_setup['manifest']['geometries'][pair['geometry']]['grid']['ny'],
             actual_setup['manifest']['geometries'][pair['geometry']]['grid']['nx'])
    if kind == 'combined':
        entry = actual_setup['physics'][pair['id']]
        path = ACTUAL/entry['path']
        if sha256(path) != entry['sha256']:
            raise ValueError(f'Operator checksum mismatch: {pair["id"]}')
        pad = actual_setup['contract']['config']['pad']
        return PaddedFFT(np.load(path), shape, pad), pad
    from synthetic_data import transfer
    grid = actual_setup['manifest']['geometries'][pair['geometry']]['grid']
    pad = run_setup['pad']
    h = transfer(shape, grid['dx'], grid['dy'], pair['blur_z_um']-pair['sharp_z_um'], pad)
    return PaddedFFT(h, shape, pad), pad


def depth_runs(model, b, operator, lam, depths):
    """Return {method: {depth: normalized prediction}} from single deepest runs."""
    import torch
    from torch_inverse import shrink
    step = .99/operator.bound
    keep = set(depths)
    out = {m: {} for m in ('lista', 'ista', 'fista')}
    # Learned update, exactly as SharedConvLISTA.forward, recorded at each depth.
    x = b.clone()
    correction_b = model.input_correction(b)
    tau = step*model.threshold.clamp_min(0)
    for k in range(1, max(depths)+1):
        x = shrink(x-step*operator.adjoint(operator(x)-b)+correction_b+model.state_correction(x), tau)
        if k in keep:
            out['lista'][k] = x[0, 0].cpu().numpy()
    # Fixed-depth ISTA/FISTA, as solve_torch(n_iter=k) with blurred start.
    x = b.clone()
    for k in range(1, max(depths)+1):
        x = shrink(x-step*operator.adjoint(operator(x)-b), step*lam)
        if k in keep:
            out['ista'][k] = x[0, 0].cpu().numpy()
    x, y, t = b.clone(), b.clone(), 1.
    for k in range(1, max(depths)+1):
        new_x = shrink(y-step*operator.adjoint(operator(y)-b), step*lam)
        next_t = (1+math.sqrt(1+4*t*t))/2
        y = new_x+(t-1)/next_t*(new_x-x)
        x, t = new_x, next_t
        if k in keep:
            out['fista'][k] = x[0, 0].cpu().numpy()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=sorted(RUNS), required=True)
    parser.add_argument('--run', type=Path, help='override the checkpoint directory for --model')
    parser.add_argument('--expected-checkpoint-sha256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depths', type=int, nargs='+', default=[8, 16, 24, 32, 40, 48, 56, 64])
    parser.add_argument('--geometry', action='append', help='restrict to geometry (repeatable)')
    parser.add_argument('--lambda-ista', type=float, default=1e-4, help='classical lambda in µT')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args(argv)
    import torch
    from combined_lista import load_shared_checkpoint

    if args.output.exists():
        raise FileExistsError('Output exists; choose a new --output directory')
    if 16 not in args.depths or min(args.depths) < 1:
        raise ValueError('Depths must be positive and include the trained depth 16')
    depths = sorted(set(args.depths))
    torch.set_num_threads(args.threads)
    device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; use --device cpu')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    spec = RUNS[args.model]
    run_dir = args.run or spec['run']
    checkpoint = run_dir/spec['checkpoint']
    if args.expected_checkpoint_sha256 and sha256(checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError('Checkpoint checksum does not match')
    actual_setup = json.loads((ACTUAL/'setup.json').read_text())
    run_setup = json.loads((run_dir/'setup.json').read_text())
    model, saved = load_shared_checkpoint(checkpoint, device)
    if saved['model_config']['layers'] != 16 or saved['scale_uT'] != run_setup['scale_uT']:
        raise ValueError('Unexpected checkpoint depth or normalization')
    if args.model == 'synthetic':
        contract = run_setup['contract']
        if sha256(ACTUAL/'setup.json') != contract['actual_sha256']['setup.json']:
            raise ValueError('Actual COMSOL setup differs from the synthetic evaluation source')
    scale = saved['scale_uT']
    lam = args.lambda_ista/scale
    pairs = [p for p in actual_setup['manifest']['pairs'] if not args.geometry or p['geometry'] in args.geometry]

    args.output.mkdir(parents=True)
    rows, checks, saved_arrays = [], [], {}
    start = perf_counter()
    with torch.inference_mode():
        for pair in pairs:
            archive_path = ACTUAL/f"comparison/{pair['id']}.npz"
            if args.model == 'synthetic' and sha256(archive_path) != run_setup['contract']['actual_sha256'][f"comparison/{pair['id']}.npz"]:
                raise ValueError(f'Archive differs from synthetic evaluation source: {pair["id"]}')
            with np.load(archive_path, allow_pickle=False) as archive:
                blurred, sharp = archive['blurred'], archive['sharp']
                archived_lista = archive['lista'] if args.model == 'combined' else None
            operator, pad = operators(args.model, actual_setup, pair, run_setup)
            operator = operator.to(device)
            for i, component in enumerate(COMPONENTS):
                t0 = perf_counter()
                b = torch.tensor(blurred[i]/scale, dtype=torch.float32, device=device)[None, None]
                runs = depth_runs(model, b, operator, lam, depths)
                for method, by_depth in runs.items():
                    for depth, x in by_depth.items():
                        prediction = x*scale
                        m = field_metrics(prediction, sharp[i].astype(np.float64))
                        rows.append(dict(model=args.model, pair_id=pair['id'], geometry=pair['geometry'],
                                         blur_z_um=pair['blur_z_um'], sharp_z_um=pair['sharp_z_um'],
                                         component=component, method=method, depth=depth,
                                         trained_depth=(method == 'lista' and depth == 16), **m))
                        if pair['id'] in SAVE_PAIRS and depth in (16, 32, 64):
                            saved_arrays[f"{pair['id']}__{component}__{method}__{depth}"] = prediction.astype(np.float32)
                if archived_lista is not None:
                    diff = runs['lista'][16]*scale-archived_lista[i]
                    checks.append(dict(pair_id=pair['id'], component=component,
                                       max_abs_diff_uT=float(np.abs(diff).max()),
                                       relative_l2_diff=float(np.linalg.norm(diff)/np.linalg.norm(archived_lista[i]))))
                print(f"{pair['id']} {component}: {perf_counter()-t0:.1f} s", flush=True)

    with open(args.output/'metrics.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output/'selected_predictions_uT.npz', **saved_arrays)
    summary = {}
    for method in ('lista', 'ista', 'fista'):
        for depth in depths:
            sel = [r for r in rows if r['method'] == method and r['depth'] == depth]
            summary[f'{method}_{depth}'] = dict(
                components=len(sel), nonfinite=sum(not r['finite'] for r in sel),
                mean_nrmse_percent=100*float(np.mean([r['nrmse_range'] for r in sel])),
                mean_relative_l2_percent=100*float(np.mean([r['relative_l2'] for r in sel])),
                mean_amplitude_slope=float(np.mean([r['amplitude_slope'] for r in sel])))
    report = dict(
        model=args.model, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        epoch=saved.get('epoch'), trained_layers=16, depths=depths, scale_uT=scale,
        learned_threshold_normalized=float(model.threshold), lambda_classical_uT=args.lambda_ista,
        operator='joint fitted, pad 32' if args.model == 'combined' else f'analytic full FFT, pad {run_setup["pad"]}',
        interpretation='Inference-only depth extrapolation of tied weights trained at 16 layers. '
                       'Same-sample COMSOL pairs. Not a model trained at the larger depth.',
        dtype='float32', device=device, gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
        torch=str(torch.__version__), seconds=perf_counter()-start,
        reproduction_check_vs_archived_16_layer=checks or None, summary=summary)
    (args.output/'summary.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
