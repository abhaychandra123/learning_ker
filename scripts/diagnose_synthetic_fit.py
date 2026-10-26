"""Read-only full-resolution probes of fit, forward mismatch and iteration budget.

No training, parameter selection, or checkpoint edits. Bz-only diagnostic cases
are intentionally small in number and do not replace the full evaluation.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from combined_lista import SharedConvLISTA, load_shared_checkpoint
from synthetic_data import digest, propagate, write_json
from synthetic_experiment import field_metrics
from synthetic_fast import RealAnalyticFFT
from torch_inverse import solve_torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/synthetic_data_hypothesis_fast_v2/augmented_1080'))
    parser.add_argument('--actual-run', type=Path, default=Path('runs/combined_full_200'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh diagnostic output directory')
    torch.set_num_threads(4)
    setup = json.loads((args.run/'setup.json').read_text())
    model, saved = load_shared_checkpoint(args.run/'final.pt')
    for key in ('contract', 'model_config', 'scale_uT'):
        if saved[key] != setup[key]:
            raise ValueError('Checkpoint/setup mismatch')
    hashes = setup['contract']['actual_sha256']
    if digest(args.actual_run/'setup.json') != hashes['setup.json']:
        raise ValueError('Actual setup mismatch')
    manifest = json.loads((args.actual_run/'setup.json').read_text())['manifest']
    initial = SharedConvLISTA(**saved['model_config']).eval()
    scale, pad = saved['scale_uT'], setup['pad']
    selected = {'L1_C1_z01.0_to_z00.5','L1_C1_z06.0_to_z00.5','L1_C1_z10.0_to_z00.5',
                'IC_2um_z02.0_to_z01.0','IC_4um_z10.0_to_z00.5'}
    result = dict(checkpoint_sha256=digest(args.run/'final.pt'), epoch=saved['epoch'],
        setup_sha256=digest(args.run/'setup.json'), pad=pad, component='Bz',
        parameter_count=sum(p.numel() for p in model.parameters()),
        parameters={name:dict(min=float(t.min()), max=float(t.max()), norm=float(t.norm()), sum=float(t.sum()))
                    for name,t in saved['model_state'].items()},
        forward_checks=[], reconstructions=[], source_sha256={
            name:digest(Path(__file__).parent/name) for name in
            ('diagnose_synthetic_fit.py','synthetic_fast.py','combined_lista.py','torch_inverse.py')},
        limits='Selected Bz cases, one trained seed; clean inputs regenerated with the original NumPy propagator. '
               'All diagnostic inverse methods use the same real-FFT analytical operator and pad. '
               'ISTA/FISTA use the original physical lambda; the trained model retains its learned threshold. '
               'Fixed-iteration FISTA probes are not convergence or optimality claims. No hyperparameter selection.')
    args.output.mkdir(parents=True)
    start = perf_counter()
    with torch.inference_mode():
        for pair in manifest['pairs']:
            if pair['id'] not in selected:
                continue
            print('Probe '+pair['id']+' Bz', flush=True)
            name = f"comparison/{pair['id']}.npz"
            if digest(args.actual_run/name) != hashes[name]:
                raise ValueError('Actual archive mismatch')
            with np.load(args.actual_run/name) as archive:
                truth, actual = archive['sharp'][2].copy(), archive['blurred'][2].copy()
            grid = manifest['geometries'][pair['geometry']]['grid']
            clean = propagate(truth, pair['dz_um'], grid['dx'], grid['dy'], pad).astype(np.float32)
            mismatch = dict(pair_id=pair['id'], **field_metrics(clean, actual),
                            archive_sha256=hashes[name])
            result['forward_checks'].append(mismatch)
            print(f"  forward relative mismatch: {100*mismatch['relative_l2']:.3f}%", flush=True)
            operator = RealAnalyticFFT(truth.shape, grid, pair['dz_um'], pad)
            for domain, observed in [('actual_comsol',actual),('clean_synthetic',clean)]:
                b = torch.from_numpy(observed/scale)[None,None]
                def record(method, prediction):
                    metrics = field_metrics(prediction, truth)
                    result['reconstructions'].append(dict(pair_id=pair['id'], domain=domain,
                        method=method, **metrics))
                    print(f"  {domain} {method}: relative L2={100*metrics['relative_l2']:.2f}%, slope={metrics['amplitude_slope']:.4f}", flush=True)
                    write_json(args.output/'diagnostics.json', result)
                record('input', observed)
                trained = model(b, operator)[0,0].numpy()*scale
                record('trained_16', trained)
                record('initial_ista_16', initial(b,operator)[0,0].numpy()*scale)
                if pair['id']=='L1_C1_z10.0_to_z00.5':
                    for iterations in (16,64,256):
                        prediction, history, seconds = solve_torch(b, operator, saved['model_config']['lam'],
                            method='fista', initial='blurred', n_iter=iterations,
                            record_every=iterations, diagnostics=False, scale=scale)
                        record(f'fista_{iterations}', prediction[0,0].numpy()*scale)
                        result['reconstructions'][-1].update(seconds=seconds,
                            relative_pg=history[-1]['components'][0]['relative_pg'], stop_reason='fixed_budget')
            del operator
    result['diagnostic_wall_seconds'] = perf_counter()-start
    write_json(args.output/'diagnostics.json', result)
    print(f'Complete: {args.output}/diagnostics.json', flush=True)


if __name__ == '__main__':
    main()
