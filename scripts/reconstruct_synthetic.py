"""Fresh local inference from a synthetic run's final.pt on an actual COMSOL pair."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from reconstruct_combined import BASELINES, COMPONENTS, fields, render, select_pair, sha256, write_json
from synthetic_data import SyntheticDataset, transfer
from synthetic_experiment import field_metrics


def add_saved_baselines(actual_root, arrays, rows, provenance):
    """Attach historical classical predictions without changing the new model output."""
    setup_path = actual_root/'setup.json'
    setup = json.loads(setup_path.read_text())
    _, pair = select_pair(setup, provenance['geometry'], provenance['standoff_um'],
                          provenance['target_standoff_um'])
    if pair is None:
        raise ValueError('No matching baseline pair')
    grid = setup['manifest']['geometries'][provenance['geometry']]['grid']
    if grid != provenance['grid']:
        raise ValueError('Baseline grid differs from reconstruction')
    archive_path = (actual_root/f"comparison/{pair['id']}.npz").resolve()
    if not archive_path.is_relative_to(actual_root.resolve()):
        raise ValueError('Archive path escapes actual run')
    if sha256(archive_path) != provenance['archive_sha256']:
        raise ValueError('Baseline archive differs from reconstructed input')
    components = provenance['components']
    with np.load(archive_path, allow_pickle=False) as archive:
        for source, name in [('sharp', 'reference'), ('blurred', 'input')]:
            if not np.array_equal(fields(archive[source], grid, components), arrays[name]):
                raise ValueError('Baseline input/reference differs from reconstruction')
        baseline_arrays = {name: fields(archive[name], grid, components) for name in BASELINES}
    csv_path = actual_root/'comparison.csv'
    with csv_path.open() as stream:
        entries = [r for r in csv.DictReader(stream) if r['pair_id']==pair['id'] and r['method'] in BASELINES]
    lookup = {(r['component'], r['method']):r for r in entries}
    if len(lookup) != len(entries) or any((c, m) not in lookup for c in components for m in BASELINES):
        raise ValueError('Missing or duplicate baseline convergence metadata')
    result_rows = [r for r in rows if r['method'] not in BASELINES]
    for i, component in enumerate(components):
        for method in BASELINES:
            original = lookup[component, method]
            flag = original['converged'].lower()
            if flag not in ('', 'true', 'false'):
                raise ValueError('Invalid baseline convergence flag')
            m = field_metrics(baseline_arrays[method][i], arrays['reference'][i])
            result_rows.append(dict(component=component, method=method, rmse_uT=m['rmse_uT'],
                range_nrmse_percent=100*m['nrmse_range'] if m['nrmse_range'] is not None else None,
                relative_l2=m['relative_l2'], amplitude_slope=m['amplitude_slope'],
                converged={'true':True, 'false':False}.get(flag), stop_reason=original['stop_reason'],
                inference_seconds=None, reused=True, iterations=int(original['iterations'])))
    config = setup['contract']['config']
    note = (f"Historical saved ISTA/FISTA: {config['operator']} operator, pad {config['pad']}. "
            f"New ConvLISTA: analytical operator, pad {provenance['operator']['pad']}. "
            'Different operator settings; contextual comparison, not a matched-operator solver benchmark. '
            'No classical solver rerun; saved iteration counts and convergence flags retained.')
    if provenance.get('baseline_note'):
        provenance['interpretation'] = provenance['interpretation'].replace(' '+provenance['baseline_note'], '')
    provenance['interpretation'] += ' '+note
    provenance['baseline_note'] = note
    provenance['baselines'] = dict(source_run=str(actual_root.resolve()), setup_sha256=sha256(setup_path),
        archive_sha256=sha256(archive_path), convergence_csv_sha256=sha256(csv_path),
        operator=config['operator'], pad=config['pad'], methods=list(BASELINES), reused=True)
    arrays.update(baseline_arrays)
    return result_rows


def save_metrics(output, rows):
    write_json(output/'metrics.json', dict(rows=rows))
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def replot_with_baselines(args):
    output = args.replot
    status = json.loads((output/'status.json').read_text())
    if status['status'] != 'complete':
        raise ValueError('Replot requires a completed report')
    provenance = json.loads((output/'provenance.json').read_text())
    if provenance['mode'] != 'fresh synthetic final checkpoint inference':
        raise ValueError('Expected a synthetic reconstruction report')
    rows = json.loads((output/'metrics.json').read_text())['rows']
    with np.load(output/'fields_uT.npz', allow_pickle=False) as saved:
        arrays = {name:saved[name] for name in saved.files}
    if args.include_saved_baselines:
        rows = add_saved_baselines(args.actual_run, arrays, rows, provenance)
    render(output, arrays, provenance['components'], provenance['grid'], rows, provenance, args.color_scale)
    np.savez_compressed(output/'fields_uT.npz', **arrays)
    save_metrics(output, rows)
    write_json(output/'provenance.json', provenance)
    status['baseline_nonconverged_rows'] = sum(r['converged'] is False for r in rows)
    write_json(output/'status.json', status)
    print(f"Updated report without inference: {(output/'index.html').resolve()}")


def run(args):
    import torch
    from combined_lista import load_shared_checkpoint
    from torch_inverse import PaddedFFT

    if args.output.exists():
        raise FileExistsError('Output exists; choose a new --output directory')
    checkpoint = args.run/'final.pt'
    setup_path = args.run/'setup.json'
    if not checkpoint.is_file() or not setup_path.is_file():
        raise FileNotFoundError('Copy the new synthetic arm final.pt and setup.json from the server first')
    checkpoint_hash = sha256(checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_hash != args.expected_checkpoint_sha256:
        raise ValueError('Checkpoint checksum does not match the selected experiment')
    setup = json.loads(setup_path.read_text())
    contract = setup['contract']
    dataset = SyntheticDataset(args.dataset)
    if sha256(args.dataset/'manifest.json') != contract['dataset_sha256']:
        raise ValueError('Synthetic dataset manifest differs from training')
    if setup['pad'] != dataset.manifest['boundary']['pad_pixels']:
        raise ValueError('Padding differs from training dataset')
    if setup['scale_uT'] != dataset.scale:
        raise ValueError('Normalization differs from training dataset')
    actual_setup_path = args.actual_run/'setup.json'
    if sha256(actual_setup_path) != contract['actual_sha256']['setup.json']:
        raise ValueError('Actual COMSOL setup differs from the evaluated source')
    actual_setup = json.loads(actual_setup_path.read_text())
    manifest = actual_setup['manifest']
    if (manifest['field_unit'] != 'uT' or manifest['length_unit'] != 'um'
            or manifest['components'] != list(COMPONENTS)):
        raise ValueError('Unsupported source units or vector convention')
    target, pair = select_pair(actual_setup, args.geometry, args.standoff, args.target_standoff)
    if pair is None:
        raise ValueError('No actual COMSOL pair at the requested heights')
    name = f"comparison/{pair['id']}.npz"
    archive_path = (args.actual_run/name).resolve()
    if not archive_path.is_relative_to(args.actual_run.resolve()):
        raise ValueError('Archive path escapes actual run')
    if sha256(archive_path) != contract['actual_sha256'][name]:
        raise ValueError('COMSOL archive differs from the evaluated source')
    grid = manifest['geometries'][args.geometry]['grid']
    components = COMPONENTS if args.component == 'all' else (args.component,)
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = dict(input=fields(archive['blurred'], grid, components),
                      reference=fields(archive['sharp'], grid, components))
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu')
    torch.set_num_threads(args.threads)
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model, saved = load_shared_checkpoint(checkpoint, device)
    for key in ('contract', 'model_config', 'scale_uT'):
        if saved[key] != setup[key]:
            raise ValueError(f'Checkpoint/setup mismatch: {key}')
    pad = setup['pad']
    # Match the completed experiment's final evaluator: full-complex analytical FFT.
    h = transfer(arrays['input'].shape[-2:], grid['dx'], grid['dy'], args.standoff-target, pad)
    operator = PaddedFFT(h, arrays['input'].shape[-2:], pad).to(device)
    scale = saved['scale_uT']
    predictions, times = [], []
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'status.json', dict(status='running'))
    try:
        with torch.inference_mode():
            for i, component in enumerate(components):
                print(f'Reconstructing {args.geometry} {component}: {args.standoff:g} -> {target:g} µm on {device}', flush=True)
                # NumPy normalization matches the final actual-COMSOL evaluator.
                b = torch.tensor(arrays['input'][i]/scale, dtype=torch.float32, device=device)[None, None]
                if device == 'cuda': torch.cuda.synchronize()
                start = perf_counter()
                prediction = model(b, operator)[0, 0].cpu().numpy()*scale
                times.append(perf_counter()-start)
                if not np.isfinite(prediction).all():
                    raise FloatingPointError('Nonfinite reconstruction')
                predictions.append(prediction)
                print(f'  completed in {times[-1]:.2f} s', flush=True)
        arrays['lista'] = np.stack(predictions)
        rows = []
        for i, component in enumerate(components):
            for method in ('input', 'lista'):
                m = field_metrics(arrays[method][i], arrays['reference'][i])
                rows.append(dict(component=component, method=method, rmse_uT=m['rmse_uT'],
                    range_nrmse_percent=100*m['nrmse_range'] if m['nrmse_range'] is not None else None,
                    relative_l2=m['relative_l2'], amplitude_slope=m['amplitude_slope'],
                    converged=None, stop_reason=None, inference_seconds=times[i] if method=='lista' else None,
                    reused=False))
        provenance = dict(geometry=args.geometry, standoff_um=args.standoff, target_standoff_um=target,
            components=list(components), grid=grid, field_unit='uT', mode='fresh synthetic final checkpoint inference',
            interpretation='Fresh inference with the synthetic-trained final model on an actual COMSOL input. '
                'Same source geometry and sharp target as training; not an unseen-geometry test. '
                'Analytical pad and normalization match the completed experiment. Reference is actual COMSOL data.',
            source_run=str(args.run.resolve()), checkpoint_sha256=checkpoint_hash,
            setup_sha256=sha256(setup_path), archive_sha256=sha256(archive_path),
            dataset_manifest_sha256=contract['dataset_sha256'], epoch=saved['epoch'],
            scale_uT=scale, device=device, torch=str(torch.__version__),
            operator=dict(kind='analytic_full_fft', pad=pad, dz_um=args.standoff-target),
            timing_note='Per-component inference plus output transfer; no training or cache preparation.')
        if args.include_saved_baselines:
            rows = add_saved_baselines(args.actual_run, arrays, rows, provenance)
        np.save(args.output/'reconstruction_uT.npy', arrays['lista'])
        np.savez_compressed(args.output/'fields_uT.npz', **arrays)
        save_metrics(args.output, rows)
        write_json(args.output/'provenance.json', provenance)
        render(args.output, arrays, components, grid, rows, provenance, args.color_scale)
        write_json(args.output/'status.json', dict(status='complete', epoch=saved['epoch'], mode=provenance['mode'],
            baseline_nonconverged_rows=sum(r['converged'] is False for r in rows)))
        print(f"Report: {(args.output/'index.html').resolve()}", flush=True)
        return arrays, rows
    except Exception as exc:
        write_json(args.output/'status.json', dict(status='failed', error=str(exc)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/synthetic_data_hypothesis_fast_v2/augmented_1080'))
    parser.add_argument('--dataset', type=Path, default=Path('runs/synthetic_v1'))
    parser.add_argument('--actual-run', type=Path, default=Path('runs/combined_full_200'))
    parser.add_argument('--geometry', default='L1_C1')
    parser.add_argument('--standoff', type=float, default=10.)
    parser.add_argument('--target-standoff', type=float, default=.5)
    parser.add_argument('--component', choices=('all', *COMPONENTS), default='all')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--expected-checkpoint-sha256')
    parser.add_argument('--include-saved-baselines', action='store_true', help='Include historical ISTA/FISTA with their original operator settings')
    parser.add_argument('--replot', type=Path, help='Update a completed synthetic report without rerunning inference')
    parser.add_argument('--color-scale', choices=('both', 'shared', 'independent'), default='both')
    parser.add_argument('--output', type=Path, default=Path('runs/reconstruction_synthetic_L1_C1_10um'))
    args = parser.parse_args(argv)
    if args.threads < 1: parser.error('--threads must be positive')
    if args.replot:
        return replot_with_baselines(args)
    return run(args)


if __name__ == '__main__':
    main()
