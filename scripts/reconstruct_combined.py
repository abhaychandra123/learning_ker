"""Reconstruct one standoff with a trained shared ConvLISTA and write a report.

Inputs are calibrated magnetic fields in microtesla, not rendered images.
Use --saved-results explicitly to report existing predictions without inference.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
from time import perf_counter

import numpy as np


COMPONENTS = ('Bx', 'By', 'Bz')
BASELINES = ('ista_16', 'fista_16', 'ista_to_epsilon', 'fista_to_epsilon')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def local_path(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Source path escapes the run directory')
    return path


def metrics(prediction, reference):
    if reference is None:
        return dict(rmse_uT=None, range_nrmse_percent=None)
    residual = prediction.astype(np.float64)-reference.astype(np.float64)
    rmse = float(np.sqrt(np.mean(residual**2)))
    span = float(np.ptp(reference.astype(np.float64)))
    return dict(rmse_uT=rmse, range_nrmse_percent=100*rmse/span if span else None)


def fields(array, grid, components):
    array = np.asarray(array)
    shape = (grid['ny'], grid['nx'])
    if array.shape == (3, *shape):
        array = array[[COMPONENTS.index(c) for c in components]]
    elif array.shape == shape and len(components) == 1:
        array = array[None]
    else:
        raise ValueError('Expected [3,ny,nx], or [ny,nx] with one --component, on the selected geometry grid')
    array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError('Fields must be finite float32 values in µT')
    return array


def select_pair(setup, geometry, standoff, target):
    manifest = setup['manifest']
    if geometry not in manifest['geometries']:
        raise ValueError('Unknown geometry; use --list')
    pairs = [p for p in manifest['pairs'] if p['geometry'] == geometry]
    if target is None:
        anchors = {p['sharp_z_um'] for p in pairs}
        if len(anchors) != 1:
            raise ValueError('Specify --target-standoff; multiple reference heights exist')
        target = next(iter(anchors))
    if not np.isfinite([standoff, target]).all() or target < 0 or standoff <= target:
        raise ValueError('Require finite input standoff > target standoff >= 0')
    matches = [p for p in pairs if np.isclose(p['blur_z_um'], standoff, rtol=0, atol=1e-9)
               and np.isclose(p['sharp_z_um'], target, rtol=0, atol=1e-9)]
    if len(matches) > 1:
        raise ValueError('Ambiguous pair')
    return target, matches[0] if matches else None


def infer(run, setup, observed, transfer, pad, device):
    checkpoint = run/'best.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'{checkpoint} is missing. Run on the training machine or copy best.pt. '
                                'Use --saved-results only to view existing predictions.')
    import torch
    from combined_lista import load_shared_checkpoint
    from torch_inverse import PaddedFFT
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu or the GPU machine')
    model, saved = load_shared_checkpoint(checkpoint, device=device)
    if (saved['contract'] != setup['contract'] or saved['model_config'] != setup['model_config']
            or saved['scale_uT'] != setup['scale_uT']):
        raise ValueError('Checkpoint does not match this run setup')
    scale = saved['scale_uT']
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid checkpoint normalization scale')
    operator = PaddedFFT(transfer, observed.shape[-2:], pad).to(device)
    predicted = np.empty_like(observed); seconds = []
    with torch.inference_mode():
        for i, field in enumerate(observed):
            b = torch.from_numpy(field)[None, None].to(device)/scale
            if device == 'cuda': torch.cuda.synchronize()
            start = perf_counter()
            result = model(b, operator)
            if device == 'cuda': torch.cuda.synchronize()
            seconds.append(perf_counter()-start)
            predicted[i] = (result[0, 0]*scale).cpu().numpy()
    if not np.isfinite(predicted).all():
        raise ValueError('Model produced nonfinite reconstructions')
    return predicted, seconds, dict(device=device, torch=torch.__version__,
                                    checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                                    best_epoch=saved['best_epoch'], scale_uT=scale)


def render(output, arrays, components, grid, rows, provenance, color_scale='both'):
    if color_scale == 'both':
        render(output, arrays, components, grid, rows, provenance, 'shared')
        render(output, arrays, components, grid, rows, provenance, 'independent')
        page = output/'index.html'
        shared = ''.join(f'<a href="figures/{c}{suffix}_shared.png"><img loading="lazy" '
                         f'src="figures/{c}{suffix}_shared.png"></a>'
                         for c in components for suffix in (('', '_baselines') if any(n in arrays for n in BASELINES) else ('',)))
        page.write_text(page.read_text()+'<details><summary>Show shared scales for amplitude comparison</summary>'
                       +shared+'</details>')
        write_json(output/'plot_settings.json', dict(color_scale='both', primary='independent'))
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = output/'figures'; figures.mkdir(exist_ok=True)
    gallery = []
    extent = [grid['x0']-grid['dx']/2, grid['x0']+(grid['nx']-.5)*grid['dx'],
              grid['y0']-grid['dy']/2, grid['y0']+(grid['ny']-.5)*grid['dy']]
    lookup = {(r['component'], r['method']): r for r in rows}
    def title(method, component):
        label = {'reference': 'Reference', 'input': 'Blurred input', 'lista': 'ConvLISTA'}.get(method, method)
        row = lookup.get((component, method))
        if row and row['range_nrmse_percent'] is not None:
            label += f'\nNRMSE {row["range_nrmse_percent"]:.3f}%'
        if row and row['converged'] is False:
            label += '\nNOT CONVERGED'
        return label
    for c, component in enumerate(components):
        names = (['reference'] if 'reference' in arrays else [])+['input', 'lista']
        with_ref = 'reference' in arrays
        fig, axes = plt.subplots(1, len(names)+int(with_ref), figsize=(5*(len(names)+int(with_ref)), 5.8),
                                 constrained_layout=True, squeeze=False)
        axes = axes[0]
        limit = max(float(np.abs(arrays[n][c]).max()) for n in names) or 1.
        for ax, name in zip(axes, names):
            bound = (float(np.abs(arrays[name][c]).max()) or 1.) if color_scale == 'independent' else limit
            im = ax.imshow(arrays[name][c], origin='lower', extent=extent, cmap='RdBu_r', vmin=-bound, vmax=bound)
            ax.set_title(title(name, component))
            if color_scale == 'independent':
                fig.colorbar(im, ax=ax, label=f'{component} (µT)', shrink=.8)
        if color_scale == 'shared':
            fig.colorbar(im, ax=list(axes[:len(names)]), label=f'{component} (µT)', shrink=.8)
        if with_ref:
            residual = arrays['lista'][c].astype(np.float64)-arrays['reference'][c]
            bound = float(np.abs(residual).max()) or 1.
            im = axes[-1].imshow(residual, origin='lower', extent=extent, cmap='RdBu_r', vmin=-bound, vmax=bound)
            axes[-1].set_title('ConvLISTA − reference\nSeparate residual scale')
            fig.colorbar(im, ax=axes[-1], label='Error (µT)', shrink=.8)
        for ax in axes: ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
        fig.suptitle(f'{provenance["geometry"]}, {component}: {provenance["standoff_um"]:g} → '
                     f'{provenance["target_standoff_um"]:g} µm — {provenance["mode"]}\n'
                     f'{color_scale.capitalize()} field scales; last panel is signed reconstruction error')
        name = f'{component}_{color_scale}.png'; fig.savefig(figures/name, dpi=140); plt.close(fig); gallery.append(name)
        baselines = [n for n in BASELINES if n in arrays]
        if baselines:
            names = ['reference', 'lista']+baselines
            fig, axes = plt.subplots(2, 3, figsize=(16, 11), constrained_layout=True)
            bound = max(float(np.abs(arrays[n][c]).max()) for n in names) or 1.
            for ax, n in zip(axes.flat, names):
                limit = (float(np.abs(arrays[n][c]).max()) or 1.) if color_scale == 'independent' else bound
                im = ax.imshow(arrays[n][c], origin='lower', extent=extent, cmap='RdBu_r', vmin=-limit, vmax=limit)
                ax.set_title(title(n, component)); ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
                if color_scale == 'independent': fig.colorbar(im, ax=ax, label=f'{component} (µT)', shrink=.8)
            if color_scale == 'shared': fig.colorbar(im, ax=list(axes.flat), label=f'{component} (µT)', shrink=.8)
            fig.suptitle(f'{component}: saved classical baselines; {color_scale} scales; no classical solver rerun')
            name = f'{component}_baselines_{color_scale}.png'; fig.savefig(figures/name, dpi=140); plt.close(fig); gallery.append(name)
    diagnostics = []
    for c, component in enumerate(components):
        truth = arrays['reference'][c].astype(np.float64) if 'reference' in arrays else None
        norm = np.linalg.norm(truth) if truth is not None else 0.
        for method, a in arrays.items():
            field = a[c].astype(np.float64)
            diagnostics.append(dict(component=component, method=method, min_uT=float(field.min()),
                                    max_uT=float(field.max()), relative_l2_error_percent=
                                    float(100*np.linalg.norm(field-truth)/norm) if norm else None))
    write_json(output/'plot_diagnostics.json', dict(definition='relative L2 error (%) = 100 * norm(pred-reference) / norm(reference)', rows=diagnostics))
    diagnostic_table = ''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k]) if r[k] is not None else "unavailable")}</td>'
                               for k in ('component','method','min_uT','max_uT','relative_l2_error_percent'))+'</tr>' for r in diagnostics)
    body = ''.join(f'<a href="figures/{name}"><img src="figures/{name}" loading="lazy"></a>' for name in gallery)
    table = ''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k]) if r[k] is not None else "unavailable")}</td>'
                                for k in ('component','method','rmse_uT','range_nrmse_percent','converged','stop_reason'))+'</tr>' for r in rows)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>ConvLISTA reconstruction</title>'
        '<style>body{font:17px system-ui;margin:30px auto;max-width:1700px;padding:20px}img{width:100%;margin:20px 0}'
        'td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style>'
        '<h1>ConvLISTA reconstruction</h1><p>'+html.escape(provenance['interpretation'])+'</p>'
        '<p>Fields are in µT. NRMSE (%) = 100 × RMSE / reference max–min range. '
        'Undefined for a constant or absent reference. Reference fields are loaded, never generated by the model. '
        f'Current view: <strong>{color_scale} field scales</strong>; residuals have their own scale. '
        'Independent scales reveal structure but equal colours do not mean equal amplitudes across panels. '
        'Colour scaling does not change any reconstruction or error metric. The last panel is reconstruction minus reference, '
        'not another reconstruction. Positive/negative errors appear red/blue.</p>'
        '<p><a href="metrics.csv">Metrics CSV</a> · <a href="provenance.json">Run details</a> · '
        '<a href="reconstruction_uT.npy">Reconstruction array</a></p>'
        '<table><tr><th>Component</th><th>Method</th><th>RMSE µT</th><th>Range NRMSE %</th>'
        '<th>Converged</th><th>Stop reason</th></tr>'+table+'</table>'
        '<h2>Amplitude and relative error</h2><p>Relative L2 error = 100 × ||prediction − reference|| / ||reference||. '
        'This uses reference energy rather than its max–min range. A small range NRMSE is not a percentage accuracy score.</p>'
        '<table><tr><th>Component</th><th>Field</th><th>Min µT</th><th>Max µT</th><th>Relative L2 error %</th></tr>'
        +diagnostic_table+'</table>'+body)
    write_json(output/'plot_settings.json', dict(color_scale=color_scale))


def replot(output, color_scale):
    """Redraw a completed output without touching arrays, scores or inference provenance."""
    output = output.resolve()
    if json.loads((output/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Replot requires a completed reconstruction report')
    provenance = json.loads((output/'provenance.json').read_text())
    rows = json.loads((output/'metrics.json').read_text())['rows']
    with np.load(output/'fields_uT.npz', allow_pickle=False) as saved:
        arrays = {n:saved[n] for n in saved.files}
    render(output, arrays, provenance['components'], provenance['grid'], rows, provenance, color_scale)
    print(f'Redrawn without inference: {output / "index.html"}')


def run(args):
    root = args.run.resolve(); setup_path = root/'setup.json'
    setup = json.loads(setup_path.read_text()); manifest = setup['manifest']
    if (manifest.get('field_unit') != 'uT' or manifest.get('length_unit') != 'um'
            or manifest.get('components') != list(COMPONENTS)):
        raise ValueError('Unsupported source units or components')
    if args.list:
        for p in manifest['pairs']:
            print(f'{p["geometry"]}: --standoff {p["blur_z_um"]:g} --target-standoff {p["sharp_z_um"]:g}')
        return
    if args.geometry is None or args.standoff is None or args.output is None:
        raise ValueError('Specify --geometry, --standoff and --output (or use --list)')
    if args.output.exists(): raise FileExistsError('Output already exists; choose a new --output folder')
    if args.reference and not args.input: raise ValueError('--reference requires --input')
    if args.saved_results and (args.input or args.reference or args.operator != 'saved'):
        raise ValueError('--saved-results only reports existing inputs and saved operators')
    if args.include_saved_baselines and (args.input or args.operator != 'saved'):
        raise ValueError('Saved baselines apply only to the original input and saved operator')
    target, pair = select_pair(setup, args.geometry, args.standoff, args.target_standoff)
    if pair is None and (args.operator == 'saved' or not args.input or args.saved_results):
        raise ValueError('No saved pair at these heights. Use --list, or --input with --operator analytic')
    grid = manifest['geometries'][args.geometry]['grid']
    components = COMPONENTS if args.component == 'all' else (args.component,)
    provenance = dict(geometry=args.geometry, standoff_um=args.standoff, target_standoff_um=target,
                      components=list(components), grid=grid, field_unit='uT',
                      setup_sha256=sha256(setup_path), source_run=str(root), numpy=np.__version__)
    arrays = {}; archive_path = None
    if args.input:
        arrays['input'] = fields(np.load(args.input, allow_pickle=False), grid, components)
        provenance['input'] = dict(path=str(args.input.resolve()), sha256=sha256(args.input))
        if args.reference:
            arrays['reference'] = fields(np.load(args.reference, allow_pickle=False), grid, components)
            provenance['reference'] = dict(path=str(args.reference.resolve()), sha256=sha256(args.reference))
    else:
        archive_path = local_path(root, f'comparison/{pair["id"]}.npz')
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays['input'] = fields(archive['blurred'], grid, components)
            arrays['reference'] = fields(archive['sharp'], grid, components)
            if args.saved_results: arrays['lista'] = fields(archive['lista'], grid, components)
            if args.include_saved_baselines:
                for name in BASELINES: arrays[name] = fields(archive[name], grid, components)
        provenance['archive'] = dict(path=str(archive_path), sha256=sha256(archive_path))
    pad = setup['contract']['config']['pad']
    if args.operator == 'saved':
        entry = setup['physics'][pair['id']]; path = local_path(root, entry['path'])
        transfer = None
        verified = path.is_file()
        if verified:
            if sha256(path) != entry['sha256']: raise ValueError('Saved operator checksum mismatch')
            transfer = np.load(path, allow_pickle=False)
        elif not args.saved_results:
            raise FileNotFoundError(f'{path} is missing; copy the physics folder from the training machine')
        provenance['operator'] = dict(kind='saved', pad=pad, sha256=entry['sha256'], path=str(path),
                                      file_verified=verified, used_for_inference=not args.saved_results)
    else:
        from propagator import analytic_transfer, k_grid
        transfer = analytic_transfer(k_grid((grid['ny']+2*pad,grid['nx']+2*pad),grid['dx'],grid['dy']),args.standoff-target)
        provenance['operator'] = dict(kind='analytic', pad=pad, dz_um=args.standoff-target)
    if transfer is not None and (transfer.shape != (grid['ny']+2*pad,grid['nx']+2*pad) or not np.isfinite(transfer).all()):
        raise ValueError('Invalid operator shape or values')
    provenance['mode'] = 'saved results (no inference)' if args.saved_results else 'fresh checkpoint inference'
    provenance['interpretation'] = (provenance['mode']+'. '+
        ('Original COMSOL pair; same-sample reconstruction, not an independent test. ' if not args.input else
         'User-supplied physical field on the selected geometry grid. Reference supplied separately if available. ')+
        ('Analytical operator substituted; this differs from a joint-fit training operator and needs validation. ' if args.operator == 'analytic' else
         'Uses the original pair-specific operator; transfer to a new measurement is not established. ')+
        ('Classical predictions reused from the source run; convergence flags retained.' if args.include_saved_baselines else ''))
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    write_json(output/'status.json', dict(status='running'))
    try:
        times = [None]*len(components)
        if not args.saved_results:
            print('Loading checkpoint and reconstructing one component at a time', flush=True)
            arrays['lista'], times, model_info = infer(root, setup, arrays['input'], transfer, pad, args.device)
            provenance['model'] = model_info
        else:
            provenance['model'] = dict(scale_uT=setup['scale_uT'], checkpoint_loaded=False)
        convergence = {}
        if args.include_saved_baselines:
            with (root/'comparison.csv').open() as stream:
                convergence = {(r['component'],r['method']):r for r in csv.DictReader(stream) if r['pair_id']==pair['id']}
            if any((c, name) not in convergence for c in components for name in BASELINES):
                raise ValueError('Missing saved baseline convergence metadata')
        rows = []
        for i, component in enumerate(components):
            for name, array in arrays.items():
                if name == 'reference': continue
                old = convergence.get((component,name), {}) if name in BASELINES else {}
                flag = old.get('converged','').lower()
                row = dict(component=component, method=name,
                           **metrics(array[i], arrays['reference'][i] if 'reference' in arrays else None),
                           converged={'true':True,'false':False}.get(flag), stop_reason=old.get('stop_reason'),
                           iterations=old.get('iterations') or None,
                           inference_seconds=times[i] if name=='lista' else None,
                           reused=name in BASELINES or (name=='lista' and args.saved_results))
                rows.append(row)
        provenance['timing_note'] = 'Fresh forward-call seconds only, CUDA synchronized; not a warmed repeated benchmark. Reused predictions have no new timing.'
        np.save(output/'reconstruction_uT.npy', arrays['lista'])
        np.savez_compressed(output/'fields_uT.npz', **arrays)
        with (output/'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write_json(output/'metrics.json', dict(definition='100 * RMSE / (max(reference)-min(reference)); full-resolution per-component', rows=rows))
        write_json(output/'provenance.json', provenance)
        render(output, arrays, components, grid, rows, provenance, args.color_scale)
        write_json(output/'status.json', dict(status='complete', mode=provenance['mode'],
                   baseline_nonconverged_rows=sum(r['converged'] is False for r in rows)))
        print(f'Report: {output / "index.html"}', flush=True)
        for r in rows:
            if r['method']=='lista': print(f'{r["component"]}: RMSE={r["rmse_uT"]} µT; range NRMSE={r["range_nrmse_percent"]}%')
    except Exception as error:
        write_json(output/'status.json', dict(status='failed', error=str(error)))
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('runs/combined_full_200'))
    p.add_argument('--geometry')
    p.add_argument('--standoff', type=float, help='Input measurement height in µm')
    p.add_argument('--target-standoff', type=float, help='Output height in µm; default is the saved geometry reference')
    p.add_argument('--component', choices=('all', *COMPONENTS), default='all')
    p.add_argument('--operator', choices=('saved','analytic'), default='saved')
    p.add_argument('--input', type=Path, help='Optional new NPY field in µT, on this geometry grid')
    p.add_argument('--reference', type=Path, help='Optional actual target NPY in µT; only with --input')
    p.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    p.add_argument('--saved-results', action='store_true', help='Only report archived predictions; no checkpoint inference')
    p.add_argument('--include-saved-baselines', action='store_true', help='Compare against original saved ISTA/FISTA predictions')
    p.add_argument('--list', action='store_true', help='List available COMSOL height pairs')
    p.add_argument('--output', type=Path)
    p.add_argument('--color-scale', choices=('shared','independent','both'), default='both',
                   help='Display scales only; default shows individual colorbars and an expandable shared-scale view')
    p.add_argument('--replot', type=Path, help='Redraw an existing completed output without inference')
    args = p.parse_args(argv)
    if args.replot:
        if args.output or args.input or args.reference or args.geometry or args.standoff is not None or args.list or args.saved_results or args.include_saved_baselines or args.target_standoff is not None:
            p.error('--replot takes an existing output directory; do not combine with reconstruction-selection flags')
        replot(args.replot, args.color_scale)
    else:
        run(args)


if __name__ == '__main__':
    main()
