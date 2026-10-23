"""Inspect synthetic fields and Gaussian noise without running any training."""
import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from synthetic_data import (SyntheticDataset, write_json, digest, inside,
                            validate_fields, error_metrics, propagate)


def propagation_table(data):
    rows = []
    for geometry in data.manifest['geometries']:
        samples = [s for s in data.samples if s['geometry'] == geometry]
        entry = data.manifest['geometries'][geometry]
        heights = sorted({s['target_z_um'] for s in samples})
        distances = sorted({s['dz_um'] for s in samples})
        fmt = lambda values: ', '.join(f'{v:.6g}' for v in values)
        rows.append(f'<tr><td>{html.escape(geometry)}</td><td>{entry["reference_z_um"]:g}</td>'
                    f'<td>{fmt(heights)}</td><td>{fmt(distances)}</td><td>{len(samples)}</td></tr>')
    count = len(data.samples)
    config = data.manifest['config']
    rotations = ', '.join(f'{r}°' for r in config['rotations_degrees'])
    noise = ', '.join(f'{n:g}%' for n in config['noise_percent'])
    # Count from actual recipes; clean levels are not repeated for --realizations > 1.
    base_count = len({(s['geometry'], s['dz_um']) for s in data.samples})
    variants = sum(config['realizations'] if n else 1 for n in config['noise_percent'])
    return ('<h2>Every propagation and the pair count</h2>'
            '<p>Each arrow runs from the lowest available reference directly to the target height; '
            'heights are not propagated sequentially. All lengths below are in µm; decimals are rounded for display.</p>'
            '<table><thead><tr><th>Geometry</th><th>Reference z</th><th>Target z</th>'
            '<th>Δz = target − reference</th><th>Vector pairs</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table>'
            f'<p><strong>{base_count} geometry/height propagations × '
            f'{len(config["rotations_degrees"])} rotations × {variants} noise variants '
            f'= {count} vector pairs. × 3 components = {3*count} scalar component pairs.</strong></p>'
            f'<p>Rotations: {rotations}. Noise settings: {noise}. '
            f'Realizations per nonzero noise setting: {config["realizations"]}. '
            'One vector pair contains Bx, By and Bz. Variants share reference geometries; '
            'this count does not represent independent physical configurations. Fields are generated on demand.</p>')


def comsol_comparisons(data, source_run, output):
    """Compare exact matching physical planes, verifying the reference and grid."""
    source_run = Path(source_run).resolve()
    setup = source_run/'setup.json'
    source = json.loads(setup.read_text())['manifest']
    if (source.get('field_unit') != 'uT' or source.get('length_unit') != 'um'
            or source.get('components') != ['Bx', 'By', 'Bz']):
        raise ValueError('COMSOL comparison source units/components mismatch')
    rows, sections = [], []
    for geometry, entry in data.manifest['geometries'].items():
        if source['geometries'][geometry]['grid'] != entry['grid']:
            raise ValueError('COMSOL comparison grid mismatch')
        available = {s['target_z_um'] for s in data.samples if s['geometry'] == geometry}
        pairs = sorted([p for p in source['pairs'] if p['geometry'] == geometry
                        and p['sharp_z_um'] == entry['reference_z_um']
                        and p['blur_z_um'] in available], key=lambda p: p['blur_z_um'])
        if not pairs:
            raise ValueError('No matching COMSOL planes for ' + geometry)
        # Predefined selection: lowest, upper-middle and highest matching height.
        for index in sorted({0, len(pairs)//2, len(pairs)-1}):
            pair = pairs[index]
            path = inside(source_run, 'comparison/'+pair['id']+'.npz')
            with np.load(path, allow_pickle=False) as archive:
                reference, actual = archive['sharp'], archive['blurred']
            if not np.array_equal(reference, data.references[geometry]):
                raise ValueError('COMSOL comparison reference differs from synthetic reference')
            validate_fields(actual, entry['grid'])
            archive_hash = digest(path)
            grid = entry['grid']; pad = data.manifest['boundary']['pad_pixels']
            dz = pair['blur_z_um']-entry['reference_z_um']
            extent = [grid['x0']-grid['dx']/2, grid['x0']+(grid['nx']-.5)*grid['dx'],
                      grid['y0']-grid['dy']/2, grid['y0']+(grid['ny']-.5)*grid['dy']]
            images = {}
            for c, component in enumerate(('Bx', 'By', 'Bz')):
                predicted = propagate(reference[c], dz, grid['dx'], grid['dy'], pad).astype(np.float32)
                residual = predicted.astype(np.float64)-actual[c]
                metrics = error_metrics(predicted, actual[c])
                margin = data.manifest['config']['interior_margin_pixels']
                interior = error_metrics(predicted, actual[c], margin)
                rows.append(dict(geometry=geometry, component=component, pair_id=pair['id'],
                                 reference_z_um=entry['reference_z_um'], target_z_um=pair['blur_z_um'],
                                 dz_um=dz, pad_pixels=pad, added_noise_percent=0,
                                 **metrics, interior_rmse_uT=interior['rmse_uT'],
                                 interior_range_nrmse_percent=interior['range_nrmse_percent'],
                                 source_archive=str(path), source_sha256=archive_hash))
                limit = max(float(np.abs(actual[c]).max()), float(np.abs(predicted).max())) or 1.
                residual_limit = float(np.abs(residual).max()) or 1.
                fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True)
                for ax, field, title in zip(axes[:2], (actual[c], predicted),
                                           ('Actual COMSOL field', 'Analytical propagation (no added noise)')):
                    im = ax.imshow(field, origin='lower', extent=extent, cmap='RdBu_r', vmin=-limit, vmax=limit)
                    ax.set_title(title)
                fig.colorbar(im, ax=list(axes[:2]), label=f'{component} (µT)', shrink=.8)
                im = axes[2].imshow(residual, origin='lower', extent=extent, cmap='RdBu_r',
                                    vmin=-residual_limit, vmax=residual_limit)
                axes[2].set_title('Analytical − COMSOL (separate scale)')
                fig.colorbar(im, ax=axes[2], label='Residual (µT)', shrink=.8)
                for ax in axes:
                    ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
                nrmse = metrics['range_nrmse_percent']
                score = f'{nrmse:.3f}%' if nrmse is not None else 'undefined (zero range)'
                fig.suptitle(f'{geometry}, {component}: reference {entry["reference_z_um"]:g} → '
                             f'target {pair["blur_z_um"]:g} µm\nForward discrepancy: '
                             f'range NRMSE {score}; RMSE {metrics["rmse_uT"]:.4g} µT')
                name = 'comsol_'+pair['id']+'_'+component+'.png'
                fig.savefig(output/name, dpi=140); plt.close(fig)
                images[component] = f'<a href="{html.escape(name)}"><img loading="lazy" src="{html.escape(name)}" alt="{html.escape(geometry)} {component} COMSOL comparison"></a>'
            sections.append(f'<section><h3>{html.escape(geometry)}: {entry["reference_z_um"]:g} → '
                            f'{pair["blur_z_um"]:g} µm</h3>'+images['Bz']+
                            '<details><summary>Show Bx and By at this same height</summary>'+
                            images['Bx']+images['By']+'</details></section>')
            print(f'COMSOL comparison: {pair["id"]}, all three components', flush=True)
    with (output/'comsol_checks.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(output/'comsol_provenance.json', dict(source_setup=str(setup),
               source_setup_sha256=digest(setup), dataset_manifest_sha256=digest(data.root/'manifest.json'),
               selection='Lowest, upper-middle and highest matching actual height per geometry', rows=rows))
    return ('<h2>Actual COMSOL fields at matching standoffs</h2>'
            '<p>Actual simulated higher-plane fields are read from the saved COMSOL inputs, not reconstructed outputs. '
            'Each is compared with analytical propagation from the verified same lowest reference. '
            'These comparisons use no rotation and <strong>no added Gaussian noise</strong>, to isolate the forward discrepancy. '
            'The two field maps share a colour scale chosen at that height; the residual has its own labelled scale. '
            'NRMSE is normalized by the actual higher-plane COMSOL range. It is not reconstruction error or a Gaussian-noise estimate.</p>'
            '<p>Selection: lowest, upper-middle and highest available matching heights per geometry. '
            'Bz is shown below; expand each case for Bx/By. Click an image for full size. '
            '<a href="comsol_checks.csv">Full and interior metrics</a> · '
            '<a href="comsol_provenance.json">Source provenance</a></p>'+''.join(sections))


def report(dataset, output, comparison_run=None, refresh=False):
    data = SyntheticDataset(dataset)
    output = Path(output).resolve()
    if refresh and not (output/'preview_selection.json').is_file():
        raise ValueError('--refresh requires an existing synthetic preview directory')
    output.mkdir(parents=True, exist_ok=refresh)
    gallery, rows = [], []
    for geometry in data.manifest['geometries']:
        eligible = [(i, s) for i, s in enumerate(data.samples) if s['geometry'] == geometry]
        # Fixed, documented selection: median available dz, largest noise,
        # smallest rotation and first realization. No selection on performance.
        heights = sorted({s['dz_um'] for _, s in eligible})
        dz = heights[len(heights)//2]
        candidates = [(i, s) for i, s in eligible if s['dz_um'] == dz]
        index, sample = sorted(candidates, key=lambda t: (-t[1]['noise_percent'],
                                                       t[1]['rotation_degrees'], t[1]['realization']))[0]
        for component in ('Bx', 'By', 'Bz'):
            item = data.component(index, component)
            meta = item['metadata']; grid = meta['grid']
            noise = item['blurred'].astype(np.float64)-item['clean']
            span = float(np.ptp(item['clean'].astype(np.float64)))
            rms = float(np.sqrt(np.mean(noise**2)))
            rows.append(dict(geometry=geometry, split=sample['split'], component=component,
                             sample_id=sample['id'], reference_z_um=sample['reference_z_um'],
                             target_z_um=sample['target_z_um'], noise_percent=sample['noise_percent'],
                             sigma_requested_uT=meta['sigma_uT'], noise_mean_uT=float(noise.mean()),
                             noise_std_uT=float(noise.std()), noise_rmse_uT=rms,
                             noise_range_nrmse_percent=100*rms/span if span else None))
            extent = [grid['x0']-grid['dx']/2, grid['x0']+(grid['nx']-.5)*grid['dx'],
                      grid['y0']-grid['dy']/2, grid['y0']+(grid['ny']-.5)*grid['dy']]
            fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
            limit = max(float(np.abs(item[k]).max()) for k in ('sharp', 'clean', 'blurred')) or 1.
            titles = ['COMSOL reference', 'Analytical propagation', 'With Gaussian noise']
            for ax, key, title in zip(axes[:3], ('sharp', 'clean', 'blurred'), titles):
                im = ax.imshow(item[key], origin='lower', extent=extent, cmap='RdBu_r', vmin=-limit, vmax=limit)
                ax.set_title(title); ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
            fig.colorbar(im, ax=list(axes[:3]), label=f'{component} (µT)', shrink=.75)
            sigma = meta['sigma_uT']
            if sigma:
                axes[3].hist(noise.ravel()[::max(1, noise.size//100000)]/sigma,
                             bins=80, density=True, color='#2872a2', alpha=.65)
                x = np.linspace(-5, 5, 300)
                axes[3].plot(x, np.exp(-x*x/2)/np.sqrt(2*np.pi), color='black', label='N(0, 1)')
                axes[3].legend()
            else:
                axes[3].text(.5, .5, 'No added noise', transform=axes[3].transAxes, ha='center')
            axes[3].set_title(f'Added noise: σ = {sigma:.4g} µT')
            axes[3].set_xlabel('Noise / requested σ'); axes[3].set_ylabel('Density')
            fig.suptitle(f'{geometry}, {component}: {sample["reference_z_um"]:g} to '
                         f'{sample["target_z_um"]:.3g} µm; {sample["noise_percent"]:g}% range noise')
            name = f'{geometry}_{component}.png'
            fig.savefig(output/name, dpi=140); plt.close(fig)
            gallery.append(name)
    with (output/'noise_checks.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(output/'preview_selection.json', rows)
    body = ''.join(f'<figure><a href="{html.escape(name)}"><img loading="lazy" src="{html.escape(name)}"></a><figcaption>{html.escape(name)}</figcaption></figure>'
                   for name in gallery)
    comparisons = comsol_comparisons(data, comparison_run, output) if comparison_run else ''
    inventory = html.escape(json.dumps(data.manifest['inventory'], indent=2))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Synthetic QDM preview</title>'
        '<style>body{font:17px system-ui;max-width:1600px;margin:32px auto;padding:20px}'
        'img{max-width:100%}figure{margin:30px 0}pre{white-space:pre-wrap}'
        'table{border-collapse:collapse;width:100%}th,td{padding:12px;text-align:left;border:1px solid #ccc}'
        'th{background:#edf2f6}td{vertical-align:top}summary{cursor:pointer;padding:12px;background:#edf2f6}'
        'section{margin:36px 0}h2{margin-top:48px}</style>'
        '<h1>Synthetic QDM dataset</h1><p>Physical fields in µT. Colour scales are labelled in each figure. '
        'Noise σ is a percentage of the clean propagated component range. COMSOL references retain their original '
        'numerical artifacts. These are synthetic variants, not independent source geometries.</p>'
        +propagation_table(data)+comparisons+
        '<h2>Synthetic inputs and added noise</h2><p>The three field maps in each figure below share one colour scale.</p>'
        '<p>Preview selection: median available propagation distance, largest configured noise, smallest rotation, '
        'first realization, for each geometry and all components. No training or reconstruction solver ran.</p>'
        f'<pre>{inventory}</pre><p><a href="noise_checks.csv">Measured noise statistics</a></p>'+body)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--comparison-run', type=Path, help='Completed combined run containing actual COMSOL planes')
    p.add_argument('--refresh', action='store_true', help='Refresh an existing synthetic preview in place')
    args = p.parse_args()
    report(args.dataset, args.output, args.comparison_run, args.refresh)
