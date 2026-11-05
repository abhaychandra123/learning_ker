"""Gallery, L1_C1 regression and realism report for a simulated dataset.

COMSOL fields are read only to judge realism here. They never feed generation,
normalisation or training, and no reconstruction score is computed.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from propagator import radial_spectrum
from simulate_fields import (COMPONENTS, DEFAULTS, IMAGE_UM, SimulatedDataset,
                             current_maps, fields_from_current, synthesise)

# Centreline traced from the 0.5 um COMSOL field peaks, in (col, row) pixels.
L1_C1_PIXELS = [(-350, 240), (-350, 180), (600, 180), (600, 600), (240, 600),
                (240, 300), (480, 300), (480, 480), (360, 480), (360, 420),
                (420, 420), (420, 360), (300, 360), (300, 540), (540, 540),
                (540, 240), (-350, 240)]
L1_C1_WIDTH_UM, L1_C1_OFFSET_UM = 2.08, 0.08
HEIGHTS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
DATA3_CONFIGS = ('C1_C2', 'C1_C2_C3', 'C1_C2_C3_C4')
EMPIRICAL_K0 = 0.095    # fitted to data3; shown for comparison only
COLOURS = ('#1f6fb2', '#c2452d', '#2e8b57', '#8a5fbf', '#c98a1b')


def load_cache(root, z, component):
    path = Path(root) / f'z{z:04.1f}_{component}.npy'
    return np.load(path).astype(np.float64) * 1e6


def load_data3(root, config, z):
    """One data3 plane as [Bx, By, Bz] in uT on the [row, col] grid."""
    whole, tenth = f'{z:.1f}'.split('.')
    path = Path(root) / f'Coil_{config}' / f'L1_{config}_Z{whole}p{tenth}um_600uA_0p1PR_MF_B.csv'
    table = pd.read_csv(path, comment='%', header=None, usecols=[2, 3, 4]).to_numpy()
    return table.T.reshape(3, 1400, 1400).astype(np.float64) * 1e6


def l1_c1_paths(copies):
    """The traced spiral plus its data3 mirror copies, in data3 order."""
    base = np.array(L1_C1_PIXELS, dtype=np.float64) * DEFAULTS['dx']
    paths = []
    for flip_rows, flip_cols in ((0, 0), (1, 0), (0, 1), (1, 1))[:copies]:
        p = base.copy()
        if flip_rows:
            p[:, 1] = IMAGE_UM - p[:, 1]
        if flip_cols:
            p[:, 0] = IMAGE_UM - p[:, 0]
        paths.append(p)
    return paths


def score(paths, heights, load, name):
    """Rebuild a coil set under both decay laws and score it against COMSOL."""
    pad, dx = DEFAULTS['pad'], DEFAULTS['dx']
    n = DEFAULTS['nx'] + 2 * pad
    kx, ky = current_maps(paths, L1_C1_WIDTH_UM, DEFAULTS['current_a'], (n, n), dx, pad)
    offset = L1_C1_OFFSET_UM
    z_ref = DEFAULTS['decay_ref_um'] + offset
    planes = [z + offset for z in heights]
    laws = (('free_space', fields_from_current(kx, ky, dx, planes)),
            ('k0', fields_from_current(kx, ky, dx, planes, EMPIRICAL_K0, z_ref)))
    rows = []
    for law, fields in laws:
        for z, stack in zip(heights, fields):
            truth = load(z)
            for c, field, b in zip(COMPONENTS, stack, truth):
                a = field[pad:-pad, pad:-pad]
                rows.append(dict(coils=name, law=law, z_um=z, component=c,
                                 relative_l2=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                                 slope=float((a * b).sum() / (a * a).sum())))
    return rows


def regression(cache, data3):
    """Rebuild real coils from the traced centreline, one and four at a time."""
    rows = score(l1_c1_paths(1), HEIGHTS,
                 lambda z: [load_cache(cache, z, c) for c in COMPONENTS], 'L1_C1')
    return rows + score(l1_c1_paths(4), (0.5, 3.0, 10.0),
                        lambda z: load_data3(data3, 'C1_C2_C3_C4', z), 'C1_C2_C3_C4')


def comsol_pairs(run, data3):
    """COMSOL 3 um input and 0.5 um target for all five configurations."""
    pairs = {}
    for geometry in ('L1_C1', 'IC_4um'):
        with np.load(Path(run) / 'comparison' / f'{geometry}_z03.0_to_z00.5.npz') as z:
            pairs[geometry] = (z['sharp'].astype(np.float64),
                               z['blurred'].astype(np.float64))
    for config in DATA3_CONFIGS:
        pairs[config] = (load_data3(data3, config, 0.5), load_data3(data3, config, 3.0))
    return pairs


def gallery(dataset, output, count=4):
    recipes = dataset.recipes('train')
    picks = [next(r for r in recipes if r['family'] == f)
             for f in ('spiral', 'meander', 'array')]
    picks.append(next(r for r in recipes if r['family'] == 'array'
                      and len(r['paths_um']) == 4))
    fig, axes = plt.subplots(count, 6, figsize=(15, 2.6 * count), squeeze=False)
    for row, recipe in enumerate(picks):
        item = synthesise(recipe, dataset.settings)
        for i, c in enumerate(COMPONENTS):
            for j, (name, stack) in enumerate((('target', item['target']),
                                               ('input', item['input']))):
                ax = axes[row, i * 2 + j]
                v = np.abs(stack[i]).max()
                ax.imshow(stack[i], origin='lower', cmap='RdBu_r', vmin=-v, vmax=v)
                ax.set_title(f'{c} {name} ±{v:.0f} µT', fontsize=9)
                ax.axis('off')
        axes[row, 0].text(0, 1.2, f"{recipe['id']} {recipe['family']}, "
                          f"{recipe['noise_percent']:g}% noise",
                          transform=axes[row, 0].transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / 'gallery.png', dpi=80)
    plt.close(fig)


def spectra(dataset, pairs, output, count=24):
    dx = dataset.settings['dx']
    curves = {'target': [], 'input': []}
    for recipe in dataset.recipes('train')[:count]:
        item = synthesise(recipe, dataset.settings)
        for name in curves:
            k, s = radial_spectrum(item[name][2], dx)
            curves[name].append(s)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name, index in ((axes[0], 'target', 0), (axes[1], 'input', 1)):
        stack = np.array(curves[name])
        keep = (k > 0.01) & (k < 2.5)
        ax.fill_between(k[keep], np.percentile(stack, 5, 0)[keep],
                        np.percentile(stack, 95, 0)[keep], color='#c9c9c9',
                        label='simulated 5–95%')
        for colour, (geometry, pair) in zip(COLOURS, pairs.items()):
            km, sm = radial_spectrum(pair[index][2], dx)
            keep_m = (km > 0.01) & (km < 2.5)
            ax.plot(km[keep_m], sm[keep_m], color=colour, lw=1.3, label=geometry)
        ax.set(xscale='log', yscale='log', xlabel='spatial frequency (cycles/µm)',
               title=f'Bz {name} spectrum')
        ax.legend(frameon=False)
    axes[0].set_ylabel('radial amplitude (µT)')
    fig.tight_layout()
    fig.savefig(output / 'spectra.png', dpi=100)
    plt.close(fig)


def amplitude_check(dataset, pairs):
    with (dataset.root / 'inventory.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    checks = []
    for name, index in (('target', 0), ('input', 1)):
        for i, c in enumerate(COMPONENTS):
            for stat, f in (('peak', lambda a: np.abs(a).max()),
                            ('rms', lambda a: np.sqrt(np.mean(a ** 2)))):
                values = [float(r[f'{name}_{c}_{stat}_ut']) for r in rows]
                low, high = np.percentile(values, [5, 95])
                for geometry, pair in pairs.items():
                    v = float(f(pair[index][i]))
                    checks.append(dict(plane=name, component=c, statistic=stat,
                                       geometry=geometry, comsol_ut=v,
                                       simulated_p5_ut=float(low),
                                       simulated_p95_ut=float(high),
                                       inside=bool(low <= v <= high)))
    return checks


def table(rows):
    head = ''.join(f'<th>{k}</th>' for k in rows[0])
    body = ''.join('<tr>' + ''.join(
        f'<td>{v:.3f}</td>' if isinstance(v, float) else f'<td>{v}</td>'
        for v in r.values()) + '</tr>' for r in rows)
    return f'<table><tr>{head}</tr>{body}</table>'


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cache', default='cache')
    p.add_argument('--comparison-run', default='runs/combined_full_200')
    p.add_argument('--data3', default='data3')
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    dataset = SimulatedDataset(args.dataset)
    pairs = comsol_pairs(args.comparison_run, args.data3)
    rows = regression(args.cache, args.data3)
    checks = amplitude_check(dataset, pairs)
    gallery(dataset, output)
    spectra(dataset, pairs, output)
    (output / 'realism.json').write_text(json.dumps(
        dict(regression=rows, amplitude=checks), indent=2) + '\n')
    (output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Simulated fields</title>'
        '<style>body{font:15px system-ui;margin:24px;max-width:1100px}'
        'img{max-width:100%}td,th{padding:2px 10px;text-align:left}</style>'
        '<h1>Simulated fields</h1><h2>Training samples</h2><img src="gallery.png">'
        '<h2>Spatial spectra against COMSOL</h2><img src="spectra.png">'
        '<h2>Real coils rebuilt from the traced centreline</h2>' + table(rows) +
        '<h2>COMSOL amplitudes against the simulated 5–95% range</h2>' + table(checks))
    for r in rows:
        if r['z_um'] in (0.5, 3.0, 10.0):
            print(r['coils'], r['law'], r['z_um'], r['component'],
                  round(r['relative_l2'], 3), round(r['slope'], 3))
    for r in checks:
        print(r['plane'], r['component'], r['statistic'], r['geometry'],
              round(r['comsol_ut'], 2), 'range', round(r['simulated_p5_ut'], 2),
              round(r['simulated_p95_ut'], 2), 'in' if r['inside'] else 'OUT')


if __name__ == '__main__':
    main()
