"""Every method on the re-solved COMSOL L1_C1 (amendment A7): one MAGIC-UNet beside FISTA, FISTA-Net and CSC-LISTA.

Usage: python3 plot_resolved_methods.py <export dir> <out.png>
<export dir>: runs/comsol_resolved/export from the server (magic, fista, fistanet3, fistanet10, csc3, csc10 parts).
Maps are Bz on the re-solved field, 2x2 averaged to 0.2 um, one colour scale per row. Scores are the mean relative L2
of Bx, By, Bz. At 10 um only MAGIC-UNet was scale-adapted, so that row is not a like-for-like comparison.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC, OUT = Path(sys.argv[1]), sys.argv[2]
arr, meta = {}, {}
for part in ('magic', 'fista', 'fistanet3', 'fistanet10', 'csc3', 'csc10'):
    with np.load(SRC / f'{part}_part.npz') as z:
        arr.update({k: z[k] for k in z.files})
    meta.update(json.loads((SRC / f'{part}_part.json').read_text()))

INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
LIGHT, DARK = '#b9c9dd', '#2a5d9e'
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'font.size': 10, 'axes.edgecolor': INK2, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})

ROWS = {10: [('run10b_dc_est', 'MAGIC-UNet\n(run B + data consistency)'),
             ('fista3000', 'FISTA, 3,000 iterations\n(depth chosen at 10 µm)'),
             ('fista_net', 'FISTA-Net\n(3 µm settings)'),
             ('csc_lista', 'CSC-LISTA\n(3 µm settings)')],
        3: [('run2_dc_est', 'MAGIC-UNet\n(run 2 + data consistency)'),
            ('fista925', 'FISTA, 925 iterations'),
            ('fista_net', 'FISTA-Net'),
            ('csc_lista', 'CSC-LISTA')]}


def mean_l2(z, data, method):
    return 100 * float(np.mean(meta[f'res{z}_{data}_{method}']['l2']))


fig = plt.figure(figsize=(18, 12.4), constrained_layout=True)
grid = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.78])
extent = (0, 140, 140, 0)
for r, z in enumerate((10, 3)):
    target = arr[f'res{z}_resolved_target']
    vmax = float(np.percentile(np.abs(target), 99.5))
    panels = [(target, f'Target, 0.5 µm\n(re-solved; input {z} µm)')]
    for method, label in ROWS[z]:
        panels.append((arr[f'res{z}_resolved_{method}'],
                       f'{label}\nre-solved {mean_l2(z, "resolved", method):.1f}%  (original {mean_l2(z, "original", method):.1f}%)'))
    for c, (img, title) in enumerate(panels):
        ax = fig.add_subplot(grid[r, c])
        im = ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent, interpolation='nearest')
        ax.set_xlim(0, 80)
        ax.set_ylim(80, 0)
        ax.set_title(title, loc='left', fontsize=9.5)
        ax.tick_params(labelsize=8)
        if c == 0:
            ax.set_ylabel(f'{z} → 0.5 µm\nµm', fontsize=11, fontweight='bold')
    bar = fig.colorbar(im, ax=fig.axes[-5:], shrink=0.8, pad=0.01, aspect=25)
    bar.set_label(f'Bz (µT), ±{vmax:.0f}')
    bar.outline.set_visible(False)

for c, z in enumerate((10, 3)):
    ax = fig.add_subplot(grid[2, (0, 3)[c]:(3, 5)[c]])
    methods = ROWS[z] + ([('tikhonov', 'Tikhonov\n(3 µm settings)')] if z == 10 else [('tikhonov', 'Tikhonov')])
    names = []
    for m in methods:
        first, *rest = m[1].split('\n')
        first = first.replace(', 3,000 iterations', '').replace(', 925 iterations', '')
        keep = z == 10 and rest and '3 µm settings' in rest[0]   # every 10 um label of a non-adapted method says so
        names.append(first + ('\n' + rest[0] if keep else ''))
    x = np.arange(len(methods))
    orig = [mean_l2(z, 'original', m) for m, _ in methods]
    res = [mean_l2(z, 'resolved', m) for m, _ in methods]
    ax.bar(x - 0.2, orig, 0.38, color=LIGHT, label='Original export')
    ax.bar(x + 0.2, res, 0.38, color=DARK, label='Re-solved (tolerance 1e-8)')
    for i in range(len(x)):
        ax.text(x[i] - 0.2, orig[i] + 1.5, f'{orig[i]:.1f}', ha='center', fontsize=8, color=INK2)
        ax.text(x[i] + 0.2, res[i] + 1.5, f'{res[i]:.1f}', ha='center', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel('Relative L2, mean of Bx, By, Bz (%)')
    ax.grid(True, axis='y', color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(f'{z} → 0.5 µm' + ('  (not like-for-like: only MAGIC-UNet was adapted to 10 µm)' if z == 10 else ''),
                 loc='left')
    if c == 0:
        ax.legend(fontsize=9, loc='upper left')

fig.suptitle('Every method on COMSOL L1_C1, original export and re-solve. One wire shape, one seed per model.',
             fontsize=14, fontweight='bold', x=0.01, ha='left')
fig.savefig(OUT, dpi=150)
for z in (10, 3):
    print(z, {m: (round(mean_l2(z, 'original', m), 1), round(mean_l2(z, 'resolved', m), 1))
              for m, _ in ROWS[z] + [('tikhonov', '')]})
