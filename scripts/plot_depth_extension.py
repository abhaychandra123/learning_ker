"""Plot depth_extension.py results: error vs depth for trained ConvLISTA and matched ISTA/FISTA.

Usage: python scripts/plot_depth_extension.py runs/depth_extension_v1
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STYLE = {'ista': ('ISTA', '#8C9EAA', 'o'), 'fista': ('FISTA', '#2872A2', 's'), 'lista': ('ConvLISTA (16-layer weights)', '#B5522B', 'D')}
MODELS = {'combined': '23 Oct model (51 COMSOL pairs)', 'synthetic': '1,080-pair synthetic model'}


def load(root):
    rows = []
    for model in MODELS:
        path = root/model/'metrics.csv'
        if path.exists():
            rows += list(csv.DictReader(open(path)))
    return rows


def mean_by(rows, keys, value):
    acc = defaultdict(list)
    for r in rows:
        acc[tuple(r[k] for k in keys)].append(float(r[value]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main(root):
    root = Path(root)
    rows = load(root)
    models = [m for m in MODELS if any(r['model'] == m for r in rows)]
    metrics = [('relative_l2', 'Relative L2 error (%)'), ('nrmse_range', 'Range NRMSE (%)'), ('amplitude_slope', 'Amplitude slope (ideal 1)')]
    fig, axs = plt.subplots(len(models), 3, figsize=(15, 4.2*len(models)), squeeze=False, constrained_layout=True)
    for i, model in enumerate(models):
        sel = [r for r in rows if r['model'] == model]
        for j, (metric, label) in enumerate(metrics):
            means = mean_by(sel, ('method', 'depth'), metric)
            ax = axs[i, j]
            for method, (name, color, marker) in STYLE.items():
                depths = sorted({int(d) for m, d in means if m == method})
                scale = 1 if metric == 'amplitude_slope' else 100
                ax.plot(depths, [scale*means[(method, str(d))] for d in depths], color=color, marker=marker, label=name, lw=2)
            values = [v for v in means.values()]
            if max(values) > 10*min(values):  # divergent runs would flatten the classical curves
                ax.set_yscale('log')
            ax.axvline(16, color='#53616A', ls=':', lw=1)
            ax.set_xlabel('Layers / iterations'); ax.set_ylabel(label)
            ax.set_xticks([8, 16, 24, 32, 40, 48, 56, 64])
            ax.grid(alpha=.3)
            if j == 0:
                ax.set_title(f'{MODELS[model]}: mean of 51 components', loc='left', fontsize=12)
        axs[i, 0].legend(frameon=False)
    fig.suptitle('Trained 16-layer ConvLISTA run deeper without retraining (dotted line = trained depth)')
    fig.savefig(root/'depth_curves.png', dpi=150)

    # Per input height, L1_C1, relative L2 at 16 / 32 / 64.
    fig, axs = plt.subplots(1, len(models), figsize=(7.5*len(models), 4.5), squeeze=False, constrained_layout=True)
    for i, model in enumerate(models):
        sel = [r for r in rows if r['model'] == model and r['geometry'] == 'L1_C1']
        means = mean_by(sel, ('method', 'depth', 'blur_z_um'), 'relative_l2')
        heights = sorted({float(h) for _, _, h in means})
        ax = axs[0, i]
        for method, depth, ls in [('lista', 16, '-'), ('lista', 32, '--'), ('lista', 64, ':'), ('fista', 64, '-')]:
            name, color, marker = STYLE[method]
            key = lambda h: (method, str(depth), next(k[2] for k in means if float(k[2]) == h))
            ax.plot(heights, [100*means[key(h)] for h in heights], color=color, ls=ls, marker=marker, ms=4,
                    label=f"{name.split(' (')[0]} {depth}")
        ax.set_title(f'{MODELS[model]}: L1_C1', loc='left', fontsize=12)
        ax.set_xlabel('Input height (µm), reference 0.5 µm'); ax.set_ylabel('Relative L2 error (%)')
        ax.set_ylim(0, 105); ax.grid(alpha=.3); ax.legend(frameon=False)
    fig.savefig(root/'depth_by_height_L1_C1.png', dpi=150)
    print('wrote', root/'depth_curves.png', root/'depth_by_height_L1_C1.png')


if __name__ == '__main__':
    main(sys.argv[1])
