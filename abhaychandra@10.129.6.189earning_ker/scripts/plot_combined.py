"""Render saved combined-experiment metrics without rerunning any solver."""
import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_results(output):
    output = Path(output)
    with (output/'comparison.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    training = json.loads((output/'training.json').read_text())
    setup = json.loads((output/'setup.json').read_text())
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    gallery = []

    def save(fig, name, caption):
        fig.savefig(figures/name, dpi=140, bbox_inches='tight')
        plt.close(fig)
        gallery.append((name, caption))

    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label in [('train_mse_normalized', 'Training'), ('validation_mse_normalized', 'Validation')]:
        points = [r for r in training if r[key] is not None]
        if points:
            ax.plot([r['epoch'] for r in points], [r[key] for r in points], label=label)
    ax.set(xlabel='Epoch (0 = initialized model)', ylabel='Normalized MSE', title='Shared ConvLISTA training')
    ax.legend()
    save(fig, 'training.png', 'Checkpoint selection uses validation MSE if available, otherwise training MSE.')

    pairs = list(dict.fromkeys(r['pair_id'] for r in rows))
    methods = list(dict.fromkeys(r['method'] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(pairs)*.35)), sharey=True)
    for ax, component in zip(axes, ('Bx', 'By', 'Bz')):
        values = np.full((len(pairs), len(methods)), np.nan)
        for r in rows:
            if r['component'] == component and r['nrmse_range']:
                values[pairs.index(r['pair_id']), methods.index(r['method'])] = 100*float(r['nrmse_range'])
        im = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap='viridis_r')
        for (y, x), v in np.ndenumerate(values):
            if np.isfinite(v):
                ax.text(x, y, f'{v:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if v > (np.nanmin(values)+np.nanmax(values))/2 else 'black')
        ax.set_xticks(range(len(methods)), methods, rotation=40, ha='right')
        ax.set_yticks(range(len(pairs)), pairs, fontsize=8)
        ax.set_title(component+' range NRMSE (%)')
        fig.colorbar(im, ax=ax, shrink=.7)
    save(fig, 'errors.png', 'Per-pair/component reconstruction error; lower is better. Blank cells have zero reference range.')

    for path in sorted((output/'comparison').glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            names = ['sharp', 'blurred', 'lista', 'fista_to_epsilon']
            fig, axes = plt.subplots(3, 4, figsize=(13, 9))
            for c, comp in enumerate(('Bx', 'By', 'Bz')):
                limit = max(float(np.abs(data['sharp'][c]).max()), 1e-12)
                stride = max(1, int(np.ceil(max(data['sharp'].shape[-2:])/600)))
                for j, name in enumerate(names):
                    axes[c, j].imshow(data[name][c, ::stride, ::stride], origin='lower',
                                      cmap='RdBu_r', vmin=-limit, vmax=limit)
                    axes[c, j].set_title(f'{comp}: {name}')
                    axes[c, j].set_xticks([])
                    axes[c, j].set_yticks([])
            fig.suptitle(path.stem+' — fields in µT; shared color limits within each row')
            save(fig, path.stem+'.png', 'Preview may be subsampled; saved arrays and scores retain full resolution.')
    body = ''.join(f'<figure><img src="figures/{html.escape(name)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
                   for name, caption in gallery)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Combined QDM comparison</title>'
        '<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px}img{max-width:100%}'
        'figure{margin:30px 0}figcaption{color:#555}</style><h1>Combined QDM comparison</h1><p>'
        +html.escape(setup['evaluation'])+'</p><p><a href="comparison.csv">Per-component metrics</a> · '
        '<a href="summary.csv">Summary</a> · <a href="metrics.json">Definitions and convergence status</a></p>'+body)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    plot_results(p.parse_args().output)
