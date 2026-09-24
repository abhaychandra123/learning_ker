"""Plot saved comparison outputs without rerunning training or inference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_results(run_dir):
    run_dir = Path(run_dir)
    result = json.loads((run_dir/'metrics.json').read_text())
    with np.load(run_dir/'reconstructions.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    grid, components = result['grid'], result['components']
    extent = [grid['x0']-grid['dx']/2, grid['x0']+grid['dx']*(grid['nx']-.5),
              grid['y0']-grid['dy']/2, grid['y0']+grid['dy']*(grid['ny']-.5)]
    methods = [('blurred', 'Observed'), ('sharp', 'Reference'),
               ('ista', f"ISTA ({result['config']['baseline_iterations']})"),
               ('lista', f"Supervised LISTA ({result['config']['layers']} layers)"),
               ('fista', f"FISTA ({result['config']['baseline_iterations']})")]
    fig, axes = plt.subplots(3, 5, figsize=(19, 11), constrained_layout=True)
    for c, component in enumerate(components):
        vmax = float(np.percentile(abs(arrays['sharp'][c]), 99.5))
        for ax, (key, title) in zip(axes[c], methods):
            ax.imshow(arrays[key][c], origin='lower', extent=extent, cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax)
            ax.set_title(f'{component}: {title}', fontsize=9)
            ax.set_xlabel('x (µm)')
        axes[c, 0].set_ylabel('y (µm)')
    fig.suptitle('Same-sample reconstruction: all reference components used in kernel fitting and LISTA training\n'
                 'Shared display range per row; µT. Iteration counts differ; see equal-depth and timing metrics.')
    fig.savefig(run_dir/'reconstructions.png', dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for c, comp in enumerate(components):
        for method in ('ista', 'lista', 'fista'):
            history = result['histories'][method]
            iterations = [r['iteration'] for r in history]
            for col, (metric, label) in enumerate([
                    ('objective', 'L1 objective'), ('nrmse_range', 'Range NRMSE (%)'),
                    ('pg_rms_uT', 'Proximal-gradient RMS (µT)')]):
                values = [r['components'][c][metric] for r in history]
                if col == 1:
                    values = [v*100 if v is not None else np.nan for v in values]
                axes[c, col].plot(iterations, values, label=method.upper())
                if col != 1:
                    axes[c, col].set_yscale('symlog', linthresh=1e-10)
                axes[c, col].set_title(f'{comp}: {label}')
                axes[c, col].set_xlabel('Iterations / trained layers')
                axes[c, col].grid(alpha=.25)
                axes[c, col].legend()
    fig.suptitle('Objective convergence and reference error are different criteria\n'
                 'LISTA intermediates are diagnostics; only its final layer is supervised.')
    fig.savefig(run_dir/'convergence.png', dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    training = result['training']
    axes[0].plot([r['epoch'] for r in training], [r['training_mse_normalized'] for r in training])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training MSE (normalized units)')
    axes[0].set_title(f"LISTA training; best training epoch {result['best_training_epoch']}")
    keys = list(result['comparisons'])
    times = [result['comparisons'][k]['inference']['median_seconds'] for k in keys]
    axes[1].bar(keys, times)
    axes[1].tick_params(axis='x', labelrotation=30)
    axes[1].set_ylabel('Median inference seconds, batch of three')
    axes[1].set_title('Same single device; training time excluded')
    fig.savefig(run_dir/'training_and_latency.png', dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    plot_results(parser.parse_args().run_dir)
