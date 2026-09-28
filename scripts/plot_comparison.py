"""Plot saved comparison outputs without rerunning training or inference."""
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_results(run_dir):
    run_dir = Path(run_dir)
    result = json.loads((run_dir/'metrics.json').read_text())
    rows = []
    for method, entry in result['comparisons'].items():
        for component, score in zip(result['components'], entry['metrics']):
            rows.append({
                'method': method, 'component': component,
                'iterations': entry.get('iterations'), 'trained_layers': entry.get('layers'),
                'stop_reason': entry.get('stop_reason', 'trained depth' if 'layers' in entry else 'fixed depth'),
                'converged': entry.get('converged'), 'epsilon': entry.get('epsilon'),
                'relative_pg': score.get('relative_pg'),
                'nrmse_range_percent': 100*score['nrmse_range'] if score['nrmse_range'] is not None else None,
                'relative_l2_percent': 100*score['relative_l2'] if score['relative_l2'] is not None else None,
                'rmse_uT': score['rmse_uT'], 'objective': score['objective'],
                'pg_rms_uT': score['pg_rms_uT'],
                'median_batch_inference_seconds': entry['inference']['median_seconds']})
    with (run_dir/'comparison_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with np.load(run_dir/'reconstructions.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    grid, components = result['grid'], result['components']
    extent = [grid['x0']-grid['dx']/2, grid['x0']+grid['dx']*(grid['nx']-.5),
              grid['y0']-grid['dy']/2, grid['y0']+grid['dy']*(grid['ny']-.5)]
    def baseline_label(method):
        baseline = result['baselines'][method]
        status = baseline.get('stop_reason', 'fixed budget')
        return f"{method.upper()} ({baseline['iterations']} updates)\n{status}"

    methods = [('blurred', 'Observed'), ('sharp', 'Reference'),
               ('ista', baseline_label('ista')),
               ('lista', f"Supervised LISTA ({result['config']['layers']} layers)"),
               ('fista', baseline_label('fista'))]
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
                 'Shared display range per row; µT. ISTA/FISTA stop by epsilon in new runs; LISTA uses its trained depth.')
    fig.savefig(run_dir/'reconstructions.png', dpi=120)
    plt.close(fig)

    metric_specs = [('objective', 'L1 objective'), ('nrmse_range', 'Range NRMSE (%)'),
                    ('pg_rms_uT', 'Proximal-gradient RMS (µT)')]
    if 'epsilon' in result['config']:
        metric_specs.append(('relative_pg', 'PG RMS / initial PG RMS'))
    fig, axes = plt.subplots(3, len(metric_specs), figsize=(5*len(metric_specs), 10), constrained_layout=True)
    for c, comp in enumerate(components):
        for method in ('ista', 'lista', 'fista'):
            history = result['histories'][method]
            iterations = [r['iteration'] for r in history]
            for col, (metric, label) in enumerate(metric_specs):
                if metric == 'relative_pg':
                    first = history[0]['components'][c]['pg_rms_uT']
                    values = [r['components'][c]['pg_rms_uT']/first if first else
                              r['components'][c]['pg_rms_uT'] for r in history]
                else:
                    values = [r['components'][c][metric] for r in history]
                if col == 1:
                    values = [v*100 if v is not None else np.nan for v in values]
                if method == 'lista':
                    curve_label = f"LISTA ({result['config']['layers']} trained layers)"
                else:
                    state = result['baselines'][method]
                    curve_label = (f"{method.upper()} ({state['iterations']} updates; "
                                   f"{state.get('stop_reason', 'fixed budget')})")
                axes[c, col].plot(iterations, values, label=curve_label)
                axes[c, col].scatter([iterations[-1]], [values[-1]], s=18, zorder=3)
                if col != 1:
                    axes[c, col].set_yscale('symlog', linthresh=1e-10)
                axes[c, col].set_title(f'{comp}: {label}')
                axes[c, col].set_xlabel('Iterations / trained layers')
                axes[c, col].grid(alpha=.25)
                if metric == 'relative_pg' and method == 'fista':
                    axes[c, col].axhline(result['config']['epsilon'], color='black', ls='--', label='epsilon')
                axes[c, col].legend(fontsize=7)
    fig.suptitle('Objective convergence and reference error are different criteria\n'
                 'LISTA intermediates are diagnostics; only its final layer is supervised.')
    fig.savefig(run_dir/'convergence.png', dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    training = result['training']
    axes[0].plot([r['epoch'] for r in training], [r['training_mse_normalized'] for r in training])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training MSE (normalized units)')
    axes[0].set_title(f"LISTA training; best training epoch {result['best_training_epoch']}")
    keys = list(result['comparisons'])
    times = [result['comparisons'][k]['inference']['median_seconds'] for k in keys]
    labels = []
    for key in keys:
        entry = result['comparisons'][key]
        if 'layers' in entry:
            labels.append(f"{key}\n{entry['layers']} trained layers")
        else:
            labels.append(f"{key}\n{entry['iterations']} updates\n{entry.get('stop_reason', 'fixed depth')}")
    axes[1].bar(labels, times)
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
