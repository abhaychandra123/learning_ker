"""Figures for README_HIGH_STANDOFF.md (10 -> 0.5 um). Runs on the Mac with NumPy and Matplotlib only.

Inputs: the arrays written by export_high_standoff.py on the server (magic_part.npz/json, fista_part.npz/json)
and the small result tables in final_evaluation/03_results/p9_high_standoff. Nothing is computed here except
means of saved scores; every number in a title comes from the export or the tables.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

COMPONENTS = ('Bx', 'By', 'Bz')
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
# Fixed colour per method in every figure (reference palette, slots in order). Method comparisons show one
# MAGIC-UNet (run B with the data-consistency step) beside FISTA, FISTA-Net and CSC-LISTA. The tile-size study
# keeps the individual runs, with their own colours and markers.
COLOR = dict(magic='#2a78d6', fista='#eb6834', fista_net='#1baf7a', csc_lista='#eda100',
             run10b='#2a78d6', run10a='#e34948', run3='#008300', run2='#e87ba4', target=INK, input='#8a8985')
NAME = dict(run10b='MAGIC-UNet, 44.8 µm tile (run B)', run10a='MAGIC-UNet, 12.8 µm tile (run A)',
            run3='Run 3 (12.8 µm tile, trained on 1–10 µm)', run2='Run 2 (12.8 µm tile, trained at 3 µm)')
MARKER = dict(run10b='o', run10a='s', run3='^', run2='D')
# shown once their 10 um runs exist; they kept their 3 um settings, so every figure says so (not like-for-like)
EXTRA = (('fista_net', 'FISTA-Net (3 µm settings)'), ('csc_lista', 'CSC-LISTA (3 µm settings)'))
MAGIC_NOTE = ('MAGIC-UNet: run B (0.7 µm pixels)\nwith the data-consistency step.\n'
              'Only MAGIC-UNet was adapted to 10 µm;\nFISTA-Net and CSC-LISTA kept\ntheir 3 µm settings.')
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2, 'legend.frameon': False,
    'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
})


def pct(v):
    return f'{100 * v:.1f}%'


def rows(path):
    return list(csv.DictReader(open(path)))


def mean3(row, prefix):
    return float(np.mean([float(row[f'{prefix}_{c}']) for c in COMPONENTS]))


# ---------------------------------------------------------------- maps on one colour scale

def maps(arr, meta, prefix, columns, title, note, path):
    """Top row: fields. Bottom row: output minus target. One colour scale for all twelve panels."""
    target = arr[f'{prefix}_target']
    vmax = float(np.percentile(np.abs(target), 99.5))
    extent = (0, 0.2 * target.shape[1], 0.2 * target.shape[0], 0)
    fig, axes = plt.subplots(2, len(columns), figsize=(3.05 * len(columns) + 0.9, 6.9), constrained_layout=True)
    for c, (label, key) in enumerate(columns):
        field, m = arr[f'{prefix}_{key}'], meta[f'{prefix}_{key}']
        top, bottom = axes[0, c], axes[1, c]
        image = top.imshow(field, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent, interpolation='nearest')
        if key == 'target':
            top.set_title(f'{label}\n ')
            bottom.axis('off')
            lines = [note, '', 'Mean of Bx, By, Bz (L2):']
            lines += [f'  {lab.split(chr(10))[0]}: {pct(np.mean(meta[f"{prefix}_{k}"]["l2"]))}'
                      for lab, k in columns[1:]]
            bottom.text(0.0, 1.0, '\n'.join(lines), va='top', ha='left', fontsize=9.2, color=INK,
                        transform=bottom.transAxes, linespacing=1.45)
        else:
            top.set_title(f'{label}\nBz: L2 {pct(m["l2"][2])} · NRMSE {pct(m["nrmse"][2])}')
            bottom.imshow(field - target, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent,
                          interpolation='nearest')
            bottom.set_title('minus target', fontweight='normal', color=INK2)
        for ax in (top, bottom):
            ax.grid(False)
            ax.set_xticks([0, 70, 140])
            ax.set_yticks([0, 70, 140])
            ax.tick_params(labelsize=8, length=2)
        top.axhline(0.1 * target.shape[0] * 2 / 2, color=INK2, lw=0.6, ls=(0, (4, 4)), alpha=0.6)
    axes[1, 1].set_xlabel('µm')
    bar = fig.colorbar(image, ax=axes, shrink=0.72, pad=0.012, aspect=34)
    bar.set_label('Bz (µT), same scale in every panel')
    bar.outline.set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight='bold', x=0.01, ha='left')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def line_cuts(arr, meta, panels, path):
    fig, axes = plt.subplots(len(panels), 1, figsize=(13.5, 2.9 * len(panels)), constrained_layout=True)
    for ax, (prefix, title, series) in zip(axes, panels):
        target = arr[f'{prefix}_target__row']
        x = 0.1 * np.arange(target.size)
        ax.plot(x, target, color=COLOR['target'], lw=2.4, label='Target, 0.5 µm')
        for label, key, colour in series:
            m = meta[f'{prefix}_{key}']['line']
            ax.plot(x, arr[f'{prefix}_{key}__row'], color=colour, lw=1.7,
                    label=f'{label}: L2 {pct(m["l2"])}, NRMSE {pct(m["nrmse"])}')
        ax.set_title(title, loc='left')
        ax.set_ylabel('Bz (µT)')
        ax.set_xlim(0, x[-1])
        ax.legend(fontsize=8.6, loc='center left', bbox_to_anchor=(1.005, 0.5))
    axes[-1].set_xlabel('Position along the middle row (µm)')
    fig.suptitle('Line cuts through the middle row, Bz, 10 → 0.5 µm', fontsize=13, fontweight='bold', x=0.01,
                 ha='left')
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- charts from the tables

def standoff_curve(sim, fista_mean, path):
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    heights = sorted({float(r['standoff']) for r in sim})
    for run in ('run10b', 'run10a', 'run3', 'run2'):
        y = [100 * np.mean([mean3(r, 'dc_est') for r in sim if r['run'] == run and float(r['standoff']) == z])
             for z in heights]
        ax.plot(heights, y, color=COLOR[run], marker=MARKER[run], ms=6, label=NAME[run])
        dx, dy = {'run10a': (-22, 7), 'run3': (26, 3)}.get(run, (0, 8))       # keep the 10 um labels apart
        ax.annotate(f'{y[2]:.1f}%', (heights[2], y[2]), textcoords='offset points', xytext=(dx, dy), ha='center',
                    fontsize=9, color=INK)
    ax.plot([10], [100 * fista_mean], color=COLOR['fista'], marker='D', ms=9, ls='none',
            label='FISTA, 3,000 iterations (first 20 layouts)')
    ax.annotate(f'{100 * fista_mean:.1f}%', (10, 100 * fista_mean), textcoords='offset points', xytext=(-30, -4),
                fontsize=9, color=INK)
    ax.set_xlabel('Input standoff (µm); target is always 0.5 µm')
    ax.set_ylabel('Relative L2 error of the 0.5 µm field (%)')
    ax.set_ylim(0, 105)
    ax.set_xticks(heights)
    ax.set_title('Tile size and training range: simulated error against standoff (50 layouts)\n'
                 'Each model with the data-consistency step at its own estimated height', loc='left')
    ax.legend(fontsize=8.8, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_layout(sim, fista_rows, extra, path):
    """One dot per layout at 10 um: MAGIC-UNet, FISTA and any further method in `extra` {label: (key, values)}."""
    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    magic = [100 * mean3(r, 'dc_est') for r in sim if r['run'] == 'run10b' and float(r['standoff']) == 10.0]
    groups = [(f'MAGIC-UNet, scale-adapted\n({len(magic)} layouts)', COLOR['magic'], magic),
              (f'FISTA\n(first {len(fista_rows)} layouts)', COLOR['fista'],
               [100 * np.mean([float(r[c]) for c in COMPONENTS]) for r in fista_rows])]
    groups += [(label.replace(' (', '\n(') + f'\n(first {len(values)} layouts)', COLOR[key], [100 * v for v in values])
               for label, (key, values) in extra.items()]
    rng = np.random.default_rng(0)
    for i, (label, colour, values) in enumerate(groups):
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(values)), values, s=30, color=colour, alpha=0.75,
                   edgecolor=SURFACE, linewidth=0.8)
        ax.hlines(np.mean(values), i - 0.3, i + 0.3, color=INK, lw=2)
        ax.annotate(f'mean {np.mean(values):.1f}%', (i, np.mean(values) + 2), ha='center', va='bottom', fontsize=9,
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', fc=SURFACE, ec='none', alpha=0.85))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(0, 106)
    ax.set_ylabel('Relative L2 error per layout (%)')
    ax.set_title('Simulation, 10 → 0.5 µm: one dot per layout\nNot like-for-like: only MAGIC-UNet was adapted to the 10 µm scale',
                 loc='left')
    ax.grid(axis='x', visible=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def height_readout(sim, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    heights = sorted({float(r['standoff']) for r in sim})
    ax.plot([4, 16], [4, 16], color=INK2, lw=1.2, ls=(0, (4, 4)), label='Perfect read-out')
    for run in ('run10b', 'run10a', 'run3', 'run2'):
        h = np.array([[float(r['h_star']) for r in sim if r['run'] == run and float(r['standoff']) == z]
                      for z in heights])
        ax.errorbar(heights, h.mean(1), yerr=h.std(1), color=COLOR[run], marker=MARKER[run], ms=6, capsize=3, lw=2,
                    elinewidth=1.2, label=NAME[run])
    ax.set_xlabel('True input standoff (µm)')
    ax.set_ylabel('Standoff read from the network\'s own current, h* (µm)')
    ax.set_xticks(heights)
    ax.set_xlim(5, 15)
    ax.set_ylim(0, 20)
    ax.set_title('Height read-out on simulation\n(mean and standard deviation over 50 layouts)', loc='left')
    ax.legend(fontsize=8.8, loc='upper left')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fista_curves(results, path):
    """Validation error against depth. Each entry is (label, results, extension or None); an extension is a later
    curve-only run over deeper iterations, merged in before the minimum is taken."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for ax, (label, r, more), corner in zip(axes, results, ('upper right', 'upper center')):
        curve = {int(k[len('fista'):]): v for k, v in r['validation'].items()}
        if more:
            curve.update({int(k[len('fista'):]): v for k, v in more['validation'].items()})
        depth = sorted(curve)
        value = [100 * curve[d] for d in depth]
        ref = min(curve, key=curve.get)
        near = [d for d in depth if curve[d] <= curve[ref] + 0.001]
        ax.axvspan(near[0], near[-1], color=COLOR['fista'], alpha=0.14, lw=0,
                   label=f'Within 0.1 point of the minimum ({near[0]:,}–{near[-1]:,})')
        ax.plot(depth, value, color=COLOR['fista'], label='Validation error, 64 simulated crops')
        ax.plot([ref], [100 * curve[ref]], marker='o', ms=8, color=INK, ls='none',
                label=f'Lowest error: {ref:,} iterations ({100 * curve[ref]:.1f}%)')
        ax.set_title(label, loc='left')
        ax.set_xlabel('FISTA iterations')
        ax.set_ylabel('Relative L2 error (%)')
        ax.legend(fontsize=8.6, loc=corner)
    fig.suptitle('FISTA depth chosen from a validation curve recorded every 25 iterations', fontsize=13,
                 fontweight='bold', x=0.01, ha='left')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def comsol_bars(magic, fista, extra, path):
    """Per coil configuration: MAGIC-UNet, FISTA and any further method in `extra` {label: (key, {config: error})}."""
    configs = ['L1_C1', 'C1_C2', 'C1_C2_C3', 'C1_C2_C3_C4', 'IC_4um']
    series = [('MAGIC-UNet (scale-adapted)', COLOR['magic'],
               [np.mean([float(r['raw']) for r in magic
                         if r['run'] == 'run10b' and r['variant'] == 'dc_est' and r['config'] == c])
                for c in configs]),
              ('FISTA', COLOR['fista'],
               [np.mean([float(r['relative_l2']) for r in fista if r['method'] == 'fista3000' and r['config'] == c])
                for c in configs])]
    series += [(label, COLOR[key], [values[c] for c in configs]) for label, (key, values) in extra.items()]
    fig, ax = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
    width = 0.8 / len(series) - 0.015
    for i, (label, colour, values) in enumerate(series):
        x = np.arange(len(configs)) + (i - (len(series) - 1) / 2) * (width + 0.015)
        ax.bar(x, 100 * np.array(values), width, color=colour, label=label)
        for xi, v in zip(x, values):
            ax.annotate(f'{100 * v:.0f}', (xi, 100 * v), textcoords='offset points', xytext=(0, 3), ha='center',
                        fontsize=8.6, color=INK)
    ax.set_xticks(np.arange(len(configs)))
    ax.set_xticklabels(configs)
    ax.set_ylabel('Relative L2 error (%), mean of Bx, By, Bz')
    ax.set_ylim(0, 112)
    ax.grid(axis='x', visible=False)
    ax.set_title('COMSOL, 10 → 0.5 µm, per coil configuration (secondary test: COMSOL differs from free-space\n'
                 'physics at this height). Not like-for-like: only MAGIC-UNet was adapted to the 10 µm scale', loc='left')
    ax.legend(fontsize=9, ncol=len(series), loc='upper center', bbox_to_anchor=(0.5, -0.1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
    for ax, (run, history) in zip(axes, histories):
        epoch = [int(r['epoch']) for r in history]
        ax.plot(epoch, [float(r['train_loss']) for r in history], color=COLOR[run], label='Training loss')
        ax.plot(epoch, [float(r['val_loss']) for r in history], color=COLOR[run], ls=(0, (4, 3)),
                label='Validation loss (4,096 crops)')
        ax.set_yscale('log')
        ax.set_title(NAME[run], loc='left')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE of the normalised current')
        ax.legend(fontsize=8.8)
    fig.suptitle('Training curves of the two 10 µm runs (50 epochs, paper recipe)', fontsize=13, fontweight='bold',
                 x=0.01, ha='left')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tiles(arr, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.2), constrained_layout=True)
    for ax, key, label in ((axes[0], 'sim_median_target', 'Target field, 0.5 µm'),
                           (axes[1], 'sim_median_input', 'Measured field, 10 µm (the network\'s input)')):
        field = arr[key]
        vmax = float(np.percentile(np.abs(field), 99.5))
        size = 0.2 * field.shape[0]
        ax.imshow(field, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=(0, size, size, 0), interpolation='nearest')
        for side, colour, name in ((12.8, COLOR['run10a'], '12.8 µm tile (run A)'),
                                   (44.8, COLOR['run10b'], '44.8 µm tile (run B)')):
            ax.add_patch(Rectangle(((size - side) / 2, (size - side) / 2), side, side, fill=False, ec=colour, lw=2.4,
                                   label=name))
        ax.set_title(f'{label}\n(own colour scale, ±{vmax:.1f} µT)', loc='left')
        ax.set_xlabel('µm')
        ax.grid(False)
    axes[1].legend(fontsize=9, loc='lower right', frameon=True, facecolor=SURFACE, edgecolor=GRID)
    fig.suptitle('What one 64 × 64 tile sees at 10 µm', fontsize=13, fontweight='bold', x=0.01, ha='left')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def k0_check(folder, path):
    """Amendment A5: simulated error and amplitude of MAGIC-UNet (run B) and FISTA against the extra decay k0."""
    sim = [r for r in rows(folder / 'sim_k0.csv') if r['run'] == 'run10b' and r['variant'] == 'dc_est']
    comsol = [r for r in rows(folder / 'comsol_shape.csv') if r['config'] != 'IC_4um']
    fista = rows(folder / 'fista_k0.csv') if (folder / 'fista_k0.csv').exists() else []
    mean = lambda rs, prefix: float(np.mean([[float(r[f'{prefix}_{c}']) for c in COMPONENTS] for r in rs]))
    values = sorted({float(r['k0']) for r in sim})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    for ax, prefix, scale, label in ((axes[0], 'l2', 100, 'Relative L2 error (%)'),
                                     (axes[1], 'amp', 1, 'Amplitude: ⟨output, target⟩ / ⟨target, target⟩')):
        y = [scale * mean([r for r in sim if float(r['k0']) == k], prefix) for k in values]
        ax.plot(values, y, color=COLOR['magic'], marker='o', ms=6, label='MAGIC-UNet, simulation (50 layouts)')
        ax.axhline(scale * mean([r for r in comsol if r['method'] == 'run10b_dc_est'], prefix), color=COLOR['magic'],
                   lw=1.4, ls=(0, (5, 4)), label='MAGIC-UNet, COMSOL C1 family')
        for k, v in zip(values, y):
            ax.annotate(f'{v:.1f}%' if scale == 100 else f'{v:.2f}', (k, v), textcoords='offset points',
                        xytext=(20, -13) if scale == 100 else (0, -15), ha='center', fontsize=9, color=INK)
        if fista:
            k = float(fista[0]['k0'])
            ax.plot([k], [scale * mean(fista, prefix)], color=COLOR['fista'], marker='D', ms=9, ls='none',
                    label=f'FISTA, simulation (first {len(fista)} layouts)')
            ax.axhline(scale * mean([r for r in comsol if r['method'] == 'fista3000'], prefix), color=COLOR['fista'],
                       lw=1.4, ls=(0, (5, 4)), label='FISTA, COMSOL C1 family')
        ax.set_xticks(values)
        ax.set_xlabel('Extra decay k₀ in the simulated 10 µm input (rad/µm)')
        ax.set_ylabel(label)
        ax.set_ylim(0, 72 if scale == 100 else 1.2)
        ax.set_xlim(-0.012, 0.165)
    axes[0].set_title('Error', loc='left')
    axes[1].set_title('Amplitude of the output relative to the target', loc='left')
    axes[1].legend(fontsize=8.6, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.suptitle('Check (amendment A5): simulated 10 → 0.5 µm with COMSOL-like extra decay', fontsize=13,
                 fontweight='bold', x=0.01, ha='left')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--export', required=True, help='Folder with magic_part.* and fista_part.*')
    p.add_argument('--results', default='final_evaluation/03_results/p9_high_standoff')
    p.add_argument('--output', default='final_evaluation/04_figures/p9_high_standoff')
    p.add_argument('--booklet', default='booklet/figures', help='Also copy the figures here as hs_<name>.png')
    a = p.parse_args()
    export, results, out = Path(a.export), Path(a.results), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    arr, meta = {}, {}
    parts = ['magic_part', 'fista_part'] + [f'{key}_part' for key, _ in EXTRA
                                              if (export / f'{key}_part.npz').exists()]
    if (export / 'comsol3_part.npz').exists():          # 3 um comparison, from export_3um_figures.py
        parts.append('comsol3_part')
    for part in parts:
        with np.load(export / f'{part}.npz') as z:
            arr.update({k: z[k] for k in z.files})
        meta.update(json.loads((export / f'{part}.json').read_text()))
    fista = json.loads((results / 'fista_10um' / 'results.json').read_text())
    depth = fista['reference_depth']

    def columns(prefix):
        cols = [('Target, 0.5 µm', 'target'), ('Input, 10 µm', 'input'), ('MAGIC-UNet', 'run10b_dc_est'),
                (f'FISTA, {depth:,} iterations', f'fista{depth}')]
        return cols + [(label, key) for key, label in EXTRA if f'{prefix}_{key}' in arr]

    def series(prefix):
        out = [('Input, 10 µm', 'input', COLOR['input']), ('MAGIC-UNet', 'run10b_dc_est', COLOR['magic']),
               ('FISTA', f'fista{depth}', COLOR['fista'])]
        return out + [(label, key, COLOR[key]) for key, label in EXTRA if f'{prefix}_{key}' in arr]

    for tag, rule in (('median', 'the layout with the\nmedian MAGIC-UNet error'),
                      ('worst', 'the layout with the\nworst MAGIC-UNet error')):
        layout = meta['chosen'][tag]
        maps(arr, meta, f'sim_{tag}', columns(f'sim_{tag}'),
             f'Simulation, 10 → 0.5 µm, Bz, held-out layout {layout}',
             f'Free-space simulation: {rule}\namong the 20 scored by FISTA.\n{MAGIC_NOTE}\nDashed line: row of the line cut.',
             out / f'sim_{tag}_maps.png')
    for config in ('L1_C1', 'IC_4um'):
        maps(arr, meta, f'comsol_{config}', columns(f'comsol_{config}'),
             f'COMSOL {config}, 10 → 0.5 µm, Bz (secondary test)',
             f'COMSOL {config}.\n{MAGIC_NOTE}\nDashed line: row of the line cut.', out / f'comsol_{config}_maps.png')

    line_cuts(arr, meta, [
        ('sim_median', f'Simulation, layout {meta["chosen"]["median"]} (median MAGIC-UNet error)',
         series('sim_median')),
        ('sim_worst', f'Simulation, layout {meta["chosen"]["worst"]} (worst MAGIC-UNet error)', series('sim_worst')),
        ('comsol_L1_C1', 'COMSOL L1_C1', series('comsol_L1_C1')),
        ('comsol_IC_4um', 'COMSOL IC_4um', series('comsol_IC_4um'))], out / 'line_cuts.png')

    if 'comsol3_L1_C1_target' in arr:                 # 3 -> 0.5 um, one colour scale per figure
        three = out.parent / 'p7_m1_m2'
        three.mkdir(parents=True, exist_ok=True)
        cols3 = [('Target, 0.5 µm', 'target'), ('Input, 3 µm', 'input'), ('MAGIC-UNet', 'magic'),
                 ('FISTA, 925 iterations', 'fista925'), ('FISTA-Net', 'fista_net'), ('CSC-LISTA', 'csc_lista')]
        for config in ('L1_C1', 'IC_4um'):
            maps(arr, meta, f'comsol3_{config}', cols3, f'COMSOL {config}, 3 → 0.5 µm, Bz',
                 f'COMSOL {config}.\nMAGIC-UNet: run 2 with the\ndata-consistency step (ours),\nstandoff not given; the\n'
                 f'others are told it.\nDashed line: row of the line cut.', three / f'comsol3_{config}_maps.png')
        if booklet.is_dir() if (booklet := Path(a.booklet)) else False:
            for f in three.glob('comsol3_*_maps.png'):
                shutil.copyfile(f, booklet / f'p7_{f.name}')

    sim = rows(results / 'rescoring' / 'sim_scores.csv')
    standoff_curve(sim, fista['sim_full_mean_relative_l2'], out / 'error_vs_standoff.png')
    extra_sim, extra_comsol = {}, {}
    for key, label in EXTRA:        # written by the scoring of the 10 um FISTA-Net and CSC-LISTA runs
        folder = results / 'methods_10um' / key
        if (folder / 'sim_full.csv').exists():
            extra_sim[label] = (key, [float(np.mean([float(r[c]) for c in COMPONENTS]))
                                      for r in rows(folder / 'sim_full.csv')])
        if (folder / 'comsol.csv').exists():
            table = rows(folder / 'comsol.csv')
            extra_comsol[label] = (key, {c: float(np.mean([float(r['relative_l2']) for r in table
                                                           if r['config'] == c]))
                                         for c in sorted({r['config'] for r in table})})
    per_layout(sim, rows(results / 'fista_10um' / 'sim_full.csv'), extra_sim, out / 'per_layout_10um.png')
    height_readout(sim, out / 'height_readout.png')
    deeper = results / 'fista_curve_10k' / 'results.json'
    fista_curves([('10 → 0.5 µm', fista, json.loads(deeper.read_text()) if deeper.exists() else None),
                  ('3 → 0.5 µm (recheck)',
                   json.loads((results / 'fista_3um_recheck' / 'results.json').read_text()), None)],
                 out / 'fista_depth_curves.png')
    comsol_bars(rows(results / 'rescoring' / 'comsol.csv'), rows(results / 'fista_10um' / 'comsol.csv'),
                extra_comsol, out / 'comsol_per_coil.png')
    training([(run, rows(results / run / 'history.csv')) for run in ('run10a', 'run10b')],
             out / 'training_curves.png')
    tiles(arr, out / 'tile_context.png')
    if (results / 'k0_check' / 'sim_k0.csv').exists():
        k0_check(results / 'k0_check', out / 'k0_check.png')
    booklet = Path(a.booklet)
    if booklet.is_dir():            # the booklet uses the same figures under an hs_ prefix
        for f in out.glob('*.png'):
            shutil.copyfile(f, booklet / f'hs_{f.name}')
    print('wrote', sorted(f.name for f in out.glob('*.png')))


if __name__ == '__main__':
    main()
