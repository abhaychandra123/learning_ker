"""Side-by-side reconstructions and line cuts for the group update (Bz, 3 -> 0.5 um).

    python3 scripts/plot_group_update.py /path/to/group_update.npz reports/group_update_magic
The .npz comes from scripts/export_group_update.py (run on the server).
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, GREEN, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#008300', '#6250d6', '#e34948'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SIGNED = LinearSegmentedColormap.from_list('signed', [BLUE, '#f0efec', RED])
METHODS = (('magic', 'MAGIC-UNet\n(paper architecture)', AQUA),
           ('magic_dc', 'MAGIC-UNet + one\ndata-consistency step', GREEN),
           ('fista1000', 'FISTA-1000\n(learned dictionary)', ORANGE),
           ('fista_net', 'FISTA-Net', BLUE),
           ('csc_lista', 'CSC-LISTA\n(64 layers)', VIOLET))
DX = 0.1
plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 200,
    'axes.titlesize': 9, 'axes.titleweight': 'bold',
})


def rel(p, t):
    return 100 * np.linalg.norm(p - t) / np.linalg.norm(t)


def nrmse(p, t):
    """RMSE divided by the target's range, as in train_csc_lista.py."""
    return 100 * np.sqrt(np.mean((p - t) ** 2)) / (t.max() - t.min())


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def cut_row(t):
    """The middle row of the crop: a fixed rule, not picked per method or per result."""
    return t.shape[0] // 2


def side_by_side(d, scores, coil, out):
    t = d[f'{coil}_target'][2]
    limit = np.percentile(np.abs(t), 99.8)
    row = cut_row(t)
    extent = (0, t.shape[1] * DX, 0, t.shape[0] * DX)
    panels = (('input', 'Input, 3 µm\n(COMSOL, measured)'),) + tuple(m[:2] for m in METHODS)
    fig, axes = plt.subplots(2, 1 + len(panels), figsize=(15.5, 5.2))
    style = dict(cmap=SIGNED, vmin=-limit, vmax=limit, origin='lower', extent=extent)
    image = axes[0, 0].imshow(t, **style)
    axes[0, 0].axhline(row * DX, color=INK, lw=.8, ls='--')
    axes[0, 0].set_title('Target, 0.5 µm\n(COMSOL truth)')
    axes[1, 0].text(.5, .5, 'Error maps\n(output minus target,\nsame colour scale)\n\nErrors are $B_z$\nover the full\n'
                    '140 µm image', ha='center', va='center', fontsize=8.5, color=MUTED,
                    transform=axes[1, 0].transAxes)
    for col, (key, label) in enumerate(panels, start=1):
        p = d[f'{coil}_{key}'][2]
        axes[0, col].imshow(p, **style)
        axes[0, col].set_title(label)
        axes[1, col].imshow(p - t, **style)
        axes[1, col].set_title(f'L2 {scores[key]["l2"][2] * 100:.1f}%\nNRMSE {scores[key]["nrmse"][2] * 100:.2f}%')
    for ax in axes.flat:
        bare(ax)
    fig.colorbar(image, ax=axes, fraction=.011, pad=.01, label='$B_z$ (µT)')
    fig.suptitle(f'{coil}, $B_z$, 3 µm → 0.5 µm (crop 51.2 µm wide). Top: target, input and reconstructions. '
                 f'Bottom: error maps. Dashed line: line cut.', fontsize=10, x=.45, y=1.01)
    fig.savefig(out / f'{coil}_side_by_side.png')
    plt.close(fig)


def line_cut(d, coil, out):
    t = d[f'{coil}_target'][2]
    row = cut_row(t)
    x = np.arange(t.shape[1]) * DX
    truth, blurred = t[row], d[f'{coil}_input'][2][row]
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(7.2, 9.0), sharex=True, sharey=True)
    for ax, (key, label, colour) in zip(axes, METHODS):
        p = d[f'{coil}_{key}'][2][row]
        ax.plot(x, blurred, color=GRID, lw=1.2, label='Input, 3 µm')
        ax.plot(x, truth, color=INK, lw=1.3, label='Target, 0.5 µm')
        ax.plot(x, p, color=colour, lw=1.3, label=label.replace('\n', ' '))
        ax.set_title(f'{label.replace(chr(10), " ")}: line-cut L2 {rel(p, truth):.1f}%, '
                     f'NRMSE {nrmse(p, truth):.2f}%', loc='left', color=colour)
        ax.set_ylabel('$B_z$ (µT)')
    axes[-1].set_xlabel(f'Position along the dashed line (µm), row {row} of the crop')
    handles = [plt.Line2D([], [], color=INK, lw=1.3), plt.Line2D([], [], color=GRID, lw=1.2)]
    fig.legend(handles, ['Target, COMSOL at 0.5 µm', f'Input, COMSOL at 3 µm (line-cut L2 {rel(blurred, truth):.1f}%, '
               f'NRMSE {nrmse(blurred, truth):.2f}%)'], loc='upper center', ncol=2,
               bbox_to_anchor=(.5, 1.02), fontsize=8)
    fig.suptitle(f'{coil}, $B_z$ line cut, 3 µm → 0.5 µm (coloured: reconstruction)', y=1.05, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f'{coil}_line_cut.png')
    plt.close(fig)
    keys = ['input'] + [key for key, _, _ in METHODS]
    return row, {key: (rel(d[f'{coil}_{key}'][2][row], truth), nrmse(d[f'{coil}_{key}'][2][row], truth))
                 for key in keys}


def main():
    d = dict(np.load(sys.argv[1]))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    scores = json.loads(str(d['scores']))
    summary = {'mean_15_pairs': json.loads(str(d['mean15']))}
    for coil in ('L1_C1', 'IC_4um'):
        side_by_side(d, scores[coil], coil, out)
        row, cuts = line_cut(d, coil, out)
        summary[coil] = dict(
            crop_origin=d[f'{coil}_origin'].tolist(), line_cut_row=row,
            full_image_mean_xyz={k: {m: round(100 * float(np.mean(x)), 2) for m, x in v.items()}
                                 for k, v in scores[coil].items()},
            full_image_bz={k: {m: round(100 * x[2], 2) for m, x in v.items()} for k, v in scores[coil].items()},
            line_cut_bz={k: {'l2': round(float(a), 2), 'nrmse': round(float(b), 2)} for k, (a, b) in cuts.items()})
    (out / 'numbers.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
