"""MAGIC-UNet at 10 -> 0.5 um on COMSOL L1_C1: the original export against the tight-tolerance re-solve (amendment A7).

Usage: python3 plot_resolved_reconstruction.py <magic_part.npz> <magic_part.json> <original pair .npz> <re-solved pair .npz> <out.png>
magic_part.*: written on the server by score_resolved_comsol.py (Bz maps at 0.2 um, 2x2 averaged, and scores).
The pairs: runs/combined_full_200/comparison/L1_C1_z10.0_to_z00.5.npz and runs/comsol_resolved/comparison/ (same name).
One colour scale for every map. Scores in titles are for Bz; the text gives the mean of Bx, By, Bz.
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PART, META, ORIG, RESOLVED, OUT = sys.argv[1:6]
arr, meta = np.load(PART), json.loads(open(META).read())
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
RED, BLUE, BLACK = '#d03b3b', '#2a78d6', '#0b0b0b'
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'font.size': 10, 'axes.edgecolor': INK2, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})


def show(a):
    z = np.asarray(a[2], np.float64)
    return z.reshape(700, 2, 700, 2).mean((1, 3))


def scores(pred, truth):
    err = pred - truth
    return (np.sqrt((err ** 2).sum((-2, -1)) / (truth ** 2).sum((-2, -1))),
            np.sqrt((err ** 2).mean((-2, -1))) / (truth.max((-2, -1)) - truth.min((-2, -1))))


rows = []
for data, path in (('original', ORIG), ('resolved', RESOLVED)):
    with np.load(path) as z:
        sharp, blurred = z['sharp'].astype(np.float64), z['blurred'].astype(np.float64)
    l2, nr = scores(blurred, sharp)
    m = meta[f'res10_{data}_run10b_dc_est']
    rows.append(dict(data=data, target=show(sharp), input=show(blurred), input_l2=l2, input_nr=nr,
                     out=arr[f'res10_{data}_run10b_dc_est'], m=m))
vmax = float(np.percentile(np.abs(rows[1]['target']), 99.5))
extent = (0, 140, 140, 0)
fig = plt.figure(figsize=(17.5, 11.6), constrained_layout=True)
grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.62])
names = {'original': 'Original COMSOL export', 'resolved': 'Re-solved COMSOL (tolerance 1e-8)'}
for r, row in enumerate(rows):
    panels = ((row['target'], f'{names[row["data"]]}\nTarget, 0.5 µm'),
              (row['input'], f'Input, 10 µm\nBz: L2 {100 * row["input_l2"][2]:.1f}% · NRMSE {100 * row["input_nr"][2]:.1f}%'),
              (row['out'], f'MAGIC-UNet (+ data consistency)\nBz: L2 {100 * row["m"]["l2"][2]:.1f}% · NRMSE {100 * row["m"]["nrmse"][2]:.1f}%'),
              (row['out'] - row['target'], f'Output minus target\nmean of Bx, By, Bz: L2 {100 * np.mean(row["m"]["l2"]):.1f}%'))
    for c, (img, title) in enumerate(panels):
        ax = fig.add_subplot(grid[r, c])
        im = ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent, interpolation='nearest')
        ax.set_title(title, loc='left')
        ax.tick_params(labelsize=8)
        if c == 0:
            ax.set_ylabel('µm')
bar = fig.colorbar(im, ax=fig.axes, shrink=0.55, pad=0.01, aspect=30, location='right')
bar.set_label(f'Bz (µT), same scale in every map (±{vmax:.0f})')
bar.outline.set_visible(False)

# line cut through the coil (row with the strongest target Bz), on the 0.2 um display grid
iy = int(np.argmax(np.abs(rows[1]['target']).max(1)))
xs = np.arange(700) * 0.2
ax = fig.add_subplot(grid[2, :])
ax.plot(xs, rows[1]['target'][iy], color=BLACK, lw=2.4, label='Target, 0.5 µm (re-solved)')
ax.plot(xs, rows[1]['out'][iy], color=BLUE, lw=1.8, label=f'MAGIC-UNet on the re-solved 10 µm input '
                                                        f'(mean L2 {100 * np.mean(rows[1]["m"]["l2"]):.1f}%)')
ax.plot(xs, rows[0]['out'][iy], color=RED, lw=1.8, label=f'MAGIC-UNet on the original 10 µm input '
                                                       f'(mean L2 {100 * np.mean(rows[0]["m"]["l2"]):.1f}%)')
ax.set_xlim(0, 140)
ax.set_xlabel(f'Position along the row through the coil, y = {iy * 0.2:.0f} µm (µm)')
ax.set_ylabel('Bz (µT)')
ax.grid(True, color=GRID)
ax.legend(fontsize=9.5, loc='lower right')
ax.set_title('Line cut, Bz', loc='left')
fig.suptitle('MAGIC-UNet, 10 → 0.5 µm, COMSOL L1_C1: the same network on the original export and on the re-solved field',
             fontsize=14, fontweight='bold', x=0.01, ha='left')
fig.savefig(OUT, dpi=150)
print('input L2 Bz original %.3f, re-solved %.3f' % (rows[0]['input_l2'][2], rows[1]['input_l2'][2]))
