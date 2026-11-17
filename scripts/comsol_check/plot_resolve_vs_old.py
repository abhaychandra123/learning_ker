"""Figures: COMSOL L1_C1 as originally exported, as re-solved at tolerance 1e-8, and as our simulator computes it.

Usage: python3 plot_resolve_vs_old.py <L1_C1_all.npz> <rt8.npz> <dxf_current.npz> <output folder> [maps|cuts|why|all]

- L1_C1_all.npz, rt8.npz: csv2npz.py output of the original export and of the re-solve (arrays [component, y, x] in
  tesla, raw CSV labels). Shown along the image axes with the frame of README_COMSOL_MODEL.md finding 1:
  B_u = -By_csv, B_v = Bx_csv, B_z = Bz_csv.
- dxf_current.npz: sheet current of the exact coil from the DXF drawing (dxf_forward.py: Laplace conduction solve in
  the 2 um conductor, 600 uA, leads ending at the box wall). Its field uses our simulator's physics (a thin current
  sheet at z = 0, thin-sheet Biot-Savart, free-space decay exp(-kz)), with mirror images at the box walls so that the
  leads close through the walls as in COMSOL. Nothing is fitted: no scale, no height, no offset; only the current's
  direction is taken from the re-solve.
Box modes (mirror images at the walls) are used for spectra and propagation, as in tolerance_check.py.

Writes resolve_vs_old_maps.png, resolve_vs_old_linecut.png and resolve_vs_old_why.png. NumPy only, a few minutes.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OLD, NEW, CUR, OUT = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
ONLY = sys.argv[5] if len(sys.argv) > 5 else 'all'
OUT.mkdir(parents=True, exist_ok=True)
old, new, cur = np.load(OLD), np.load(NEW), np.load(CUR)
x, y = new['x'].astype(float), new['y'].astype(float)
N, D, MU0 = 1400, 0.1, 4e-7 * np.pi
HEIGHTS = ('0.5', '3.0', '10.0')
NAMES = ('Bx', 'By', 'Bz')
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
RED, BLUE, GREEN = '#d03b3b', '#2a78d6', '#1baf7a'          # original, re-solve, our simulator (fixed everywhere)
LABEL = dict(old='Original export (default tolerance)', new='Re-solve (tolerance 1e-8)', sim='Our simulator, exact coil')
COLOR = dict(old=RED, new=BLUE, sim=GREEN)
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'font.size': 10, 'axes.edgecolor': INK2, 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'axes.titlesize': 10.5, 'axes.titleweight': 'bold'})


def frame(B):
    """Microtesla, components along the image axes (u = columns, v = rows) and z."""
    B = B.astype(np.float64) * 1e6
    return np.stack([-B[1], B[0], B[2]])


def mirror(a, pu, pv):
    a = np.concatenate([a, pu * a[:, ::-1]], axis=1)
    return np.concatenate([a, pv * a[::-1, :]], axis=0)


PARITY = [(-1, 1), (1, -1), (1, 1)]
k1 = 2 * np.pi * np.fft.fftfreq(2 * N, D)                    # rad/um on the mirrored (box) grid
KV, KU = np.meshgrid(k1, k1, indexing='ij')
K = np.hypot(KU, KV)
Ku, Kv = cur['Ku'].astype(np.float64), cur['Kv'].astype(np.float64)
FU, FV = np.fft.fft2(mirror(Ku, 1, -1)), np.fft.fft2(mirror(Kv, -1, 1))


def simulate(z):
    """Our simulator's thin-sheet physics for the exact coil at height z (um), box walls, in uT."""
    kap = K.copy()
    kap[0, 0] = 1.0
    e = 0.5 * MU0 * np.exp(-K * z) * 1e6
    parts = (e * FV, -e * FU, 1j * (KU * FV - KV * FU) * e / kap)
    return np.stack([np.real(np.fft.ifft2(p))[:N, :N] for p in parts])


def modes(B):
    return [np.fft.fft2(mirror(b, *p)) for b, p in zip(B, PARITY)]


rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
F = {'old': {z: frame(old[f'L1_C1_z{z}']) for z in HEIGHTS},
     'new': {z: frame(new[f'L1_C1_z{z}']) for z in HEIGHTS},
     'sim': {z: simulate(float(z)) for z in HEIGHTS}}
sign = np.sign(sum(np.vdot(F['sim']['0.5'][c], F['new']['0.5'][c]) for c in range(3)))   # current direction only
F['sim'] = {z: sign * v for z, v in F['sim'].items()}
extent = (x[0], x[-1], y[0], y[-1])
diff = {v: {z: [100 * rel(F[v][z][c], F['new'][z][c]) for c in range(3)] for z in HEIGHTS} for v in ('old', 'sim')}

# 1. maps: Bz at each height; the three versions and both differences, one colour scale per row
if ONLY in ('all', 'maps'):
    fig, axes = plt.subplots(3, 5, figsize=(19.5, 12.0), constrained_layout=True)
    for row, z in enumerate(HEIGHTS):
        vmax = float(np.percentile(np.abs(F['new'][z][2]), 99.7))
        panels = ((F['old'][z][2], LABEL['old']), (F['new'][z][2], LABEL['new']), (F['sim'][z][2], LABEL['sim']),
                  (F['old'][z][2] - F['new'][z][2], 'Original minus re-solve'),
                  (F['sim'][z][2] - F['new'][z][2], 'Our simulator minus re-solve'))
        for col, (img, title) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(img, origin='lower', extent=extent, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           interpolation='nearest')
            ax.set_title(f'{title}\n{float(z):g} µm' if col < 3 else f'{title}\n{float(z):g} µm', loc='left')
            ax.grid(False)
            ax.tick_params(labelsize=8)
            if col == 0:
                ax.set_ylabel('y (µm)', fontsize=9)
            if row == 2:
                ax.set_xlabel('x (µm)', fontsize=9)
        for col, v in ((3, 'old'), (4, 'sim')):
            axes[row, col].text(0.03, 0.04, 'Bx / By / Bz off the re-solve:\n' + ' / '.join(f'{d:.0f}%' for d in diff[v][z]),
                                transform=axes[row, col].transAxes, fontsize=9, va='bottom',
                                bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle='round,pad=0.3'))
        bar = fig.colorbar(im, ax=axes[row, :], shrink=0.88, pad=0.008, aspect=24)
        bar.set_label(f'Bz (µT), same scale in this row (±{vmax:.1f})')
        bar.outline.set_visible(False)
    fig.suptitle('COMSOL L1_C1, Bz: the original export, the tight-tolerance re-solve, and our simulator for the same coil',
                 fontsize=14, fontweight='bold', x=0.01, ha='left')
    fig.savefig(OUT / 'resolve_vs_old_maps.png', dpi=160)
    plt.close(fig)

# 2. line cuts through the coil at 3 and 10 um, all three components and all three versions
if ONLY in ('all', 'cuts'):
    iy = int(np.argmax(np.abs(F['new']['10.0'][2]).max(1)))
    fig, axes = plt.subplots(3, 2, figsize=(15.5, 10.5), constrained_layout=True, sharex=True)
    for col, z in enumerate(('3.0', '10.0')):
        for c in range(3):
            ax = axes[c, col]
            for v, lw, ls in (('new', 2.6, '-'), ('sim', 1.8, (0, (5, 2))), ('old', 1.8, '-')):
                ax.plot(x, F[v][z][c][iy], color=COLOR[v], lw=lw, ls=ls, label=LABEL[v])
            ax.set_title(f'{NAMES[c]} at {float(z):g} µm: original {diff["old"][z][c]:.0f}% off the re-solve, '
                         f'our simulator {diff["sim"][z][c]:.0f}%', loc='left')
            if col == 0:
                ax.set_ylabel(f'{NAMES[c]} (µT)')
    for ax in axes[-1]:
        ax.set_xlabel(f'x (µm), along y = {y[iy]:.1f} µm (the row through the strongest Bz)')
    axes[0, 1].legend(fontsize=9, loc='upper right')
    fig.suptitle('Line cuts: the re-solve and our simulator agree; the original export is too weak, more so higher up',
                 fontsize=14, fontweight='bold', x=0.01, ha='left')
    fig.savefig(OUT / 'resolve_vs_old_linecut.png', dpi=160)
    plt.close(fig)

# 3. why: amplitude per spatial frequency relative to the re-solve, decay rate per band, and where the field sits
if ONLY in ('all', 'why'):
    edges = np.array([0.0, 0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7])
    centre = np.sqrt(edges[:-1] * edges[1:])

    def amplitude(B, R):
        Mb, Mr = modes(B), modes(R)
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (K >= lo) & (K < hi) & (K > 0)
            num = sum(np.vdot(r[m], b[m]).real for b, r in zip(Mb, Mr))
            den = sum(np.vdot(r[m], r[m]).real for r in Mr)
            out.append(num / den if den > 0 else np.nan)
        return np.array(out)

    rate_edges = np.array([0.015, 0.035, 0.06, 0.09, 0.13, 0.18, 0.25, 0.35, 0.45])

    def rates(B1, B2, dz):
        M1, M2 = modes(B1), modes(B2)
        out, kc = [], []
        for lo, hi in zip(rate_edges[:-1], rate_edges[1:]):
            m = (K >= lo) & (K < hi)
            num = sum(np.vdot(a[m], b[m]).real for a, b in zip(M1, M2))
            den = sum(np.vdot(a[m], a[m]).real for a in M1)
            out.append(-np.log(num / den) / dz)
            kc.append(np.sqrt(np.mean(K[m] ** 2)))
        return np.array(kc), np.array(out)

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.6), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.25, 1.1, 0.8]))
    ax = axes[0]
    for z, mk in (('3.0', 's'), ('10.0', 'o')):
        for v in ('old', 'sim'):
            amp = amplitude(F[v][z], F['new'][z])
            ok = np.isfinite(amp)
            ax.plot(centre[ok], amp[ok], color=COLOR[v], marker=mk, ms=6, lw=2 if z == '10.0' else 1.4,
                    ls='-' if z == '10.0' else (0, (4, 2)), label=f'{LABEL[v].split(" (")[0]}, {float(z):g} µm')
    ax.axhline(1, color=BLUE, lw=2.4, label='Re-solve (reference)')
    ax.set_xscale('log')
    ax.set_xticks([0.03, 0.05, 0.1, 0.2, 0.3, 0.5])
    ax.set_xticklabels(['0.03', '0.05', '0.1', '0.2', '0.3', '0.5'])
    ax.minorticks_off()
    ax.set_ylim(0, 1.25)
    ax.set_xlabel('Spatial frequency k (rad/µm): broad structure left, fine detail right')
    ax.set_ylabel('Amplitude relative to the re-solve')
    ax.set_title('Our simulator keeps the full amplitude at every scale;\nthe original export loses the broad structure',
                 loc='left')
    ax.legend(fontsize=8.6, loc='lower right')

    ax = axes[1]
    kk = np.linspace(0, 0.43, 200)
    ax.plot(kk, kk, color=INK, lw=1.2, label='Free-space physics: rate = k')
    ax.plot(kk, np.sqrt(kk ** 2 + 0.1 ** 2), color=INK2, lw=1.2, ls=(0, (1, 2)), label='Screened: √(k² + 0.10²)')
    for v, mk, ms, lw in (('old', 'o', 6, 1.8), ('new', 'o', 11, 4.5), ('sim', 'D', 5, 1.6)):
        kc, r = rates(F[v]['3.0'], F[v]['10.0'], 7.0)
        ax.plot(kc, r, color=COLOR[v], marker=mk, ms=ms, lw=lw, alpha=0.55 if v == 'new' else 1.0,
                label=LABEL[v] + (' (under the green)' if v == 'new' else ''))
    ax.set_xlim(0, 0.43)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel('Spatial frequency k (rad/µm)')
    ax.set_ylabel('Measured decay rate from 3 to 10 µm (1/µm)')
    ax.set_title('How fast each scale fades with height:\nre-solve and simulator follow physics', loc='left')
    ax.legend(fontsize=8.4, loc='upper left')

    ax = axes[2]
    shares = []
    for z in HEIGHTS:
        M = modes(F['new'][z])
        total = sum((np.abs(m) ** 2).sum() for m in M)
        low = sum((np.abs(m[K < 0.1]) ** 2).sum() for m in M)
        shares.append(100 * low / total)
    bars = ax.bar([f'{float(z):g} µm' for z in HEIGHTS], shares, color=['#9ec5f4', '#5598e7', '#184f95'], width=0.55)
    for b, v in zip(bars, shares):
        ax.annotate(f'{v:.0f}%', (b.get_x() + b.get_width() / 2, v), textcoords='offset points', xytext=(0, 4),
                    ha='center', fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Share of the field\'s energy below k = 0.1 rad/µm (%)')
    ax.set_title('Higher up, the field is mostly\nbroad structure, where the error sits', loc='left')
    ax.grid(axis='x', visible=False)
    fig.suptitle('Why they differ: the early-stopped solve weakens the broadest structure, which dominates at 10 µm',
                 fontsize=14, fontweight='bold', x=0.01, ha='left')
    fig.savefig(OUT / 'resolve_vs_old_why.png', dpi=160)
    plt.close(fig)
    print('energy share below 0.1 rad/um:', dict(zip(HEIGHTS, [round(s, 1) for s in shares])))

print('off the re-solve, Bx/By/Bz %:', {v: {z: [round(d, 1) for d in diff[v][z]] for z in HEIGHTS} for v in diff})
