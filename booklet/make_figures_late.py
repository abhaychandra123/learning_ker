"""Teaching figures for Chapters 22 to 27 of the booklet (written 16 November 2026).

Run from the repository root:  python3 booklet/make_figures_late.py
NumPy and Matplotlib only, about two minutes on the Mac. One simulated layout from simulate_fields.py is used for the
band picture and the height read-out; the other two figures are formulas.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import m12_common as mc                                                     # noqa: E402
from simulate_fields import DEFAULTS, current_maps, draw_recipe, fields_from_current     # noqa: E402

OUT = ROOT / 'booklet' / 'figures'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8985'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'font.size': 10,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2, 'legend.frameon': False, 'axes.titlesize': 10.5,
    'axes.titleweight': 'bold'})


def layout():
    """One simulated layout: equivalent sheet current on the image, target at 0.5 um and clean field at 3 um."""
    s = dict(DEFAULTS)
    ny, nx, pad, dx = s['ny'], s['nx'], s['pad'], s['dx']
    recipe = draw_recipe(0, 'booklet_late_000', s)
    kx, ky = current_maps(recipe['paths_um'], recipe['width_um'], s['current_a'], (ny + 2 * pad, nx + 2 * pad), dx,
                          pad)
    d = recipe['offset_um']
    crop = (slice(pad, pad + ny), slice(pad, pad + nx))
    low, high = fields_from_current(kx, ky, dx, [0.5 + d, 3.0 + d])
    # equivalent sheet at the zero-offset plane: divergence-free part, continued upward by d (as magic_unet.py)
    wx = 2 * np.pi * np.fft.rfftfreq(kx.shape[1], dx)[None, :]
    wy = 2 * np.pi * np.fft.fftfreq(kx.shape[0], dx)[:, None]
    k = np.hypot(wx, wy)
    k2 = k ** 2
    k2[0, 0] = 1.0
    fx, fy = np.fft.rfft2(kx), np.fft.rfft2(ky)
    lon = (wx * fx + wy * fy) / k2
    fx, fy = (fx - wx * lon) * np.exp(-k * d), (fy - wy * lon) * np.exp(-k * d)
    fx[0, 0] = fy[0, 0] = 0
    j = np.stack([np.fft.irfft2(f, s=kx.shape)[crop] for f in (fx, fy)])
    return j, np.stack([c[crop] for c in low]), np.stack([c[crop] for c in high])


def radial(power, k, edges):
    idx = np.digitize(k, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    total = np.bincount(idx[ok], power[ok], len(edges) - 1)
    count = np.bincount(idx[ok], minlength=len(edges) - 1)
    return total / np.maximum(count, 1)


def bands(target, blurred):
    rng = np.random.default_rng(0)
    sigma = 0.02 * float(np.ptp(blurred[2]))                # 2% of the input's range, as in the simulator
    ft = mc.padded_fft(target[2])
    _, _, k = mc.wavenumbers(ft.shape[-1])
    edges = np.arange(0, 5.001, 0.05)
    centre = edges[:-1] + 0.025
    sharp = radial(np.abs(ft) ** 2, k, edges)
    kept = radial(np.abs(np.exp(-k * 2.5) * ft) ** 2, k, edges)
    noise = radial(np.abs(mc.padded_fft(sigma * rng.standard_normal(target[2].shape))) ** 2, k, edges)
    kc = float(centre[np.argmax(kept < noise)])
    fig, ax = plt.subplots(figsize=(8.6, 4.4), constrained_layout=True)
    top = sharp.max() * 3
    ax.axvspan(0, kc, color=BLUE, alpha=0.08, lw=0)
    ax.axvspan(kc, 2.9, color=ORANGE, alpha=0.10, lw=0)
    ax.axvspan(2.9, 5, color=GREY, alpha=0.16, lw=0)
    ax.semilogy(centre, sharp, color=BLUE, label='Target at 0.5 µm')
    ax.semilogy(centre, kept, color=ORANGE, label='What survives the blur to 3 µm')
    ax.semilogy(centre, noise, color=GREY, label='Noise in the measurement (2%)')
    for x, text in ((kc / 2, 'measured\nband'), ((kc + 2.9) / 2, 'prior\nband'), (3.95, 'noise band\n(on COMSOL targets)')):
        ax.text(x, top / 2.2, text, ha='center', va='top', fontsize=9.5, color=INK)
    ax.annotate(f'$k_c$ = {kc:.2f}', (kc, noise[np.argmax(kept < noise)]), textcoords='offset points',
                xytext=(8, -26), fontsize=9.5, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    ax.annotate('$k_n$ = 2.9', (2.9, top / 40), textcoords='offset points', xytext=(6, 0), fontsize=9.5)
    ax.set_xlim(0, 5)
    ax.set_ylim(noise.min() / 30, top)
    ax.set_xlabel('Spatial frequency k (rad/µm)')
    ax.set_ylabel('Power of $B_z$ (log scale)')
    ax.legend(fontsize=9, loc='lower left')
    fig.savefig(OUT / 'late_bands.png', dpi=150)
    plt.close(fig)
    return kc


def dial():
    k = np.linspace(0, 3, 600)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True, sharey=True)
    for ax, dz, title, curves in (
            (axes[0], 2.5, '3 → 0.5 µm', ((1.0, GREY, 'μ = 1'), (0.1, BLUE, 'μ = 0.1 (chosen for run 2)'),
                                          (0.01, AQUA, 'μ = 0.01'))),
            (axes[1], 9.5, '10 → 0.5 µm', ((0.178, BLUE, 'μ = 0.178 (chosen for run B)'),
                                           (0.0032, ORANGE, 'μ = 0.0032 (chosen for run A)'),
                                           (1e-4, AQUA, 'μ = 0.0001')))):
        h = np.exp(-k * dz)
        for mu, colour, label in curves:
            ax.plot(k, h * h / (h * h + mu), color=colour, label=label)
        ax.set_title(title, loc='left')
        ax.set_xlabel('Spatial frequency k (rad/µm)')
        ax.legend(fontsize=9, loc='upper right')
        ax.text(0.03, 0.08, 'use the\nmeasurement', transform=ax.transAxes, fontsize=9, color=INK2, va='bottom')
        ax.text(0.97, 0.08, 'keep the\nnetwork', transform=ax.transAxes, fontsize=9, color=INK2, va='bottom',
                ha='right')
    axes[0].set_ylabel('w: share of the answer taken from the measurement')
    fig.savefig(OUT / 'late_dc_gain.png', dpi=150)
    plt.close(fig)


def height(j, blurred):
    rng = np.random.default_rng(1)
    sigma = 0.02 * (blurred.max((1, 2)) - blurred.min((1, 2)))
    b = blurred + sigma[:, None, None] * rng.standard_normal(blurred.shape)
    heights = np.round(np.arange(1.5, 5.01, 0.125), 3)
    f = np.fft.fft2(j)
    _, _, k = mc.wavenumbers(j.shape[-1])
    smooth = np.real(np.fft.ifft2(f * np.exp(-k * 0.5)))    # the same current, smoothed by exp(-0.5 k)
    fig, ax = plt.subplots(figsize=(7.4, 4.3), constrained_layout=True)
    out = {}
    for label, current, colour in (('True current', j, BLUE), ('The same current, slightly too smooth', smooth, ORANGE)):
        m = 100 * mc.height_misfit(current, b, heights)
        best = float(heights[int(np.argmin(m))])
        out[label] = best
        ax.plot(heights, m, color=colour, label=f'{label}: minimum at {best:.2f} µm')
        ax.plot([best], [m.min()], marker='o', ms=7, color=colour)
    ax.axvline(3.0, color=INK2, lw=1, ls=(0, (4, 4)))
    ax.text(3.04, 3, 'true height', fontsize=9, color=INK2)
    ax.set_xlabel('Trial height h (µm)')
    ax.set_ylabel('Misfit with the measured field (%)')
    ax.set_ylim(0, None)
    ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, 1.12))
    fig.savefig(OUT / 'late_height_misfit.png', dpi=150)
    plt.close(fig)
    return out


def decay():
    wavelength = np.logspace(0, 2.3, 400)
    k = 2 * np.pi / wavelength
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    ax.axvspan(1, 4, color=GREY, alpha=0.16, lw=0)
    ax.text(2.0, 0.5, 'sharp detail of\n2 µm wires', ha='center', fontsize=9, color=INK2)
    ax.plot(wavelength, np.exp(-k * 2.5), color=BLUE, label='3 → 0.5 µm (2.5 µm of travel)')
    ax.plot(wavelength, np.exp(-k * 9.5), color=ORANGE, label='10 → 0.5 µm (9.5 µm of travel)')
    ax.set_xscale('log')
    ax.set_xlabel('The pattern repeats every ... µm')
    ax.set_ylabel('Share of its amplitude that survives')
    ax.set_xticks([1, 2, 4, 10, 20, 40, 100, 200])
    ax.set_xticklabels(['1', '2', '4', '10', '20', '40', '100', '200'])
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, loc='upper left')
    fig.savefig(OUT / 'late_decay.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    current, target, blurred = layout()
    print('k_c on this layout:', bands(target, blurred))
    dial()
    decay()
    print('height read-out:', height(current, blurred))
