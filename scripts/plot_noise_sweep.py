"""Figures for the noise sweep (README_NOISE_SSIM.md §2), from the copied scores.csv files. Runs on the Mac.

  python scripts/plot_noise_sweep.py

(a, b) B relative L2 at 0.5 um against noise at 3 and 10 um, MAGIC-UNet (corrected) and FISTA, on the layouts both
    were scored on; FISTA with lambda re-tuned per level (an oracle choice that favours it) and with lambda fixed.
(c) J SSIM of MAGIC-UNet against noise (20 layouts), with the two values Reed et al. report as grey reference marks.
Writes final_evaluation/04_figures/p11_noise_ssim/noise_sweep.png and copies it to booklet/figures/ns_noise_sweep.png.
"""
import csv
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'final_evaluation/03_results/p11_noise_ssim'
OUT = ROOT / 'final_evaluation/04_figures/p11_noise_ssim'
BLUE, ORANGE, GREY, INK, MUTED = '#2a78d6', '#eb6834', '#8a8984', '#0b0b0b', '#52514e'
TICKS = [0, 1, 2, 5, 10, 25, 50, 100]


def rows(name):
    path = RES / name / 'scores.csv'
    return list(csv.DictReader(open(path))) if path.exists() else []


def levels(rs):
    return sorted({float(r['level']) for r in rs if r['level'] != 'native'})


def magic_curve(rs, key, layouts=None):
    out = []
    for lv in levels(rs):
        v = [float(r[key]) for r in rs if r['level'] != 'native' and float(r['level']) == lv
             and (layouts is None or r['layout'] in layouts)]
        out.append((lv, np.mean(v)))
    return out


def fista_curve(rs, fixed=False):
    """Best lambda multiple per level (mean over that level's layouts), or lambda x 1 throughout (fixed)."""
    out = []
    for lv in levels(rs):
        at = [r for r in rs if r['level'] != 'native' and float(r['level']) == lv]
        by = {}
        for r in at:
            by.setdefault(float(r['lambda_mult']), []).append(float(r['fista_l2']))
        out.append((lv, np.mean(by[1.0]) if fixed else min(np.mean(v) for v in by.values())))
    return out


def style(ax):
    ax.set_xscale('symlog', linthresh=1, linscale=0.6)
    ax.set_xticks(TICKS)
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.set_xlim(-0.1, 110)
    ax.axvspan(1, 2.5, color='#ecebe8', zorder=0)
    ax.grid(axis='y', color='#e4e3df', linewidth=0.8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#b9b8b2')
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel('Noise added (% of each component\'s range)', color=MUTED, fontsize=9.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    cap = 1.5
    for ax, z, title in ((axes[0], '3um', '(a) 3 → 0.5 µm: field error'), (axes[1], '10um', '(b) 10 → 0.5 µm: field error')):
        f = rows(f'fista_{z}')
        layouts = {r['layout'] for r in f} or None
        series = [(magic_curve(rows(f'magic_{z}'), 'dc_l2', layouts), BLUE, 'o', 'MAGIC-UNet + correction')]
        if f:
            series.append((fista_curve(f), ORANGE, 's', 'FISTA, λ re-tuned per level'))
            series.append((fista_curve(f, fixed=True), ORANGE, 'hollow', 'FISTA, λ fixed'))
        for curve, colour, mk, label in series:
            if not curve:
                continue
            x = [p[0] for p in curve]
            y = [min(p[1], cap) * 100 for p in curve]
            if mk == 'hollow':
                ax.plot(x, y, color=colour, lw=1.2, ls='--', marker='o', ms=6, markerfacecolor='white',
                        markeredgecolor=colour, markeredgewidth=1.4, label=label)
            else:
                ax.plot(x, y, color=colour, lw=2, marker=mk, ms=6, markeredgecolor='white', markeredgewidth=1.2,
                        label=label)
        style(ax)
        ax.set_ylim(0, cap * 100 + 8)
        ax.set_ylabel('B error at 0.5 µm (relative L2, %)', color=MUTED, fontsize=9.5)
        ax.set_title(title, loc='left', color=INK, fontsize=11)
        ax.text(1.6, 3, 'training\nnoise', ha='center', va='bottom', fontsize=7.5, color=MUTED)
        n = len(layouts) if layouts else 20
        ax.text(0.98, 0.03, f'{n} layouts' + ('' if f else '; FISTA still running'), transform=ax.transAxes,
                ha='right', fontsize=7.5, color=MUTED)
        ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    axes[0].text(0.02, 0.70, 'Points on the top edge are off scale\n(MAGIC-UNet\'s correction, tuned at low\n'
                 'noise, amplifies the noise above 10 %)', transform=axes[0].transAxes, ha='left', va='top',
                 fontsize=7.5, color=MUTED)

    b = axes[2]
    for z, ls, mk in (('10um', '-', 'o'), ('3um', '--', 's')):
        m = magic_curve(rows(f'magic_{z}'), 'J_ssim')
        if m:
            b.plot([p[0] for p in m], [p[1] for p in m], ls=ls, color=BLUE, lw=2, marker=mk, ms=6,
                   markeredgecolor='white', markeredgewidth=1.2,
                   label=f'MAGIC-UNet, {z.replace("um", " µm")} (ours)')
    # Reed et al.: SSIM 0.99 without noise, 0.84 at sigma_n = 1 (about 67 % of the range on our layouts)
    b.scatter([0, 67], [0.99, 0.84], marker='D', s=46, color=GREY, zorder=5,
              label='Reed et al. (50 µm standoff, own training noise)')
    style(b)
    b.set_ylim(0, 1.03)
    b.set_ylabel('Current-density SSIM (J)', color=MUTED, fontsize=9.5)
    b.set_title('(c) Current-density SSIM, as the paper reports it', loc='left', color=INK, fontsize=11)
    b.annotate('paper at σ_n = 1', (67, 0.84), textcoords='offset points', xytext=(-6, 8), ha='right',
               fontsize=7.5, color=MUTED)
    b.text(1.6, 0.03, 'training\nnoise', ha='center', va='bottom', fontsize=7.5, color=MUTED)
    b.text(0.98, 0.97, '20 layouts', transform=b.transAxes, ha='right', va='top', fontsize=7.5, color=MUTED)
    b.legend(frameon=False, fontsize=8.5, loc='lower left', bbox_to_anchor=(0.0, 0.1))
    fig.supxlabel('Simulated layouts (free-space simulator). Shaded: the 1–2.5 % noise the networks were trained '
                  'with. Filled orange: FISTA\'s λ re-tuned per level (best of ×1, ×3, ×10, chosen on the same layouts, '
                  'which favours FISTA); hollow: λ fixed at its training-noise value, as MAGIC-UNet\'s settings are.',
                  fontsize=8, color=MUTED, x=0.0, ha='left')
    path = OUT / 'noise_sweep.png'
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    shutil.copy(path, ROOT / 'booklet/figures/ns_noise_sweep.png')
    print('wrote', path)


if __name__ == '__main__':
    main()
