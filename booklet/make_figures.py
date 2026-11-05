"""Draw every booklet figure from saved run data. Needs NumPy and Matplotlib.

Run from the repository root:  python3 booklet/make_figures.py
Inputs: runs/ metrics, runs/simulated_3_to_05_v3, booklet/data/. The file
booklet/data/booklet_export.npz comes from booklet/export_from_server.py.
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from simulate_fields import current_maps, synthesise  # noqa: E402

OUT = ROOT / 'booklet' / 'figures'
DATA = ROOT / 'booklet' / 'data'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SIGNED = LinearSegmentedColormap.from_list('signed', [BLUE, '#f0efec', RED])
SEQUENTIAL = LinearSegmentedColormap.from_list('single', ['#f4f8fd', BLUE, '#0d3f7c'])

plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
    'axes.axisbelow': True, 'lines.linewidth': 1.8, 'legend.frameon': False,
    'savefig.bbox': 'tight', 'savefig.dpi': 220, 'figure.dpi': 220,
    'axes.titlesize': 9.5, 'axes.titleweight': 'bold',
})


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig, name):
    fig.savefig(OUT / f'{name}.png')
    plt.close(fig)


def figure_filters():
    f = np.linspace(0, .6, 1200)
    h = np.exp(-2 * np.pi * f * 2.5)
    mu = 0.03162277660168379
    smooth = .001 * 4 * 2 * np.sin(np.pi * f * .1 / 2 ** .5) ** 2
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    a.semilogy(f, h, color=BLUE)
    a.set_xlabel('Spatial frequency $f$ (cycles/µm)')
    a.set_ylabel('Fraction of the signal that survives')
    a.set_title('The blur: $h(f)=e^{-2\\pi f\\,\\Delta z}$, $\\Delta z=2.5$ µm')
    for frequency, label in ((.1, '10 µm'), (.2, '5 µm'), (.5, '2 µm')):
        value = np.exp(-2 * np.pi * frequency * 2.5)
        a.plot([frequency], [value], 'o', color=BLUE, ms=4)
        a.annotate(f'{label} period\n{value:.2g}', (frequency, value), xytext=(6, 5),
                   textcoords='offset points', fontsize=7.5, color=MUTED)
    b.semilogy(f, 1 / h, color=RED, label='Exact inverse $1/h$')
    b.semilogy(f, h / (h * h + mu), color=BLUE, label='Wiener, $\\mu=0.0316$')
    b.semilogy(f, h / (h * h + smooth), color=ORANGE, label='FISTA-Net start, Eq. 16')
    b.axhline(1, color=MUTED, lw=.8)
    b.set_ylim(.1, 1e7)
    b.set_xlabel('Spatial frequency $f$ (cycles/µm)')
    b.set_ylabel('Gain applied to the input')
    b.set_title('Three ways to undo it')
    b.legend(fontsize=7.5, loc='upper left')
    fig.tight_layout()
    save(fig, 'filters')


def figure_shrink():
    x = np.linspace(-3, 3, 400)
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ax.plot(x, x, color=MUTED, lw=1, ls='--', label='No change')
    ax.plot(x, np.sign(x) * np.maximum(np.abs(x) - 1, 0), color=BLUE, label='$\\eta_\\theta(x)$, $\\theta=1$')
    ax.set_xlabel('Input $x$')
    ax.set_ylabel('Output')
    ax.legend(fontsize=7.5, loc='upper left')
    save(fig, 'shrink')


def figure_simulated():
    manifest = json.loads((ROOT / 'runs/simulated_3_to_05_v3/manifest.json').read_text())
    settings = manifest['settings']
    recipes = manifest['splits']['train']
    chosen = [next(r for r in recipes if r['family'] == family)
              for family in ('spiral', 'meander', 'array')]
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.5))
    for row, recipe in zip(axes, chosen):
        pad = settings['pad']
        shape = (settings['ny'] + 2 * pad, settings['nx'] + 2 * pad)
        kx, ky = current_maps(recipe['paths_um'], recipe['width_um'], settings['current_a'],
                              shape, settings['dx'], pad)
        wire = np.hypot(kx, ky)[pad:-pad, pad:-pad]
        item = synthesise(recipe, settings)
        row[0].imshow(wire > 0, cmap=SEQUENTIAL, origin='lower')
        limit = np.abs(item['target'][2]).max()
        row[1].imshow(item['target'][2], cmap=SIGNED, vmin=-limit, vmax=limit, origin='lower')
        limit = np.abs(item['clean'][2]).max()
        row[2].imshow(item['clean'][2], cmap=SIGNED, vmin=-limit, vmax=limit, origin='lower')
        row[3].imshow(item['input'][2], cmap=SIGNED, vmin=-limit, vmax=limit, origin='lower')
        row[0].set_ylabel(f"{recipe['family']}\nwidth {recipe['width_um']:.1f} µm", fontsize=8)
        for ax in row:
            bare(ax)
    for ax, title in zip(axes[0], ('Drawn wire', 'Target: $B_z$, 0.5 µm', 'Clean $B_z$, 3 µm',
                                   'Input: 3 µm + speckle')):
        ax.set_title(title)
    fig.tight_layout()
    save(fig, 'simulated')


def figure_atoms(data):
    atoms = data['atoms']
    fig, axes = plt.subplots(4, 8, figsize=(7.2, 3.7))
    limit = np.abs(atoms).max()
    for ax, atom in zip(axes.flat, atoms):
        ax.imshow(atom, cmap=SIGNED, vmin=-limit, vmax=limit)
        bare(ax)
    fig.tight_layout(pad=.3)
    save(fig, 'atoms')


def figure_filters_w(data):
    start = data['w_start']
    panels = (('Start: $d_1/L$', start), ('Trained, layer 1', data['w_first']),
              ('Trained, layer 16', data['w_mid']), ('Trained, layer 32', data['w_last']))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.95))
    for ax, (title, w) in zip(axes, panels):
        limit = np.abs(w).max()
        ax.imshow(w, cmap=SIGNED, vmin=-limit, vmax=limit)
        ax.set_title(title)
        ax.set_xlabel(f'peak {limit:.1e}', fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    save(fig, 'filters_w')


def figure_thresholds(data):
    start = float(data['theta_start'])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for ax, name, title in ((a, 'theta_32', '32 layers, best step 300'),
                            (b, 'theta_64', '64 layers, best step 200')):
        ratio = data[name] / start
        ax.bar(np.arange(1, len(ratio) + 1), ratio, color=BLUE, width=.75)
        ax.axhline(1, color=ORANGE, lw=1.2)
        ax.set_title(title)
        ax.set_xlabel('Layer')
    a.set_ylabel('Threshold ÷ starting value')
    a.annotate('Sparse-run floor', (31.5, 1), xytext=(13, 1.9), fontsize=7.5, color=MUTED,
               arrowprops=dict(arrowstyle='-', color=MUTED, lw=.6))
    fig.tight_layout()
    save(fig, 'thresholds')


RUNS = (
    ('Strict, 64 layers', DATA / 'strict_layers_64.json', BLUE),
    ('Strict, 32 layers', DATA / 'strict_layers_32.json', ORANGE),
    ('Plain, 16 layers', ROOT / 'runs/csc_lista_v2/full_plain/training.json', AQUA),
    ('Sparse MSE, 16', ROOT / 'runs/csc_lista_v2/full_sparse/training.json', YELLOW),
    ('Sparse relative, 16', ROOT / 'runs/csc_lista_v3/relative_16/training.json', MAGENTA),
    ('Sparse relative, 32', ROOT / 'runs/csc_lista_v3/relative_32/training.json', GREEN),
    ('LFISTA-style, 32', ROOT / 'runs/csc_lista_lfista/layers_32/training.json', VIOLET),
)


def figure_curves():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3))
    for name, path, colour in RUNS:
        log = json.loads(path.read_text())
        steps = [row['step'] for row in log]
        a.plot(steps, [100 * row['validation'] for row in log], color=colour, label=name)
        b.plot(steps, [100 * row['density'] for row in log], color=colour)
    a.set_xlabel('Training step')
    a.set_ylabel('Validation relative $L_2$ (%)')
    a.set_title('Validation error')
    a.set_ylim(40, 60)
    b.set_xlabel('Training step')
    b.set_ylabel('Density of the codes (%)')
    b.set_title('Density')
    b.set_ylim(0, 105)
    a.legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(1.1, -.2))
    fig.tight_layout()
    save(fig, 'curves')


SCORES = (
    ('ISTA-16', 64.3, 64.4, MUTED),
    ('Wiener', 60.3, 49.3, MUTED),
    ('Sparse relative, 16', 45.6, 42.0, BLUE),
    ('FISTA-32', 50.9, 41.8, ORANGE),
    ('Sparse MSE, 16', 48.6, 40.2, BLUE),
    ('FISTA-Net, run 1', 65.2, 39.4, BLUE),
    ('Plain, 16', 53.4, 39.1, BLUE),
    ('Sparse relative, 32', 42.5, 38.9, BLUE),
    ('FISTA-64', 45.3, 37.9, ORANGE),
    ('Strict, 32', 51.9, 35.4, BLUE),
    ('LFISTA-style, 32', 42.8, 34.2, BLUE),
    ('Strict, 64', 48.8, 33.8, BLUE),
    ('FISTA-Net, run 2', 47.9, 33.3, BLUE),
    ('FISTA-300, $\\lambda$ 0.03', 36.6, 32.1, ORANGE),
    ('FISTA-300, $\\lambda$ 0.009', 35.5, 30.5, ORANGE),
)


def figure_results():
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    y = np.arange(len(SCORES))
    values = [s[2] for s in SCORES]
    ax.axvspan(15, 19, color=GRID, zorder=0)
    ax.text(17, -.72, 'Speckle\nfloor', ha='center', va='bottom', fontsize=7, color=MUTED)
    ax.barh(y, values, color=[s[3] for s in SCORES], height=.68)
    for i, value in enumerate(values):
        ax.text(value + .6, i, f'{value:.1f}%', va='center', fontsize=8)
    ax.set_yticks(y, [s[0] for s in SCORES])
    ax.invert_yaxis()
    ax.set_ylim(len(SCORES) - .5, -.8)
    ax.set_xlim(0, 72)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('COMSOL relative $L_2$, mean of 15 pairs (%). Lower is better.')
    save(fig, 'results')


def figure_rank():
    fig, ax = plt.subplots(figsize=(5.8, 4))
    offsets = {'Plain, 16': (6, -3), 'Sparse MSE, 16': (-7, -2), 'Sparse relative, 16': (-6, 7),
               'Sparse relative, 32': (-6, -2), 'Strict, 32': (6, -3), 'Strict, 64': (6, -8),
               'FISTA-32': (6, 3), 'FISTA-64': (-6, -9), 'LFISTA-style, 32': (-6, 3),
               'FISTA-Net, run 1': (-6, 5), 'FISTA-Net, run 2': (3, -14), 'FISTA-300, $\\lambda$ 0.03': (6, 0),
               'FISTA-300, $\\lambda$ 0.009': (6, -6), 'Wiener': (-6, -11), 'ISTA-16': (-6, -11)}
    for name, simulated, comsol, colour in SCORES:
        ax.plot(simulated, comsol, 'o', color=colour, ms=6, mec='#fcfcfb', mew=1)
        x, y = offsets[name]
        ax.annotate(name, (simulated, comsol), xytext=(x, y), textcoords='offset points',
                    fontsize=7.5, ha='left' if x > 0 else 'right')
    ax.set_xlabel('Simulated validation error (%)')
    ax.set_ylabel('COMSOL error (%)')
    ax.set_xlim(30, 69)
    ax.set_ylim(28, 68)
    save(fig, 'rank')


def figure_depth():
    fig, ax = plt.subplots(figsize=(5, 2.9))
    ax.plot((32, 64, 300), (41.8, 37.9, 30.5), 'o-', color=ORANGE, label='FISTA, no training')
    ax.plot((32, 64), (35.4, 33.8), 'o-', color=BLUE, label='Strict Eq. 20')
    ax.plot((32,), (34.2,), 's', color=VIOLET, ms=7, label='LFISTA-style')
    ax.set_xscale('log')
    ax.set_xticks((32, 64, 128, 300), ('32', '64', '128', '300'))
    ax.minorticks_off()
    ax.set_ylim(28, 44)
    ax.set_xlabel('Iterations or layers')
    ax.set_ylabel('COMSOL error (%)')
    ax.legend(fontsize=7.5, loc='upper right')
    save(fig, 'depth')


def figure_fistanet():
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for name, colour, label in (('run_1', ORANGE, 'Run 1, paper start values'),
                                ('run_2', BLUE, "Run 2, authors' start values")):
        log = json.loads((ROOT / f'runs/fista_net/{name}/training.json').read_text())
        ax.semilogy([r['step'] for r in log], [100 * r['validation'] for r in log], color=colour,
                    lw=1.2, label=label)
    ax.axhline(60.3, color=MUTED, lw=1, ls='--')
    ax.annotate('Wiener, 60.3%', (300, 60.3), xytext=(0, -11), textcoords='offset points',
                fontsize=7.5, color=MUTED)
    ax.set_yticks((50, 100, 400, 1600, 6400), ('50', '100', '400', '1,600', '6,400'))
    ax.minorticks_off()
    ax.set_ylim(38, 16000)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Validation error (%), log scale')
    ax.legend(fontsize=7.5, loc='upper right')
    save(fig, 'fistanet_curve')


def figure_reconstruction(data):
    target = data['target']
    panels = (('Target (COMSOL, 0.5 µm)', target, None), ('Input (COMSOL, 3 µm)', data['input'], None),
              ('Wiener', data['wiener'], None), ('Strict Eq. 20, 64 layers', data['strict_64'], None))
    limit = np.percentile(np.abs(target), 99.8)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
    for ax, (title, image, _) in zip(axes[0], panels):
        ax.imshow(image, cmap=SIGNED, vmin=-limit, vmax=limit, origin='lower')
        ax.set_title(title, fontsize=8)
        bare(ax)
    bare(axes[1, 0])
    axes[1, 0].text(.5, .5, 'Error maps\n(same colour scale,\nsmaller is better)', ha='center',
                    va='center', fontsize=8, color=MUTED, transform=axes[1, 0].transAxes)
    for ax, (title, image, _) in zip(axes[1, 1:], panels[1:]):
        ax.imshow(image - target, cmap=SIGNED, vmin=-limit, vmax=limit, origin='lower')
        score = 100 * np.linalg.norm(image - target) / np.linalg.norm(target)
        ax.set_title(f'{score:.1f}% on this crop', fontsize=8)
        bare(ax)
    fig.tight_layout()
    save(fig, 'reconstruction')


def figure_reconstruction_new(data):
    target = data['target']
    new = dict(np.load(DATA / 'booklet_new.npz'))
    run2 = np.load(DATA / 'booklet_run2.npz')['fista_net_2']
    panels = (('FISTA-300, $\\lambda$ 0.009', new['fista300']), ('LFISTA-style, 32 layers', new['lfista']),
              ('FISTA-Net, run 2', run2))
    limit = np.percentile(np.abs(target), 99.8)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
    axes[0, 0].imshow(target, cmap=SIGNED, vmin=-limit, vmax=limit, origin='lower')
    axes[0, 0].set_title('Target (COMSOL, 0.5 µm)', fontsize=8)
    bare(axes[0, 0])
    bare(axes[1, 0])
    axes[1, 0].text(.5, .5, 'Error maps\n(same colour scale,\nsmaller is better)', ha='center',
                    va='center', fontsize=8, color=MUTED, transform=axes[1, 0].transAxes)
    for top, low, (title, image) in zip(axes[0, 1:], axes[1, 1:], panels):
        top.imshow(image, cmap=SIGNED, vmin=-limit, vmax=limit, origin='lower')
        top.set_title(title, fontsize=8)
        bare(top)
        low.imshow(image - target, cmap=SIGNED, vmin=-limit, vmax=limit, origin='lower')
        score = 100 * np.linalg.norm(image - target) / np.linalg.norm(target)
        low.set_title(f'{score:.1f}% on this crop', fontsize=8)
        bare(low)
    fig.tight_layout()
    save(fig, 'reconstruction_new')


def figure_schedules():
    softplus = lambda x: np.log1p(np.exp(x))
    k = np.arange(1, 8)
    mu = softplus(-.5 * k - 2)
    theta = softplus(-.2 * k - 1)
    rho = (softplus(k) - softplus(1.)) / softplus(k)
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    for values, colour, label in ((rho, AQUA, 'Momentum $\\rho^{(k)}$'),
                                  (theta, ORANGE, 'Threshold $\\theta^{(k)}$'),
                                  (mu, BLUE, 'Step $\\mu^{(k)}$')):
        ax.plot(k, values, 'o-', color=colour, ms=4, label=label)
    ax.set_xlabel('Layer $k$')
    ax.set_ylabel('Value before training')
    ax.legend(fontsize=7.5, loc='center right', bbox_to_anchor=(1, .55))
    save(fig, 'schedules')


def main():
    data = dict(np.load(DATA / 'booklet_export.npz'))
    figure_filters()
    figure_shrink()
    figure_simulated()
    figure_atoms(data)
    figure_filters_w(data)
    figure_thresholds(data)
    figure_curves()
    figure_results()
    figure_rank()
    figure_reconstruction(data)
    figure_reconstruction_new(data)
    figure_depth()
    figure_fistanet()
    figure_schedules()


if __name__ == '__main__':
    main()
