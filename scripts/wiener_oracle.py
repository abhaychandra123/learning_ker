"""Is Wiener filtering a good method for COMSOL standoff reconstruction? Oracle and held-out tests.

The Wiener filter is the best linear shift-invariant estimator given the true
second-order statistics. This script estimates those statistics from COMSOL pairs,
one group per (input height, target height). In-sample ("oracle") scores are an
upper bound for any such filter, not a deployable result. Held-out scores fit on
some configurations or heights and test on the rest.

Model: B_z = H_z X + n_z, H_z = exp(-2 pi |f| (z - z_target)), all on the same
zero-padded canvas as the existing baseline. Spectra are radial averages, and every
image is weighted by 1 / ||X||^2 so fits target the mean squared relative L2.

Filters
  tikhonov          H / (H^2 + mu), mu = 0.0316: the existing baseline (49.3% at 3 -> 0.5).
  tikhonov_best_mu  the same form with mu chosen on the fitting pairs.
  cross             E[X B*] / E|B|^2 (Wiener-Hopf); no noise model needed.
  model             H S_x / (H^2 S_x + S_n), S_x = E|X|^2, S_n = E|B - H X|^2, this group only.
  shared            the model form with S_x from every truth image at this target height
                    (all input heights share one truth), and S_n from this input height.
  shared_trend      the shared form with S_n at this height interpolated, in log, from the
                    S_n of the other input heights; this height's input is never used.
  cross_2d          E[X B*] / E|B|^2 at every 2D frequency. Memorises a few pairs, so its
                    in-sample score is optimistic. Oracle only.

Protocols
  oracle      fit and score on the same pairs.
  loco        leave one configuration out: fit on the group's other configurations.
  loso        leave one wire shape out (only when it differs from loco).
  loho        leave this input height out; configurations in-sample (shared_trend only).
  loho_loco   leave this height and this configuration out: S_x and S_n come only from
              the other configurations at the other heights (shared_trend only).
"""
import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_fields import COMPONENTS  # noqa: E402

DX, IMAGE, PAD, MU = 0.1, 1400, 128, 0.03162277660168379
N = IMAGE + 2 * PAD
DATA3 = ('C1_C2', 'C1_C2_C3', 'C1_C2_C3_C4')
# Wire shapes. IC_2um and IC_4um are the same coil; data3 arrays mirror the L1_C1 coil.
SHAPE = {'IC_2um': 'IC', 'IC_4um': 'IC', 'L1_C1': 'C1',
         'C1_C2': 'C1', 'C1_C2_C3': 'C1', 'C1_C2_C3_C4': 'C1'}
MU_GRID = 10.0 ** np.arange(-6, 0.01, 0.25)
TINY = 1e-300
PAIR_FILE = re.compile(r'^(?P<name>.+)_z(?P<z_in>\d+(?:\.\d+)?)_to_z(?P<z_tg>\d+(?:\.\d+)?)\.npz$')


# ---------------------------------------------------------------- data

@dataclass(frozen=True)
class Pair:
    name: str
    shape: str
    z_in: float
    z_tg: float
    sharp: tuple   # (path, npz key or None for a COMSOL csv)
    blurred: tuple

    @property
    def group(self):
        return (self.z_in, self.z_tg)


def load_csv(path):
    import pandas as pd
    table = pd.read_csv(path, comment='%', header=None, usecols=[2, 3, 4]).to_numpy()
    return table.T.reshape(3, IMAGE, IMAGE) * 1e6


def load(source):
    """A [3, 1400, 1400] field in uT from a COMSOL csv (Tesla) or an npz array (uT)."""
    path, key = source
    if key is None:
        return load_csv(path)
    with np.load(path) as z:
        return z[key].astype(np.float64)


def discover(run, data3):
    """Every <name>_z<in>_to_z<target>.npz in <run>/comparison, plus data3 at 3 -> 0.5."""
    pairs = []
    for path in sorted((Path(run) / 'comparison').glob('*.npz')):
        m = PAIR_FILE.match(path.name)
        if m:
            name = m['name']
            pairs.append(Pair(name, SHAPE.get(name, name), float(m['z_in']), float(m['z_tg']),
                              (path, 'sharp'), (path, 'blurred')))
    if data3 is not None:
        for config in DATA3:
            stem = Path(data3) / f'Coil_{config}' / f'L1_{config}'
            pairs.append(Pair(config, SHAPE[config], 3.0, 0.5,
                              (Path(f'{stem}_Z0p5um_600uA_0p1PR_MF_B.csv'), None),
                              (Path(f'{stem}_Z3p0um_600uA_0p1PR_MF_B.csv'), None)))
    return pairs


def manifest(path):
    """Pairs from a CSV with columns name, shape, z_input, z_target, sharp, blurred.

    Relative paths resolve against the manifest's folder. npz files use the arrays
    'sharp' and 'blurred'; csv files are COMSOL exports in Tesla.
    """
    root = Path(path).parent
    pairs = []
    with open(path, newline='') as stream:
        for row in csv.DictReader(stream):
            def source(column):
                p = root / row[column]
                return (p, None) if p.suffix == '.csv' else (p, column)
            pairs.append(Pair(row['name'], row['shape'], float(row['z_input']),
                              float(row['z_target']), source('sharp'), source('blurred')))
    return pairs


# ---------------------------------------------------------------- spectra

def grid(n=N, dx=DX):
    """|f| (cycles/um) of each rfft2 coefficient and its radial ring index."""
    f = np.hypot(np.fft.fftfreq(n, dx)[:, None], np.fft.rfftfreq(n, dx)[None, :])
    return f, np.rint(f * n * dx).astype(np.int64)


def transfer(f, dz):
    return np.exp(-2 * np.pi * dz * f)


def spectrum(x, pad=PAD):
    return np.fft.rfft2(np.pad(x, ((pad, pad), (pad, pad))))


def output(g, B, pad=PAD):
    n = B.shape[0]
    return np.fft.irfft2(B * g, s=(n, n))[pad:-pad, pad:-pad]


def relative_l2(y, x):
    return float(np.linalg.norm(y - x) / np.linalg.norm(x))


class Rings:
    """Radial averaging on one canvas."""

    def __init__(self, n=N, dx=DX):
        self.f, self.bins = grid(n, dx)
        self.count = np.bincount(self.bins.ravel())
        self.freq = np.arange(len(self.count)) / (n * dx)

    def __call__(self, a):
        """Ring means of a 2D (real or complex) array."""
        flat = self.bins.ravel()
        total = np.bincount(flat, a.real.ravel())
        if np.iscomplexobj(a):
            total = total + 1j * np.bincount(flat, a.imag.ravel())
        return total / self.count

    def spread(self, ring):
        return ring[self.bins]


@dataclass
class Item:
    """One component of one pair, with spectra on the padded canvas."""
    name: str
    shape: str
    component: int
    x: np.ndarray     # truth, cropped
    X: np.ndarray     # truth spectrum
    B: np.ndarray     # input spectrum
    w: float          # 1 / ||x||^2


def moments(items, h, rings):
    """Weighted means: 2D E|B|^2 and E[X B*]; ring E|X|^2 and E|B - H X|^2."""
    bb = np.zeros(h.shape)
    xb = np.zeros(h.shape, complex)
    xx = np.zeros(h.shape)
    nn = np.zeros(h.shape)
    for it in items:
        bb += it.w * np.abs(it.B) ** 2
        xb += it.w * it.X * np.conj(it.B)
        xx += it.w * np.abs(it.X) ** 2
        nn += it.w * np.abs(it.B - h * it.X) ** 2
    k = len(items)
    return dict(bb=bb / k, xb=xb / k, xx=rings(xx / k), nn=rings(nn / k))


def model_filter(h, sx, sn, rings):
    """H S_x / (H^2 S_x + S_n) with ring spectra S_x, S_n."""
    sx, sn = rings.spread(sx), rings.spread(sn)
    return h * sx / np.maximum(h * h * sx + sn, TINY)


def filters(m, h, rings):
    bb, xb = rings(m['bb']), rings(m['xb'])
    return dict(cross=rings.spread(xb.real / np.maximum(bb, TINY)),
                model=model_filter(h, m['xx'], m['nn'], rings))


def best_mu(items, h):
    def mean_error(mu):
        g = h / (h * h + mu)
        return np.mean([relative_l2(output(g, it.B), it.x) for it in items])
    return float(min(MU_GRID, key=mean_error))


def log_interp(z, heights, rings_by_height):
    """Interpolate ring spectra linearly in log between neighbouring heights; clamp outside."""
    heights = np.asarray(heights, float)
    order = np.argsort(heights)
    heights = heights[order]
    logs = np.log(np.maximum(np.stack([rings_by_height[i] for i in order]), TINY))
    if z <= heights[0]:
        return np.exp(logs[0])
    if z >= heights[-1]:
        return np.exp(logs[-1])
    i = int(np.searchsorted(heights, z)) - 1
    t = (z - heights[i]) / (heights[i + 1] - heights[i])
    return np.exp((1 - t) * logs[i] + t * logs[i + 1])


# ---------------------------------------------------------------- experiment

class Experiment:
    def __init__(self, pairs):
        self.pairs = pairs
        self.rings = Rings()
        self.truth_cache = {}
        # Per (name, target): ring E|X|^2 summed over components, and component count.
        self.sx = {}
        # Per (name, z_in, target): ring E|B - H X|^2 summed over components.
        self.sn = {}

    def truth(self, pair):
        key = (pair.name, pair.z_tg)
        if key not in self.truth_cache:
            x = load(pair.sharp)
            self.truth_cache[key] = [(x[i], spectrum(x[i]), 1 / np.sum(x[i] ** 2)) for i in range(3)]
        return self.truth_cache[key]

    def items(self, pair):
        b = load(pair.blurred)
        return [Item(pair.name, pair.shape, i, x, X, spectrum(b[i]), w)
                for i, (x, X, w) in enumerate(self.truth(pair))]

    def first_pass(self):
        """Ring S_x per truth and ring S_n per (configuration, height), for sharing."""
        for pair in self.pairs:
            h = transfer(self.rings.f, pair.z_in - pair.z_tg)
            items = self.items(pair)
            if (pair.name, pair.z_tg) not in self.sx:
                self.sx[(pair.name, pair.z_tg)] = sum(
                    self.rings(it.w * np.abs(it.X) ** 2) for it in items) / 3
            self.sn[(pair.name, pair.z_in, pair.z_tg)] = sum(
                self.rings(it.w * np.abs(it.B - h * it.X) ** 2) for it in items) / 3
            print(f'  spectra {pair.name} {pair.z_in:g} -> {pair.z_tg:g}', flush=True)

    def shared_sx(self, z_tg, exclude=()):
        rings = [v for (name, t), v in self.sx.items() if t == z_tg and name not in exclude]
        return np.mean(rings, axis=0) if rings else None

    def sn_mean(self, names, z_in, z_tg):
        rings = [self.sn[(n, z_in, z_tg)] for n in names if (n, z_in, z_tg) in self.sn]
        return np.mean(rings, axis=0) if rings else None

    def trend_sn(self, names, z_in, z_tg):
        """S_n at z_in interpolated from other input heights of the same configurations."""
        by_height = {}
        for (n, zi, zt) in self.sn:
            if zt == z_tg and zi != z_in and n in names:
                by_height.setdefault(zi, []).append(self.sn[(n, zi, zt)])
        if len(by_height) < 2:
            return None
        heights = sorted(by_height)
        return log_interp(z_in, heights, [np.mean(by_height[z], axis=0) for z in heights])

    def score(self, filters_, items, protocol, pair_group, rows):
        z_in, z_tg = pair_group
        for method, g in filters_.items():
            for it in items:
                rows.append(dict(z_input=z_in, z_target=z_tg, dz=z_in - z_tg, config=it.name,
                                 component=COMPONENTS[it.component], method=method,
                                 protocol=protocol, relative_l2=relative_l2(output(g, it.B), it.x)))

    def group(self, key, rows):
        z_in, z_tg = key
        group_pairs = [p for p in self.pairs if p.group == key]
        names = [p.name for p in group_pairs]
        h = transfer(self.rings.f, z_in - z_tg)
        items = [it for p in group_pairs for it in self.items(p)]
        info = dict(z_input=z_in, z_target=z_tg, dz=z_in - z_tg, configs=names)

        # Oracle.
        m = moments(items, h, self.rings)
        mu = best_mu(items, h)
        info['best_mu'] = mu
        f = dict(tikhonov=h / (h * h + MU), tikhonov_best_mu=h / (h * h + mu))
        f.update(filters(m, h, self.rings))
        f['shared'] = model_filter(h, self.shared_sx(z_tg), m['nn'], self.rings)
        f['cross_2d'] = m['xb'] / np.maximum(m['bb'], TINY)
        self.score(f, items, 'oracle', key, rows)
        snr = transfer(self.rings.freq, z_in - z_tg) ** 2 * m['xx'] / np.maximum(m['nn'], TINY)
        below = np.nonzero((snr < 1) & (self.rings.freq > 0))[0]
        info['snr_crossover_cycles_per_um'] = float(self.rings.freq[below[0]]) if len(below) else None
        info['S_x'], info['S_n'], info['snr'] = m['xx'], m['nn'], snr

        # Leave height out, configurations in-sample.
        sn = self.trend_sn(set(names), z_in, z_tg)
        if sn is not None:
            self.score({'shared_trend': model_filter(h, self.shared_sx(z_tg), sn, self.rings)},
                       items, 'loho', key, rows)

        if len(names) < 2:
            return info
        shapes = sorted({p.shape for p in group_pairs})
        folds = [('loco', [n]) for n in names]
        if 1 < len(shapes) < len(names):
            folds += [('loso', [p.name for p in group_pairs if p.shape == s]) for s in shapes]
        for protocol, held in folds:
            train = [it for it in items if it.name not in held]
            test = [it for it in items if it.name in held]
            m = moments(train, h, self.rings)
            mu = best_mu(train, h)
            f = dict(tikhonov_best_mu=h / (h * h + mu))
            f.update(filters(m, h, self.rings))
            sx = self.shared_sx(z_tg, exclude=held)
            if sx is not None:
                f['shared'] = model_filter(h, sx, m['nn'], self.rings)
            self.score(f, test, protocol, key, rows)
            if protocol == 'loco':
                others = set(names) - set(held)
                sn = self.trend_sn(others, z_in, z_tg)
                if sn is not None and sx is not None:
                    self.score({'shared_trend': model_filter(h, sx, sn, self.rings)},
                               test, 'loho_loco', key, rows)
        return info

    def run(self, keys):
        rows, infos = [], []
        print('first pass: spectra per configuration and height', flush=True)
        self.first_pass()
        for key in keys:
            print(f'group {key[0]:g} -> {key[1]:g}', flush=True)
            infos.append(self.group(key, rows))
        return rows, infos


# ---------------------------------------------------------------- reporting

def summarise(rows):
    """Mean relative L2 per (group, method, protocol), and per configuration."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r['z_input'], r['z_target'], r['method'], r['protocol'], r['config'])].append(r['relative_l2'])
    out = defaultdict(lambda: dict(by_config={}))
    for (zi, zt, method, protocol, config), v in acc.items():
        out[(zi, zt, f'{method}@{protocol}')]['by_config'][config] = float(np.mean(v))
    for entry in out.values():
        # Mean over components and configurations, components weighted equally.
        entry['mean'] = float(np.mean(list(entry['by_config'].values())))
    return out


TABLE = ['tikhonov@oracle', 'tikhonov_best_mu@oracle', 'model@oracle', 'shared@oracle',
         'tikhonov_best_mu@loco', 'model@loco', 'shared@loco', 'shared_trend@loho',
         'shared_trend@loho_loco']


def table(summary, infos):
    lines = ['| z_in → z_target | configs | best μ | SNR = 1 at (cyc/µm) | ' +
             ' | '.join(TABLE) + ' |', '|' + '---|' * (4 + len(TABLE))]
    for info in infos:
        zi, zt = info['z_input'], info['z_target']
        cells = []
        for k in TABLE:
            s = summary.get((zi, zt, k))
            cells.append(f"{100 * s['mean']:.1f}" if s else '—')
        cross = info['snr_crossover_cycles_per_um']
        lines.append(f"| {zi:g} → {zt:g} | {len(info['configs'])} | {info['best_mu']:.2g} | "
                     f"{'—' if cross is None else f'{cross:.3f}'} | " + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def plot(summary, infos, freq, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    targets = sorted({i['z_target'] for i in infos})
    keep = (freq > 0) & (freq <= 1.0)
    for zt in targets:
        group = sorted((i for i in infos if i['z_target'] == zt), key=lambda i: i['z_input'])
        if not group:
            continue
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        cmap = plt.get_cmap('viridis')
        for j, info in enumerate(group):
            c = cmap(j / max(1, len(group) - 1))
            ax[0].semilogy(freq[keep], info['S_n'][keep], color=c, label=f"{info['z_input']:g} um")
            ax[1].semilogy(freq[keep], info['snr'][keep], color=c)
        sx = np.mean([i['S_x'] for i in group], axis=0)
        ax[0].semilogy(freq[keep], sx[keep], 'k--', label='$S_x$ (truth)')
        ax[0].set(xlabel='|f| (cycles/um)', title=f'$S_n = E|B_z - H_z X|^2$, target {zt:g} um',
                  ylim=(1e-10 * sx[keep].max(), None))
        ax[0].legend(fontsize=7, ncol=2)
        ax[1].axhline(1, color='k', lw=.6)
        ax[1].set(xlabel='|f| (cycles/um)', title='Input SNR $H_z^2 S_x / S_n$', ylim=(1e-8, None))
        # One curve per configuration, so heights with different configurations do not mix.
        z = [i['z_input'] for i in group]
        configs = sorted({c for i in group for c in i['configs']})
        styles = [('tikhonov@oracle', ':', 'existing mu'), ('model@oracle', '-', 'Wiener oracle'),
                  ('model@loco', '--', 'Wiener, other config'),
                  ('shared_trend@loho_loco', '-.', 'Wiener, other config + height')]
        for ci, config in enumerate(configs):
            for k, ls, label in styles:
                pts = [(zi, 100 * summary[(zi, zt, k)]['by_config'][config]) for zi in z
                       if (zi, zt, k) in summary and config in summary[(zi, zt, k)]['by_config']]
                if pts:
                    ax[2].plot(*zip(*pts), ls, color=f'C{ci}', marker='o', ms=3,
                               label=f'{config}: {label}')
        ax[2].set(xlabel='input height (um)', ylabel='relative L2 (%)',
                  title=f'Reconstruction to {zt:g} um')
        ax[2].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(output_dir / f'standoff_target_{zt:g}um.png', dpi=130)
        plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--output', required=True)
    p.add_argument('--comparison-run', default='runs/combined_full_200',
                   help='Run whose comparison/ folder holds <name>_z<in>_to_z<target>.npz pairs')
    p.add_argument('--data3', default='data/data3',
                   help="Folder of the three data3 configurations (3 -> 0.5 um), or 'none'")
    p.add_argument('--manifest',
                   help='CSV of pairs (name, shape, z_input, z_target, sharp, blurred); '
                        'replaces discovery from --comparison-run and --data3')
    p.add_argument('--groups', nargs='+', metavar='IN:TARGET',
                   help='Score only these height pairs, e.g. 3:0.5 (spectra still use all pairs)')
    return p


def main():
    args = parser().parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    if args.manifest:
        pairs = manifest(args.manifest)
    else:
        data3 = None if args.data3.lower() == 'none' or not Path(args.data3).exists() else args.data3
        if data3 is None and args.data3.lower() != 'none':
            print(f'data3 folder {args.data3} not found; continuing without it', flush=True)
        pairs = discover(args.comparison_run, data3)
    seen = set()
    for p in pairs:
        if (p.name, p.z_in, p.z_tg) in seen:
            raise SystemExit(f'duplicate pair {p.name} {p.z_in} -> {p.z_tg}')
        seen.add((p.name, p.z_in, p.z_tg))
    keys = sorted({p.group for p in pairs}, key=lambda k: (k[1], k[0]))
    if args.groups:
        wanted = {tuple(float(v) for v in g.split(':')) for g in args.groups}
        keys = [k for k in keys if k in wanted]
    print(f'{len(pairs)} pairs in {len(keys)} groups', flush=True)

    experiment = Experiment(pairs)
    rows, infos = experiment.run(keys)
    with (out / 'scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows)
    freq = experiment.rings.freq
    np.savez(out / 'spectra.npz', freq=freq, **{
        f"{k}_{i['z_input']:g}_to_{i['z_target']:g}": i[k]
        for i in infos for k in ('S_x', 'S_n', 'snr')})
    plot(summary, infos, freq, out)
    report = table(summary, infos)
    (out / 'table.md').write_text(report + '\n')
    (out / 'results.json').write_text(json.dumps(dict(
        pairs=[dict(name=p.name, shape=p.shape, z_input=p.z_in, z_target=p.z_tg) for p in pairs],
        mu_existing=MU, pad=PAD,
        groups=[{k: v for k, v in i.items() if k not in ('S_x', 'S_n', 'snr')} for i in infos],
        summary=[dict(z_input=k[0], z_target=k[1], method=k[2], **v) for k, v in summary.items()]),
        indent=2))
    print(report)


if __name__ == '__main__':
    main()
