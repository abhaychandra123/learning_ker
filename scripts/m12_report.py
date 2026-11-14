"""Tables for M1_M2_RESULTS.md from the saved M1 and M2 outputs (NumPy/CSV only; runs anywhere).

  python3 scripts/m12_report.py --root runs/m1_m2 > runs/m1_m2/tables.md
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

C = ('Bx', 'By', 'Bz')


def read(path):
    return list(csv.DictReader(open(path)))


def f(x, d=1):
    return '–' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{100 * x:.{d}f}%'


def mean(rows, key):
    v = [float(r[key]) for r in rows]
    return float(np.mean(v)) if v else float('nan')


def split(rows, key):
    """15-pair mean at 3 -> 0.5, C1 family (12 rows), IC_4um (3 rows)."""
    three = [r for r in rows if float(r.get('z_input', 3)) == 3.0 and float(r.get('z_target', .5)) == 0.5]
    return (mean(three, key), mean([r for r in three if r['shape'] == 'C1'], key),
            mean([r for r in three if r['shape'] == 'IC'], key))


def m1_tables(root):
    m1 = root / 'm1'
    out = []
    gate = json.loads((m1 / 'gate.json').read_text())
    out.append(f"Reproduction gate: worst |re-run − saved| = {gate['worst_abs_diff']:.1e} over 120 rows; "
               f"float32 read-out vs float64 NumPy: {gate['readout_precision']:.1e}.\n")
    curve = json.loads((m1 / 'mu_curve.json').read_text())
    sim = read(m1 / 'sim_scores.csv')
    out.append('### M1 on simulation (50 validation layouts per standoff, target 0.5 µm)\n')
    out.append('| Run | Standoff | Raw | Corrected, true h | Corrected, h* | μ(h) | median h* | median |h*−h|/h | '
               'worst |h*−h|/h | edge |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for run in curve:
        for z, mu in zip(curve[run]['standoffs'], curve[run]['mu']):
            rows = [r for r in sim if r['run'] == run and float(r['standoff']) == z]
            raw = np.mean([[float(r[f'raw_{c}']) for c in C] for r in rows])
            dt = np.mean([[float(r[f'dc_true_{c}']) for c in C] for r in rows])
            de = np.mean([[float(r[f'dc_est_{c}']) for c in C] for r in rows])
            h = np.array([float(r['h_star']) for r in rows])
            relerr = np.abs(h - z) / z
            edges = sum(r['edge'] == 'True' for r in rows)
            out.append(f'| {run} | {z:g} | {f(raw)} | {f(dt)} | {f(de)} | {mu:.2g} | {np.median(h):.2f} | '
                       f'{f(np.median(relerr), 0)} | {f(relerr.max(), 0)} | {edges} |')
    out.append('')
    rows = read(m1 / 'comsol.csv')
    out.append('### M1 on COMSOL, 3 → 0.5 µm (relative L2, mean over components)\n')
    out.append('| Run | Variant | 15 pairs | C1 family (12) | IC_4um (3) | low-passed target, 15 pairs | '
               'in-band | prior band |')
    out.append('|---|---|---:|---:|---:|---:|---:|---:|')
    for run in curve:
        for v in ('raw', 'dc_true', 'dc_est'):
            sel = [r for r in rows if r['run'] == run and r['variant'] == v]
            a, b, c = split(sel, 'raw')
            lp = split(sel, 'lowpass')[0]
            ib = split(sel, 'in_band')[0]
            pb = split(sel, 'prior_band')[0]
            out.append(f'| {run} | {v} | {f(a)} | {f(b)} | {f(c)} | {f(lp)} | {f(ib)} | {f(pb)} |')
    out.append('')
    out.append('### M1 on COMSOL, every pair (mean over components)\n')
    out.append('| Run | Pair | z_in | h* | Raw | Corrected, true h | Corrected, h* |')
    out.append('|---|---|---:|---:|---:|---:|---:|')
    keys = sorted({(r['run'], r['config'], float(r['z_input']), float(r['z_target'])) for r in rows},
                  key=lambda t: (t[0], t[1], t[2]))
    for run, config, zi, zt in keys:
        sel = lambda v: [r for r in rows if (r['run'], r['config'], float(r['z_input']), float(r['z_target']),
                                             r['variant']) == (run, config, zi, zt, v)]
        h = float(sel('raw')[0]['h_star'])
        edge = ' (edge)' if sel('raw')[0]['edge'] == 'True' else ''
        name = config if zt == 0.5 else f'{config} (→{zt:g})'
        out.append(f'| {run} | {name} | {zi:g} | {h:.2f}{edge} | {f(mean(sel("raw"), "raw"))} | '
                   f'{f(mean(sel("dc_true"), "raw"))} | {f(mean(sel("dc_est"), "raw"))} |')
    out.append('')
    return out


FOURWAY_FOLDERS = ('fourway_partial', 'fourway_fista_depth')  # same function, same k_n
# m1_rescore.py scores every variant with the same four-way function, so seed rows need no separate m2_fourway run.
M1_ROWS = {('m1', 'run2', 'dc_est'): 'MAGIC-UNet run 2 + correction at h*',
           ('m1', 'run2', 'dc_true'): 'MAGIC-UNet run 2 + correction at true h',
           ('m1', 'run3', 'dc_est'): 'MAGIC-UNet run 3 + correction at h*',
           ('m1_seeds', 'magic_seed1', 'raw'): 'MAGIC-UNet run 2, seed 1',
           ('m1_seeds', 'magic_seed2', 'raw'): 'MAGIC-UNet run 2, seed 2',
           ('m1_seeds', 'magic_seed1', 'dc_est'): 'MAGIC-UNet run 2, seed 1 + correction at h*',
           ('m1_seeds', 'magic_seed2', 'dc_est'): 'MAGIC-UNet run 2, seed 2 + correction at h*'}


def fourway_rows(root):
    """All four-way rows at 3 -> 0.5 um: m2_fourway folders, plus the corrected variants from the M1 re-scores."""
    rows = []
    for folder in FOURWAY_FOLDERS:
        path = root / folder / 'fourway.csv'
        if path.exists():
            # fista300_rerun is the gate copy of fista300 (same predictions, re-run); keep the original only.
            rows += [r for r in read(path) if r['method'] != 'fista300_rerun']
    for (folder, run, variant), label in M1_ROWS.items():
        path = root / folder / 'comsol.csv'
        if path.exists():
            for r in read(path):
                if (r['run'], r['variant']) == (run, variant) and \
                        (float(r['z_input']), float(r['z_target'])) == (3.0, 0.5):
                    rows.append(dict(r, method=label))
    return rows


def fourway_tables(root):
    rows = fourway_rows(root)
    if not rows:
        return []
    out = ['### Four-way scores, 3 → 0.5 µm (mean over components; k_n from M2a)\n',
           '| Method | Raw 15 | Raw C1 (12) | Raw IC (3) | Low-passed 15 | Low-passed C1 | Low-passed IC | '
           'In-band 15 | Prior band 15 | Noise band 15 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    methods = list(dict.fromkeys(r['method'] for r in rows))
    for m in sorted(methods, key=lambda m: mean([r for r in rows if r['method'] == m], 'raw')):
        sel = [r for r in rows if r['method'] == m]
        raw, lp = split(sel, 'raw'), split(sel, 'lowpass')
        out.append(f'| {m} | ' + ' | '.join(f(x) for x in raw + lp) + ' | ' +
                   ' | '.join(f(split(sel, k)[0]) for k in ('in_band', 'prior_band', 'noise_band')) + ' |')
    t = rows[0]
    out.append('')
    sel = [r for r in rows if r['method'] == methods[0]]
    out.append('Target energy share (fraction of the target norm, mean over the 15 pairs): '
               f"in-band {f(split(sel, 'target_in')[0])}, prior band {f(split(sel, 'target_prior')[0])}, "
               f"noise band {f(split(sel, 'target_noise')[0])}.\n")
    return out


def fista_tables(root):
    path = root / 'fista_depth' / 'results.json'
    if not path.exists():
        return []
    r = json.loads(path.read_text())
    rows = read(root / 'fista_depth' / 'comsol.csv')
    out = ['### FISTA beyond 300 iterations\n', '| Depth | λ scale | λ | 16-crop selection | 64-crop validation | '
           'COMSOL 15 | C1 (12) | IC (3) |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for depth, scale in r['choice'].items():
        sel = [x for x in rows if x['method'] == f'fista{depth}']
        for x in sel:
            x['shape'] = 'IC' if x['config'] == 'IC_4um' else 'C1'
        a, b, c = split(sel, 'relative_l2')
        out.append(f"| {depth} | {float(scale):g} | {0.3 * float(scale):.3g} | "
                   f"{f(r['trials'][str(float(scale))][depth])} | {f(r['validation'][f'fista{depth}'])} | "
                   f'{f(a)} | {f(b)} | {f(c)} |')
    out.append('\nSelection trials (mean relative L2 on the first 16 validation crops):\n')
    out.append('| λ scale | ' + ' | '.join(f'FISTA-{d}' for d in r['choice']) + ' |')
    out.append('|---:|' + '---:|' * len(r['choice']))
    for scale, by_depth in sorted(r['trials'].items(), key=lambda t: float(t[0])):
        out.append(f'| {float(scale):g} | ' + ' | '.join(f(by_depth[d]) for d in r['choice']) + ' |')
    out.append('')
    return out


def seed_tables(root):
    out = []
    runs = {'seed 0 (run 2)': Path('runs/magic_unet/run2_server_outputs/results.json')}
    for s in (1, 2):
        runs[f'seed {s}'] = root / f'magic_seed{s}' / 'results.json'
    have = {k: json.loads(p.read_text()) for k, p in runs.items() if p.exists()}
    if len(have) < 2:
        return out
    out += ['### MAGIC-UNet run 2, three seeds (standard protocol of train_magic_unet.py)\n',
            '| Seed | Stride | Simulated validation | COMSOL 3 → 0.5, 15 pairs | 2 → 0.5 | 4 → 0.5 | IC_2um 2 → 1 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for k, r in have.items():
        c = r['comsol_mean_relative_l2']
        out.append(f"| {k} | {r['stride']} | {f(r['sim_validation_relative_l2'])} | {f(c['3->0.5']['mean'])} | "
                   f"{f(c['2->0.5']['mean'])} | {f(c['4->0.5']['mean'])} | {f(c['2->1']['mean'])} |")
    v = [r['comsol_mean_relative_l2']['3->0.5']['mean'] for r in have.values()]
    out.append(f'\nRange at 3 → 0.5 µm over {len(v)} seeds: {f(min(v), 2)} to {f(max(v), 2)} '
               f'(spread {100 * (max(v) - min(v)):.2f} points).\n')
    out += seed_correction_tables(root)
    return out


def seed_correction_tables(root):
    """Amendment A4: per-variant, per-subset seed ranges, and the all-seeds-below rule against FISTA-1000."""
    seeds = {'seed 0 (run 2)': (root / 'm1' / 'comsol.csv', 'run2')}
    for s in (1, 2):
        seeds[f'seed {s}'] = (root / 'm1_seeds' / 'comsol.csv', f'magic_seed{s}')
    scores = {}
    for name, (path, run) in seeds.items():
        if not path.exists():
            return [f'(seed re-score: missing {path})\n']
        rows = [r for r in read(path) if r['run'] == run]
        scores[name] = {v: split([r for r in rows if r['variant'] == v], 'raw') for v in ('raw', 'dc_est')}
    gate = root / 'm1_seeds' / 'gate.json'
    fista = [x for x in read(root / 'fista_depth' / 'comsol.csv') if x['method'] == 'fista1000']
    for x in fista:
        x['shape'] = 'IC' if x['config'] == 'IC_4um' else 'C1'
    ref = split(fista, 'relative_l2')
    subsets = ('15 pairs', 'C1 family (12)', 'IC_4um (3)')
    out = ['### Seeds re-scored with the correction (amendment A3), 3 → 0.5 µm\n']
    if gate.exists():
        out.append(f"Seed re-score gate: worst |re-run − saved| = {json.loads(gate.read_text())['worst_abs_diff']:.1e}.\n")
    out += ['| Seed | Variant | ' + ' | '.join(subsets) + ' |', '|---|---|---:|---:|---:|']
    for v, label in (('raw', 'raw'), ('dc_est', 'corrected at h*')):
        for name in scores:
            out.append(f'| {name} | {label} | ' + ' | '.join(f(x, 2) for x in scores[name][v]) + ' |')
    out.append('| FISTA-1000 (deterministic) | | ' + ' | '.join(f(x, 2) for x in ref) + ' |')
    out += ['', '| Variant | Subset | Range over 3 seeds (points) | Seeds below FISTA-1000 | A4 verdict |',
            '|---|---|---:|---:|---|']
    for v, label in (('raw', 'raw'), ('dc_est', 'corrected at h*')):
        for i, sub in enumerate(subsets):
            vals = [scores[n][v][i] for n in scores]
            below = sum(x < ref[i] for x in vals)
            verdict = ('beats FISTA-1000' if below == 3 else 'not distinguishable on this test set' +
                       (' (FISTA-1000 lower than every seed)' if below == 0 else ''))
            out.append(f'| {label} | {sub} | {100 * (max(vals) - min(vals)):.2f} | {below} of 3 | {verdict} |')
    out.append('')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--root', default='runs/m1_m2')
    root = Path(p.parse_args().root)
    lines = []
    for part in (m1_tables, fourway_tables, fista_tables, seed_tables):
        try:
            lines += part(root)
        except FileNotFoundError as e:
            lines.append(f'({part.__name__}: missing {e.filename})\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
