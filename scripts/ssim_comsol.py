"""SSIM and RMSE of B at 0.5 um for every method, from the COMSOL predictions already saved (CPU only).

Nothing is re-run: each method's saved prediction is compared with the COMSOL target. Relative L2 is recomputed
too and checked against the method's saved comsol.csv where one exists (gate), so the SSIM column belongs to the
same predictions as the published relative L2.

Sets:
  3um   - the 15 pairs at 3 -> 0.5 um (original exports): MAGIC-UNet run 2 raw and corrected, FISTA-925,
          FISTA-1000, CSC-LISTA strict 64, FISTA-Net run 2;
  10um  - the 15 pairs at 10 -> 0.5 um (original exports): MAGIC-UNet run B raw and corrected, FISTA-3000;
          CSC-LISTA and FISTA-Net (3 um settings) only where their figure exports hold the field;
  resolved - L1_C1 re-solved at tolerance 1e-8, 3 and 10 um, from runs/comsol_resolved/export (whatever
          components those exports hold; the figure exports keep Bz).

SSIM: Wang et al. 2004 as magic_unet.ssim (11 x 11 Gaussian, sigma 1.5), data range = target max - min, per
component, at the 0.1 um export pitch.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from magic_unet import ssim
from wiener_oracle import discover, load

COMPONENTS = ('Bx', 'By', 'Bz')


def metrics(pred, truth):
    pred, truth = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    err = pred - truth
    return dict(l2=float(np.linalg.norm(err) / np.linalg.norm(truth)), ssim=ssim(pred, truth),
                rmse=float(np.sqrt(np.mean(err ** 2))), nrmse=float(np.sqrt(np.mean(err ** 2)) /
                                                                   (truth.max() - truth.min())))


def saved_l2(path, method_filter):
    """{(config, component): relative L2} from a saved comsol.csv, rows selected by method_filter(row)."""
    if not Path(path).exists():
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        if method_filter(r):
            out[r['config'], r['component']] = float(r.get('relative_l2') or r.get('raw'))
    return out


def comsol_sets(a):
    """Yield (set, method, config, prediction [3,H,W] or [H,W], truth, saved-l2 lookup)."""
    npz = lambda p: np.load(p) if Path(p).exists() else None
    hs, m12 = Path(a.root) / 'runs/high_standoff', Path(a.root) / 'runs/m1_m2'
    for z, label in ((3.0, '3um'), (10.0, '10um')):
        pairs = [q for q in discover(a.comparison_run, a.data3, (z,)) if q.z_in == z and q.z_tg == 0.5]
        if z == 3.0:
            magic = npz(m12 / 'm1/comsol_3_to_05_predictions.npz')
            m1csv = m12 / 'm1/comsol.csv'
            sources = [
                ('MAGIC-UNet run 2 + correction', magic, 'run2_dc_est_{}',
                 saved_l2(m1csv, lambda r: r['run'] == 'run2' and r['variant'] == 'dc_est'
                          and float(r['z_input']) == 3 and float(r['z_target']) == 0.5)),
                ('MAGIC-UNet run 2 raw', magic, 'run2_raw_{}',
                 saved_l2(m1csv, lambda r: r['run'] == 'run2' and r['variant'] == 'raw'
                          and float(r['z_input']) == 3 and float(r['z_target']) == 0.5)),
                ('FISTA-925', npz(hs / 'fista_3um_dense/predictions.npz'), 'fista925_{}',
                 saved_l2(hs / 'fista_3um_dense/comsol.csv', lambda r: r['method'] == 'fista925')),
                ('FISTA-1000', npz(hs / 'fista_3um_dense/predictions.npz'), 'fista1000_{}',
                 saved_l2(hs / 'fista_3um_dense/comsol.csv', lambda r: r['method'] == 'fista1000')),
                ('CSC-LISTA strict 64', npz(m12 / 'group_b/csc_strict_64/csc_strict_64_predictions.npz'), '{}', {}),
                ('FISTA-Net run 2', npz(m12 / 'group_b/fista_net_run2/fista_net_run2_predictions.npz'), '{}', {}),
            ]
        else:
            magic = npz(hs / 'm1/comsol_10_to_05_predictions.npz')
            m1csv = hs / 'm1/comsol.csv'
            sources = [
                ('MAGIC-UNet run B + correction', magic, 'run10b_dc_est_{}',
                 saved_l2(m1csv, lambda r: r['run'] == 'run10b' and r['variant'] == 'dc_est'
                          and float(r['z_input']) == 10 and float(r['z_target']) == 0.5)),
                ('MAGIC-UNet run B raw', magic, 'run10b_raw_{}',
                 saved_l2(m1csv, lambda r: r['run'] == 'run10b' and r['variant'] == 'raw'
                          and float(r['z_input']) == 10 and float(r['z_target']) == 0.5)),
                ('FISTA-3000', npz(hs / 'fista/predictions.npz'), 'fista3000_{}',
                 saved_l2(hs / 'fista/comsol.csv', lambda r: r['method'] == 'fista3000')),
                ('CSC-LISTA (3 um settings)', npz(hs / 'export/csc_lista_part.npz'), 'comsol_{}_csc_lista', {}),
                ('FISTA-Net (3 um settings)', npz(hs / 'export/fista_net_part.npz'), 'comsol_{}_fista_net', {}),
            ]
        for q in pairs:
            truth = load(q.sharp)
            for method, store, key, ref in sources:
                if store is None or key.format(q.name) not in store.files:
                    continue
                yield label, method, q.name, store[key.format(q.name)], truth, ref


def resolved_sets(a):
    folder = Path(a.root) / 'runs/comsol_resolved/export'
    for path in sorted(folder.glob('*_part.npz')):
        store = np.load(path)
        for key in store.files:
            if key.endswith('__row') or key.endswith('_target'):
                continue
            z, data, method = key.split('_', 2)
            target = f'{z}_{data}_target'
            if target in store.files:
                yield f'L1_C1 {z[3:]}um {data}', method, 'L1_C1', store[key], store[target], {}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--root', default='.')
    p.add_argument('--comparison-run', default='runs/combined_full_200')
    p.add_argument('--data3', default='data/data3')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    rows, gate = [], []
    avg2 = lambda z: z.reshape(z.shape[0] // 2, 2, z.shape[1] // 2, 2).mean((1, 3))
    for label, method, config, pred, truth, ref in list(comsol_sets(a)) + list(resolved_sets(a)):
        pred, truth = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
        if pred.ndim == 2:
            # figure exports keep Bz only, averaged 2 x 2 (export_high_standoff.py); compare at that pitch
            t = truth[2] if truth.ndim == 3 else truth
            if t.shape != pred.shape:
                t = avg2(t)
            comps = [('Bz_avg2', pred, t)]
        else:
            comps = [(c, pred[i], truth[i]) for i, c in enumerate(COMPONENTS)]
            comps.append(('Bz_avg2', avg2(pred[2]), avg2(truth[2])))   # same footing as the figure exports
        for comp, pc, tc in comps:
            m = metrics(pc, tc)
            rows.append(dict(set=label, method=method, config=config, component=comp, **m))
            if (config, comp) in ref:
                gate.append(dict(set=label, method=method, config=config, component=comp,
                                 diff=m['l2'] - ref[config, comp]))
        print(label, method, config, {r['component']: (round(r['l2'], 4), round(r['ssim'], 4))
                                      for r in rows[-len(comps):]}, flush=True)
    with open(out / 'scores.csv', 'w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for r in rows:
        group = 'Bz_avg2' if r['component'] == 'Bz_avg2' else 'Bx_By_Bz'
        s = summary.setdefault(f"{r['set']} [{group}]", {}).setdefault(
            r['method'], dict(l2=[], ssim=[], rmse=[], configs=set(), components=set()))
        for k in ('l2', 'ssim', 'rmse'):
            s[k].append(r[k])
        s['configs'].add(r['config'])
        s['components'].add(r['component'])
    summary = {st: {m: dict(l2=float(np.mean(v['l2'])), ssim=float(np.mean(v['ssim'])),
                            rmse_ut=float(np.mean(v['rmse'])), configs=sorted(v['configs']),
                            components=sorted(v['components'])) for m, v in ms.items()}
               for st, ms in sorted(summary.items())}
    worst = max((abs(g['diff']) for g in gate), default=float('nan'))
    (out / 'summary.json').write_text(json.dumps(dict(summary=summary, gate_worst=worst, gate_rows=len(gate),
                                                      gate_passed=bool(worst <= 1e-4)), indent=2))
    print(json.dumps(dict(gate_worst=worst, gate_rows=len(gate)), indent=2))
    for st, ms in summary.items():
        print(st)
        for m, v in ms.items():
            print(f'  {m:34s} L2 {v["l2"]:.4f}  SSIM {v["ssim"]:.4f}  RMSE {v["rmse_ut"]:.4f} uT  '
                  f'({len(v["configs"])} configs, {",".join(v["components"])})')


if __name__ == '__main__':
    main()
