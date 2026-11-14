"""M2b (M1_M2_PROTOCOL.md): four-way scores for every saved 3 -> 0.5 um COMSOL prediction.

Sources are given as NAME=NPZ:KEYPATTERN[:CSV:METHOD], where KEYPATTERN contains {config} (for example
'{config}_B' for MAGIC-UNet files, '{config}' for Group B files, 'fista1000_{config}'). If a saved comsol.csv and
its method label are given, the raw relative L2 recomputed here must match it within 1e-4 (gate); a source that
fails is reported and left out. k_n comes from M2a (runs/m1_m2/cutoff/cutoff.json).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import m12_common as mc
from wiener_oracle import discover, load

COMPONENTS = ('Bx', 'By', 'Bz')


def parse(spec):
    name, rest = spec.split('=', 1)
    parts = rest.split(':')
    return dict(name=name, npz=parts[0], key=parts[1],
                csv=parts[2] if len(parts) > 2 else None, method=parts[3] if len(parts) > 3 else None)


def saved_scores(path, method):
    out = {}
    for r in csv.DictReader(open(path)):
        if method is not None and r.get('method', method) != method:
            continue
        if 'z_input' in r and (float(r['z_input']), float(r['z_target'])) != (3.0, 0.5):
            continue
        out[(r['config'], r['component'])] = float(r['relative_l2'])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--source', action='append', required=True)
    p.add_argument('--comparison-run', default='runs/combined_full_200')
    p.add_argument('--data3', default='data/data3')
    p.add_argument('--cutoff', default='runs/m1_m2/cutoff/cutoff.json')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    k_noise = json.loads(Path(a.cutoff).read_text())['k_noise']
    pairs = [q for q in discover(a.comparison_run, a.data3) if (q.z_in, q.z_tg) == (3.0, 0.5)]
    data = {q.name: (load(q.sharp), load(q.blurred)) for q in pairs}
    rows, gates = [], []
    for spec in map(parse, a.source):
        z = np.load(spec['npz'])
        ref = saved_scores(spec['csv'], spec['method']) if spec['csv'] else {}
        mine, worst = [], 0.0
        for config, (sharp, blurred) in data.items():
            pred = z[spec['key'].format(config=config)].astype(np.float64)
            for c, row in enumerate(mc.fourway(pred, sharp, blurred, 2.5, k_noise)):
                comp = COMPONENTS[c]
                if (config, comp) in ref:
                    worst = max(worst, abs(row['raw'] - ref[(config, comp)]))
                mine.append(dict(method=spec['name'], config=config, shape=mc.SHAPE[config], component=comp,
                                 **row))
        passed = (not ref) or worst <= 1e-4
        gates.append(dict(method=spec['name'], checked=bool(ref), worst_abs_diff=worst, passed=passed))
        print(spec['name'], 'gate', 'n/a' if not ref else f'{worst:.2e}', 'ok' if passed else 'FAILED',
              flush=True)
        if passed:
            rows.extend(mine)
    with open(out / 'fourway.csv', 'w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / 'gates.json').write_text(json.dumps(dict(k_noise=k_noise, gates=gates), indent=2))


if __name__ == '__main__':
    main()
