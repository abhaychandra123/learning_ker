"""M2a (M1_M2_PROTOCOL.md): the noise cutoff k_n, from the five 0.5 um COMSOL targets alone.

No prediction is read. Pools the radial profile of total and Maxwell-inconsistent energy over L1_C1, IC_4um and
the three data3 configurations, and applies the protocol rule. Also prints the SNR = 1 crossings of the five
3 -> 0.5 um pairs as a check against information_budget.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import m12_common as mc
from wiener_oracle import discover, load


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--comparison-run', default='runs/combined_full_200')
    p.add_argument('--data3', default='data/data3')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    pairs = [q for q in discover(args.comparison_run, args.data3) if q.z_in == 3.0 and q.z_tg == 0.5]
    names = [q.name for q in pairs]
    assert sorted(names) == sorted(['L1_C1', 'IC_4um', 'C1_C2', 'C1_C2_C3', 'C1_C2_C3_C4']), names
    targets = [load(q.sharp) for q in pairs]
    lower, total, bad = mc.inconsistency_profile(targets)
    k_n = mc.noise_cutoff(lower, total, bad)
    per_target = {}
    for q, t in zip(pairs, targets):
        lo, tt, bb = mc.inconsistency_profile([t])
        try:
            per_target[q.name] = mc.noise_cutoff(lo, tt, bb)
        except ValueError:
            per_target[q.name] = None
    crossings = {q.name: [mc.snr_crossing(load(q.sharp)[c], load(q.blurred)[c], 2.5) for c in range(3)]
                 for q in pairs}
    frac = bad / np.maximum(total, 1e-300)
    np.savetxt(out / 'profile.csv', np.column_stack([lower, total, bad, frac]), delimiter=',',
               header='k_lower_rad_per_um,total_energy,inconsistent_energy,inconsistent_share', comments='')
    result = dict(k_noise=k_n, rule='first 0.1 rad/um bin at or above 1.0 rad/um with inconsistent share > 0.5, '
                  'pooled over the five 0.5 um targets', per_target_cutoff=per_target,
                  snr_crossing_3_to_05=crossings, targets=names)
    (out / 'cutoff.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    for a, f in zip(lower, frac):
        if a < 4.0:
            print(f'{a:4.1f}  {f:.3f}')


if __name__ == '__main__':
    main()
