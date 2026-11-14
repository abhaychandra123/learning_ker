"""Export side-by-side reconstructions for the group update. Read-only, CPU, runs on the server.

From ~/learning_ker on the server:
    PYTHONPATH=experiments/magic_unet/scripts ~/qdm-env/bin/python export_group_update.py
Writes /tmp/group_update.npz: for L1_C1 and IC_4um at 3 -> 0.5 um, the COMSOL target and input and
every method's prediction, cropped to the 512 x 512 window (0.1 um pixels) holding the most target
Bz energy, plus full-image relative L2 and NRMSE (RMSE / target range, as in train_csc_lista.py) per
component, for these two coils and as means over all 15 pairs at 3 -> 0.5 um.
"""
import json

import numpy as np

from wiener_oracle import discover, load

M = 'runs/m1_m2'
SOURCES = {
    'magic': (f'{M}/m1/comsol_3_to_05_predictions.npz', 'run2_raw_{c}'),
    'magic_dc': (f'{M}/m1/comsol_3_to_05_predictions.npz', 'run2_dc_est_{c}'),
    'fista1000': (f'{M}/fista_depth/predictions.npz', 'fista1000_{c}'),
    'fista_net': (f'{M}/group_b/fista_net_run2/fista_net_run2_predictions.npz', '{c}'),
    'csc_lista': (f'{M}/group_b/csc_strict_64/csc_strict_64_predictions.npz', '{c}'),
    'tikhonov': (f'{M}/group_b/csc_strict_64/wiener_predictions.npz', '{c}'),
}
N, STEP = 512, 8


def window(t):
    """Top-left corner of the N x N window with the most energy (box sums on a STEP grid)."""
    c = np.pad(np.cumsum(np.cumsum(t.astype(np.float64) ** 2, 0), 1), ((1, 0), (1, 0)))
    best, where = -1.0, (0, 0)
    for r in range(0, t.shape[0] - N + 1, STEP):
        s = c[r + N, N::STEP] - c[r, N::STEP] - c[r + N, 0:-N:STEP] + c[r, 0:-N:STEP]
        j = int(s.argmax())
        if s[j] > best:
            best, where = s[j], (r, j * STEP)
    return where


def rel(p, t):
    return float(np.linalg.norm(p - t) / np.linalg.norm(t))


def nrmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)) / (t.max() - t.min()))


def both(p, t):
    return {'l2': [rel(p[i], t[i]) for i in range(3)], 'nrmse': [nrmse(p[i], t[i]) for i in range(3)]}


out, scores, every = {}, {}, {}
pairs = {p.name: p for p in discover('runs/combined_full_200', 'data/data3') if p.z_in == 3.0 and p.z_tg == 0.5}
files = {}
for coil, pair in pairs.items():
    target, blurred = load(pair.sharp), load(pair.blurred)
    every[coil] = {'input': both(blurred, target)}
    preds = {}
    for name, (path, key) in SOURCES.items():
        if path not in files:
            files[path] = np.load(path)
        preds[name] = files[path][key.format(c=coil)].astype(np.float64)
        every[coil][name] = both(preds[name], target)
    if coil not in ('L1_C1', 'IC_4um'):
        continue
    r, c = window(target[2])
    crop = (slice(None), slice(r, r + N), slice(c, c + N))
    out[f'{coil}_origin'] = np.array([r, c])
    out[f'{coil}_target'] = target[crop].astype(np.float32)
    out[f'{coil}_input'] = blurred[crop].astype(np.float32)
    for name, pred in preds.items():
        out[f'{coil}_{name}'] = pred[crop].astype(np.float32)
    scores[coil] = every[coil]
methods = ['input'] + list(SOURCES)
mean15 = {m: {k: float(np.mean([every[c][m][k] for c in every])) for k in ('l2', 'nrmse')} for m in methods}
out['scores'] = np.array(json.dumps(scores))
out['mean15'] = np.array(json.dumps(dict(pairs=sorted(every), n=3 * len(every), **mean15)))
np.savez_compressed('/tmp/group_update.npz', **out)
print(sorted(every))
print(json.dumps({m: {k: round(100 * v, 2) for k, v in d.items()} for m, d in mean15.items()}, indent=1))
