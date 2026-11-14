"""Export the MAGIC-UNet numbers the booklet needs. Runs on the server, CPU only.

Copy to the server as logs/booklet_export_magic.py, then from ~/learning_ker run:
    ~/qdm-env/bin/python logs/booklet_export_magic.py
Writes /tmp/booklet_magic.npz; copy it to booklet/data/.
"""
import csv
import json

import numpy as np

RUN = 'runs/magic_unet'
SCALE = 145.48712158203125     # booklet_export.npz 'scale' (v3 normalisation, uT)
R, C, N = 128, 112, 512         # booklet_export.npz 'crop_origin', L1_C1 Bz 3 -> 0.5 um

out = {}
with np.load(f'{RUN}/run2/comsol_3_to_05_predictions.npz') as z:
    out['magic_run2'] = (z['L1_C1_B'][2, R:R + N, C:C + N] / SCALE).astype(np.float32)
for run in ('run', 'run2'):
    rows = list(csv.DictReader(open(f'{RUN}/{run}/history.csv')))
    out[f'history_{run}'] = np.array([[float(r['epoch']), float(r['train_loss']), float(r['val_loss'])]
                                      for r in rows])
rows = list(csv.DictReader(open(f'{RUN}/run2/comsol.csv')))
out['comsol_z_in'] = np.array([float(r['z_input']) for r in rows])
out['comsol_z_target'] = np.array([float(r['z_target']) for r in rows])
out['comsol_error'] = np.array([float(r['relative_l2']) for r in rows])
out['comsol_config'] = np.array([r['config'] for r in rows])
out['results_run2'] = np.array(json.dumps(json.load(open(f'{RUN}/run2/results.json'))))
np.savez_compressed('/tmp/booklet_magic.npz', **out)
print({k: v.shape for k, v in out.items()})
