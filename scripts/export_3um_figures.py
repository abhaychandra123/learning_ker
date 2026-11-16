"""Export the saved 3 -> 0.5 um COMSOL predictions for the one-colour-scale comparison figure (presentation, TODO.md).

Server side, read-only apart from the output file. Methods, all from saved predictions (nothing re-run):
  MAGIC-UNet  run 2 with the data-consistency step at its own estimated height (runs/m1_m2/m1)
  FISTA       925 iterations, the 3 um baseline of amendment A3 (runs/high_standoff/fista_3um_dense)
  FISTA-Net   run 2 (runs/m1_m2/group_b/fista_net_run2)
  CSC-LISTA   strict, 64 layers (runs/m1_m2/group_b/csc_strict_64)
  Tikhonov    mu = 0.0316 (runs/m1_m2/group_b/csc_strict_64/wiener_predictions.npz)
Output: runs/high_standoff/export/comsol3_part.npz / .json, keys comsol3_<config>_<method>, the same format as
export_high_standoff.py so plot_high_standoff.py can draw it.
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_high_standoff import add             # noqa: E402

SOURCES = {
    'magic': ('runs/m1_m2/m1/comsol_3_to_05_predictions.npz', 'run2_dc_est_{c}'),
    'fista925': ('runs/high_standoff/fista_3um_dense/predictions.npz', 'fista925_{c}'),
    'fista_net': ('runs/m1_m2/group_b/fista_net_run2/fista_net_run2_predictions.npz', '{c}'),
    'csc_lista': ('runs/m1_m2/group_b/csc_strict_64/csc_strict_64_predictions.npz', '{c}'),
    'tikhonov': ('runs/m1_m2/group_b/csc_strict_64/wiener_predictions.npz', '{c}'),
}


def main():
    out = Path('runs/high_standoff/export')
    out.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, {}
    stores = {m: np.load(path) for m, (path, _) in SOURCES.items()}
    for config in ('L1_C1', 'IC_4um'):
        with np.load(f'runs/combined_full_200/comparison/{config}_z03.0_to_z00.5.npz') as z:
            sharp, blurred = z['sharp'].astype(np.float64), z['blurred'].astype(np.float64)
        add(arrays, meta, f'comsol3_{config}_target', sharp, sharp)
        add(arrays, meta, f'comsol3_{config}_input', blurred, sharp)
        for method, (_, key) in SOURCES.items():
            add(arrays, meta, f'comsol3_{config}_{method}', stores[method][key.format(c=config)], sharp)
            print(config, method, np.round(meta[f'comsol3_{config}_{method}']['l2'], 4), flush=True)
    np.savez_compressed(out / 'comsol3_part.npz', **arrays)
    (out / 'comsol3_part.json').write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
