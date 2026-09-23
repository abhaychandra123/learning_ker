"""
One-time conversion of the COMSOL CSV exports into .npy arrays.

Each CSV is ~230 MB of text (2.5 GB total) and takes ~20 s to parse. The cached
arrays are 7.8 MB each and load instantly, so every downstream experiment reads
those instead.

Layout of a source file:
    9 comment lines starting with '%'
    1,960,000 data rows: x, y, Bx, By, Bz, |B|
    positions in um, fields in tesla, x varies fastest

Run once:
    python scripts/convert.py
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"

N_HEADER = 9
COMPONENTS = ("Bx", "By", "Bz", "Bnorm")
COL_INDEX = {"x": 0, "y": 1, "Bx": 2, "By": 3, "Bz": 4, "Bnorm": 5}


def parse_z(name):
    """L1_C1_Z0p5um_600uA_... -> 0.5"""
    m = re.search(r"_Z(\d+)p(\d+)um_", name)
    if not m:
        raise ValueError(f"no standoff in filename: {name}")
    return float(f"{m.group(1)}.{m.group(2)}")


def z_tag(z):
    """0.5 -> 'z00.5'. Zero-padded so string sort matches numeric sort."""
    return f"z{z:04.1f}"


def infer_grid(x, y):
    """Recover (nx, ny, x0, y0, dx, dy) from the coordinate columns.

    Deliberately does NOT use np.unique: the y coordinates differ between files
    at the 1e-14 level (float printing), so unique-grouping can return a
    different count per file. The export is a regular raster with x fastest, so
    the grid is recoverable from where y first changes.
    """
    nx = int(np.argmax(y != y[0]))
    if nx == 0:
        raise ValueError("y never changes in the first pass - not an x-fastest raster")
    ny, rem = divmod(len(x), nx)
    if rem:
        raise ValueError(f"{len(x)} rows is not divisible by nx={nx}")

    dx = np.diff(x[:nx])
    dy = np.diff(y[::nx])
    if not np.allclose(dx, dx[0], rtol=1e-6):
        raise ValueError("x spacing is not uniform")
    if not np.allclose(dy, dy[0], rtol=1e-6):
        raise ValueError("y spacing is not uniform")

    return dict(nx=nx, ny=ny, x0=float(x[0]), y0=float(y[0]),
                dx=float(dx[0]), dy=float(dy[0]))


def convert_one(path):
    t0 = time.time()
    df = pd.read_csv(path, skiprows=N_HEADER, header=None,
                     names=list(COL_INDEX), engine="c")
    arr = df.to_numpy(dtype=np.float64)

    grid = infer_grid(arr[:, COL_INDEX["x"]], arr[:, COL_INDEX["y"]])
    shape = (grid["ny"], grid["nx"])          # [row=y, col=x]

    z = parse_z(path.name)
    for comp in COMPONENTS:
        field = arr[:, COL_INDEX[comp]].reshape(shape).astype(np.float32)
        np.save(CACHE / f"{z_tag(z)}_{comp}.npy", field)

    print(f"  z={z:5.1f} um  {shape[1]}x{shape[0]}  "
          f"peak|Bz|={np.abs(arr[:, COL_INDEX['Bz']]).max()*1e6:8.2f} uT  "
          f"({time.time()-t0:.1f}s)")
    return z, grid


def main():
    paths = sorted(DATA.glob("*.csv"), key=lambda p: parse_z(p.name))
    if not paths:
        sys.exit(f"no CSVs found in {DATA}")

    CACHE.mkdir(exist_ok=True)
    print(f"converting {len(paths)} files -> {CACHE}")

    zs, grids = [], []
    for p in paths:
        z, g = convert_one(p)
        zs.append(z)
        grids.append(g)

    # Every plane must share one grid, otherwise pairing them is meaningless.
    ref = grids[0]
    for z, g in zip(zs, grids):
        for key in ("nx", "ny"):
            if g[key] != ref[key]:
                sys.exit(f"z={z} has {key}={g[key]}, expected {ref[key]}")
        for key in ("x0", "y0", "dx", "dy"):
            if abs(g[key] - ref[key]) > 1e-6:
                sys.exit(f"z={z} has {key}={g[key]}, expected {ref[key]}")

    meta = dict(ref)
    meta["z_values"] = zs
    meta["components"] = list(COMPONENTS)
    meta["length_unit"] = "um"
    meta["field_unit"] = "T"
    (CACHE / "grid.json").write_text(json.dumps(meta, indent=2))

    print(f"\ngrid {ref['nx']}x{ref['ny']} @ {ref['dx']} um, "
          f"x0={ref['x0']:.2f} y0={ref['y0']:.2f}")
    print(f"z planes: {zs}")
    print(f"wrote {len(paths)*len(COMPONENTS)} arrays + grid.json")


if __name__ == "__main__":
    main()
