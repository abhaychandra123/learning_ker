"""
Loading layer over the .npy cache written by convert.py.

Everything downstream (visualisation, kernel estimation, dataset export) reads
fields through here so there is exactly one place that knows about paths, grid
metadata and array orientation.

Orientation convention, used everywhere:
    field[row, col] with row -> y, col -> x
    imshow(field, origin="lower", extent=grid.extent)
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
FIGURES = ROOT / "figures"

COMPONENTS = ("Bx", "By", "Bz", "Bnorm")
SIGNED = ("Bx", "By", "Bz")          # diverging colour scale
LABEL = {"Bx": "$B_x$", "By": "$B_y$", "Bz": "$B_z$", "Bnorm": "$|B|$"}


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    x0: float
    y0: float
    dx: float
    dy: float

    @property
    def x(self):
        return self.x0 + self.dx * np.arange(self.nx)

    @property
    def y(self):
        return self.y0 + self.dy * np.arange(self.ny)

    @property
    def extent(self):
        """(left, right, bottom, top) in um, for imshow(origin='lower')."""
        return (self.x0 - self.dx / 2,
                self.x0 + self.dx * (self.nx - 0.5),
                self.y0 - self.dy / 2,
                self.y0 + self.dy * (self.ny - 0.5))

    def ix(self, x_um):
        """Nearest column index for a physical x, clipped to the grid."""
        return int(np.clip(round((x_um - self.x0) / self.dx), 0, self.nx - 1))

    def iy(self, y_um):
        return int(np.clip(round((y_um - self.y0) / self.dy), 0, self.ny - 1))

    def window(self, x_range, y_range):
        """Slices for a physical bounding box -> field[sy, sx]."""
        sx = slice(self.ix(x_range[0]), self.ix(x_range[1]) + 1)
        sy = slice(self.iy(y_range[0]), self.iy(y_range[1]) + 1)
        return sy, sx


def _meta():
    path = CACHE / "grid.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run `python scripts/convert.py` first")
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def grid():
    m = _meta()
    return Grid(nx=m["nx"], ny=m["ny"], x0=m["x0"], y0=m["y0"],
                dx=m["dx"], dy=m["dy"])


@lru_cache(maxsize=1)
def z_values():
    """Standoff heights in um, ascending."""
    return tuple(_meta()["z_values"])


def z_tag(z):
    return f"z{z:04.1f}"


@lru_cache(maxsize=64)
def load(z, comp="Bz", unit="uT"):
    """Field plane at standoff z. Cached, so repeated calls are free.

    unit: 'T' as stored, or 'uT' (default) which is friendlier for plotting
    and keeps the least-squares systems away from 1e-6 scaling.
    """
    if comp not in COMPONENTS:
        raise ValueError(f"{comp!r} not in {COMPONENTS}")
    path = CACHE / f"{z_tag(z)}_{comp}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/convert.py")
    field = np.load(path)
    if unit == "uT":
        return field * 1e6
    if unit == "T":
        return field
    raise ValueError(f"unknown unit {unit!r}")


def pairs(reference_z, targets=None, components=SIGNED):
    """Enumerate (sharp, blurred) pairs anchored on one plane.

    Yields dicts so callers can add fields without breaking unpacking:
        {z_sharp, z_blur, dz, comp, sharp, blur}

    This is the seam Phase 2 plugs into - a tile/patch exporter consumes the
    same enumeration.
    """
    if targets is None:
        targets = [z for z in z_values() if z > reference_z]
    for z in targets:
        for comp in components:
            yield dict(z_sharp=reference_z, z_blur=z,
                       dz=z - reference_z, comp=comp,
                       sharp=load(reference_z, comp),
                       blur=load(z, comp))


def summary():
    """Per-plane statistics, for a quick sanity table."""
    g = grid()
    rows = []
    for z in z_values():
        row = {"z": z}
        for comp in COMPONENTS:
            f = load(z, comp)
            row[f"peak_{comp}"] = float(np.abs(f).max())
            row[f"rms_{comp}"] = float(np.sqrt(np.mean(f.astype(np.float64) ** 2)))
        rows.append(row)
    return g, rows


if __name__ == "__main__":
    g, rows = summary()
    print(f"grid {g.nx} x {g.ny} @ {g.dx} um   "
          f"x {g.x[0]:.2f}..{g.x[-1]:.2f}   y {g.y[0]:.2f}..{g.y[-1]:.2f} um")
    print(f"\n{'z (um)':>7} " + " ".join(f"{'peak '+c:>12}" for c in COMPONENTS)
          + " " + " ".join(f"{'rms '+c:>11}" for c in COMPONENTS))
    for r in rows:
        print(f"{r['z']:7.1f} "
              + " ".join(f"{r['peak_'+c]:12.3f}" for c in COMPONENTS)
              + " " + " ".join(f"{r['rms_'+c]:11.4f}" for c in COMPONENTS)
              + "   uT")
