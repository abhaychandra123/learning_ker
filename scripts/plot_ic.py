"""
Plot any single-standoff COMSOL CSV using the visualize.py colour conventions.

Usage:
    python3 scripts/plot_ic.py data/IC_4um_Z0p5um_600uA_0p1PR__MF.csv
    python3 scripts/plot_ic.py data/L1_C1_Z2p0um_600uA_0p1PR_MF_B.csv
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fields as F
from visualize import DIVERGING, SEQUENTIAL, CLIP_PCT

FIGURES = F.ROOT / "figures"
N_HEADER = 9
COMPONENTS = ("Bx", "By", "Bz", "Bnorm")

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 9,
})


def parse_z_from_name(name):
    m = re.search(r"_Z(\d+)p(\d+)um_", name)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    return None


def load_csv(path):
    df = pd.read_csv(path, skiprows=N_HEADER, header=None,
                     names=["x", "y", "Bx", "By", "Bz", "Bnorm"], engine="c")
    arr = df.to_numpy(dtype=np.float64)
    x, y = arr[:, 0], arr[:, 1]

    nx = int(np.argmax(y != y[0]))
    ny = len(x) // nx
    dx = float(x[1] - x[0])
    dy = float(y[nx] - y[0])
    x0, y0 = float(x[0]), float(y[0])
    extent = (x0 - dx / 2, x0 + dx * (nx - 0.5),
              y0 - dy / 2, y0 + dy * (ny - 0.5))

    planes = {}
    col = {"Bx": 2, "By": 3, "Bz": 4, "Bnorm": 5}
    for comp in COMPONENTS:
        planes[comp] = arr[:, col[comp]].reshape(ny, nx).astype(np.float32) * 1e6

    return planes, extent, nx, ny, dx


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 scripts/plot_ic.py <path-to-csv>")

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        sys.exit(f"file not found: {csv_path}")

    FIGURES.mkdir(exist_ok=True)
    planes, extent, nx, ny, dx = load_csv(csv_path)

    z = parse_z_from_name(csv_path.name)
    z_str = f"z = {z} um" if z is not None else "unknown z"
    tag = csv_path.stem

    print(f"{tag}: grid {nx}x{ny} @ {dx:.3f} um, "
          f"extent x=[{extent[0]:.1f}, {extent[1]:.1f}] "
          f"y=[{extent[2]:.1f}, {extent[3]:.1f}]")

    labels = {"Bx": "$B_x$", "By": "$B_y$", "Bz": "$B_z$", "Bnorm": "$|B|$"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for ax, comp in zip(axes.ravel(), COMPONENTS):
        data = planes[comp]
        if comp in F.SIGNED:
            v = np.percentile(np.abs(data), CLIP_PCT)
            im = ax.imshow(data, origin="lower", extent=extent,
                           cmap=DIVERGING, vmin=-v, vmax=v,
                           interpolation="nearest")
        else:
            v = np.percentile(data, CLIP_PCT)
            im = ax.imshow(data, origin="lower", extent=extent,
                           cmap=SEQUENTIAL, vmin=0, vmax=v,
                           interpolation="nearest")
        peak = np.abs(data).max()
        ax.set_title(f"{labels[comp]}   {z_str}   (peak {peak:.2f} uT)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="uT")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.grid(False)

    fig.suptitle(f"{tag}  {z_str}  (all four field components)", fontsize=11)
    out = FIGURES / f"{tag}_overview.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
