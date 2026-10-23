"""
Measure mesh noise in the COMSOL data.

The physics guarantees: B(k, z2) = B(k, z1) * exp(-2*pi*k*dz)
exactly in source-free space. So if we propagate z1 analytically
to z2 and subtract the actual z2 data, the residual is everything
in the data that ISN'T physics -- i.e. mesh interpolation artifacts.

    python scripts/mesh_noise.py
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fields as F
import propagator as P


def nrmse(predicted, actual):
    mse = np.mean((predicted - actual) ** 2)
    return np.sqrt(mse) / (actual.max() - actual.min())


def main():
    grid = F.grid()
    dx = grid.dx
    zs = sorted(F.z_values())

    comp = "Bz"
    print(f"Component: {comp}")
    print(f"Available z-planes: {zs}")
    print()

    # --- For every pair (z1, z2), propagate z1 -> z2 and measure residual ---
    print(f"{'z1':>5}  {'z2':>5}  {'dz':>5}  {'NRMSE':>10}  {'max_resid':>12}  {'signal_range':>14}")
    print("-" * 70)

    all_results = []
    for i, z1 in enumerate(zs):
        for z2 in zs[i+1:]:
            dz = z2 - z1
            actual = F.load(z2, comp)
            sharp = F.load(z1, comp)
            predicted = P.propagate(sharp, dz, dx)
            residual = actual - predicted

            err = nrmse(predicted, actual)
            sig_range = actual.max() - actual.min()
            max_res = np.max(np.abs(residual))

            all_results.append({
                "z1": z1, "z2": z2, "dz": dz,
                "nrmse": err, "max_resid": max_res,
                "sig_range": sig_range,
                "residual": residual, "actual": actual,
                "predicted": predicted,
            })
            print(f"{z1:5.1f}  {z2:5.1f}  {dz:5.1f}  {err:10.6f}  {max_res:12.4f}  {sig_range:14.4f}")

    # --- FIGURE 1: NRMSE heatmap (z1 vs z2) ---
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    nz = len(zs)
    matrix = np.full((nz, nz), np.nan)
    for r in all_results:
        i = zs.index(r["z1"])
        j = zs.index(r["z2"])
        matrix[i, j] = r["nrmse"] * 100

    im = ax.imshow(matrix, origin="lower", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_xticks(range(nz))
    ax.set_xticklabels([f"{z}" for z in zs], rotation=45)
    ax.set_yticks(range(nz))
    ax.set_yticklabels([f"{z}" for z in zs])
    ax.set_xlabel("z2 (target plane, um)")
    ax.set_ylabel("z1 (source plane, um)")
    ax.set_title("Mesh noise: NRMSE (%) of analytic propagation\n"
                 "residual = actual(z2) - propagate(z1, dz)")

    for r in all_results:
        i = zs.index(r["z1"])
        j = zs.index(r["z2"])
        val = r["nrmse"] * 100
        color = "white" if val > 3 else "black"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="NRMSE (%)", fraction=0.046)
    fig.savefig(F.FIGURES / "mesh_noise_heatmap.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"\nSaved figures/mesh_noise_heatmap.png")

    # --- FIGURE 2: example residual maps for z1=0.5, varying z2 ---
    z1_fixed = 0.5
    show_z2s = [1.0, 2.0, 5.0, 10.0]
    show_results = [r for r in all_results if r["z1"] == z1_fixed and r["z2"] in show_z2s]

    fig, axes = plt.subplots(2, len(show_results), figsize=(5 * len(show_results), 8),
                             constrained_layout=True)

    for col, r in enumerate(show_results):
        actual = r["actual"]
        residual = r["residual"]
        extent = grid.extent

        vmax = np.percentile(np.abs(actual), 99.5)
        ax = axes[0, col]
        ax.imshow(actual, origin="lower", extent=extent, cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Actual Bz at z={r['z2']}")
        ax.grid(False)

        rmax = np.percentile(np.abs(residual), 99.5)
        ax = axes[1, col]
        im = ax.imshow(residual, origin="lower", extent=extent, cmap="RdBu_r",
                       vmin=-rmax, vmax=rmax, interpolation="nearest")
        ax.set_title(f"Residual (NRMSE={r['nrmse']:.4f})\n"
                     f"propagate z={z1_fixed} by dz={r['dz']}")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes.ravel():
        ax.set_xlabel("x (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (um)")

    fig.suptitle(f"Mesh noise measurement: propagate z={z1_fixed} analytically, subtract actual\n"
                 f"Any nonzero residual violates Laplace -- it is mesh artifact, not physics",
                 fontsize=12)
    fig.savefig(F.FIGURES / "mesh_noise_residuals.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("Saved figures/mesh_noise_residuals.png")

    # --- FIGURE 3: spectrum of signal vs residual ---
    fig, axes = plt.subplots(1, len(show_results), figsize=(5 * len(show_results), 4),
                             constrained_layout=True)

    for col, r in enumerate(show_results):
        actual = r["actual"]
        residual = r["residual"]
        k = P.k_grid(actual.shape, dx)

        k_c, sig_spec = P.radial_average(P.amplitude_spectrum(actual), k, k_max=2.0)
        k_c, res_spec = P.radial_average(P.amplitude_spectrum(residual), k, k_max=2.0)

        ax = axes[col]
        ax.semilogy(k_c, sig_spec, "b-", lw=1.2, label="signal (actual)")
        ax.semilogy(k_c, res_spec, "r-", lw=1.0, label="residual (mesh noise)")
        ax.set_xlabel("k (cycles/um)")
        ax.set_title(f"z1={r['z1']}, z2={r['z2']}, dz={r['dz']}")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1.5)

    axes[0].set_ylabel("amplitude spectrum")
    fig.suptitle("Signal vs mesh noise spectrum\n"
                 "Where red exceeds blue, the data is dominated by mesh artifacts", fontsize=12)
    fig.savefig(F.FIGURES / "mesh_noise_spectrum.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("Saved figures/mesh_noise_spectrum.png")

    # --- FIGURE 4: NRMSE vs dz, grouped by source z1 ---
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    source_zs = sorted(set(r["z1"] for r in all_results))

    for z1 in source_zs[:6]:
        subset = [r for r in all_results if r["z1"] == z1]
        dzs = [r["dz"] for r in subset]
        nrmses = [r["nrmse"] * 100 for r in subset]
        ax.plot(dzs, nrmses, "o-", ms=4, lw=1.2, label=f"z1={z1}")

    ax.set_xlabel("dz (um)")
    ax.set_ylabel("NRMSE (%)")
    ax.set_title("Mesh noise vs propagation distance\n"
                 "each curve: fixed source z1, varying target z2")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(F.FIGURES / "mesh_noise_vs_dz.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("Saved figures/mesh_noise_vs_dz.png")


if __name__ == "__main__":
    main()
