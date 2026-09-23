"""
Learn the blur kernel K_i for each standoff pair by least-squares in Fourier space.

For each pair (s, b_i) where s is the sharp plane at z=0.5 and b_i is
a blurred plane at some higher z:

    minimize ||b_i - K_i * s||^2

In Fourier space this is pointwise division: K_hat_i = B_i / S.

One pair (z=0.5 -> z=2.0, i.e. dz=1.5) is held out for testing.

    python scripts/kernel.py
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import fields as F
import propagator as P

REF_Z = 0.5
HOLDOUT_Z = 2.0
COMPONENTS = ("Bx", "By", "Bz")


def learn_kernel(sharp, blurred):
    S = np.fft.fft2(sharp.astype(np.float64))
    B = np.fft.fft2(blurred.astype(np.float64))
    K_hat = B / S
    return K_hat


def learn_kernel_joint(sharps, blurreds):
    """Joint kernel from multiple (sharp, blurred) pairs.

    K_hat = sum_i conj(S_i) B_i / sum_i |S_i|^2
    """
    num = None
    den = None
    for sharp, blurred in zip(sharps, blurreds):
        S = np.fft.fft2(sharp.astype(np.float64))
        B = np.fft.fft2(blurred.astype(np.float64))
        if num is None:
            num = np.conj(S) * B
            den = np.abs(S) ** 2
        else:
            num += np.conj(S) * B
            den += np.abs(S) ** 2
    return num / den


def apply_kernel(sharp, K_hat):
    S = np.fft.fft2(sharp.astype(np.float64))
    return np.real(np.fft.ifft2(K_hat * S))


def nrmse(predicted, actual):
    mse = np.mean((predicted - actual) ** 2)
    return np.sqrt(mse) / (actual.max() - actual.min())


def to_real_space(K_hat):
    return np.real(np.fft.ifftshift(np.fft.ifft2(K_hat)))


def main():
    grid = F.grid()
    dx = grid.dx
    zs = list(F.z_values())
    target_zs = [z for z in zs if z > REF_Z]

    print(f"Reference plane: z = {REF_Z} um")
    print(f"Holdout pair:    z = {REF_Z} -> {HOLDOUT_Z} um (dz = {HOLDOUT_Z - REF_Z})")
    print(f"Components:      {COMPONENTS}")
    print()

    # ---- ORIGINAL per-component kernels (commented out) ----
    # kernels = {}
    # results = []
    # for z_target in target_zs:
    #     dz = z_target - REF_Z
    #     is_holdout = (z_target == HOLDOUT_Z)
    #     for comp in COMPONENTS:
    #         sharp = F.load(REF_Z, comp)
    #         blurred = F.load(z_target, comp)
    #         if is_holdout:
    #             print(f"  dz={dz:5.1f}  {comp}  ** HELD OUT **")
    #             continue
    #         K_hat = learn_kernel(sharp, blurred)
    #         kernels[(dz, comp)] = K_hat
    #         predicted = apply_kernel(sharp, K_hat)
    #         err = nrmse(predicted, blurred)
    #         results.append({"z_target": z_target, "dz": dz, "comp": comp, "nrmse": err})
    #         print(f"  dz={dz:5.1f}  {comp}  NRMSE = {err:.6f}")

    # ---- JOINT kernel across Bx, By, Bz ----
    kernels = {}
    results = []

    for z_target in target_zs:
        dz = z_target - REF_Z
        is_holdout = (z_target == HOLDOUT_Z)

        if is_holdout:
            for comp in COMPONENTS:
                print(f"  dz={dz:5.1f}  {comp}  ** HELD OUT **")
            continue

        sharps = [F.load(REF_Z, c) for c in COMPONENTS]
        blurreds = [F.load(z_target, c) for c in COMPONENTS]
        K_hat = learn_kernel_joint(sharps, blurreds)
        kernels[dz] = K_hat

        for comp, s, b in zip(COMPONENTS, sharps, blurreds):
            predicted = apply_kernel(s, K_hat)
            err = nrmse(predicted, b)
            results.append({"z_target": z_target, "dz": dz, "comp": comp, "nrmse": err})
            print(f"  dz={dz:5.1f}  {comp}  NRMSE = {err:.6f}  (joint kernel)")

    # summary table
    print("\n--- Summary (training NRMSE, joint kernel) ---")
    print(f"{'dz':>6}  {'Bx':>10}  {'By':>10}  {'Bz':>10}")
    dzs_done = sorted(set(r["dz"] for r in results))
    for dz in dzs_done:
        vals = {r["comp"]: r["nrmse"] for r in results if r["dz"] == dz}
        print(f"{dz:6.1f}  {vals.get('Bx',0):10.6f}  {vals.get('By',0):10.6f}  {vals.get('Bz',0):10.6f}")

    k = P.k_grid(F.load(REF_Z, "Bz").shape, dx)

    # ---- FIGURE 1: learned |K_hat(k)| vs analytic ----
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    norm = Normalize(vmin=min(dzs_done), vmax=max(dzs_done))
    cmap = matplotlib.colormaps["viridis"]

    for dz in dzs_done:
        K_hat = kernels[dz]
        k_centres, K_radial = P.radial_average(np.abs(K_hat), k, k_max=2.0)
        color = cmap(norm(dz))
        ax.semilogy(k_centres, K_radial, color=color, lw=1.3)
        ax.semilogy(k_centres, P.analytic_transfer(k_centres, dz),
                    color=color, lw=1.0, ls="--", alpha=0.6)

    ax.set_xlim(0, 1.5)
    ax.set_ylim(1e-6, 3)
    ax.set_xlabel("spatial frequency k (cycles / um)")
    ax.set_ylabel("|K_hat(k)|")
    ax.set_title("Joint kernel (solid) vs analytic exp(-2pi*k*dz) (dashed)")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="dz (um)", pad=0.01, fraction=0.04)
    fig.savefig(F.FIGURES / "kernel_vs_analytic_joint.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"\nSaved figures/kernel_vs_analytic_joint.png")

    # ---- FIGURE 2: kernel in real space (central crop) ----
    pick_dzs = [0.5, 2.5, 4.5, 9.5]
    crop_px = 150  # pixels from center to show

    fig, axes = plt.subplots(1, len(pick_dzs), figsize=(4 * len(pick_dzs), 3.8),
                             constrained_layout=True)
    for ax, dz in zip(axes, pick_dzs):
        K_real = to_real_space(kernels[dz])
        ny, nx = K_real.shape
        cy, cx = ny // 2, nx // 2
        crop = K_real[cy - crop_px:cy + crop_px, cy - crop_px:cy + crop_px]
        extent_um = crop_px * dx
        v = np.percentile(np.abs(crop), 99.5)
        im = ax.imshow(crop, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                       extent=[-extent_um, extent_um, -extent_um, extent_um],
                       interpolation="nearest")
        ax.set_title(f"dz = {dz} um")
        ax.set_xlabel("um")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("um")
    fig.suptitle("Joint kernel in real space (central crop)", fontsize=12)
    fig.savefig(F.FIGURES / "kernel_real_space_joint.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("Saved figures/kernel_real_space_joint.png")

    # ---- FIGURE 3: 1D radial profile of real-space kernel vs analytic Poisson ----
    fig, axes = plt.subplots(1, len(pick_dzs), figsize=(4 * len(pick_dzs), 3.5),
                             constrained_layout=True)
    for ax, dz in zip(axes, pick_dzs):
        K_real = to_real_space(kernels[dz])
        ny, nx = K_real.shape
        cy, cx = ny // 2, nx // 2
        line = K_real[cy, cx:]
        r_px = np.arange(len(line))
        r_um = r_px * dx

        r_analytic = r_um[1:]
        poisson = dz / (2 * np.pi * (r_analytic**2 + dz**2)**1.5)

        ax.plot(r_um[:200], line[:200], "b-", lw=1.2, label="joint learned")
        ax.plot(r_analytic[:200], poisson[:200], "r--", lw=1.0, label="Poisson kernel")
        ax.set_xlabel("r (um)")
        ax.set_title(f"dz = {dz} um")
        ax.legend(fontsize=8)
        ax.set_xlim(0, min(20, 200 * dx))
    axes[0].set_ylabel("K(r)")
    fig.suptitle("Radial profile of joint kernel vs analytic Poisson kernel", fontsize=12)
    fig.savefig(F.FIGURES / "kernel_radial_profile_joint.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("Saved figures/kernel_radial_profile_joint.png")

    # ---- HOLDOUT TEST ----
    # Predict z=2.0 WITHOUT ever using z=2.0 data.
    # Method: apply the learned dz=0.5 kernel three times:
    #   0.5 -> 1.0 -> 1.5 -> 2.0
    # The physics says the propagator depends only on dz (not absolute z),
    # so the same dz=0.5 kernel should work at every height.
    # Compare against: (a) analytic propagator, (b) actual z=2.0 data.

    holdout_dz = HOLDOUT_Z - REF_Z  # 1.5
    n_steps = int(holdout_dz / 0.5)  # 3 steps of 0.5 um each

    print(f"\n--- Holdout Test (predicting z = {HOLDOUT_Z} um) ---")
    print(f"    Apply learned dz=0.5 kernel {n_steps} times: ", end="")
    print(" -> ".join(f"{REF_Z + i*0.5}" for i in range(n_steps + 1)))
    print()

    holdout_errors = []

    fig, axes = plt.subplots(len(COMPONENTS), 4, figsize=(20, 4 * len(COMPONENTS)),
                             constrained_layout=True)

    for i, comp in enumerate(COMPONENTS):
        sharp = F.load(REF_Z, comp)
        actual = F.load(HOLDOUT_Z, comp)

        # method 1: analytic propagator from 0.5 -> 2.0 (one step)
        pred_analytic = P.propagate(sharp, holdout_dz, dx)
        err_analytic = nrmse(pred_analytic, actual)

        # method 2: apply joint learned K(dz=0.5) three times
        K_half = kernels[0.5]
        pred = sharp.copy()
        for step in range(n_steps):
            pred = apply_kernel(pred, K_half)
        err_learned = nrmse(pred, actual)

        holdout_errors.append({
            "comp": comp,
            "analytic": err_analytic,
            "learned_3x": err_learned,
        })

        print(f"  {comp}:")
        print(f"    analytic propagator (1 step)      NRMSE = {err_analytic:.6f}")
        print(f"    learned K(dz=0.5) x{n_steps}             NRMSE = {err_learned:.6f}")

        # plot: actual, analytic, learned, residual
        vmax = np.percentile(np.abs(actual), 99.5)
        extent = grid.extent

        ax = axes[i, 0]
        ax.imshow(actual, origin="lower", extent=extent, cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Actual {comp} at z={HOLDOUT_Z}")
        ax.grid(False)

        ax = axes[i, 1]
        ax.imshow(pred_analytic, origin="lower", extent=extent, cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Analytic (NRMSE={err_analytic:.4f})")
        ax.grid(False)

        ax = axes[i, 2]
        ax.imshow(pred, origin="lower", extent=extent, cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Learned x{n_steps} (NRMSE={err_learned:.4f})")
        ax.grid(False)

        residual = actual - pred
        rmax = np.percentile(np.abs(residual), 99.5)
        ax = axes[i, 3]
        im = ax.imshow(residual, origin="lower", extent=extent, cmap="RdBu_r",
                       vmin=-rmax, vmax=rmax, interpolation="nearest")
        ax.set_title(f"Residual (actual - learned)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, label="uT")

    for ax in axes.ravel():
        ax.set_xlabel("x (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (um)")

    fig.suptitle(f"Holdout test (joint kernel): predicting z = {HOLDOUT_Z} um by applying dz=0.5 kernel {n_steps} times\n"
                 f"(z=2.0 data was NOT used in training)", fontsize=13)
    fig.savefig(F.FIGURES / "kernel_holdout_test_joint.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"\nSaved figures/kernel_holdout_test_joint.png")

    # summary
    print("\n--- Holdout Summary ---")
    print(f"{'comp':>6}  {'analytic':>10}  {'learned_3x':>12}")
    for r in holdout_errors:
        print(f"{r['comp']:>6}  {r['analytic']:10.6f}  {r['learned_3x']:12.6f}")


if __name__ == "__main__":
    main()
