"""
Learn the blur kernel with Wiener regularisation (L2-penalised least squares).

For each pair (s, b_i):

    minimize ||b_i - K_i * s||^2 + lambda * ||K_i||^2

Closed-form solution in Fourier space (Wiener deconvolution):

    K_hat = conj(S) * B / (|S|^2 + lambda)

When lambda = 0 this reduces to K_hat = B / S (the unregularised version).
When |S|^2 >> lambda, K_hat ~ B / S  (low k: data dominates, kernel is free).
When |S|^2 << lambda, K_hat ~ 0      (high k: penalty dominates, kernel suppressed).

One pair (z=0.5 -> z=2.0, dz=1.5) is held out for testing.

    python scripts/kernel_regularised.py
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


def learn_kernel_wiener(sharp, blurred, lam):
    S = np.fft.fft2(sharp.astype(np.float64))
    B = np.fft.fft2(blurred.astype(np.float64))
    K_hat = np.conj(S) * B / (np.abs(S)**2 + lam)
    return K_hat


def learn_kernel_wiener_joint(sharps, blurreds, lam):
    """Joint Wiener kernel from multiple (sharp, blurred) pairs.

    K_hat = sum_i conj(S_i) B_i / (sum_i |S_i|^2 + lambda)
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
    return num / (den + lam)


def apply_kernel(sharp, K_hat):
    S = np.fft.fft2(sharp.astype(np.float64))
    return np.real(np.fft.ifft2(K_hat * S))


def nrmse(predicted, actual):
    mse = np.mean((predicted - actual) ** 2)
    return np.sqrt(mse) / (actual.max() - actual.min())


def estimate_lambda(sharp, dx):
    """Estimate lambda from the noise floor of the sharp plane.

    The idea: at high spatial frequencies, |S|^2 is just noise.
    We take the median of |S|^2 in the high-k region as our lambda.
    This way lambda sits right at the noise level: frequencies with
    signal above the noise pass through, frequencies below get killed.
    """
    S = np.fft.fft2(sharp.astype(np.float64))
    power = np.abs(S)**2
    k = P.k_grid(sharp.shape, dx)
    high_k_mask = k > 1.0  # well above any physical content
    return float(np.median(power[high_k_mask]))


def main():
    grid = F.grid()
    dx = grid.dx
    zs = list(F.z_values())
    target_zs = [z for z in zs if z > REF_Z]

    # estimate lambda from the reference plane
    sharp_bz = F.load(REF_Z, "Bz")
    lam = estimate_lambda(sharp_bz, dx)
    print(f"Estimated lambda = {lam:.2e} (from high-k noise floor of reference plane)")
    print(f"Reference plane: z = {REF_Z} um")
    print(f"Holdout pair:    z = {REF_Z} -> {HOLDOUT_Z} um (dz = {HOLDOUT_Z - REF_Z})")
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
    #         K_hat = learn_kernel_wiener(sharp, blurred, lam)
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
        K_hat = learn_kernel_wiener_joint(sharps, blurreds, lam)
        kernels[dz] = K_hat

        for comp, s, b in zip(COMPONENTS, sharps, blurreds):
            predicted = apply_kernel(s, K_hat)
            err = nrmse(predicted, b)
            results.append({"z_target": z_target, "dz": dz, "comp": comp, "nrmse": err})
            print(f"  dz={dz:5.1f}  {comp}  NRMSE = {err:.6f}  (joint kernel)")

    print("\n--- Summary (training NRMSE, joint kernel) ---")
    print(f"{'dz':>6}  {'Bx':>10}  {'By':>10}  {'Bz':>10}")
    dzs_done = sorted(set(r["dz"] for r in results))
    for dz in dzs_done:
        vals = {r["comp"]: r["nrmse"] for r in results if r["dz"] == dz}
        print(f"{dz:6.1f}  {vals.get('Bx',0):10.6f}  {vals.get('By',0):10.6f}  {vals.get('Bz',0):10.6f}")

    k = P.k_grid(sharp_bz.shape, dx)

    # ---- FIGURE 1: learned |K_hat(k)| vs analytic ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
    norm = Normalize(vmin=min(dzs_done), vmax=max(dzs_done))
    cmap = matplotlib.colormaps["viridis"]

    for dz in dzs_done:
        K_hat = kernels[dz]
        k_centres, K_radial = P.radial_average(np.abs(K_hat), k, k_max=2.0)
        color = cmap(norm(dz))
        axes[0].semilogy(k_centres, K_radial, color=color, lw=1.3)
        axes[0].semilogy(k_centres, P.analytic_transfer(k_centres, dz),
                         color=color, lw=1.0, ls="--", alpha=0.6)

    axes[0].set_xlim(0, 1.5)
    axes[0].set_ylim(1e-6, 3)
    axes[0].set_xlabel("spatial frequency k (cycles / um)")
    axes[0].set_ylabel("|K_hat(k)|")
    axes[0].set_title("Joint regularised kernel (solid) vs analytic (dashed)")

    # also show the unregularised for comparison (just dz=0.5)
    sharp_ref = F.load(REF_Z, "Bz")
    blurred_1 = F.load(1.0, "Bz")
    S = np.fft.fft2(sharp_ref.astype(np.float64))
    B = np.fft.fft2(blurred_1.astype(np.float64))
    K_unreg = B / S
    K_reg = kernels[0.5]
    k_c, unreg_radial = P.radial_average(np.abs(K_unreg), k, k_max=2.0)
    k_c, reg_radial = P.radial_average(np.abs(K_reg), k, k_max=2.0)

    axes[1].semilogy(k_c, unreg_radial, "b-", lw=1.3, label="unregularised (B/S)")
    axes[1].semilogy(k_c, reg_radial, "r-", lw=1.3, label=f"regularised (lambda={lam:.1e})")
    axes[1].semilogy(k_c, P.analytic_transfer(k_c, 0.5), "k--", lw=1.0, label="analytic exp(-2pi*k*dz)")
    axes[1].set_xlim(0, 1.5)
    axes[1].set_ylim(1e-6, 3)
    axes[1].set_xlabel("spatial frequency k (cycles / um)")
    axes[1].set_ylabel("|K_hat(k)|")
    axes[1].set_title("dz=0.5: unregularised vs regularised vs analytic")
    axes[1].legend(fontsize=10)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes[0], label="dz (um)", pad=0.01, fraction=0.04)
    fig.savefig(F.FIGURES / "kernel_reg_vs_analytic_joint.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"\nSaved figures/kernel_reg_vs_analytic_joint.png")

    # ---- FIGURE 2: kernel in real space ----
    pick_dzs = [0.5, 2.5, 4.5, 9.5]
    crop_px = 150

    fig, axes = plt.subplots(1, len(pick_dzs), figsize=(4 * len(pick_dzs), 3.8),
                             constrained_layout=True)
    for ax, dz in zip(axes, pick_dzs):
        K_real = np.real(np.fft.ifftshift(np.fft.ifft2(kernels[dz])))
        ny, nx = K_real.shape
        cy, cx = ny // 2, nx // 2
        crop = K_real[cy - crop_px:cy + crop_px, cy - crop_px:cy + crop_px]
        extent_um = crop_px * dx
        v = np.percentile(np.abs(crop), 99.5)
        if v == 0:
            v = 1e-10
        im = ax.imshow(crop, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                       extent=[-extent_um, extent_um, -extent_um, extent_um],
                       interpolation="nearest")
        ax.set_title(f"dz = {dz} um")
        ax.set_xlabel("um")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("um")
    fig.suptitle("Joint regularised kernel in real space (central crop)", fontsize=12)
    fig.savefig(F.FIGURES / "kernel_reg_real_space_joint.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("Saved figures/kernel_reg_real_space_joint.png")

    # ---- FIGURE 3: radial profile vs Poisson ----
    fig, axes = plt.subplots(1, len(pick_dzs), figsize=(4 * len(pick_dzs), 3.5),
                             constrained_layout=True)
    for ax, dz in zip(axes, pick_dzs):
        K_real = np.real(np.fft.ifftshift(np.fft.ifft2(kernels[dz])))
        ny, nx = K_real.shape
        cy, cx = ny // 2, nx // 2
        line = K_real[cy, cx:]
        r_um = np.arange(len(line)) * dx

        r_analytic = r_um[1:]
        poisson = dz / (2 * np.pi * (r_analytic**2 + dz**2)**1.5)

        n_show = min(200, len(line))
        ax.plot(r_um[:n_show], line[:n_show], "b-", lw=1.2, label="joint reg.")
        ax.plot(r_analytic[:n_show], poisson[:n_show], "r--", lw=1.0, label="Poisson kernel")
        ax.set_xlabel("r (um)")
        ax.set_title(f"dz = {dz} um")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 20)
    axes[0].set_ylabel("K(r)")
    fig.suptitle("Radial profile: joint regularised kernel vs analytic Poisson", fontsize=12)
    fig.savefig(F.FIGURES / "kernel_reg_radial_profile_joint.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("Saved figures/kernel_reg_radial_profile_joint.png")

    # ---- HOLDOUT TEST ----
    holdout_dz = HOLDOUT_Z - REF_Z
    n_steps = int(holdout_dz / 0.5)

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

        # analytic propagator
        pred_analytic = P.propagate(sharp, holdout_dz, dx)
        err_analytic = nrmse(pred_analytic, actual)

        # apply joint regularised K(dz=0.5) three times
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
        print(f"    regularised K(dz=0.5) x{n_steps}         NRMSE = {err_learned:.6f}")

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
        ax.set_title(f"Reg. learned x{n_steps} (NRMSE={err_learned:.4f})")
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

    fig.suptitle(f"Holdout test (joint regularised): predicting z = {HOLDOUT_Z} um by applying dz=0.5 kernel {n_steps} times\n"
                 f"lambda = {lam:.1e}  |  z=2.0 data NOT used in training", fontsize=13)
    fig.savefig(F.FIGURES / "kernel_reg_holdout_test_joint.png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"\nSaved figures/kernel_reg_holdout_test_joint.png")

    print("\n--- Holdout Summary ---")
    print(f"{'comp':>6}  {'analytic':>10}  {'reg_3x':>10}")
    for r in holdout_errors:
        print(f"{r['comp']:>6}  {r['analytic']:10.6f}  {r['learned_3x']:10.6f}")


if __name__ == "__main__":
    main()
