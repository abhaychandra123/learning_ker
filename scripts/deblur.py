"""
Deblurring engine using ISTA with Total Variation regularisation.

Given a blurred image at height z, recover the sharp image at z=0.5
using the learned joint kernel.

ISTA solves:   minimize ||K*s - b||^2 + lam_tv * TV(s)

  1. gradient step:   s_temp = s - (1/L) * K^T(K*s - b)
  2. proximal step:   s = prox_TV(s_temp, lam_tv/L)

K^T in Fourier space is conj(K_hat). L = max(|K_hat|^2).

Usage:
    python scripts/deblur.py                   # default z=3.0, all components
    python scripts/deblur.py --z 5.0           # deblur from z=5.0
    python scripts/deblur.py --z 3.0 --comp Bz # single component
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fields as F
import propagator as P

REF_Z = 0.5
COMPONENTS = ("Bx", "By", "Bz")


# ---- Joint kernel learning ----

def learn_kernel_wiener_joint(sharps, blurreds, lam):
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


def estimate_lambda(sharp, dx):
    S = np.fft.fft2(sharp.astype(np.float64))
    power = np.abs(S) ** 2
    k = P.k_grid(sharp.shape, dx)
    return float(np.median(power[k > 1.0]))


# ---- ISTA-TV deblurring ----

def tv_grad_mag(s):
    dy = np.diff(s, axis=0, append=s[-1:, :])
    dx = np.diff(s, axis=1, append=s[:, -1:])
    return np.sqrt(dx ** 2 + dy ** 2 + 1e-8)


def tv_prox_chambolle(s, lam_tv, n_inner=30):
    """Proximal operator for lam_tv * TV(s) via Chambolle's dual projection.

    Solves: minimize (1/2)||x - s||^2 + lam_tv * TV(x)

    The dual variables (py, px) represent the gradient field.
    The algorithm alternates between:
      - computing the divergence of the dual field
      - updating the dual field by projecting onto the TV ball
    """
    py = np.zeros_like(s)
    px = np.zeros_like(s)
    tau = 0.25

    for _ in range(n_inner):
        # divergence of dual field
        div_p = np.zeros_like(s)
        div_p[1:, :] += py[1:, :] - py[:-1, :]
        div_p[0, :] += py[0, :]
        div_p[:, 1:] += px[:, 1:] - px[:, :-1]
        div_p[:, 0] += px[:, 0]

        # gradient of (s + lam_tv * div_p)
        grad_val = s + lam_tv * div_p
        gy = np.diff(grad_val, axis=0, append=grad_val[-1:, :])
        gx = np.diff(grad_val, axis=1, append=grad_val[:, -1:])

        # update dual with projection
        mag = np.sqrt(gx ** 2 + gy ** 2 + 1e-10)
        py = (py + tau * gy) / (1 + tau * mag / lam_tv)
        px = (px + tau * gx) / (1 + tau * mag / lam_tv)

    # final divergence
    div_p = np.zeros_like(s)
    div_p[1:, :] += py[1:, :] - py[:-1, :]
    div_p[0, :] += py[0, :]
    div_p[:, 1:] += px[:, 1:] - px[:, :-1]
    div_p[:, 0] += px[:, 0]

    return s + lam_tv * div_p


def deblur_ista_tv(blurred, K_hat, lam_tv, n_iter=100, n_inner=20):
    """ISTA with TV regularisation.

    Each iteration:
      1. Compute gradient of data term: K^T(K s - b)
         In Fourier space: IFFT(conj(K_hat) * (K_hat * FFT(s) - FFT(b)))
      2. Gradient descent step: s = s - (1/L) * gradient
      3. TV proximal step: s = prox_TV(s, lam_tv/L)

    L = max(|K_hat|^2) is the Lipschitz constant of the data gradient.
    """
    B = np.fft.fft2(blurred.astype(np.float64))
    L = float(np.max(np.abs(K_hat) ** 2))
    step = 1.0 / L

    s = blurred.astype(np.float64).copy()
    history = []

    for it in range(n_iter):
        # gradient of ||K*s - b||^2
        S_hat = np.fft.fft2(s)
        grad = np.real(np.fft.ifft2(np.conj(K_hat) * (K_hat * S_hat - B)))

        # gradient descent + TV proximal
        s_temp = s - step * grad
        s = tv_prox_chambolle(s_temp, lam_tv * step, n_inner=n_inner)

        if it % 10 == 0 or it == n_iter - 1:
            resid = np.real(np.fft.ifft2(K_hat * np.fft.fft2(s) - B))
            data_err = float(np.mean(resid ** 2))
            tv_val = float(np.sum(tv_grad_mag(s)))
            history.append({"iter": it, "data": data_err, "tv": tv_val,
                            "cost": data_err + lam_tv * tv_val})
            print(f"    iter {it:4d}  data={data_err:.4f}  TV={tv_val:.1f}")

    return s, history


def nrmse(predicted, actual):
    mse = np.mean((predicted - actual) ** 2)
    return np.sqrt(mse) / (actual.max() - actual.min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--z", type=float, default=3.0)
    parser.add_argument("--comp", type=str, default=None,
                        help="Single component, or omit for all three")
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--lam_tv", type=float, default=None)
    args = parser.parse_args()

    grid = F.grid()
    dx = grid.dx
    z_target = args.z
    dz = z_target - REF_Z
    comps = [args.comp] if args.comp else list(COMPONENTS)

    print(f"Deblurring: z={z_target} -> z={REF_Z} (dz={dz})")
    print(f"Components: {comps}")
    print()

    # ---- learn the joint kernel ----
    sharps_all = [F.load(REF_Z, c) for c in COMPONENTS]
    blurreds_all = [F.load(z_target, c) for c in COMPONENTS]
    lam_est = estimate_lambda(sharps_all[2], dx)
    K_hat = learn_kernel_wiener_joint(sharps_all, blurreds_all, lam_est)
    print(f"Joint kernel learned (lambda={lam_est:.2e})")

    # auto TV lambda
    gt_ref = F.load(REF_Z, "Bz")
    sig_range = gt_ref.max() - gt_ref.min()
    if args.lam_tv is None:
        lam_tv = 0.001 * sig_range
    else:
        lam_tv = args.lam_tv
    print(f"TV lambda = {lam_tv:.4f}")
    print()

    all_results = {}

    for comp in comps:
        blurred = F.load(z_target, comp)
        ground_truth = F.load(REF_Z, comp)

        print(f"--- {comp} ---")
        deblurred, history = deblur_ista_tv(
            blurred, K_hat, lam_tv, n_iter=args.n_iter, n_inner=20
        )
        err = nrmse(deblurred, ground_truth)
        print(f"  NRMSE vs ground truth: {err*100:.2f}%\n")

        all_results[comp] = {
            "blurred": blurred, "ground_truth": ground_truth,
            "deblurred": deblurred, "nrmse": err, "history": history,
        }

    # ---- FIGURE 1: full comparison (all components) ----
    n_comp = len(comps)
    fig, axes = plt.subplots(n_comp, 4, figsize=(20, 4.5 * n_comp),
                             constrained_layout=True)
    if n_comp == 1:
        axes = axes[np.newaxis, :]

    for i, comp in enumerate(comps):
        r = all_results[comp]
        vmax = np.percentile(np.abs(r["ground_truth"]), 99.5)
        extent = grid.extent

        for j, (img, title) in enumerate([
            (r["blurred"], f"Blurred {comp} (z={z_target})"),
            (r["ground_truth"], f"Ground truth {comp} (z={REF_Z})"),
            (r["deblurred"], f"ISTA-TV {comp} (NRMSE={r['nrmse']*100:.2f}%)"),
        ]):
            ax = axes[i, j]
            ax.imshow(img, origin="lower", extent=extent, cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.grid(False)

        residual = r["ground_truth"] - r["deblurred"]
        rmax = np.percentile(np.abs(residual), 99.5)
        ax = axes[i, 3]
        im = ax.imshow(residual, origin="lower", extent=extent, cmap="RdBu_r",
                       vmin=-rmax, vmax=rmax, interpolation="nearest")
        ax.set_title(f"Residual (truth - deblurred)", fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, label="uT")

    for ax in axes.ravel():
        ax.set_xlabel("x (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (um)")

    fig.suptitle(f"ISTA-TV deblurring: z={z_target} -> z={REF_Z} (dz={dz}), "
                 f"{args.n_iter} iterations", fontsize=13)
    fig.savefig(F.FIGURES / f"deblur_z{z_target:.0f}.png",
                bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"Saved figures/deblur_z{z_target:.0f}.png")

    # ---- FIGURE 2: zoom into coil ----
    cy, cx = grid.ny // 2, grid.nx // 2
    crop = 250
    sl = (slice(cy - crop, cy + crop), slice(cx - crop, cx + crop))
    ext_crop = [grid.x0 + (cx - crop) * dx, grid.x0 + (cx + crop) * dx,
                grid.y0 + (cy - crop) * dx, grid.y0 + (cy + crop) * dx]

    fig, axes = plt.subplots(n_comp, 3, figsize=(15, 4.5 * n_comp),
                             constrained_layout=True)
    if n_comp == 1:
        axes = axes[np.newaxis, :]

    for i, comp in enumerate(comps):
        r = all_results[comp]
        vmax = np.percentile(np.abs(r["ground_truth"]), 99.5)
        for j, (img, title) in enumerate([
            (r["blurred"], f"Blurred {comp} (z={z_target})"),
            (r["ground_truth"], f"Ground truth (z={REF_Z})"),
            (r["deblurred"], f"ISTA-TV ({r['nrmse']*100:.2f}%)"),
        ]):
            ax = axes[i, j]
            ax.imshow(img[sl], origin="lower", extent=ext_crop, cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.grid(False)

    for ax in axes.ravel():
        ax.set_xlabel("x (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (um)")
    fig.suptitle(f"Zoom: ISTA-TV deblurring z={z_target} -> z={REF_Z}", fontsize=13)
    fig.savefig(F.FIGURES / f"deblur_z{z_target:.0f}_zoom.png",
                bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"Saved figures/deblur_z{z_target:.0f}_zoom.png")

    # ---- FIGURE 3: convergence ----
    comp0 = comps[0]
    history = all_results[comp0]["history"]
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        iters = [h["iter"] for h in history]

        axes[0].semilogy(iters, [h["data"] for h in history], "b-o", ms=3)
        axes[0].set_xlabel("iteration")
        axes[0].set_ylabel("||Ks - b||^2")
        axes[0].set_title("Data fidelity term")
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(iters, [h["cost"] for h in history], "k-o", ms=3)
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("data + lam_tv * TV")
        axes[1].set_title("Total cost")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"ISTA convergence ({comp0}, {args.n_iter} iterations)", fontsize=12)
        fig.savefig(F.FIGURES / f"deblur_z{z_target:.0f}_convergence.png",
                    bbox_inches="tight", dpi=130)
        plt.close(fig)
        print(f"Saved figures/deblur_z{z_target:.0f}_convergence.png")

    # ---- FIGURE 4: spectra ----
    k = P.k_grid(gt_ref.shape, dx)
    fig, axes = plt.subplots(1, n_comp, figsize=(6 * n_comp, 5),
                             constrained_layout=True)
    if n_comp == 1:
        axes = [axes]

    for ax, comp in zip(axes, comps):
        r = all_results[comp]
        for img, label, color in [
            (r["blurred"], f"blurred (z={z_target})", "red"),
            (r["ground_truth"], f"ground truth (z={REF_Z})", "black"),
            (r["deblurred"], "ISTA-TV deblurred", "green"),
        ]:
            k_c, spec = P.radial_average(P.amplitude_spectrum(img), k, k_max=2.0)
            ax.semilogy(k_c, spec, color=color, lw=1.3, label=label)
        ax.set_xlabel("k (cycles/um)")
        ax.set_ylabel("amplitude spectrum")
        ax.set_title(f"{comp}")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1.5)

    fig.suptitle("Spectra: blurred vs deblurred vs ground truth", fontsize=12)
    fig.savefig(F.FIGURES / f"deblur_z{z_target:.0f}_spectra.png",
                bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"Saved figures/deblur_z{z_target:.0f}_spectra.png")

    # ---- Summary ----
    print(f"\n{'='*50}")
    print(f"ISTA-TV Deblurring Summary (z={z_target} -> z={REF_Z})")
    print(f"{'='*50}")
    print(f"{'Component':>10}  {'NRMSE':>10}")
    for comp in comps:
        err = all_results[comp]["nrmse"]
        print(f"{comp:>10}  {err*100:9.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
