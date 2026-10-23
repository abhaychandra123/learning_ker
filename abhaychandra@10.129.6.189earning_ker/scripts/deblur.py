"""
ISTA deblurring with L1 regularization.

We want to solve:

    minimize_s  0.5 * ||A(s) - b||_2^2 + lambda_ista * ||s||_1

where:
    s = sharp/source field at the reference plane
    b = observed blurred field at z_target
    A = forward propagation / blur operator
    A^T = adjoint of A
    lambda_ista >= 0 is the L1 sparsity parameter

For a Fourier-domain convolution operator with transfer K_hat(k),
A(s) = Crop( F^{-1}[ K_hat * F(Pad(s)) ] )
and the adjoint is
A^T(r) = Crop( F^{-1}[ conj(K_hat) * F(Pad(r)) ] ).

The standard ISTA iteration is:

    v_t = s_t - (1/L) * A^T(A(s_t) - b)
    s_{t+1} = soft_threshold(v_t, lambda_ista / L)

with the soft-threshold operator:
    soft(x, theta) = sign(x) * max(|x| - theta, 0)

and Lipschitz constant:
    L = max_k |K_hat(k)|^2

This implementation keeps the existing joint kernel learning idea but removes all
kernel-estimation regularization (no Wiener/Tikhonov term). The only
regularization is the L1 ISTA penalty.

Usage:
    python scripts/deblur.py --z 3.0 --comp Bz --epsilon 1e-3 --lambda_ista 1e-4 --pad 32
    python scripts/deblur.py --z 5.0 --comp Bz
    python scripts/deblur.py --z 3.0
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fields as F
import propagator as P

REF_Z = 0.5
COMPONENTS = ("Bx", "By", "Bz")


def pad_image(x, pad):
    """Zero-pad an image symmetrically around its center."""
    x = np.asarray(x, dtype=np.float64)
    if pad < 0:
        raise ValueError(f"pad must be >= 0, got {pad}")
    if pad == 0:
        return x.copy()
    ny, nx = x.shape
    padded = np.zeros((ny + 2 * pad, nx + 2 * pad), dtype=np.float64)
    padded[pad:pad + ny, pad:pad + nx] = x
    return padded


def crop_image(x, original_shape, pad):
    """Crop a padded image back to the original shape."""
    ny, nx = original_shape
    return np.asarray(x, dtype=np.float64)[pad:pad + ny, pad:pad + nx].copy()


def learn_kernel_joint(sharps, blurreds, pad=32, den_threshold=1e-12):
    """Unregularized joint least-squares kernel on the padded grid.

    K_hat = sum_c conj(S_c) * B_c / sum_c |S_c|^2
    with a zero mask only at numerically tiny denominators.
    """
    num = None
    den = None
    for sharp, blurred in zip(sharps, blurreds):
        sharp_p = pad_image(sharp, pad)
        blurred_p = pad_image(blurred, pad)
        S_hat = np.fft.fft2(sharp_p.astype(np.float64))
        B_hat = np.fft.fft2(blurred_p.astype(np.float64))
        if num is None:
            num = np.conj(S_hat) * B_hat
            den = np.abs(S_hat) ** 2
        else:
            num += np.conj(S_hat) * B_hat
            den += np.abs(S_hat) ** 2

    K_hat = np.zeros_like(num, dtype=np.complex128)
    mask = np.abs(den) > den_threshold
    K_hat[mask] = num[mask] / den[mask]
    return K_hat


def forward_operator(s, K_hat, pad):
    """Forward model A(s) = Crop( F^{-1}[K_hat * F(Pad(s))] )."""
    s_p = pad_image(s, pad)
    spec = np.fft.fft2(s_p.astype(np.float64))
    out_p = np.fft.ifft2(K_hat * spec)
    out = crop_image(np.real(out_p), s.shape, pad)
    return out.astype(np.float64)


def adjoint_operator(r, K_hat, pad):
    """Adjoint operator A^T(r) = Crop( F^{-1}[conj(K_hat) * F(Pad(r))] )."""
    r_p = pad_image(r, pad)
    spec = np.fft.fft2(r_p.astype(np.float64))
    out_p = np.fft.ifft2(np.conj(K_hat) * spec)
    out = crop_image(np.real(out_p), r.shape, pad)
    return out.astype(np.float64)


def soft_threshold(x, threshold):
    """Soft-thresholding operator: sign(x) * max(|x| - theta, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def deblur_ista(blurred, K_hat, lambda_ista, n_iter=None, pad=32, *, epsilon=1e-3):
    """ISTA to relative proximal-gradient tolerance (fixed n_iter for old API tests)."""
    from iterative import solve_l1, console_progress
    print(f"    Starting ISTA; stopping at relative PG <= {epsilon:g}. "
          "Progress every 10 updates.", flush=True)
    result, rows = solve_l1(blurred, K_hat, lambda_ista, method="ista",
                            n_iter=n_iter, pad=pad, epsilon=epsilon,
                            progress=console_progress("ISTA"))
    for row in rows:
        row.update(iter=row["iteration"], cost=row["objective"])
    last = rows[-1]
    print(f"    ISTA {last['stop_reason']}: {last['iteration']} updates, "
          f"relative residual={last['relative_pg']:.6g}, epsilon={epsilon:g}")
    return result, rows


def nrmse(predicted, actual):
    mse = np.mean((predicted - actual) ** 2)
    return np.sqrt(mse) / (actual.max() - actual.min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--z", type=float, default=3.0)
    parser.add_argument("--comp", type=str, default=None,
                        help="Single component, or omit for all three")
    parser.add_argument("--epsilon", type=float, default=1e-3,
                        help="relative proximal-gradient tolerance (default: 0.001)")
    parser.add_argument("--lambda_ista", type=float, default=1e-4)
    parser.add_argument("--pad", type=int, default=32)
    args = parser.parse_args()

    if not np.isfinite(args.epsilon) or not 0 < args.epsilon < 1:
        parser.error("epsilon must be finite and strictly between 0 and 1")

    F.FIGURES.mkdir(parents=True, exist_ok=True)
    grid = F.grid()
    dx = grid.dx
    z_target = args.z
    dz = z_target - REF_Z
    comps = [args.comp] if args.comp else list(COMPONENTS)

    print(f"Deblurring: z={z_target} -> z={REF_Z} (dz={dz})")
    print(f"Components: {comps}")
    print(f"Padding: {args.pad}px")
    print(f"Lambda_ISTA: {args.lambda_ista:.6e}")
    print()

    sharps_all = [F.load(REF_Z, c) for c in COMPONENTS]
    blurreds_all = [F.load(z_target, c) for c in COMPONENTS]
    K_hat = learn_kernel_joint(sharps_all, blurreds_all, pad=args.pad)
    print(f"Joint kernel learned on padded grid (shape={K_hat.shape})")
    print()

    all_results = {}
    all_forward_nrmse = {}

    for comp in comps:
        blurred = F.load(z_target, comp)
        ground_truth = F.load(REF_Z, comp)

        forward_pred = forward_operator(ground_truth, K_hat, pad=args.pad)
        forward_err = nrmse(forward_pred, blurred)
        all_forward_nrmse[comp] = forward_err

        print(f"--- {comp} ---")
        print(f"  forward model A(ground_truth): NRMSE = {forward_err * 100:.3f}%")

        deblurred, history = deblur_ista(
            blurred, K_hat, args.lambda_ista,
            epsilon=args.epsilon, pad=args.pad
        )

        err = nrmse(deblurred, ground_truth)
        print(f"  ISTA reconstruction vs ground truth: NRMSE = {err * 100:.3f}%\n")

        all_results[comp] = {
            "blurred": blurred,
            "ground_truth": ground_truth,
            "deblurred": deblurred,
            "nrmse": err,
            "forward_nrmse": forward_err,
            "history": history,
        }

    summary = [{"component": comp,
                "iterations": all_results[comp]["history"][-1]["iteration"],
                "converged": all_results[comp]["history"][-1]["converged"],
                "stop_reason": all_results[comp]["history"][-1]["stop_reason"],
                "epsilon": args.epsilon,
                "relative_pg": all_results[comp]["history"][-1]["relative_pg"],
                "forward_nrmse_range": all_results[comp]["forward_nrmse"],
                "reconstruction_nrmse_range": all_results[comp]["nrmse"]} for comp in comps]
    prefix = F.FIGURES / f"deblur_ista_l1_z{z_target:g}"
    prefix.with_name(prefix.name+"_metrics.json").write_text(json.dumps({
        "algorithm": "ISTA", "config": vars(args), "summary": summary,
        "history": {comp: all_results[comp]["history"] for comp in comps}}, indent=2, allow_nan=False))
    with prefix.with_name(prefix.name+"_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)

    def stop_label(result):
        last = result["history"][-1]
        return f"{last['iteration']} updates; {last['stop_reason']}"

    # ---- FIGURE 1: comparison (blurred, ground truth, deblurred, residual) ----
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
            (r["deblurred"], f"ISTA (L1) {comp} (NRMSE={r['nrmse'] * 100:.2f}%)\n{stop_label(r)}"),
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
        ax.set_title("Residual (truth - reconstruction)", fontsize=10)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, label="uT")

    for ax in axes.ravel():
        ax.set_xlabel("x (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (um)")

    fig.suptitle(f"ISTA (L1) deblurring: z={z_target} -> z={REF_Z}, "
                 f"lambda_ista={args.lambda_ista:.2e}, pad={args.pad}, "
                 f"epsilon={args.epsilon:g}", fontsize=12)
    fig.savefig(F.FIGURES / f"deblur_ista_l1_z{z_target:.0f}.png",
                bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"Saved figures/deblur_ista_l1_z{z_target:.0f}.png")

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
            (r["deblurred"], f"ISTA (L1) ({r['nrmse'] * 100:.2f}%)\n{stop_label(r)}"),
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
    fig.suptitle(f"Zoom: ISTA (L1) deblurring z={z_target} -> z={REF_Z}", fontsize=13)
    fig.savefig(F.FIGURES / f"deblur_ista_l1_z{z_target:.0f}_zoom.png",
                bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"Saved figures/deblur_ista_l1_z{z_target:.0f}_zoom.png")

    # ---- FIGURE 3: every component's actual stopping iteration ----
    fig, axes = plt.subplots(n_comp, 3, figsize=(17, 4*n_comp),
                             constrained_layout=True, squeeze=False)
    for row, comp in enumerate(comps):
        history = all_results[comp]["history"]
        iters = [h["iteration"] for h in history]
        label = f"{comp}: {stop_label(all_results[comp])}"
        for col, (key, ylabel) in enumerate([
                ("data", "0.5 ||A(s)-b||²"), ("cost", "L1 objective"),
                ("relative_pg", "PG RMS / initial PG RMS")]):
            values = [h[key] for h in history]
            ax = axes[row, col]
            ax.plot(iters, values, "o-", ms=3, label=label)
            ax.scatter([iters[-1]], [values[-1]], s=35, zorder=3)
            ax.set(xlabel="Completed updates", ylabel=ylabel, title=label)
            ax.set_yscale("symlog", linthresh=1e-10)
            if key == "relative_pg":
                ax.axhline(args.epsilon, color="black", ls="--", label=f"epsilon={args.epsilon:g}")
                ax.legend(fontsize=8)
            ax.grid(alpha=.3)
    fig.suptitle(f"ISTA convergence: actual update counts per component | epsilon={args.epsilon:g}")
    fig.savefig(F.FIGURES / f"deblur_ista_l1_z{z_target:.0f}_convergence.png",
                bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"Saved figures/deblur_ista_l1_z{z_target:.0f}_convergence.png")

    # ---- FIGURE 4: spectra ----
    fig, axes = plt.subplots(1, n_comp, figsize=(6 * n_comp, 5),
                             constrained_layout=True)
    if n_comp == 1:
        axes = [axes]

    for ax, comp in zip(axes, comps):
        r = all_results[comp]
        gt_ref = r["ground_truth"]
        k = P.k_grid(gt_ref.shape, dx)
        for img, label, color in [
            (r["blurred"], f"blurred (z={z_target})", "red"),
            (r["ground_truth"], f"ground truth (z={REF_Z})", "black"),
            (r["deblurred"], "ISTA (L1) reconstruction", "green"),
        ]:
            k_c, spec = P.radial_average(P.amplitude_spectrum(img), k, k_max=2.0)
            ax.semilogy(k_c, spec, color=color, lw=1.3, label=label)
        ax.set_xlabel("k (cycles/um)")
        ax.set_ylabel("amplitude spectrum")
        ax.set_title(f"{comp}: {stop_label(r)}")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1.5)

    fig.suptitle("Spectra: blurred vs ISTA reconstruction vs ground truth", fontsize=12)
    fig.savefig(F.FIGURES / f"deblur_ista_l1_z{z_target:.0f}_spectra.png",
                bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"Saved figures/deblur_ista_l1_z{z_target:.0f}_spectra.png")

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print(f"ISTA (L1) Deblurring Summary (z={z_target} -> z={REF_Z})")
    print(f"{'=' * 70}")
    print(f"{'Component':>10}  {'Updates':>8}  {'Stop reason':>22}  {'Forward NRMSE':>16}  {'Reconstruction NRMSE':>22}")
    for comp in comps:
        fw = all_forward_nrmse[comp]
        err = all_results[comp]["nrmse"]
        last = all_results[comp]["history"][-1]
        print(f"{comp:>10}  {last['iteration']:8d}  {last['stop_reason']:>22}  {fw * 100:15.3f}%  {err * 100:21.3f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
