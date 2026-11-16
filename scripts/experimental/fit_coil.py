"""Physics fit of the experimental B_NV maps with the designed coil (COMSOL_params/Gemetry_C1.dxf).

Model, per case:   B_NV(X, Y) = I * u . B_coil(X - x0, Y - y0; orientation, h) + background
  B_coil   field of the DXF spiral (2 um wide thin sheet, leads extended and closed far away), free space,
           computed with the repository's validated generator (simulate_fields.rasterize / fields_from_current)
  orientation  one of 8 (4 rotations x mirror) of the design in the image
  h        standoff of the NV layer above the coil (um)
  u        NV axis (theta from +z, phi from +X); u and -u are equivalent (the sign goes into I)
  I        current (A), background = offset + plane + a step at the camera seam x < 86 um
Coarse search (FFT correlation over position, linear I and background), then continuous refinement.
Fields are in physical axes: X = column direction, Y = row direction (rows ascending in Y), z up.

Usage: python3 fit_coil.py <exp.npz from load_experimental.py> <out_dir> [cases e.g. 1,2,3,4,6]
"""
import json, os, sys
import numpy as np
from scipy import ndimage, optimize
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulate_fields import rasterize, fields_from_current

DX = 0.25                      # fine grid (um)
NF = 2048                      # fine canvas (512 um)
OFF = 255.875                  # coil coordinate -> canvas coordinate; makes 1 um block centres half-integers
NB = NF // 4                   # 1 um blocks
LEAD_END = 200.0
I_DESIGN = 600e-6
SPIRAL = [(-21, -18), (21, -18), (21, 18), (-9, 18), (-9, -6), (9, -6), (9, 6), (3, 6), (3, 0), (-3, 0),
          (-3, 12), (15, 12), (15, -12), (-15, -12)]
PATH = [(-21, LEAD_END)] + SPIRAL + [(-15, LEAD_END), (-21, LEAD_END)]     # closed far away
ORIENTS = [(r, m) for m in (0, 1) for r in range(4)]


def transform(points, orient, shift=(0.0, 0.0)):
    r, m = orient
    p = np.asarray(points, float)
    if m:
        p = p*np.array([-1.0, 1.0])
    for _ in range(r):
        p = np.stack([-p[:, 1], p[:, 0]], 1)
    return p + np.asarray(shift)


def coil_fields(h_list, orient, frac=(0.0, 0.0), current=1.0):
    """Physical (Bx, By, Bz) in uT per unit 'current' on 1 um blocks [len(h), 3, NB, NB], coil centre at block
    coordinate (0,0) + frac. Block centre m sits at coil coordinate m - 255.5."""
    p = transform(PATH, orient, frac) + OFF
    kx, ky = rasterize(p, 2.0, current, (NF, NF), DX, 0)
    out = []
    for f in fields_from_current(kx, ky, DX, list(h_list)):
        phys = np.stack([f[1], -f[0], -f[2]])                    # validated frame conversion (README_COMSOL_MODEL §8)
        out.append(phys.reshape(3, NB, 4, NB, 4).mean((2, 4)))
    return np.array(out)


def unit(theta, phi):
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])


def good_mask(d):
    """Drop isolated fit-failure spikes and the noisy image corners; keep strong but smooth coil signal."""
    med = ndimage.median_filter(d, 3)
    res = d - med
    noise = 1.4826*np.median(np.abs(res))
    loc = np.sqrt(ndimage.median_filter(res**2, 15))
    spike = np.abs(res) > np.maximum(8.0, 8*noise)
    return ~spike & (loc < 4*noise) & (np.abs(d) < 80)


def window(X, Y, x0, y0, orient, margin=6.0):
    """Pixels within the spiral's bounding box (+margin), oriented like the fitted coil."""
    half = np.array([22.0, 19.0]) + margin
    if orient[0] % 2:
        half = half[::-1]
    return (np.abs(X - x0) < half[0]) & (np.abs(Y - y0) < half[1])


def background_cols(X, Y):
    Xc, Yc = (X - 82.5)/82.5, (Y - 82.5)/82.5
    return [np.ones_like(X), Xc, Yc, (X < 86).astype(float)]


def linear_fit(d, m, T, bg):
    A = np.stack([T[m]] + [b[m] for b in bg], 1)
    coef, *_ = np.linalg.lstsq(A, d[m], rcond=None)
    model = coef[0]*T + sum(c*b for c, b in zip(coef[1:], bg))
    return coef, model


def correlate(a, t, size=1024):
    """S[ty, tx] = sum_j a[j] t[j + t] (2D), a and t zero-padded."""
    A = np.zeros((size, size)); A[:a.shape[0], :a.shape[1]] = a
    Tt = np.zeros((size, size)); Tt[:t.shape[0], :t.shape[1]] = t
    return np.real(np.fft.ifft2(np.conj(np.fft.fft2(A))*np.fft.fft2(Tt)))


def coarse(d, m, X, Y, heights):
    bg = background_cols(X, Y)
    coef, _ = linear_fit(d, m, np.zeros_like(d), bg)                  # background-only fit first
    r = np.where(m, d - sum(c*b for c, b in zip(coef[1:], bg)), 0.0)
    M = m.astype(float)
    thetas = np.deg2rad(np.r_[0, 15, 30, 45, 54.7356, 60, 75, 90])
    phis = np.deg2rad(np.arange(0, 360, 15))
    U = np.array([unit(t, p) for t in thetas for p in phis])           # [nu, 3]
    best = []
    n = d.shape[0]
    sl = slice(256 - n, 257)                                           # t = 256 - x0, x0 in [0, n]
    for orient in ORIENTS:
        F = coil_fields(heights, orient)
        for ih, h in enumerate(heights):
            C = np.array([correlate(r, F[ih, c])[sl, sl] for c in range(3)])           # [3, ty, tx]
            CC = np.array([[correlate(M, F[ih, a]*F[ih, b])[sl, sl] for b in range(3)] for a in range(3)])
            num = np.einsum("uc,cyx->uyx", U, C)
            den = np.einsum("ua,ub,abyx->uyx", U, U, CC)
            score = num**2/np.maximum(den, 1e-30)
            iu, ty, tx = np.unravel_index(np.argmax(score), score.shape)
            # sliced index ty <-> t = 256 - n + ty <-> centre y0 = 256 - t = n - ty
            best.append((score[iu, ty, tx], orient, h, U[iu], float(n - tx), float(n - ty)))
    best.sort(key=lambda b: -b[0])
    return best, U


def evaluate(params, orient, d, m, X, Y):
    h, theta, phi, x0, y0 = params
    ix, iy = int(np.floor(x0)), int(np.floor(y0))
    F = coil_fields([h], orient, (x0 - ix, y0 - iy))[0]
    j = np.arange(d.shape[1]); i = np.arange(d.shape[0])
    T = np.einsum("c,cyx->yx", unit(theta, phi), F[:, (i - iy + 256)[:, None], (j - ix + 256)[None, :]])
    coef, model = linear_fit(d, m, T, background_cols(X, Y))
    return coef, model, T, F


def main():
    src, out = sys.argv[1], sys.argv[2]
    cases = [int(c) for c in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3,4,6").split(",")]
    os.makedirs(out, exist_ok=True)
    E = np.load(src)
    Xv, Yv = E["X"], E["Y"]
    X, Y = np.meshgrid(Xv, Yv)
    heights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    summary = {}
    for n in cases:
        d = E[f"case{n}"].astype(float)
        m = good_mask(d)
        best, _ = coarse(d, m, X, Y, heights)
        top = best[0]
        _, orient, h0, u0, x0, y0 = top
        theta0 = float(np.arccos(np.clip(u0[2], -1, 1))); phi0 = float(np.arctan2(u0[1], u0[0]))
        print(f"case{n}: coarse best orient {orient} h {h0} theta {np.rad2deg(theta0):.1f} phi {np.rad2deg(phi0):.1f} "
              f"centre ({x0:.1f}, {y0:.1f})", flush=True)
        per_orient = {}
        for b in best:
            per_orient.setdefault(b[1], b)
        for o, b in sorted(per_orient.items(), key=lambda kv: -kv[1][0]):
            print(f"   orient {o}: score {b[0]/best[0][0]:.3f} of best, h {b[2]}, centre ({b[4]:.0f}, {b[5]:.0f})")
        full_mask = m
        m = full_mask & window(X, Y, x0, y0, orient)

        def resid(p):
            _, model, _, _ = evaluate(p, orient, d, m, X, Y)
            return (model - d)[m]
        res = optimize.least_squares(resid, [h0, theta0, phi0, x0, y0], diff_step=[0.02, 0.01, 0.01, 0.02, 0.02],
                                     bounds=([0.3, 0, -2*np.pi, -50, -50], [40, np.pi/2 + 1e-6, 4*np.pi, 215, 215]),
                                     x_scale=[1, 0.2, 0.2, 1, 1], max_nfev=200)
        h, theta, phi, x0, y0 = res.x
        coef, model, T, F = evaluate(res.x, orient, d, m, X, Y)
        r = (d - model)[m]
        # crude 1-sigma uncertainties from the Jacobian
        J = res.jac; cov = np.linalg.pinv(J.T @ J)*np.mean(r**2)
        sig = np.sqrt(np.diag(cov))
        core = m & (np.hypot(X - x0, Y - y0) < 35)
        sig_core = coef[0]*T
        explained = 1 - np.sum((d - model)[core]**2)/np.sum((d - (model - sig_core))[core]**2)
        summary[f"case{n}"] = dict(
            orientation=dict(rot90=orient[0], mirror=orient[1]), standoff_um=float(h), standoff_sigma_um=float(sig[0]),
            nv_theta_deg=float(np.rad2deg(theta)), nv_phi_deg=float(np.rad2deg(phi) % 360),
            nv_sigma_deg=[float(np.rad2deg(sig[1])), float(np.rad2deg(sig[2]))],
            centre_um=[float(x0), float(y0)], current_uA=float(coef[0]*1e6), current_over_design=float(coef[0]/I_DESIGN),
            background_uT=[float(c) for c in coef[1:]], residual_rms_uT=float(np.sqrt(np.mean(r**2))),
            signal_explained_near_coil=float(explained), good_pixels=int(m.sum()),
            coarse_top5=[dict(orient=list(b[1]), h=b[2]) for b in best[:5]])
        print(json.dumps(summary[f"case{n}"], indent=1), flush=True)
        np.savez_compressed(f"{out}/fit_case{n}.npz", data=d, mask=m, full_mask=full_mask, model=model, signal=coef[0]*T,
                            background=model - coef[0]*T, X=Xv, Y=Yv)
    json.dump(summary, open(f"{out}/fit_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
