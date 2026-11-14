"""Side by side at one height: COMSOL, free-space simulator, exact coil + walls + k0.

Usage: python3 plot_10um_compare.py <dir with L1_C1_all.npz and dxf_current.npz> <out.png> [z_um]
dxf_current.npz is written by dxf_forward.py (sheet current of the DXF coil, 600 uA).
Components are shown with the CSV labels (Bx, By, Bz as exported) and CSV axes (cpl1x, cpl1y).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP, OUT = sys.argv[1], sys.argv[2]
Z = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
d, N, mu0, K0 = 0.1, 1400, 4e-7*np.pi, 0.10
old = np.load(f"{SP}/L1_C1_all.npz")
cur = np.load(f"{SP}/dxf_current.npz")
Ku, Kv = cur["Ku"].astype(float), cur["Kv"].astype(float)
u = old["x"].astype(float); v = old["y"].astype(float)


def mirror(a, pu, pv):
    a = np.concatenate([a, pu*a[:, ::-1]], axis=1)
    return np.concatenate([a, pv*a[::-1, :]], axis=0)


def field(z, mode, k0=0.0, pad=1400):
    """Sheet current at z=0 -> (Bu, Bv, Bz) at height z (um), tesla."""
    if mode == "free":
        Fu, Fv = np.fft.fft2(np.pad(Ku, pad)), np.fft.fft2(np.pad(Kv, pad)); n = N + 2*pad
        sl = (slice(pad, pad + N), slice(pad, pad + N))
    else:
        Fu, Fv = np.fft.fft2(mirror(Ku, 1, -1)), np.fft.fft2(mirror(Kv, -1, 1)); n = 2*N
        sl = (slice(0, N), slice(0, N))
    k = 2*np.pi*np.fft.fftfreq(n, d*1e-6)
    KV, KU = np.meshgrid(k, k, indexing="ij")
    kap = np.sqrt(KU**2 + KV**2 + (k0*1e6)**2)
    if k0 == 0:
        kap[0, 0] = 1.0
    E = 0.5*mu0*np.exp(-kap*z*1e-6)
    out = (E*Fv, -E*Fu, 1j*(KU*Fv - KV*Fu)*E/kap)
    return [np.real(np.fft.ifft2(a))[sl] for a in out]


def to_csv_labels(Bu, Bv, Bz):
    return [Bv, -Bu, Bz]          # CSV Bx = B along rows, CSV By = -(B along columns)


comsol = [c.astype(float) for c in old[f"L1_C1_z{Z:.1f}"]]
free = to_csv_labels(*field(Z, "free"))
boxk = to_csv_labels(*field(Z, "box", K0))
T = 1e6                                            # tesla -> microtesla
rel = lambda a, b: 100*np.linalg.norm(a - b)/np.linalg.norm(b)

names = ["Bx", "By", "Bz"]
cols = ["COMSOL (Extra Fine)", "Free-space simulator\n(exact coil, no walls, no $k_0$)",
        "Exact coil + walls + $k_0$=0.10", "Free space − COMSOL", "Walls + $k_0$ − COMSOL"]
ext = [u[0] - d/2, u[-1] + d/2, v[0] - d/2, v[-1] + d/2]
fig = plt.figure(figsize=(22, 17.5), dpi=110)
gs = fig.add_gridspec(4, 5, height_ratios=[1, 1, 1, 0.8], hspace=0.38, wspace=0.28, top=0.95, bottom=0.04)
iy = int(np.argmin(np.abs(v - 0.0)))               # horizontal line through the coil centre
for r, nm in enumerate(names):
    c, f, b = comsol[r]*T, free[r]*T, boxk[r]*T
    lim = np.max(np.abs(c))
    dlim = max(np.max(np.abs(f - c)), np.max(np.abs(b - c)))
    panels = [(c, lim, "RdBu_r", ""), (f, lim, "RdBu_r", f"error {rel(f, c):.1f}%"),
              (b, lim, "RdBu_r", f"error {rel(b, c):.1f}%"), (f - c, dlim, "PuOr_r", ""), (b - c, dlim, "PuOr_r", "")]
    for j, (img, L, cmap, note) in enumerate(panels):
        ax = fig.add_subplot(gs[r, j])
        im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=-L, vmax=L)
        ax.axhline(v[iy], color="k", lw=0.5, ls=":")
        title = (cols[j] + "\n" if r == 0 else "") + f"{nm}" + (f"   ({note})" if note else "")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (µm)", fontsize=9); ax.set_ylabel("y (µm)", fontsize=9); ax.tick_params(labelsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=8)
        cb.set_label("µT", fontsize=8)
for j, nm in enumerate(names):
    ax = fig.add_subplot(gs[3, j])
    ax.plot(u, comsol[j][iy]*T, "k", lw=2.2, label="COMSOL")
    ax.plot(u, free[j][iy]*T, color="tab:red", lw=1.4, label="free space")
    ax.plot(u, boxk[j][iy]*T, color="tab:blue", lw=1.4, ls="--", label="walls + $k_0$")
    ax.set_title(f"{nm} along y = {v[iy]:.2f} µm (dotted line)", fontsize=11)
    ax.set_xlabel("x (µm)", fontsize=9); ax.set_ylabel("µT", fontsize=9); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    if j == 0:
        ax.legend(fontsize=9)
ax = fig.add_subplot(gs[3, 3:])
ax.axis("off")
ax.text(0.0, 0.95,
        f"L1_C1 at z = {Z:g} µm, 600 µA. Same colour scale across each row (COMSOL's max |B|);\n"
        "difference maps share their own scale per row.\n\n"
        "Free space: the DXF coil as a thin current sheet, field by Fourier Biot–Savart\n"
        "with zero padding, decay exp(−k z). This is the physics of simulate_fields.py.\n\n"
        "Walls + k$_0$: same current, mirror images at the 140 µm box walls (B·n = 0),\n"
        "decay exp(−√(k² + k$_0$²) z) with k$_0$ = 0.10 rad/µm measured from COMSOL.\n\n"
        "Errors are relative L2 against COMSOL over the full image, unscaled.\n"
        "Script: scripts/comsol_check/plot_10um_compare.py",
        va="top", fontsize=11, family="sans-serif")
fig.suptitle(f"COMSOL vs simulators at {Z:g} µm above the coil", fontsize=16, y=0.985)
fig.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
for r, nm in enumerate(names):
    print(f"{nm}: max|B| COMSOL {np.max(np.abs(comsol[r]))*T:.3f} µT, free {np.max(np.abs(free[r]))*T:.3f}, "
          f"walls+k0 {np.max(np.abs(boxk[r]))*T:.3f};  error free {rel(free[r], comsol[r]):.1f}%, walls+k0 {rel(boxk[r], comsol[r]):.1f}%")
