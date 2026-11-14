"""Ampère's law in the air: is curl B = 0 in COMSOL's solution?

In source-free air, curl B = 0 exactly. We estimate the horizontal components of curl B half-way between two
COMSOL planes 1 um apart:
    (curl B)_u = dBz/dv - dBv/dz,     (curl B)_v = dBu/dz - dBz/du
dz from the two planes, du/dv spectrally on the plane average. Fields are low-passed (Gaussian, k < ~0.5 rad/um)
first, so mesh noise and finite-difference error stay small. Controls: the same estimate on simulated fields
from the DXF coil, (a) exact free-space magnetostatics, where curl B must be ~0, and (b) the screened model with
k0 = 0.10, where curl B = -k0^2 A is predicted.

Usage: python3 ampere_test.py <dir with L1_C1_all.npz, dxf_current.npz> <out.png>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SP, OUT = sys.argv[1], sys.argv[2]
d, N, mu0 = 0.1, 1400, 4e-7*np.pi
old = np.load(f"{SP}/L1_C1_all.npz")
cur = np.load(f"{SP}/dxf_current.npz")
Ku, Kv = cur["Ku"].astype(float), cur["Kv"].astype(float)
u = old["x"].astype(float); v = old["y"].astype(float)


def mirror(a, pu, pv):
    a = np.concatenate([a, pu*a[:, ::-1]], axis=1)
    return np.concatenate([a, pv*a[::-1, :]], axis=0)


k = 2*np.pi*np.fft.fftfreq(2*N, d)            # rad/um, on the mirrored (box) grid
KV, KU = np.meshgrid(k, k, indexing="ij"); K = np.hypot(KU, KV)
G = np.exp(-(K/0.35)**2)                       # Gaussian low-pass, 1/e at 0.35 rad/um


def sim(z, k0):
    """Box-mirrored DXF sheet field at z (um), in tesla, components along cols (u), rows (v), z."""
    Fu, Fv = np.fft.fft2(mirror(Ku, 1, -1)), np.fft.fft2(mirror(Kv, -1, 1))
    kap = np.sqrt(K**2 + k0**2)*1e6
    kk = np.where(kap == 0, 1.0, kap)
    E = 0.5*mu0*np.exp(-kap*z*1e-6)
    return [np.real(np.fft.ifft2(a))[:N, :N] for a in (E*Fv, -E*Fu, 1j*(KU*Fv - KV*Fu)*1e6*E/kk)]


def frame(B):
    Bx, By, Bz = B.astype(float)
    return [-By, Bx, Bz]


def curl_h(B1, B2, dz=1.0):
    """Horizontal curl B half-way between planes; returns (curl_u, curl_v, scale) in T/um, low-passed."""
    parity = [(-1, 1), (1, -1), (1, 1)]
    F1 = [np.fft.fft2(mirror(b, *p))*G for b, p in zip(B1, parity)]
    F2 = [np.fft.fft2(mirror(b, *p))*G for b, p in zip(B2, parity)]
    Fm = [(a + b)/2 for a, b in zip(F1, F2)]
    dzB = [np.real(np.fft.ifft2(b - a))[:N, :N]/dz for a, b in zip(F1, F2)]
    dvBz = np.real(np.fft.ifft2(1j*KV*Fm[2]))[:N, :N]
    duBz = np.real(np.fft.ifft2(1j*KU*Fm[2]))[:N, :N]
    cu = dvBz - dzB[1]
    cv = dzB[0] - duBz
    scale = np.sqrt(np.sum(dvBz**2 + dzB[1]**2 + dzB[0]**2 + duBz**2)/2)   # size of the terms that should cancel
    return cu, cv, scale


rows = []
maps = {}
for z1, z2 in ((2.0, 3.0), (3.0, 4.0), (6.0, 7.0), (9.0, 10.0)):
    out = {}
    out["COMSOL"] = curl_h(frame(old[f"L1_C1_z{z1}"]), frame(old[f"L1_C1_z{z2}"]))
    out["exact physics"] = curl_h(sim(z1, 0.0), sim(z2, 0.0))
    out["screened k0=0.10"] = curl_h(sim(z1, 0.10), sim(z2, 0.10))
    line = f"  {z1:.0f}->{z2:.0f} um:"
    for name, (cu, cv, s) in out.items():
        line += f"  {name}: |curl B| / |terms| = {np.sqrt(np.sum(cu**2 + cv**2))/s:.3f}"
    # pattern match COMSOL vs screened prediction
    cuC, cvC, _ = out["COMSOL"]; cuS, cvS, _ = out["screened k0=0.10"]
    a = (np.vdot(cuS, cuC) + np.vdot(cvS, cvC))/(np.vdot(cuS, cuS) + np.vdot(cvS, cvS))
    corr = (np.vdot(cuS, cuC) + np.vdot(cvS, cvC))/np.sqrt((np.vdot(cuS, cuS) + np.vdot(cvS, cvS))*(np.vdot(cuC, cuC) + np.vdot(cvC, cvC)))
    line += f"   | COMSOL vs screened pattern: correlation {corr:.2f}, amplitude ratio {a:.2f}"
    print(line)
    maps[(z1, z2)] = out

# implied "air current" (A/m^2) and comparison with the coil's own sheet current
cuC, cvC, _ = maps[(3.0, 4.0)]["COMSOL"]
Jair = np.hypot(cuC, cvC)/1e-6/mu0                 # T/um -> T/m -> A/m^2
print(f"\nImplied current density in COMSOL's air at 3.5 um: max {Jair.max():.3g} A/m^2, rms {np.sqrt(np.mean(Jair**2)):.3g} A/m^2")
print("For scale: the coil carries 600 uA in a 2 um x 0.2 um cross-section = 1.5e9 A/m^2.")
# net spurious current crossing the plane x = const under the coil region? total horizontal air current through a
# vertical half-plane is hard to get from two planes; report the current per unit height through the line y=0, u<0..
iy = int(np.argmin(np.abs(v)))
print(f"Air current per unit height crossing the line y = 0 (left half, x < 0): "
      f"{np.sum(cvC[iy, u < 0])*d*1e-6/1e-6/mu0*1e-6:.3g} A per um of height  (coil current 6e-4 A)")

# figure: curl maps at 3.5 um
fig, axs = plt.subplots(2, 3, figsize=(17, 10.5), dpi=110)
ext = [u[0] - d/2, u[-1] + d/2, v[0] - d/2, v[-1] + d/2]
names = ["COMSOL", "exact physics", "screened k0=0.10"]
titles = ["COMSOL (Extra Fine)", "Exact magnetostatics (control)", "Screened model, $k_0$ = 0.10"]
for r, (comp, idx) in enumerate((("(curl B)$_x$  [along CSV x]", 0), ("(curl B)$_y$  [along CSV y]", 1))):
    L = np.max(np.abs(maps[(3.0, 4.0)]["COMSOL"][idx]))/1e-6/mu0
    for j, nm in enumerate(names):
        img = maps[(3.0, 4.0)][nm][idx]/1e-6/mu0   # A/m^2
        ax = axs[r, j]
        im = ax.imshow(img, origin="lower", extent=ext, cmap="RdBu_r", vmin=-L, vmax=L)
        cu, cv, s = maps[(3.0, 4.0)][nm]
        ax.set_title(f"{titles[j]}\n{comp}   |curl B|/|terms| = {np.sqrt(np.sum(cu**2 + cv**2))/s:.2f}", fontsize=11)
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("current density in air, A/m²")
fig.suptitle("Ampère's law in the air at 3.5 µm above the coil: curl B / µ0 (must be zero in empty air)\n"
             "low-passed below ~0.35 rad/µm; same colour scale across each row", fontsize=14)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
