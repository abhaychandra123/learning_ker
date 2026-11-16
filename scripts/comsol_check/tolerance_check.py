"""Does a tighter iterative tolerance (FGMRES, relative tolerance 1e-8) remove COMSOL's extra decay k0?

Usage: python3 tolerance_check.py <old dir: L1_C1_all.npz, dxf_current.npz> <rt8.npz>
rt8.npz: csv2npz.py output of COMSOL_params_v2/Extra_Fine_Mesh/FGMRES_RT1e-8/*.csv (0.5, 3, 10 um).
"""
import sys
import numpy as np

OLD, RT8 = sys.argv[1], sys.argv[2]
d, N, mu0 = 0.1, 1400, 4e-7*np.pi
old = np.load(f"{OLD}/L1_C1_all.npz"); new = np.load(RT8)
cur = np.load(f"{OLD}/dxf_current.npz"); Ku, Kv = cur["Ku"].astype(float), cur["Kv"].astype(float)


def frame(B):
    Bx, By, Bz = B.astype(float)
    return [-By, Bx, Bz]


def mirror(a, pu, pv):
    a = np.concatenate([a, pu*a[:, ::-1]], axis=1)
    return np.concatenate([a, pv*a[::-1, :]], axis=0)


k = 2*np.pi*np.fft.fftfreq(2*N, d); KV, KU = np.meshgrid(k, k, indexing="ij"); K = np.hypot(KU, KV)
PAR = [(-1, 1), (1, -1), (1, 1)]
rel = lambda a, b: np.linalg.norm(a - b)/np.linalg.norm(b)


def modes(B):
    return [np.fft.fft2(mirror(b, *p)) for b, p in zip(frame(B), PAR)]


print("== 1. Re-solve (RT 1e-8) vs original solve, relative L2 difference Bx/By/Bz (CSV labels)")
for z in ("0.5", "3.0", "10.0"):
    a, b = new[f"L1_C1_z{z}"].astype(float), old[f"L1_C1_z{z}"].astype(float)
    print(f"  z={z:>4}: " + "/".join(f"{100*rel(a[c], b[c]):.3f}%" for c in range(3)))

bands = [(0.015, 0.035), (0.035, 0.06), (0.06, 0.09), (0.09, 0.13), (0.13, 0.18), (0.18, 0.25), (0.25, 0.35)]
print("\n== 2. Decay rate per band in box modes (all components pooled); free space predicts rate = k")
for tag, D in (("original", old), ("RT 1e-8", new)):
    for z1, z2 in ((0.5, 3.0), (3.0, 10.0)):
        M1, M2 = modes(D[f"L1_C1_z{z1}"]), modes(D[f"L1_C1_z{z2}"])
        cells = []
        for lo, hi in bands:
            m = (K >= lo) & (K < hi)
            num = sum(np.vdot(x[m], y[m]).real for x, y in zip(M1, M2)); den = sum(np.vdot(x[m], x[m]).real for x in M1)
            rate = -np.log(num/den)/(z2 - z1); kc = np.sqrt(np.mean(K[m]**2))
            cells.append(f"k={kc:.3f}:{rate:.3f}(k0={np.sqrt(max(rate**2 - kc**2, 0)):.3f})")
        print(f"  {tag:9s} {z1}->{z2}: " + "  ".join(cells))


def prop(src, dz, k0):
    f = np.exp(-np.sqrt(K**2 + k0**2)*dz)
    return [np.real(np.fft.ifft2(np.fft.fft2(mirror(b, *p))*f))[:N, :N] for b, p in zip(src, PAR)]


print("\n== 3. COMSOL 0.5 um propagated to 10 um (box walls), relative L2 vs COMSOL 10 um, Bu/Bv/Bz")
for tag, D in (("original", old), ("RT 1e-8", new)):
    src, tgt = frame(D["L1_C1_z0.5"]), frame(D["L1_C1_z10.0"])
    for k0 in (0.0, 0.10):
        p = prop(src, 9.5, k0)
        print(f"  {tag:9s} k0={k0:.2f}: " + "/".join(f"{100*rel(p[c], tgt[c]):.1f}%" for c in range(3)))


def sim(z, k0):
    Fu, Fv = np.fft.fft2(mirror(Ku, 1, -1)), np.fft.fft2(mirror(Kv, -1, 1))
    kap = np.sqrt(K**2 + k0**2)*1e6; kk = np.where(kap == 0, 1.0, kap)
    E = 0.5*mu0*np.exp(-kap*z*1e-6)
    return [np.real(np.fft.ifft2(a))[:N, :N] for a in (E*Fv, -E*Fu, 1j*(KU*Fv - KV*Fu)*1e6*E/kk)]


print("\n== 4. Exact DXF coil (walls) vs COMSOL, relative L2 Bu/Bv/Bz")
for z in ("3.0", "10.0"):
    for tag, D in (("original", old), ("RT 1e-8", new)):
        tgt = frame(D[f"L1_C1_z{z}"])
        line = f"  z={z:>4} {tag:9s}"
        for k0 in (0.0, 0.10):
            s = sim(float(z), k0)
            line += f"  k0={k0:.2f}: " + "/".join(f"{100*rel(s[c], tgt[c]):.1f}%" for c in range(3))
        print(line)
