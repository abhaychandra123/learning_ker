"""Exact box-mode decay test (no taper) + wall-condition checks."""
import numpy as np, sys
SP = sys.argv[1]; d = 0.1; N = 1400
old = np.load(f"{SP}/L1_C1_all.npz"); eef = np.load(f"{SP}/new_EEF.npz"); ef = np.load(f"{SP}/new_EF.npz")
def frame(B):
    Bx, By, Bz = B.astype(float); return -By, Bx, Bz

print("== Wall checks (L1_C1). Normal component at the wall vs typical interior |B| (rms)")
for z in ("0.5", "3.0", "10.0"):
    Bu, Bv, Bz = frame(old[f"L1_C1_z{z}"])
    rms = np.sqrt(np.mean(Bu**2 + Bv**2 + Bz**2))
    print(f"  z={z:>4}: |Bu| at u-walls {np.sqrt(np.mean(Bu[:, [0, -1]]**2))/rms:.3f}, |Bv| at u-walls {np.sqrt(np.mean(Bv[:, [0, -1]]**2))/rms:.3f}, "
          f"|Bv| at v-walls {np.sqrt(np.mean(Bv[[0, -1], :]**2))/rms:.3f}, |Bu| at v-walls {np.sqrt(np.mean(Bu[[0, -1], :]**2))/rms:.3f}; "
          f"mean Bz / rms Bz = {Bz.mean()/np.sqrt(np.mean(Bz**2)):+.4f}")

def mirror(a, pu, pv):
    a = np.concatenate([a, pu*a[:, ::-1]], axis=1)
    return np.concatenate([a, pv*a[::-1, :]], axis=0)
k1 = 2*np.pi*np.fft.fftfreq(2*N, d)
KV, KU = np.meshgrid(k1, k1, indexing="ij"); K = np.hypot(KU, KV)
def modes(B):
    Bu, Bv, Bz = frame(B)
    return [np.fft.fft2(mirror(Bu, -1, +1)), np.fft.fft2(mirror(Bv, +1, -1)), np.fft.fft2(mirror(Bz, +1, +1))]

bands = [(0.015, 0.035), (0.035, 0.06), (0.06, 0.09), (0.09, 0.13), (0.13, 0.18), (0.18, 0.25), (0.25, 0.35), (0.35, 0.5), (0.5, 0.7)]
H = 45.0
def rates(B1, B2, z1, z2, comps=(0, 1, 2)):
    M1, M2 = modes(B1), modes(B2)
    out = []
    for lo, hi in bands:
        m = (K >= lo) & (K < hi)
        r = []
        for c in comps:
            num = np.vdot(M1[c][m], M2[c][m]).real; den = np.vdot(M1[c][m], M1[c][m]).real
            r.append(-np.log(max(num/den, 1e-12))/(z2 - z1))
        kc = np.sqrt(np.mean(K[m]**2))
        # exact box prediction for this band (tangential cosh, normal sinh)
        pt = -np.log(np.cosh(kc*(H - z2))/np.cosh(kc*(H - z1)))/(z2 - z1)
        pz = -np.log(np.sinh(kc*(H - z2))/np.sinh(kc*(H - z1)))/(z2 - z1)
        out.append((kc, r, pt, pz))
    return out

print("\n== Measured decay rate per band in exact box modes (no taper). Columns: k | Bu Bv Bz measured | box prediction tangential / normal")
for (z1, z2) in ((0.5, 3.0), (3.0, 6.0), (6.0, 10.0), (2.0, 4.0), (7.0, 9.0)):
    print(f" L1_C1 {z1} -> {z2}")
    for kc, r, pt, pz in rates(old[f"L1_C1_z{z1}"], old[f"L1_C1_z{z2}"], z1, z2):
        print(f"   k={kc:5.3f} | {r[0]:6.3f} {r[1]:6.3f} {r[2]:6.3f} | {pt:6.3f} / {pz:6.3f}   excess Bu,Bv vs box: {r[0]-pt:+.3f} {r[1]-pt:+.3f}  Bz: {r[2]-pz:+.3f}")
for name, D in (("EF", ef), ("EEF", eef)):
    print(f" {name} 0.5 -> 3.0")
    for kc, r, pt, pz in rates(D["L1_C1_z0.5"], D["L1_C1_z3.0"], 0.5, 3.0):
        print(f"   k={kc:5.3f} | {r[0]:6.3f} {r[1]:6.3f} {r[2]:6.3f} | {pt:6.3f} / {pz:6.3f}   excess: {r[0]-pt:+.3f} {r[1]-pt:+.3f}  {r[2]-pz:+.3f}")

# which modes carry the field at 10 um? share of field energy below k
print("\n== Share of field norm below k at each height (all components, box modes)")
for z in ("0.5", "3.0", "10.0"):
    M = modes(old[f"L1_C1_z{z}"]); E = sum(np.abs(x)**2 for x in M); tot = E.sum()
    print(f"  z={z:>4}: " + "  ".join(f"k<{kc}: {np.sqrt(E[K < kc].sum()/tot):.2f}" for kc in (0.1, 0.25, 0.5, 1.0)))
