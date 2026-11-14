"""Measured decay rate per spatial frequency between COMSOL planes, vs free-space k."""
import numpy as np, sys
SP = sys.argv[1]; d = 0.1
old = np.load(f"{SP}/L1_C1_all.npz"); ef = np.load(f"{SP}/new_EF.npz"); eef = np.load(f"{SP}/new_EEF.npz")
N = 1400
def frame(B):
    Bx, By, Bz = B.astype(float); return -By, Bx, Bz
pad = 300
k0 = 2*np.pi*np.fft.fftfreq(N + 2*pad, d)
KV, KU = np.meshgrid(k0, k0, indexing="ij"); K = np.hypot(KU, KV)
win = np.outer(np.hanning(N), np.hanning(N))   # taper so edges do not leak
def F(a): return np.fft.fft2(np.pad(a*win, pad))
bands = [(0.02, 0.06), (0.06, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.3)]

def rate(B1, B2, dz):
    """Least-squares complex ratio per band over all three components -> effective decay rate."""
    out = []
    F1 = [F(c) for c in frame(B1)]; F2 = [F(c) for c in frame(B2)]
    for lo, hi in bands:
        m = (K >= lo) & (K < hi)
        num = sum(np.vdot(a[m], b[m]) for a, b in zip(F1, F2))
        den = sum(np.vdot(a[m], a[m]) for a in F1)
        r = (num/den).real
        kc = np.sqrt(np.mean(K[m]**2))
        out.append((kc, -np.log(max(r, 1e-12))/dz))
    return out

zs = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
print("Effective decay rate (1/um) per band, between consecutive COMSOL planes (L1_C1, Extra Fine).")
print("Free space predicts rate = k (first row).")
hdr = rate(old["L1_C1_z3.0"], old["L1_C1_z4.0"], 1.0)
print(" pair      " + " ".join(f"{kc:6.3f}" for kc, _ in hdr) + "   <- band rms k (rad/um)")
for z1, z2 in zip(zs[:-1], zs[1:]):
    r = rate(old[f"L1_C1_z{z1}"], old[f"L1_C1_z{z2}"], z2 - z1)
    print(f" {z1:4.1f}->{z2:4.1f} " + " ".join(f"{x:6.3f}" for _, x in r) + "   excess: " + " ".join(f"{x-kc:+.3f}" for kc, x in r))
print("\nSame for the new meshes 0.5 -> 3:")
for name, D in (("EF ", ef), ("EEF", eef)):
    r = rate(D["L1_C1_z0.5"], D["L1_C1_z3.0"], 2.5)
    print(f" {name}       " + " ".join(f"{x:6.3f}" for _, x in r) + "   excess: " + " ".join(f"{x-kc:+.3f}" for kc, x in r))

# amplitude (projection slope) of COMSOL onto free-space prediction, per height
print("\nTotal-field projection slope of COMSOL(z) onto free-space-propagated COMSOL(0.5), k<1, Hann taper:")
src = [F(c) for c in frame(old["L1_C1_z0.5"])]
for z in zs[1:]:
    tg = [F(c) for c in frame(old[f"L1_C1_z{z}"])]
    f = np.exp(-K*(z - 0.5)); m = K < 1.0
    num = sum(np.vdot(s[m]*f[m], t[m]) for s, t in zip(src, tg)).real
    den = sum(np.vdot(s[m]*f[m], s[m]*f[m]) for s in src).real
    # best effective dz
    best = min(((np.mean([np.linalg.norm((s*np.exp(-K*dz))[m] - t[m])/np.linalg.norm(t[m]) for s, t in zip(src, tg)]), dz)
                for dz in np.arange(0.5*(z-0.5), 2.0*(z-0.5)+1e-9, 0.05)))
    print(f"  z={z:4.1f}: slope {num/den:.3f}   best effective dz {best[1]:.2f} um (nominal {z-0.5:.1f}), error there {100*best[0]:.1f}%")
