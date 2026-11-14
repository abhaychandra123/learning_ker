"""Test: does COMSOL's closed box (B.n = 0 on all walls) explain the faster decay?

Arrays are [comp, row=v (cpl1y), col=u (cpl1x)], tesla, 0.1 um pitch, walls at the window edges.
"""
import numpy as np, sys
SP = sys.argv[1]
d = 0.1
old = np.load(f"{SP}/L1_C1_all.npz")
ef = np.load(f"{SP}/new_EF.npz"); eef = np.load(f"{SP}/new_EEF.npz")
c4 = np.load(f"{SP}/C1234_sel.npz")
u = old["x"].astype(float); v = old["y"].astype(float)
N = len(u)

def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)

def to_frame(B, hyp):
    """Return (Bu, Bv, Bz) along array columns (u) and rows (v)."""
    Bx, By, Bz = B.astype(float)
    if hyp == "H0":   # naive labels
        return Bx, By, Bz
    if hyp == "H1":   # u = -Y, v = X  ->  Bu = -BY, Bv = BX
        return -By, Bx, Bz
    raise ValueError

# ---------------------------------------------------------------- 1. frame / Maxwell consistency
def kgrid(n0, n1, pitch):
    k0 = 2*np.pi*np.fft.fftfreq(n0, pitch); k1 = 2*np.pi*np.fft.fftfreq(n1, pitch)
    KV, KU = np.meshgrid(k0, k1, indexing="ij")
    return KU, KV, np.hypot(KU, KV)

def consistency(B, hyp, kmin=0.3, kmax=1.5, pad=400):
    Bu, Bv, Bz = to_frame(B, hyp)
    P = lambda a: np.pad(a, pad)
    FU, FV, FZ = (np.fft.fft2(P(a)) for a in (Bu, Bv, Bz))
    KU, KV, K = kgrid(N + 2*pad, N + 2*pad, d)
    Ks = np.where(K == 0, 1, K)
    pred = 1j*(KU*FU + KV*FV)/Ks
    m = (K > kmin) & (K < kmax)
    return np.linalg.norm((FZ - pred)[m]) / np.linalg.norm(FZ[m]), np.linalg.norm((FZ + pred)[m]) / np.linalg.norm(FZ[m])

print("== 1. Component frame: residual of Bz = i(k.Bt)/k in 0.3-1.5 rad/um (sign +, sign -)")
for hyp in ("H0", "H1"):
    r = consistency(old["L1_C1_z3.0"], hyp)
    print(f"  {hyp}: {r[0]:.3f} / {r[1]:.3f}")

# ---------------------------------------------------------------- 2. mesh noise: old vs EF vs EEF
def radial_rel(a, b, edges):
    Fa, Fb = np.fft.fft2(a - b), np.fft.fft2(b)
    _, _, K = kgrid(N, N, d)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (K >= lo) & (K < hi)
        out.append(np.linalg.norm(Fa[m]) / np.linalg.norm(Fb[m]))
    return out

print("\n== 2. Mesh: relative L2 differences (Bx, By, Bz)")
for z in ("0.5", "3.0"):
    o, e, ee = old[f"L1_C1_z{z}"], ef[f"L1_C1_z{z}"], eef[f"L1_C1_z{z}"]
    print(f"  z={z}: old vs new EF  ", [f"{rel(o[c], e[c]):.4f}" for c in range(3)])
    print(f"  z={z}: EF vs EEF      ", [f"{rel(e[c], ee[c]):.4f}" for c in range(3)])
edges = [0, 0.5, 1, 1.5, 2, 2.9, 4, 6, 10, 32]
print("  Spectral |EF-EEF|/|EEF| per band at z=0.5 (bands rad/um):", edges)
for c, n in enumerate("xyz"):
    print(f"   B{n}:", " ".join(f"{x:.3f}" for x in radial_rel(ef["L1_C1_z0.5"][c], eef["L1_C1_z0.5"][c], edges)))
print("  Same at z=3:")
for c, n in enumerate("xyz"):
    print(f"   B{n}:", " ".join(f"{x:.3f}" for x in radial_rel(ef["L1_C1_z3.0"][c], eef["L1_C1_z3.0"][c], edges)))

# ---------------------------------------------------------------- 3. propagation 0.5 -> z
def mirror(a, parity_u, parity_v):
    """Half-sample mirror extension: parity +1 even, -1 odd. Period 2N along each axis."""
    a = np.concatenate([a, parity_u*a[:, ::-1]], axis=1)
    a = np.concatenate([a, parity_v*a[::-1, :]], axis=0)
    return a

def propagate_box(Bu, Bv, Bz, z1, z2, H):
    """Exact upward continuation inside a box: walls at window edges, top wall at z=H, B.n=0."""
    Eu = mirror(Bu, -1, +1); Ev = mirror(Bv, +1, -1); Ez = mirror(Bz, +1, +1)
    _, _, K = kgrid(2*N, 2*N, d)
    if np.isinf(H):
        ft = fz = np.exp(-K*(z2 - z1))
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            ft = np.cosh(K*(H - z2))/np.cosh(K*(H - z1))
            fz = np.where(K == 0, (H - z2)/(H - z1), np.sinh(K*(H - z2))/np.sinh(K*(H - z1)))
        ft = np.nan_to_num(ft, nan=0.0); fz = np.nan_to_num(fz, nan=0.0)
        big = K*(H - z1) > 300   # avoid overflow; ratio -> exp(-k dz)
        ft[big] = np.exp(-K[big]*(z2 - z1)); fz[big] = ft[big]
    out = [np.real(np.fft.ifft2(np.fft.fft2(E)*f))[:N, :N] for E, f in ((Eu, ft), (Ev, ft), (Ez, fz))]
    return out

def propagate_free(Bu, Bv, Bz, z1, z2, pad=700):
    P = lambda a: np.pad(a, pad)
    _, _, K = kgrid(N + 2*pad, N + 2*pad, d)
    f = np.exp(-K*(z2 - z1))
    return [np.real(np.fft.ifft2(np.fft.fft2(P(a))*f))[pad:pad+N, pad:pad+N] for a in (Bu, Bv, Bz)]

def lowpass(a, kc=1.0):
    _, _, K = kgrid(N, N, d)
    return np.real(np.fft.ifft2(np.fft.fft2(a)*(K < kc)))

hyp = "H1"
src = to_frame(old["L1_C1_z0.5"], hyp)
print("\n== 3. COMSOL L1_C1 0.5 um propagated upward, relative L2 vs COMSOL at z (Bu, Bv, Bz) [frame H1]")
print("   z | free space (pad 700)     | box lateral only (H=inf) | full box H=45            | box H=45, low-passed k<1")
rows = []
for z in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
    tgt = to_frame(old[f"L1_C1_z{z}"], hyp)
    fr = propagate_free(*src, 0.5, z)
    bl = propagate_box(*src, 0.5, z, np.inf)
    bx = propagate_box(*src, 0.5, z, 45.0)
    e = lambda p: [rel(p[c], tgt[c]) for c in range(3)]
    el = [rel(lowpass(bx[c]), lowpass(tgt[c])) for c in range(3)]
    efl = [rel(lowpass(fr[c]), lowpass(tgt[c])) for c in range(3)]
    rows.append((z, e(fr), e(bl), e(bx), el, efl))
    f3 = lambda xs: "/".join(f"{100*x:5.1f}" for x in xs)
    print(f"  {z:4.1f} | {f3(e(fr))} | {f3(e(bl))} | {f3(e(bx))} | {f3(el)}   (free low-passed {f3(efl)})")

# fit H from the 0.5 -> 10 pair
print("\n== 3b. Top-wall height that best explains 0.5 -> 10 um (full box), mean rel L2 over components")
tgt = to_frame(old["L1_C1_z10.0"], hyp)
for H in (20, 30, 40, 45, 50, 60, 90, 200):
    bx = propagate_box(*src, 0.5, 10.0, float(H))
    print(f"  H={H:4d}: {100*np.mean([rel(bx[c], tgt[c]) for c in range(3)]):6.2f}%")

# 4-coil configuration
print("\n== 3c. data3 C1+C2+C3+C4: 0.5 -> 3, 10, 12.5 um (Bu/Bv/Bz)")
s4 = to_frame(c4["L1_C1_C2_C3_C4_z0.5"], hyp)
for z in (3.0, 10.0, 12.5):
    tgt = to_frame(c4[f"L1_C1_C2_C3_C4_z{z}"], hyp)
    fr = propagate_free(*s4, 0.5, z); bx = propagate_box(*s4, 0.5, z, 45.0)
    f3 = lambda p: "/".join(f"{100*rel(p[c], tgt[c]):5.1f}" for c in range(3))
    print(f"  z={z}: free {f3(fr)}   box {f3(bx)}")

np.save(f"{SP}/prop_rows.npy", np.array([(r[0], *r[1], *r[2], *r[3], *r[4], *r[5]) for r in rows]))
