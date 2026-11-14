"""Forward-simulate L1_C1 from the DXF geometry and compare with COMSOL.

Current: Laplace conduction solve in the 2 um aluminium sheet, terminals where the leads meet the box wall,
total 600 uA (COMSOL's coil geometry analysis does the same). Field: thin sheet at z=0.
Variants: free space (zero padding); box walls (mirror images); box walls + k0 (modified Helmholtz).
"""
import numpy as np, sys
import scipy.sparse as sp, scipy.sparse.linalg as spl
SP = sys.argv[1]; d = 0.1; N = 1400; mu0 = 4e-7*np.pi
old = np.load(f"{SP}/L1_C1_all.npz"); ef = np.load(f"{SP}/new_EF.npz"); eef = np.load(f"{SP}/new_EEF.npz")
u = -41.95 + d*np.arange(N); v = -38.95 + d*np.arange(N)
V, U = np.meshgrid(v, u, indexing="ij")          # rows v, cols u
X, Y = V, -U                                       # geometry frame
rects = []
L = [l.strip() for l in open(sys.argv[2], encoding="latin-1").read().splitlines()]
P = list(zip(L[0::2], L[1::2]))
cur = None
for c, val in P:
    if c == "0":
        if cur and len(cur) == 8: xs, ys = cur[0::2], cur[1::2]; rects.append((min(xs), max(xs), min(ys), max(ys)))
        cur = [] if val == "LWPOLYLINE" else None
    elif cur is not None and c in ("10", "20"):
        cur.append(float(val))
if cur and len(cur) == 8: xs, ys = cur[0::2], cur[1::2]; rects.append((min(xs), max(xs), min(ys), max(ys)))
mask = np.zeros((N, N), bool)
for x0, x1, y0, y1 in rects:
    mask |= (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
print(f"{len(rects)} rectangles, conductor area {mask.sum()*d*d:.1f} um^2, widths "
      f"{sorted(set(round(min(x1-x0, y1-y0), 3) for x0, x1, y0, y1 in rects))} um")
# terminals: first column (u = -41.95, i.e. at the wall) of each lead
termA = mask & (U < -41.9) & (X > -16) & (X < -14)
termB = mask & (U < -41.9) & (X > -22) & (X < -20)
print("terminal pixels", termA.sum(), termB.sum())
idx = -np.ones((N, N), int); free = mask & ~termA & ~termB
idx[free] = np.arange(free.sum())
phi = np.zeros((N, N)); phi[termA] = 1.0
rows, cols, vals = [], [], []; b = np.zeros(free.sum())
fi, fj = np.nonzero(free)
for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    ni, nj = fi + di, fj + dj
    ok = (ni >= 0) & (ni < N) & (nj >= 0) & (nj < N)
    ok[ok] &= mask[ni[ok], nj[ok]]
    me = idx[fi[ok], fj[ok]]
    rows += list(me); cols += list(me); vals += [1.0]*len(me)
    nb = idx[ni[ok], nj[ok]]; isfree = nb >= 0
    rows += list(me[isfree]); cols += list(nb[isfree]); vals += [-1.0]*int(isfree.sum())
    np.add.at(b, me[~isfree], phi[ni[ok][~isfree], nj[ok][~isfree]])
A = sp.csr_matrix((vals, (rows, cols)), shape=(free.sum(), free.sum()))
phi[free] = spl.spsolve(A.tocsc(), b)
# face flows (conductance 1): flow in +u across face (j, j+1), +v across (i, i+1)
fu = np.zeros((N, N + 1)); fv = np.zeros((N + 1, N))
mu_ = mask[:, :-1] & mask[:, 1:]; fu[:, 1:-1][mu_] = (phi[:, :-1] - phi[:, 1:])[mu_]
mv_ = mask[:-1, :] & mask[1:, :]; fv[1:-1, :][mv_] = (phi[:-1, :] - phi[1:, :])[mv_]
# lead continues through the wall: left face of terminal pixels carries the same flow as their right face
fu[:, 0] = np.where(termA[:, 0] | termB[:, 0], fu[:, 1], 0)
Itot = fu[:, 0][termA[:, 0]].sum()
s = 600e-6/abs(Itot)
Ku = s*0.5*(fu[:, :-1] + fu[:, 1:])/(d*1e-6)      # A/m
Kv = s*0.5*(fv[:-1, :] + fv[1:, :])/(d*1e-6)
w = 2e-6
print(f"sheet current in a straight wire: expected I/w = {600e-6/w:.0f} A/m; max |K| = {np.max(np.hypot(Ku, Kv)):.0f} A/m (corners crowd)")
div = (fu[:, 1:] - fu[:, :-1]) + (fv[1:, :] - fv[:-1, :])
print(f"current conservation: max |net flow| per pixel / total = {np.max(np.abs(div[free]))/abs(Itot):.2e}")

def mirror(a, pu, pv):
    a = np.concatenate([a, pu*a[:, ::-1]], axis=1); return np.concatenate([a, pv*a[::-1, :]], axis=0)
def field(z, mode, k0=0.0, pad=1400):
    if mode == "free":
        Fu, Fv = np.fft.fft2(np.pad(Ku, pad)), np.fft.fft2(np.pad(Kv, pad)); n = N + 2*pad
    else:
        Fu, Fv = np.fft.fft2(mirror(Ku, 1, -1)), np.fft.fft2(mirror(Kv, -1, 1)); n = 2*N
    k = 2*np.pi*np.fft.fftfreq(n, d*1e-6); KV, KU = np.meshgrid(k, k, indexing="ij")
    kap = np.sqrt(KU**2 + KV**2 + (k0*1e6)**2); kap[0, 0] = kap[0, 0] if k0 > 0 else 1.0
    E = 0.5*mu0*np.exp(-kap*z*1e-6)
    Bu = E*Fv; Bv = -E*Fu; Bz = 1j*(KU*Fv - KV*Fu)*E/kap
    sl = (slice(pad, pad + N), slice(pad, pad + N)) if mode == "free" else (slice(0, N), slice(0, N))
    return [np.real(np.fft.ifft2(a))[sl] for a in (Bu, Bv, Bz)]
def frame(B):
    Bx, By, Bz = B.astype(float); return [-By, Bx, Bz]
def rel(a, b): return np.linalg.norm(a - b)/np.linalg.norm(b)

def compare(sim, tgt):
    num = sum(np.vdot(s_, t_) for s_, t_ in zip(sim, tgt)); den = sum(np.vdot(s_, s_) for s_ in sim)
    a = num/den
    return a, [rel(sim[c], tgt[c]) for c in range(3)], [rel(a*sim[c], tgt[c]) for c in range(3)]

print("\nSimulated (exact DXF geometry, 600 uA, sheet at z=0) vs COMSOL. Relative L2 Bu/Bv/Bz, unscaled; [best scale a]")
targets = [("0.5", "EF ", old["L1_C1_z0.5"]), ("0.5", "EEF", eef["L1_C1_z0.5"]), ("3.0", "EF ", old["L1_C1_z3.0"]),
           ("3.0", "EEF", eef["L1_C1_z3.0"]), ("5.0", "EF ", old["L1_C1_z5.0"]), ("10.0", "EF ", old["L1_C1_z10.0"])]
sign = None
for z, name, B in targets:
    tgt = frame(B); line = f"  z={z:>4} {name}:"
    for mode, k0 in (("free", 0.0), ("box", 0.0), ("box", 0.10)):
        sim = field(float(z), mode, k0)
        if sign is None:
            sign = np.sign(compare(sim, tgt)[0]); print(f"  (terminal polarity sign = {sign:+.0f})")
        sim = [sign*x for x in sim]
        a, e, ea = compare(sim, tgt)
        line += f"  {mode}{'+k0' if k0 else '   '} " + "/".join(f"{100*x:5.1f}" for x in e) + f" [a={a:.3f}]"
    print(line)

print("\nNoise reference: EF vs EEF at 0.5 um 19/22/26%, at 3 um 12/13/16% (two meshes of the same model).")
print("Low-passed (k<1 rad/um) comparison, box+k0 vs COMSOL:")
k = 2*np.pi*np.fft.fftfreq(N, d); KV, KU = np.meshgrid(k, k, indexing="ij"); Kl = np.hypot(KU, KV) < 1.0
lp = lambda a: np.real(np.fft.ifft2(np.fft.fft2(a)*Kl))
for z, name, B in targets:
    tgt = frame(B); sim = [sign*x for x in field(float(z), "box", 0.10)]; sim0 = [sign*x for x in field(float(z), "free", 0.0)]
    print(f"  z={z:>4} {name}: free " + "/".join(f"{100*rel(lp(s_), lp(t_)):5.1f}" for s_, t_ in zip(sim0, tgt))
          + "   box+k0 " + "/".join(f"{100*rel(lp(s_), lp(t_)):5.1f}" for s_, t_ in zip(sim, tgt)))
# height offset check at 3 um
tgt = frame(old["L1_C1_z3.0"])
print("\nBest sheet height at the 3 um plane (box+k0):", ", ".join(
    f"z={zz:.2f}: {100*np.mean(compare([sign*x for x in field(zz, 'box', 0.10)], tgt)[1]):.2f}%" for zz in (2.8, 2.9, 3.0, 3.1, 3.2)))
np.savez_compressed(f"{SP}/dxf_current.npz", Ku=Ku.astype(np.float32), Kv=Kv.astype(np.float32), mask=mask)
