"""Does decay rate sqrt(k^2 + k0^2) close the COMSOL propagation gap? Box vs free, several k0."""
import numpy as np, sys
SP = sys.argv[1]; d = 0.1; N = 1400
old = np.load(f"{SP}/L1_C1_all.npz")
def frame(B):
    Bx, By, Bz = B.astype(float); return -By, Bx, Bz
def mirror(a, pu, pv):
    a = np.concatenate([a, pu*a[:, ::-1]], axis=1); return np.concatenate([a, pv*a[::-1, :]], axis=0)
def kg(n):
    k = 2*np.pi*np.fft.fftfreq(n, d); KV, KU = np.meshgrid(k, k, indexing="ij"); return np.hypot(KU, KV)
K2 = kg(2*N); PAD = 700; KP = kg(N + 2*PAD); K1 = kg(N)
def rel(a, b): return np.linalg.norm(a - b)/np.linalg.norm(b)
def lp(a, kc=1.0): return np.real(np.fft.ifft2(np.fft.fft2(a)*(K1 < kc)))

def prop(src, dz, k0, box):
    Bu, Bv, Bz = src
    if box:
        f = np.exp(-np.sqrt(K2**2 + k0**2)*dz)   # top wall at 45 um is negligible (checked: H barely matters)
        E = [mirror(Bu, -1, 1), mirror(Bv, 1, -1), mirror(Bz, 1, 1)]
        return [np.real(np.fft.ifft2(np.fft.fft2(e)*f))[:N, :N] for e in E]
    f = np.exp(-np.sqrt(KP**2 + k0**2)*dz)
    return [np.real(np.fft.ifft2(np.fft.fft2(np.pad(a, PAD))*f))[PAD:PAD+N, PAD:PAD+N] for a in src]

src = frame(old["L1_C1_z0.5"])
print("COMSOL L1_C1 0.5 um -> z, mean relative L2 over Bu/Bv/Bz: full band | low-passed k<1 rad/um")
print("   z  | free k0=0   | box k0=0    | box k0=0.08 | box k0=0.10 | box k0=0.12 | free k0=0.10")
for z in (2.0, 3.0, 5.0, 7.0, 10.0):
    tgt = frame(old[f"L1_C1_z{z}"])
    cells = []
    for k0, box in ((0, False), (0, True), (0.08, True), (0.10, True), (0.12, True), (0.10, False)):
        p = prop(src, z - 0.5, k0, box)
        cells.append(f"{100*np.mean([rel(p[c], tgt[c]) for c in range(3)]):4.1f}|{100*np.mean([rel(lp(p[c]), lp(tgt[c])) for c in range(3)]):4.1f}")
    print(f"  {z:4.1f} | " + " | ".join(f"{c:>11}" for c in cells))

# per-component at 10 um, best model
tgt = frame(old["L1_C1_z10.0"]); p = prop(src, 9.5, 0.10, True); p0 = prop(src, 9.5, 0.0, False)
print("\nAt 10 um, Bu/Bv/Bz: free k0=0", [f"{100*rel(p0[c], tgt[c]):.1f}" for c in range(3)], " box k0=0.10", [f"{100*rel(p[c], tgt[c]):.1f}" for c in range(3)])
# the reverse direction, our benchmark: forward-model the 0.5 target to the 3 um input
tgt3 = frame(old["L1_C1_z3.0"])
for k0, box in ((0, False), (0.10, True)):
    p = prop(src, 2.5, k0, box)
    print(f"Benchmark forward mismatch A(target 0.5) vs input 3 um, {'box' if box else 'free'} k0={k0}: "
          + "/".join(f"{100*rel(p[c], tgt3[c]):.1f}" for c in range(3)) + "  (low-passed k<1: "
          + "/".join(f"{100*rel(lp(p[c]), lp(tgt3[c])):.1f}" for c in range(3)) + ")")
