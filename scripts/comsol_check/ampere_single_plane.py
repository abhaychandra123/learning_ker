"""Ampere's law tested on ONE exported plane, for the original export and the tight-tolerance re-solve.

Why one plane suffices: B = curl A, so div B = 0 always, and every Fourier mode B ~ exp(i k.r - g z) has
B_z = i (k . B_t) / g. Curl B = 0 (Ampere's law in source-free air) holds if and only if g = k. So the ratio
    R(k) = B_z / [ i (k . B_t) / k ]      (box modes, all walls; times tanh(k (H - z)) for the top wall, H = 45 um)
must be 1 in every band; COMSOL's screened field (g = sqrt(k^2 + k0^2)) would give k / sqrt(k^2 + k0^2) < 1 at low k.
The 1 um-spaced curl test of ampere_test.py needs neighbouring planes, which the re-solve does not have.
Usage: python3 ampere_single_plane.py <old dir: L1_C1_all.npz> <rt8.npz> [out.png]
"""
import sys
import numpy as np

OLD, RT8 = sys.argv[1:3]
d, N, H = 0.1, 1400, 45.0
old = np.load(f"{OLD}/L1_C1_all.npz"); new = np.load(RT8)


def mirror(a, pu, pv):
    a = np.concatenate([a, pu*a[:, ::-1]], axis=1)
    return np.concatenate([a, pv*a[::-1, :]], axis=0)


k = 2*np.pi*np.fft.fftfreq(2*N, d); KV, KU = np.meshgrid(k, k, indexing="ij"); K = np.hypot(KU, KV)
Ks = np.where(K == 0, 1, K)
bands = [(0.015, 0.035), (0.035, 0.06), (0.06, 0.09), (0.09, 0.13), (0.13, 0.18), (0.18, 0.25), (0.25, 0.35), (0.35, 0.5)]


def ratios(B, z):
    Bx, By, Bz = B.astype(float)
    Fu, Fv, Fz = (np.fft.fft2(mirror(b, *p)) for b, p in ((-By, (-1, 1)), (Bx, (1, -1)), (Bz, (1, 1))))
    P = 1j*(KU*Fu + KV*Fv)/Ks*np.tanh(K*(H - z))
    out = []
    for lo, hi in bands:
        m = (K >= lo) & (K < hi)
        r = np.vdot(P[m], Fz[m]).real/np.vdot(P[m], P[m]).real
        res = np.linalg.norm(Fz[m] - P[m])/np.linalg.norm(Fz[m])
        out.append((np.sqrt(np.mean(K[m]**2)), r, res))
    return out


rows = {}
print("R(k) = Bz / [i k.Bt / k] per band (1 = Ampere satisfied); screened k0 = 0.10 predicts k/sqrt(k^2+0.01)")
for z in ("3.0", "10.0"):
    for name, D in (("original", old), ("re-solve 1e-8", new)):
        r = ratios(D[f"L1_C1_z{z}"], float(z))
        rows[(name, z)] = r
        print(f"  z={z:>4} {name:14s} " + "  ".join(f"k={kc:.3f}:{R:.2f}" for kc, R, _ in r))
    kc = [x[0] for x in r]
    print(f"  z={z:>4} {'screened model':14s} " + "  ".join(f"k={c:.3f}:{c/np.sqrt(c*c + 0.01):.2f}" for c in kc))

if len(sys.argv) > 3:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    kk = np.linspace(0.01, 0.45, 200)
    for a, z in zip(ax, ("3.0", "10.0")):
        a.axhline(1, color="k", lw=1.4, label="Ampère satisfied (free space)")
        a.plot(kk, kk/np.sqrt(kk**2 + 0.01), "k:", lw=1.4, label=r"screened, $k_0$ = 0.10")
        for name, col in (("original", "tab:red"), ("re-solve 1e-8", "tab:blue")):
            r = rows[(name, z)]
            a.plot([x[0] for x in r], [x[1] for x in r], "o-", color=col,
                   label="original export" if name == "original" else "re-solve, tolerance 1e-8")
        a.set_title(f"COMSOL L1_C1 plane at {float(z):g} µm"); a.set_xlabel("spatial frequency k (rad/µm)")
        a.grid(alpha=0.3)
    ax[0].set_ylabel(r"$B_z \,/\, [\,i\,\mathbf{k}\cdot\mathbf{B}_t / k\,]$")
    ax[0].legend(fontsize=9)
    fig.suptitle("Ampère's law on a single plane: the ratio must be 1 in source-free air", fontsize=12)
    fig.tight_layout(); fig.savefig(sys.argv[3], dpi=130)
    print("saved", sys.argv[3])
