"""Figure: COMSOL's extra decay disappears with a tight solver tolerance.

Left: decay rate per spatial-frequency band between the 3 and 10 um planes (box modes, all components pooled), for the
original export (default tolerance) and the re-solve (FGMRES, relative tolerance 1e-8), with free space (rate = k)
and the screened law sqrt(k^2 + 0.10^2). Right: COMSOL 0.5 um propagated to 10 um in free space (with the box walls)
against COMSOL's own 10 um plane, Bx/By/Bz, for both solves.
Usage: python3 plot_tolerance.py <old dir: L1_C1_all.npz> <rt8.npz> <out.png>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLD, RT8, OUT = sys.argv[1:4]
d, N = 0.1, 1400
old = np.load(f"{OLD}/L1_C1_all.npz"); new = np.load(RT8)
frame = lambda B: [-B[1].astype(float), B[0].astype(float), B[2].astype(float)]


def mirror(a, pu, pv):
    a = np.concatenate([a, pu*a[:, ::-1]], axis=1)
    return np.concatenate([a, pv*a[::-1, :]], axis=0)


k = 2*np.pi*np.fft.fftfreq(2*N, d); KV, KU = np.meshgrid(k, k, indexing="ij"); K = np.hypot(KU, KV)
PAR = [(-1, 1), (1, -1), (1, 1)]
modes = lambda B: [np.fft.fft2(mirror(b, *p)) for b, p in zip(frame(B), PAR)]
edges = np.array([0.015, 0.035, 0.06, 0.09, 0.13, 0.18, 0.25, 0.35, 0.45])


def rates(D, z1, z2):
    M1, M2 = modes(D[f"L1_C1_z{z1}"]), modes(D[f"L1_C1_z{z2}"])
    kc, r = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (K >= lo) & (K < hi)
        num = sum(np.vdot(x[m], y[m]).real for x, y in zip(M1, M2)); den = sum(np.vdot(x[m], x[m]).real for x in M1)
        kc.append(np.sqrt(np.mean(K[m]**2))); r.append(-np.log(num/den)/(z2 - z1))
    return np.array(kc), np.array(r)


def prop_err(D):
    src, tgt = frame(D["L1_C1_z0.5"]), frame(D["L1_C1_z10.0"])
    f = np.exp(-K*9.5)
    p = [np.real(np.fft.ifft2(np.fft.fft2(mirror(b, *q))*f))[:N, :N] for b, q in zip(src, PAR)]
    return [100*np.linalg.norm(p[c] - tgt[c])/np.linalg.norm(tgt[c]) for c in range(3)]


fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
kk = np.linspace(0, 0.42, 200)
ax[0].plot(kk, kk, "k-", lw=1.5, label="free space: rate = k")
ax[0].plot(kk, np.sqrt(kk**2 + 0.10**2), "k:", lw=1.5, label=r"screened: $\sqrt{k^2 + 0.10^2}$")
for D, name, col in ((old, "original export (default tolerance)", "tab:red"), (new, "re-solve, tolerance 1e-8", "tab:blue")):
    kc, r = rates(D, 3.0, 10.0)
    ax[0].plot(kc, r, "o-", color=col, label=name)
ax[0].set_xlabel("spatial frequency k (rad/µm)"); ax[0].set_ylabel("measured decay rate, 3 → 10 µm (1/µm)")
ax[0].set_title("COMSOL L1_C1: decay per frequency band"); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
e_old, e_new = prop_err(old), prop_err(new)
x = np.arange(3); w = 0.38
ax[1].bar(x - w/2, e_old, w, color="tab:red", label="original export")
ax[1].bar(x + w/2, e_new, w, color="tab:blue", label="re-solve, tolerance 1e-8")
for i in range(3):
    ax[1].text(x[i] - w/2, e_old[i] + 1, f"{e_old[i]:.0f}%", ha="center", fontsize=9)
    ax[1].text(x[i] + w/2, e_new[i] + 1, f"{e_new[i]:.1f}%", ha="center", fontsize=9)
ax[1].set_xticks(x); ax[1].set_xticklabels(["Bx", "By", "Bz"])
ax[1].set_ylabel("relative L2 error (%)")
ax[1].set_title("COMSOL 0.5 µm propagated to 10 µm (free space)\nvs COMSOL's own 10 µm plane")
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(OUT, dpi=130)
print("saved", OUT, "errors old", np.round(e_old, 1), "new", np.round(e_new, 1))
