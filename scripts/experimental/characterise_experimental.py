"""What the experimental maps look like: noise, outliers, spectra against noise, similarity between cases, and the
vector field rebuilt from B_NV next to the fitted coil.

Usage: python3 characterise_experimental.py <exp.npz> <fit_all dir> <prepared_own.npz> <figure dir> <out json>
"""
import json, os, sys
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import fit_coil as fc

EXP, FIT, PREP, FIGS, OUT = sys.argv[1:6]
E = np.load(EXP); S = json.load(open(f"{FIT}/fit_summary.json")); P = np.load(PREP); meta = json.loads(str(P["meta"]))
X, Y = np.meshgrid(E["X"], E["Y"])
cases = [f"case{n}" for n in range(1, 7)]
stats = {}
for c in cases:
    d = E[c].astype(float)
    dd = np.diff(d[20:-20, 100:-20], axis=1)
    sigma = 1.4826*np.median(np.abs(dd - np.median(dd)))/np.sqrt(2)
    med = ndimage.median_filter(d, 3)
    spikes = int((np.abs(d - med) > np.maximum(8.0, 8*sigma)).sum())
    loc = np.sqrt(ndimage.median_filter((d - med)**2, 15))
    noisy = float((loc > 4*sigma).mean())
    left, right = d[40:120, 70:85], d[40:120, 87:102]
    stats[c] = dict(noise_uT=float(sigma), spike_pixels=spikes, noisy_area_fraction=noisy,
                    seam_step_uT=float(np.median(left) - np.median(right)),
                    p1_p99_uT=[float(np.percentile(d, 1)), float(np.percentile(d, 99))])
good = np.all([np.abs(E[c]) < 50 for c in cases], axis=0)
reg = (X > 25) & (X < 80) & (Y > 20) & (Y < 75) & good
corr = np.array([[np.corrcoef(E[a][reg], E[b][reg])[0, 1] for b in cases] for a in cases])
json.dump(dict(stats=stats, correlation_bottom_left_region=corr.round(3).tolist(), cases=cases), open(OUT, "w"), indent=1)
for c, s in stats.items():
    print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3)
# (a) noise map case 1
a = fig.add_subplot(gs[0, 0])
d = E["case1"].astype(float); med = ndimage.median_filter(d, 3)
loc = np.sqrt(ndimage.median_filter((d - med)**2, 15))
im = a.imshow(loc, origin="lower", extent=[0, 165, 0, 165], cmap="viridis", vmax=3)
fig.colorbar(im, ax=a, fraction=0.046, label="local noise (µT)")
a.set_title(f"Case 1: local noise (median {stats['case1']['noise_uT']:.2f} µT per pixel)\nnoisy corners, camera seam at x ≈ 86 µm")
# (b) spectra vs noise
a = fig.add_subplot(gs[0, 1])
k1 = 2*np.pi*np.fft.fftfreq(165, 1.0); K1 = np.hypot(*np.meshgrid(k1, k1))
edges = np.arange(0.05, np.pi, 0.08)
for c, col in (("case1", "tab:blue"), ("case4", "tab:cyan"), ("case3", "tab:red"), ("case6", "tab:orange")):
    f = np.load(f"{FIT}/fit_{c}.npz")
    Sg = np.abs(np.fft.fft2(f["signal"]))
    prof = [np.sqrt(np.mean(Sg[(K1 >= lo) & (K1 < hi)]**2)) for lo, hi in zip(edges[:-1], edges[1:])]
    a.semilogy(0.5*(edges[:-1] + edges[1:]), prof, color=col, label=f"{c} coil signal (fit, h = {S[c]['standoff_um']:.1f} µm)")
a.axhline(0.37*165, color="k", ls="--", label="white noise 0.37 µT per pixel")
a.set_xlabel("k (rad/µm)"); a.set_ylabel("|FFT| per mode"); a.legend(fontsize=8); a.grid(alpha=0.3)
a.set_title("Signal spectrum against the noise floor\n(SNR = 1 at ~0.85 rad/µm near, ~0.55 rad/µm deep)")
# (c) correlation matrix
a = fig.add_subplot(gs[0, 2])
im = a.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
a.set_xticks(range(6)); a.set_yticks(range(6)); a.set_xticklabels(range(1, 7)); a.set_yticklabels(range(1, 7))
for i in range(6):
    for j in range(6):
        a.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
a.set_title("Correlation between cases, bottom-left coil region\n(1, 4, 5 share the bottom-left coil)")
fig.colorbar(im, ax=a, fraction=0.046)
# (d) data vs fitted coil, case 1
b = P["case1_bnv_clean"]; a = fig.add_subplot(gs[1, 0])
a.imshow(b, origin="lower", extent=[0, 165, 0, 165], cmap="RdBu_r", vmin=-10, vmax=10)
a.set_xlim(15, 95); a.set_ylim(8, 88); a.set_title("Case 1: B_NV after background removal")
a = fig.add_subplot(gs[1, 1])
f = np.load(f"{FIT}/fit_case1.npz")
a.imshow(f["signal"], origin="lower", extent=[0, 165, 0, 165], cmap="RdBu_r", vmin=-10, vmax=10)
p = fc.transform(fc.PATH[1:-2] + [(-15, 30), (-21, 30)], (1, 0), S["case1"]["centre_um"])
a.set_xlim(15, 95); a.set_ylim(8, 88)
a.set_title(f"Fitted design (h {S['case1']['standoff_um']:.2f} µm, I {S['case1']['current_uA']:.0f} µA)\nexplains {100*S['case1']['signal_explained_near_coil']:.0f}% near the coil")
a = fig.add_subplot(gs[1, 2])
a.imshow(np.where(f["full_mask"], f["data"] - f["model"], np.nan), origin="lower", extent=[0, 165, 0, 165], cmap="RdBu_r", vmin=-6, vmax=6)
a.set_xlim(15, 95); a.set_ylim(8, 88)
a.set_title("Data − fitted design: the real leads run along y ≈ 20–25 µm,\nnot where the drawing puts them")
fig.suptitle("The experimental QDM maps: noise, spectra, similarity and fit", fontsize=14)
fig.tight_layout(); fig.savefig(f"{FIGS}/data_characterisation.png", dpi=95); plt.close(fig)

# vector field rebuilt from B_NV vs fitted coil (case 1)
MU0 = 4e-7*np.pi*1e6
def sheet_field(j, h, dx=0.1, pad=400):
    jx, jy = np.pad(j[0], pad), np.pad(j[1], pad); ny, nx = jx.shape
    wx = 2*np.pi*np.fft.rfftfreq(nx, dx)[None, :]; wy = 2*np.pi*np.fft.fftfreq(ny, dx)[:, None]; k = np.hypot(wx, wy)
    s = k.copy(); s[0, 0] = 1
    fx, fy = np.fft.rfft2(jx), np.fft.rfft2(jy); fz = 1j*(wy*fx - wx*fy)/s; e = 0.5*MU0*np.exp(-k*h)
    return np.stack([np.fft.irfft2(q*e, s=(ny, nx))[pad:pad + j.shape[1], pad:pad + j.shape[2]] for q in (fx, fy, fz)])
v = P["case1_input"].astype(float); cl = sheet_field(P["case1_ref_j"].astype(float), meta["case1"]["standoff_fit_um"])
phys = lambda B: [B[1], -B[0], -B[2]]                       # model frame -> physical Bx, By, Bz
fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
for i, nm in enumerate(("Bx", "By", "Bz")):
    L = np.percentile(np.abs(cl[i]), 99.5)
    for r, (img, t) in enumerate(((phys(v)[i], "rebuilt from B_NV"), (phys(cl)[i], "fitted design"))):
        ax[r, i].imshow(img, origin="lower", extent=[0, 165, 0, 165], cmap="RdBu_r", vmin=-L, vmax=L)
        ax[r, i].set_title(f"Case 1, {nm} at the measurement height: {t}"); ax[r, i].set_xlim(10, 100); ax[r, i].set_ylim(5, 95)
fig.suptitle("The three field components rebuilt from one NV projection (top) and the fitted design (bottom)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIGS}/vector_from_bnv_case1.png", dpi=90); plt.close(fig)
print("saved figures")
