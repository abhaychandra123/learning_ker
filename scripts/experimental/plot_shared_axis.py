"""Shared-NV-axis pass (theta 54.7, phi 157.5 deg): MAGIC-UNet current maps for all six cases, and a summary table.

Usage: python3 plot_shared_axis.py <prepared_shared.npz> <pass2 predictions npz> <pass2 results.json>
                                   <own-axis scores json> <figure dir> <out json>
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import fit_coil as fc

PREP, PRED, RES, OWN, FIGS, OUT = sys.argv[1:7]
P = np.load(PREP); meta = json.loads(str(P["meta"])); M = np.load(PRED); R = json.load(open(RES)); own = json.load(open(OWN))
cen = 0.25 + 0.5*np.arange(322); X, Y = np.meshgrid(cen, cen)


def b5(a):
    s = a.shape
    return a.reshape(s[:-2] + (s[-2]//5, 5, s[-1]//5, 5)).mean((-3, -1))


def outline(ax, case, colour="lime"):
    for c in (("case1", "case2") if case == "case5" else (case,)):
        m = meta[c]; o = (m["orientation"]["rot90"], m["orientation"]["mirror"])
        p = fc.transform(fc.SPIRAL, o, m["centre_um"])
        ax.plot(p[:, 0], p[:, 1], "-", color=colour, lw=0.6, alpha=0.6)


def boxmask(case, margin=1.0):
    out = np.zeros_like(X, bool)
    for c in (("case1", "case2") if case == "case5" else (case,)):
        m = meta[c]; o = (m["orientation"]["rot90"], m["orientation"]["mirror"])
        half = np.array([22.0, 19.0]) + margin
        if o[0] % 2:
            half = half[::-1]
        out |= (np.abs(X - m["centre_um"][0]) < half[0]) & (np.abs(Y - m["centre_um"][1]) < half[1])
    return out


summary = {}
cases = sorted(meta)
for case in cases:
    refj = b5(P[f"{case}_ref_j"][:, :1610, :1610].astype(float)); msk = boxmask(case)
    row = dict(standoff_fit_um=meta[case]["standoff_fit_um"])
    for run in ("run2", "run3", "run10b"):
        j = M[f"{case}_{run}_j"].astype(float)
        corr = float((np.sum(j[0][msk]*refj[0][msk]) + np.sum(j[1][msk]*refj[1][msk])) /
                     np.sqrt(np.sum(j[:, msk]**2)*np.sum(refj[:, msk]**2)))
        row[run] = dict(h_star=R[f"{case}/{run}"]["h_star"], current_corr_box=corr)
        if case in own and run in own[case]["magic"]:
            row[run]["own_axis"] = dict(h_star=own[case]["magic"][run]["h_star"],
                                        current_corr_box=own[case]["magic"][run]["current_corr_box"])
    summary[case] = row
json.dump(summary, open(OUT, "w"), indent=1)
for c, r in summary.items():
    print(c, f"h_fit {r['standoff_fit_um']:.2f}", {run: (round(r[run]["h_star"], 2), round(r[run]["current_corr_box"], 2),
                                                          (round(r[run]["own_axis"]["h_star"], 2), round(r[run]["own_axis"]["current_corr_box"], 2))
                                                          if "own_axis" in r[run] else None) for run in ("run2", "run3", "run10b")})

fig, ax = plt.subplots(2, 6, figsize=(30, 10.5))
for col, case in enumerate(cases):
    m = meta[case]
    if case == "case5":
        xs = [meta["case1"]["centre_um"][0], meta["case2"]["centre_um"][0]]; ys = [meta["case1"]["centre_um"][1], meta["case2"]["centre_um"][1]]
        lim = (min(xs) - 34, max(xs) + 34, min(ys) - 34, max(ys) + 34)
    else:
        lim = (m["centre_um"][0] - 34, m["centre_um"][0] + 34, m["centre_um"][1] - 34, m["centre_um"][1] + 34)
    a = ax[0, col]
    a.imshow(P[f"{case}_bnv_clean"], origin="lower", extent=[0, 165, 0, 165], cmap="RdBu_r", vmin=-10, vmax=10)
    a.set_title(f"{case}: measured B_NV, fitted standoff {m['standoff_fit_um']:.1f} µm", fontsize=11)
    a.set_xlim(lim[:2]); a.set_ylim(lim[2:])
    a = ax[1, col]
    j = M[f"{case}_run10b_j"].astype(float); mag = np.hypot(*j)
    a.imshow(mag, origin="lower", extent=[0, 161, 0, 161], cmap="magma", vmax=np.percentile(mag, 99.7))
    outline(a, case)
    r = summary[case]["run10b"]
    a.set_title(f"MAGIC-UNet run10b |J|: h* = {r['h_star']:.1f} µm, corr {r['current_corr_box']:+.2f}", fontsize=11)
    a.set_xlim(lim[:2]); a.set_ylim(lim[2:])
fig.suptitle("All six experimental cases, common NV axis (θ 54.7°, φ 157.5°): MAGIC-UNet run10b current "
             "(green: designed spiral at the fitted position; case 5 has both left coils)", fontsize=13)
fig.tight_layout(); fig.savefig(f"{FIGS}/magic_currents_all_cases.png", dpi=80); plt.close(fig)
print("saved", f"{FIGS}/magic_currents_all_cases.png")
