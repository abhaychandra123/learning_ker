"""Score and plot every method on the experimental QDM maps.

Reference: the designed coil (DXF) fitted to each map (fit_coil.py): orientation, centre, standoff, NV axis, current.
It is a model fit, not a measurement of the near field. The real leads are routed differently from the drawing, so
all scores are taken inside the spiral's bounding box (+1 um), where the drawing matches the device.
Common grid: 0.5 um pixels (MAGIC outputs were saved binned to 0.5 um; field methods are binned here).
MAGIC-UNet in method comparisons follows the rule fixed before this test: run2 + data consistency for cases fitted
below 6.5 um, run10b + data consistency above.
Usage: python3 evaluate_experimental.py <prepared.npz> <magic predictions npz> <magic results.json>
                                        <field-methods dir> <exp.npz> <figure dir> <out json>
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import fit_coil as fc

PREP, MAGIC, MAGIC_JSON, FIELD_DIR, EXP, FIGS, OUT = sys.argv[1:8]
P = np.load(PREP); meta = json.loads(str(P["meta"]))
M = np.load(MAGIC); MJ = json.load(open(MAGIC_JSON))
E = np.load(EXP)
field = {}
for f in glob.glob(f"{FIELD_DIR}/*.npz"):
    js = json.load(open(f[:-4] + ".json"))
    z = np.load(f)
    for case, r in js.items():
        field[(case, r["method"])] = (z[f"{case}_{r['method']}_f05"], r)
NB = 322                                                     # 161 um at 0.5 um
cen = 0.25 + 0.5*np.arange(NB)
X, Y = np.meshgrid(cen, cen)
LABEL = dict(magic="MAGIC-UNet", fista="FISTA", fistanet="FISTA-Net", csc="CSC-LISTA", tikhonov="Tikhonov")


def b5(a):
    s = a.shape
    return a.reshape(s[:-2] + (s[-2]//5, 5, s[-1]//5, 5)).mean((-3, -1))


def box(case, margin=1.0):
    m = meta[case]; x0, y0 = m["centre_um"]; o = (m["orientation"]["rot90"], m["orientation"]["mirror"])
    half = np.array([22.0, 19.0]) + margin
    if o[0] % 2:
        half = half[::-1]
    return (np.abs(X - x0) < half[0]) & (np.abs(Y - y0) < half[1])


def field_on_grid(arr, r):
    """0.1 um crop at (rows, cols) -> 0.5 um common grid (NaN outside the crop)."""
    r0, c0 = r["crop_rows"][0], r["crop_cols"][0]
    dr, dc = (-r0) % 5, (-c0) % 5
    a = arr[:, dr:, dc:]
    a = a[:, :a.shape[1]//5*5, :a.shape[2]//5*5]
    g = np.full((3, NB, NB), np.nan)
    i0, j0 = (r0 + dr)//5, (c0 + dc)//5
    sub = b5(a)
    ni, nj = min(sub.shape[1], NB - i0), min(sub.shape[2], NB - j0)
    g[:, i0:i0 + ni, j0:j0 + nj] = sub[:, :ni, :nj]
    return g


def scores(pred, ref, m):
    e, c = [], []
    for i in range(3):
        p, t = pred[i][m], ref[i][m]
        e.append(float(np.linalg.norm(p - t)/np.linalg.norm(t)))
        c.append(float(np.corrcoef(p, t)[0, 1]))
    return dict(rel_l2=float(np.mean(e)), rel_l2_components=e, corr=float(np.mean(c)))


results, preds = {}, {}
for case, m in meta.items():
    if f"{case}_ref_f05" not in P.files:
        continue
    ref = b5(P[f"{case}_ref_f05"][:, :1610, :1610].astype(float))
    refj = b5(P[f"{case}_ref_j"][:, :1610, :1610].astype(float))
    msk = box(case)
    regime = 10 if m["standoff_fit_um"] >= 6.5 else 3
    best = "run10b" if regime == 10 else "run2"
    res = dict(standoff_fit_um=m["standoff_fit_um"], regime_um=regime, magic_rule_run=best, magic={}, methods={})
    for run in ("run2", "run3", "run10b"):
        k = f"{case}_{run}"
        if f"{k}_dc05" not in M.files:
            continue
        j = M[f"{k}_j"].astype(float)
        jc = float((np.sum(j[0][msk]*refj[0][msk]) + np.sum(j[1][msk]*refj[1][msk])) /
                   np.sqrt(np.sum(j[:, msk]**2)*np.sum(refj[:, msk]**2)))
        out_share = float(np.sqrt(np.sum(j[:, ~box(case, 12.0)]**2)/np.sum(j**2)))
        r = MJ[f"{case}/{run}"]
        res["magic"][run] = dict(h_star=r["h_star"], raw=scores(M[f"{k}_raw05"], ref, msk),
                                 dc=scores(M[f"{k}_dc05"], ref, msk), current_corr_box=jc,
                                 current_share_far_from_coil=out_share)
        preds[(case, f"magic_{run}")] = M[f"{k}_dc05"].astype(float)
    preds[(case, "magic")] = preds.get((case, f"magic_{best}"))
    res["methods"]["magic"] = res["magic"].get(best, {}).get("dc")
    for meth in ("fista", "fistanet", "csc", "tikhonov"):
        if (case, meth) in field:
            g = field_on_grid(*field[(case, meth)])
            ok = msk & np.isfinite(g[0])
            res["methods"][meth] = dict(scores(g, ref, ok), setting=field[(case, meth)][1]["setting"],
                                        dz_used_um=field[(case, meth)][1]["dz_used_um"])
            preds[(case, meth)] = g
    preds[(case, "ref")] = ref
    results[case] = res
json.dump(results, open(OUT, "w"), indent=1)

# ------------------------------------------------------------------ console table
print(f"{'case':6s} {'h_fit':>6s}  " + "  ".join(f"{LABEL[k]:>12s}" for k in ("magic", "fista", "fistanet", "csc", "tikhonov")))
for case, r in results.items():
    cells = []
    for k in ("magic", "fista", "fistanet", "csc", "tikhonov"):
        s = r["methods"].get(k)
        cells.append(f"{100*s['rel_l2']:5.0f}% c{s['corr']:+.2f}" if s else f"{'n/a':>12s}")
    print(f"{case:6s} {r['standoff_fit_um']:6.2f}  " + "  ".join(f"{c:>12s}" for c in cells))
print("\nMAGIC runs (corrected): h* / field rel L2 in box / current corr in box / current share far from coil")
for case, r in results.items():
    print(case, {run: (round(v["h_star"], 2), round(100*v["dc"]["rel_l2"]), round(v["current_corr_box"], 2),
                       round(v["current_share_far_from_coil"], 2)) for run, v in r["magic"].items()})

# ------------------------------------------------------------------ figures
os.makedirs(FIGS, exist_ok=True)


def zoom(ax, case, pad=6.0):
    x0, y0 = meta[case]["centre_um"]
    ax.set_xlim(x0 - 28 - pad, x0 + 28 + pad); ax.set_ylim(y0 - 28 - pad, y0 + 28 + pad)


def outline(ax, case):
    m = meta[case]; o = (m["orientation"]["rot90"], m["orientation"]["mirror"])
    p = fc.transform(fc.SPIRAL, o, m["centre_um"])
    ax.plot(p[:, 0], p[:, 1], "-", color="lime", lw=0.6, alpha=0.6)


# 1. the six measured maps
fig, ax = plt.subplots(2, 3, figsize=(15, 9.6))
for n, a in zip(range(1, 7), ax.ravel()):
    d = E[f"case{n}"]
    im = a.imshow(d, origin="lower", extent=[0, 165, 0, 165], cmap="RdBu_r", vmin=-8, vmax=8)
    a.set_title(f"Case {n}" + (f": fitted standoff {meta[f'case{n}']['standoff_fit_um']:.1f} µm" if f"case{n}" in meta else ""))
    a.set_xlabel("X (µm)"); a.set_ylabel("Y (µm)")
fig.colorbar(im, ax=ax, fraction=0.025, label="B_NV (µT)")
fig.suptitle("Experimental QDM maps (B along the NV axis), 1 µm pixels", fontsize=14)
fig.savefig(f"{FIGS}/experimental_maps.png", dpi=110, bbox_inches="tight"); plt.close(fig)

# 2. method comparison per case: physical Bz at 0.5 um (model frame Bz = -physical Bz)
for case in results:
    cols = [("ref", "Fitted design (reference)")] + [(k, LABEL[k]) for k in ("magic", "fista", "fistanet", "csc", "tikhonov")
                                                     if preds.get((case, k)) is not None]
    fig, ax = plt.subplots(1, len(cols) + 1, figsize=(3.6*(len(cols) + 1), 4.3))
    ax[0].imshow(P[f"{case}_bnv_clean"], origin="lower", extent=[0, 165, 0, 165], cmap="RdBu_r", vmin=-10, vmax=10)
    ax[0].set_title(f"{case}: measured B_NV\n(fitted standoff {meta[case]['standoff_fit_um']:.1f} µm)", fontsize=10)
    zoom(ax[0], case)
    L = np.nanpercentile(np.abs(preds[(case, 'ref')][2]), 99.5)
    for a, (k, name) in zip(ax[1:], cols):
        img = -preds[(case, k)][2]
        a.imshow(img, origin="lower", extent=[0, 161, 0, 161], cmap="RdBu_r", vmin=-L, vmax=L)
        s = results[case]["methods"].get(k) if k != "ref" else None
        tag = "" if s is None else f"\nerror {100*s['rel_l2']:.0f}%, corr {s['corr']:+.2f}"
        if k == "magic":
            name += f" ({results[case]['magic_rule_run']} + DC)"
        a.set_title(name + tag, fontsize=10); zoom(a, case); outline(a, case)
    fig.suptitle(f"{case}: Bz at 0.5 µm. Scores inside the spiral box against the fitted design", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIGS}/methods_{case}.png", dpi=110); plt.close(fig)

# 3. MAGIC current maps, every run
cases = list(results)
fig, ax = plt.subplots(len(cases), 4, figsize=(15.5, 3.9*len(cases)))
ax = np.atleast_2d(ax)
for r, case in enumerate(cases):
    refj = b5(P[f"{case}_ref_j"][:, :1610, :1610].astype(float))
    ax[r, 0].imshow(np.hypot(*refj), origin="lower", extent=[0, 161, 0, 161], cmap="magma")
    ax[r, 0].set_title(f"{case}: fitted design |J| (h = {meta[case]['standoff_fit_um']:.1f} µm)", fontsize=10)
    zoom(ax[r, 0], case)
    for c, run in enumerate(("run2", "run3", "run10b")):
        key = f"{case}_{run}_j"
        if key not in M.files:
            continue
        mag = np.hypot(*M[key].astype(float))
        a = ax[r, c + 1]
        a.imshow(mag, origin="lower", extent=[0, 161, 0, 161], cmap="magma", vmax=np.percentile(mag, 99.7))
        v = results[case]["magic"][run]
        a.set_title(f"MAGIC-UNet {run}: h* = {v['h_star']:.1f} µm\ncurrent corr {v['current_corr_box']:+.2f}", fontsize=10)
        zoom(a, case); outline(a, case)
fig.suptitle("Predicted sheet current |J| on the real QDM maps (green: designed spiral, fitted position)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{FIGS}/magic_currents.png", dpi=95); plt.close(fig)

# 4. heights
fig, ax = plt.subplots(figsize=(7.5, 4.3))
x = np.arange(len(cases)); w = 0.2
ax.bar(x - 1.5*w, [meta[c]["standoff_fit_um"] for c in cases], w, color="k", label="physics fit (design + NV axis)")
for i, (run, col) in enumerate((("run2", "tab:orange"), ("run3", "tab:green"), ("run10b", "tab:blue"))):
    ax.bar(x + (i - 0.5)*w, [results[c]["magic"].get(run, {}).get("h_star", np.nan) for c in cases], w, color=col,
           label=f"MAGIC-UNet {run} read-out h*")
ax.set_xticks(x); ax.set_xticklabels(cases); ax.set_ylabel("standoff (µm)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
ax.set_title("Standoff: physics fit vs the network's own read-out")
fig.tight_layout(); fig.savefig(f"{FIGS}/heights.png", dpi=120); plt.close(fig)
print("figures in", FIGS)
