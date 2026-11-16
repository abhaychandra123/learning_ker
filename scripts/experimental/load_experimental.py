"""Load the QDM experimental B_NV maps (Experimental data/B_field_map_CaseN.xlsx) into one npz, and summarise them.

Each workbook: one sheet "B_NV (µT)", first row = X centres (µm), first column = Y centres (µm, descending),
165 x 165 values in µT. The *_table.xlsx files hold the same values as (x, y, B_NV in T); checked here.
Usage: python3 load_experimental.py "<Experimental data dir>" <out.npz>   (needs openpyxl)
"""
import sys
import numpy as np
import openpyxl

SRC, OUT = sys.argv[1], sys.argv[2]
arr, X, Y = {}, None, None
for n in range(1, 7):
    ws = openpyxl.load_workbook(f"{SRC}/B_field_map_Case{n}.xlsx", read_only=True, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    x = np.array(rows[0][1:], float); y = np.array([r[0] for r in rows[1:]], float)
    B = np.array([r[1:] for r in rows[1:]], float)
    order = np.argsort(y); y, B = y[order], B[order]          # rows ascending in Y
    if X is None:
        X, Y = x, y
    assert np.allclose(x, X) and np.allclose(y, Y)
    # cross-check against the table file
    wt = openpyxl.load_workbook(f"{SRC}/B_field_map_case{n}_table.xlsx", read_only=True, data_only=True).worksheets[0]
    t = np.array([r for r in wt.iter_rows(min_row=2, values_only=True)], float)
    ix = np.searchsorted(X, t[:, 0]); iy = np.searchsorted(Y, t[:, 1])
    Bt = np.full_like(B, np.nan); Bt[iy, ix] = t[:, 2]*1e6
    diff = np.nanmax(np.abs(Bt - B))
    arr[f"case{n}"] = B
    print(f"case{n}: grid {B.shape}, table rows {len(t)}, max |grid - table| = {diff:.3g} µT")
np.savez(OUT, X=X, Y=Y, **arr)

print(f"\nX {X[0]}..{X[-1]} (step {X[1]-X[0]}), Y {Y[0]}..{Y[-1]}")
cases = [arr[f"case{n}"] for n in range(1, 7)]
for n, B in enumerate(cases, 1):
    med = np.median(B); mad = 1.4826*np.median(np.abs(B - med))
    bad = np.abs(B - med) > 50
    # noise: std of the difference between horizontal neighbours / sqrt 2, in the quiet right half
    q = B[20:-20, 100:-20]
    noise = np.std(np.diff(q, axis=1))/np.sqrt(2)
    core = B[25:75, 30:75]
    print(f"case{n}: median {med:+.2f} µT, robust sd {mad:.2f}, |B-med|>50 µT: {bad.sum()} px, "
          f"neighbour noise (right half) {noise:.2f} µT, coil region 1/99 pct {np.percentile(core, 1):+.1f}/{np.percentile(core, 99):+.1f} µT")

good = np.all([np.abs(B) < 50 for B in cases], axis=0)
print(f"\npixels |B| < 50 µT in every case: {good.mean()*100:.1f}%")
print("Correlation between cases over the coil region (x 25-80, y 20-75 µm):")
reg = (X[None, :] > 25) & (X[None, :] < 80) & (Y[:, None] > 20) & (Y[:, None] < 75) & good
for i in range(6):
    print("  " + " ".join(f"{np.corrcoef(cases[i][reg], cases[j][reg])[0, 1]:+.3f}" for j in range(6)))
print("Least-squares scale of case n onto case 1 over the coil region:")
print("  " + " ".join(f"{np.vdot(cases[0][reg], c[reg])/np.vdot(cases[0][reg], cases[0][reg]):+.3f}" for c in cases))
