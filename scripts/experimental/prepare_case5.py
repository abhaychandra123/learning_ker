"""Case 5 (bottom-left + top-left coils together) for MAGIC-UNet, with the common NV axis.

No single-coil fit exists for this map. Background (offset, plane, camera-seam step) is fitted on good pixels more than
12 um outside both coil boxes; the approximate reference is the sum of the case-1 and case-2 fitted coils (common-axis
fits, same chip positions), with their own fitted currents. Appended to an existing prepared npz as "case5".
Usage: python3 prepare_case5.py <exp.npz> <prepared_shared.npz> <theta,phi deg>
"""
import json, os, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
import fit_coil as fc
import prepare_vector as pv

E = np.load(sys.argv[1]); PATH = sys.argv[2]
th, ph = (np.deg2rad(float(v)) for v in sys.argv[3].split(","))
P = dict(np.load(PATH)); meta = json.loads(str(P["meta"]))
d = E["case5"].astype(float)
X, Y = np.meshgrid(E["X"], E["Y"])
m = fc.good_mask(d)
far = m.copy()
for c in ("case1", "case2"):
    o = (meta[c]["orientation"]["rot90"], meta[c]["orientation"]["mirror"])
    far &= ~fc.window(X, Y, *meta[c]["centre_um"], o, margin=12.0)
cols = fc.background_cols(X, Y)
bgc, *_ = np.linalg.lstsq(np.stack([c_[far] for c_ in cols], 1), d[far], rcond=None)
bg = sum(c_*b for c_, b in zip(bgc, cols))
med = ndimage.generic_filter(np.where(m, d, np.nan), np.nanmedian, 5, mode="nearest")
b = np.where(m, d, np.where(np.isfinite(med), med, bg)) - bg
u = fc.unit(th, ph)
spectra, n = pv.nv_to_vector(b, u)
phys = pv.fine_shift(pv.upsample(spectra, n), 0.45)
P["case5_input"] = np.stack([-phys[1], phys[0], -phys[2]]).astype(np.float32)
P["case5_bnv_clean"] = b.astype(np.float32)
P["case5_ref_j"] = (P["case1_ref_j"] + P["case2_ref_j"]).astype(np.float32)
P["case5_ref_f05"] = (P["case1_ref_f05"] + P["case2_ref_f05"]).astype(np.float32)
h = 0.5*(meta["case1"]["standoff_fit_um"] + meta["case2"]["standoff_fit_um"])
meta["case5"] = dict(background_full_uT=[float(v) for v in bgc], u=u.tolist(), standoff_fit_um=h,
                     current_fit_uA=None, centre_um=[0.5*(meta["case1"]["centre_um"][0] + meta["case2"]["centre_um"][0]),
                                                     0.5*(meta["case1"]["centre_um"][1] + meta["case2"]["centre_um"][1])],
                     orientation=None, note="two coils; reference = case1 + case2 fitted coils; standoff = their mean")
P["meta"] = json.dumps(meta)
np.savez_compressed(PATH, **P)
print("case5 added: max |B|", float(np.abs(P["case5_input"]).max()), "uT, standoff used", round(h, 2))
