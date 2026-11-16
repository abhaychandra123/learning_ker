"""MAGIC-UNet on the experimental QDM maps (server GPU, inference only).

Input: prepared.npz from prepare_vector.py (vector field at 0.1 um in the model frame, fitted-coil reference).
For each case and each trained model (run2: standoff 3 +- 0.6 um, 0.2 um pixels; run3: 1-10 um, 0.2 um;
run10b: 10 +- 2 um, 0.7 um pixels), exactly the M1 pipeline:
  current -> 0.5 um field (raw) -> height read-out h* -> one Tikhonov step at h* with mu(h*) from the
  run's simulation curve (never chosen on these data).
Metrics: h* vs the fitted standoff; re-blur misfit ||forward(J, h_fit) - input|| / ||input||; agreement with the
fitted-coil reference (field at 0.5 um, current), which is a model fit, not a measurement.
Usage (server, from ~/learning_ker):
  ~/qdm-env/bin/python scripts/experimental/run_magic_experimental.py --prepared <npz> --out <dir>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "experiments/magic_unet_10um/scripts")   # frozen code of the 10 um phase (M1 pipeline with h_max)
import m12_common as mc
from m1_rescore import Corrector, Sheet, current, load_model, read_height

RUNS = {"run2": ("runs/magic_unet/run2", "runs/m1_m2/m1/mu_curve.json"),
        "run3": ("runs/magic_unet/run3", "runs/m1_m2/m1/mu_curve.json"),
        "run10b": ("runs/magic_unet/run10b", "runs/high_standoff/m1/mu_curve.json")}


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b)/np.linalg.norm(b))


def binned(a, f):
    s = a.shape
    return a.reshape(s[:-2] + (s[-2]//f, f, s[-1]//f, f)).mean((-3, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="")
    ap.add_argument("--control", action="store_true",
                    help="feed the clean field of the fitted coil at the fitted standoff instead of the measurement")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    dev = torch.device("cuda")
    P = np.load(a.prepared)
    meta = json.loads(str(P["meta"]))
    if a.cases:
        meta = {k: v for k, v in meta.items() if k in a.cases.split(",")}
    results, arrays = {}, {}
    for run, (folder, mu_file) in RUNS.items():
        model, beta, factor = load_model(Path(folder)/"final.pt", dev)
        stride = json.loads((Path(folder)/"results.json").read_text())["stride"]
        curve = json.loads(Path(mu_file).read_text())[run]
        for case, m in meta.items():
            b = P[f"{case}_input"].astype(np.float64)
            n = b.shape[-1]
            usable = (n//70)*70                             # 161 um: divisible by every bin factor (2, 5, 7, 10)
            b = b[:, :usable, :usable]
            ref05 = P[f"{case}_ref_f05"][:, :usable, :usable]; refj = P[f"{case}_ref_j"][:, :usable, :usable]
            from magic_unet import field_from_sheet
            clean = field_from_sheet(refj, m["standoff_fit_um"], 0.1)
            data_vs_model = rel(b, clean)
            if a.control:
                b = clean
            j = current(model, beta, factor, stride, b, dev)
            sheet = Sheet(j, dev, image=usable)
            raw = sheet.field64(0.5).cpu().numpy()
            h, misfit, edge = read_height(sheet, b, dev, 20.0)
            mu = mc.mu_at(h, curve["standoffs"], curve["mu"])
            dc = Corrector(raw, b, h - 0.5, dev)(mu).cpu().numpy()
            fit_h = m["standoff_fit_um"]
            reblur = sheet.field64(fit_h).cpu().numpy()
            r = dict(input_vs_fitted_coil_field=data_vs_model, stride=stride, pixel_um=0.1*factor, h_star=float(h), h_star_at_edge=bool(edge),
                     standoff_fit_um=fit_h, mu=mu, misfit_at_h_star=float(misfit),
                     reblur_misfit_at_fit_h=rel(reblur, b),
                     field05_vs_ref_raw=rel(raw, ref05), field05_vs_ref_dc=rel(dc, ref05),
                     current_vs_ref_1um=rel(binned(j, 10), binned(refj, 10)),
                     current_corr_1um=float(np.corrcoef(binned(j, 10).ravel(), binned(refj, 10).ravel())[0, 1]),
                     max_abs_current=float(np.abs(j).max()), max_abs_ref_current=float(np.abs(refj).max()))
            results[f"{case}/{run}"] = r
            print(case, run, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
            arrays[f"{case}_{run}_j"] = binned(j, 5).astype(np.float32)          # 0.5 um
            arrays[f"{case}_{run}_raw05"] = binned(raw, 5).astype(np.float32)
            arrays[f"{case}_{run}_dc05"] = binned(dc, 5).astype(np.float32)
        del model; torch.cuda.empty_cache()
    json.dump(results, open(out/"results.json", "w"), indent=1)
    np.savez_compressed(out/"predictions_0p5um.npz", **arrays)
    print("done")


if __name__ == "__main__":
    main()
