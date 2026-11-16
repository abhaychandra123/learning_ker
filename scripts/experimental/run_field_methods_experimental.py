"""Field-domain methods on the experimental QDM maps (server GPU, inference only), one method per process.

Each method is rebuilt from its own frozen code and saved settings. These methods are TOLD the standoff: the operator
uses dz = (fitted standoff - 0.5 um), with the fitted standoff from fit_coil.py. The learned parts (CSC-LISTA,
FISTA-Net) were trained at one dz (2.5 um for the 3 um models, 9.5 um for the 10 um models); the regime nearest to
each case's fitted standoff is used. Nothing is chosen on these data.

  method     regime 3 (fitted standoff < 6.5 um)               regime 10 (>= 6.5 um)
  fista      dictionary, 925 iterations, lambda 0.009          dictionary, 3,000 iterations, lambda 0.00009
  tikhonov   mu 0.0316 (CSC-LISTA strict 64 run)               not available (no mu was ever chosen at 10 um)
  csc        runs/csc_lista_strict/layers_64                   runs/high_standoff/csc_lista_10um
  fistanet   runs/fista_net/run_2                              runs/high_standoff/fista_net_10um (best checkpoint)

Input: the vector field of prepared.npz (model frame, 0.1 um), cropped to 100 um around the fitted coil centre.
Usage (server, ~/learning_ker):
  ~/qdm-env/bin/python scripts/experimental/run_field_methods_experimental.py --method fista \
      --prepared runs/experimental_qdm/prepared_own.npz --out runs/experimental_qdm/field_methods
"""
import argparse, importlib, inspect, json, sys, time
from pathlib import Path
import numpy as np

CROP = 1000          # 100 um
SETTINGS = {
    ("fista", 3): dict(frozen="experiments/fista_depths", steps=925, scale=0.03),
    ("fista", 10): dict(frozen="experiments/fista_depths", steps=3000, scale=0.0003),
    ("tikhonov", 3): dict(frozen="experiments/csc_lista_v4", run="runs/csc_lista_strict/layers_64"),
    ("csc", 3): dict(frozen="experiments/csc_lista_v4", run="runs/csc_lista_strict/layers_64"),
    ("csc", 10): dict(frozen="experiments/high_standoff_methods/csc_lista", run="runs/high_standoff/csc_lista_10um"),
    ("fistanet", 3): dict(frozen="experiments/fista_net_code_start", run="runs/fista_net/run_2"),
    ("fistanet", 10): dict(frozen="experiments/high_standoff_methods/fista_net", run="runs/high_standoff/fista_net_10um"),
}
DICTIONARY = "runs/csc_lista_v2/dictionary_strong/dictionary_0.3.pt"
DATASET = "runs/simulated_3_to_05_v3"


def crop_box(centre_um, n):
    c = [int(round(v/0.1)) for v in centre_um]           # (x, y) -> column, row
    lo = [min(max(v - CROP//2, 0), n - CROP) for v in c]
    return slice(lo[1], lo[1] + CROP), slice(lo[0], lo[0] + CROP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=("fista", "tikhonov", "csc", "fistanet"))
    ap.add_argument("--prepared", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P = np.load(a.prepared); meta = json.loads(str(P["meta"]))
    if a.cases:
        meta = {k: v for k, v in meta.items() if k in a.cases.split(",")}
    import torch
    dev = torch.device("cuda")
    loaded = {}
    results, arrays = {}, {}
    for case, m in meta.items():
        h = m["standoff_fit_um"]; regime = 10 if h >= 6.5 else 3
        key = (a.method, regime)
        if key not in SETTINGS:
            print(case, a.method, "skipped: no saved setting for regime", regime, flush=True); continue
        s = SETTINGS[key]
        if not loaded:                                     # one frozen tree per process
            sys.path.insert(0, str(Path(s["frozen"]).resolve()/"scripts"))
        elif loaded["frozen"] != s["frozen"]:
            print(case, a.method, "skipped: needs another frozen tree; rerun with --cases", flush=True); continue
        legacy = (Path(s["frozen"])/"scripts"/"train_conv_lista.py").exists()
        tr = importlib.import_module("train_conv_lista" if legacy else "train_csc_lista")
        mod = importlib.import_module("conv_lista" if legacy else "csc_lista")
        from simulate_fields import SimulatedDataset
        loaded["frozen"] = s["frozen"]
        scale = SimulatedDataset(DATASET).scale
        dz = h - 0.5
        if a.method == "fista":
            saved = torch.load(DICTIONARY, weights_only=True)
            method = tr.CodedSolver(saved["g"].to(dev), dz, s["scale"]*float(saved["lam"]), s["steps"], 128, True, 8)
            pad, desc = 128, f"FISTA-{s['steps']}, lambda {s['scale']*float(saved['lam']):.3g}"
        else:
            run = Path(s["run"]); args = json.loads((run/"args.json").read_text())
            state = torch.load(run/"best.pt", map_location=dev, weights_only=True)
            pad = args["pad"]
            if a.method == "tikhonov":
                mu = json.loads((run/"results.json").read_text())["wiener"][1]
                method, desc = mod.Dataless(tr.DX, dz, mu), f"Tikhonov mu {mu:.4g}"
            elif a.method == "fistanet":
                fn = importlib.import_module("fista_net")
                model = fn.FISTANet(tr.DX, dz, args["layers"], start=args["start"]).to(dev)
                model.load_state_dict(state["state"]); model.eval()
                method, desc = (lambda b: model(b)[0]), f"FISTA-Net {run} step {state.get('step')}"
            else:
                saved = torch.load(args["dictionary"], weights_only=True)
                cls = getattr(mod, "CSCLISTA", None) or getattr(mod, "ConvLISTA")
                kw = dict(chunk=args["chunk"])
                if "momentum" in inspect.signature(cls.__init__).parameters:
                    kw["momentum"] = args.get("momentum", False)
                model = cls(saved["g"].to(dev), tr.DX, dz, args["crop"] + 2*args["pad"], args["pad"], saved["lam"],
                            args["layers"], args["filter_width"], **kw).to(dev)
                model.load_state_dict(state["state"]); model.eval()
                method, desc = model, f"CSC-LISTA {run} step {state.get('step')}"
        b = P[f"{case}_input"].astype(np.float64)
        box = crop_box(m["centre_um"], b.shape[-1])
        t0 = time.time()
        y = tr.predict(method, torch.tensor(b[:, box[0], box[1]]/scale, dtype=torch.float32), pad, dev)
        y = y.numpy().astype(np.float64)*scale
        ref = P[f"{case}_ref_f05"][:, box[0], box[1]].astype(np.float64)
        e = np.sqrt(((y - ref)**2).sum((-2, -1))/(ref**2).sum((-2, -1)))
        results[case] = dict(method=a.method, regime=regime, setting=desc, dz_used_um=dz, standoff_fit_um=h,
                             crop_rows=[box[0].start, box[0].stop], crop_cols=[box[1].start, box[1].stop],
                             field05_vs_ref_crop=[float(v) for v in e], seconds=time.time() - t0)
        arrays[f"{case}_{a.method}_f05"] = y.astype(np.float32)
        print(case, json.dumps(results[case]), flush=True)
    tag = a.method + ("_" + a.cases.replace(",", "_") if a.cases else "")
    json.dump(results, open(out/f"{tag}.json", "w"), indent=1)
    np.savez_compressed(out/f"{tag}.npz", **arrays)


if __name__ == "__main__":
    main()
