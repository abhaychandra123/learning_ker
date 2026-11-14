"""M1 (M1_M2_PROTOCOL.md): re-score MAGIC-UNet run 2 and run 3 with a height read-out and one
data-consistency correction. Inference only; runs on the server GPU.

Order of work, so that COMSOL cannot influence any choice:
  1. reproduction gate: re-run inference on every COMSOL pair and compare the raw score with the saved comsol.csv;
  2. simulation pass 1 (per standoff, 50 validation layouts): raw score, height read-out, correction at the true
     height for every mu on the grid -> mu(h) per run;
  3. simulation pass 2: correction at the read-out height with mu(h*);
  4. COMSOL: raw, correction at the true height, correction at h*, h*, and the four-way scores (k_n from M2a).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import m12_common as mc
from magic_data import binned
from magic_unet import MAGICUNet, predict_tiled
from simulate_fields import COMPONENTS, SimulatedDataset, synthesise
from wiener_oracle import discover, load

MU0_UT = 4e-7 * np.pi * 1e6
SHEET_PAD = 700


# ---------------------------------------------------------------- torch physics (GPU)

class Sheet:
    """Thin-sheet Biot-Savart fields of one predicted current, same formulas as magic_unet.field_from_sheet."""

    def __init__(self, j, device, image=1400):
        self.image, self.device = image, device
        jx = np.pad(np.asarray(j[0], np.float64), SHEET_PAD)
        jy = np.pad(np.asarray(j[1], np.float64), SHEET_PAD)
        self.shape = jx.shape
        ny, nx = self.shape
        wx = 2 * np.pi * torch.fft.rfftfreq(nx, mc.DX, dtype=torch.float64, device=device)[None, :]
        wy = 2 * np.pi * torch.fft.fftfreq(ny, mc.DX, dtype=torch.float64, device=device)[:, None]
        self.k = torch.hypot(wx, wy)
        safe = self.k.clone()
        safe[0, 0] = 1.0
        fx = torch.fft.rfft2(torch.as_tensor(jx, device=device))
        fy = torch.fft.rfft2(torch.as_tensor(jy, device=device))
        fz = 1j * (wy * fx - wx * fy) / safe
        self.spec64 = torch.stack([fx, fy, fz])
        self.spec32 = self.spec64.to(torch.complex64)
        self.k32 = self.k.float()

    def crop(self, a):
        p = SHEET_PAD
        return a[..., p:p + self.image, p:p + self.image]

    def field64(self, height):
        decay = 0.5 * MU0_UT * torch.exp(-self.k * height)
        return self.crop(torch.fft.irfft2(self.spec64 * decay, s=self.shape))

    def misfit(self, heights, b32, b_norm, chunk=6):
        """mean over components of ||forward(J, h) - b|| / ||b|| for each h (float32 on the GPU)."""
        out = []
        for i in range(0, len(heights), chunk):
            h = torch.as_tensor(np.asarray(heights[i:i + chunk], np.float32), device=self.device)
            decay = 0.5 * MU0_UT * torch.exp(-self.k32[None] * h[:, None, None])
            f = self.crop(torch.fft.irfft2(self.spec32[None] * decay[:, None], s=self.shape))
            err = ((f - b32[None]) ** 2).sum((-2, -1)).sqrt() / b_norm[None]
            out.append(err.mean(1).double().cpu().numpy())
        return np.concatenate(out)


class Corrector:
    """M1b on the GPU: X + H/(H^2+mu) (B - H X) per component, 128 px padding, float64."""

    def __init__(self, pred, b, dz, device):
        pad = mc.PAD
        self.n = pred.shape[-1] + 2 * pad
        t = lambda a: torch.nn.functional.pad(torch.as_tensor(np.asarray(a, np.float64), device=device),
                                              (pad, pad, pad, pad))
        self.fx, self.fb = torch.fft.rfft2(t(pred)), torch.fft.rfft2(t(b))
        wx = 2 * np.pi * torch.fft.rfftfreq(self.n, mc.DX, dtype=torch.float64, device=device)[None, :]
        wy = 2 * np.pi * torch.fft.fftfreq(self.n, mc.DX, dtype=torch.float64, device=device)[:, None]
        self.h = torch.exp(-torch.hypot(wx, wy) * dz)

    def __call__(self, mu):
        h = self.h
        out = torch.fft.irfft2(self.fx + h / (h * h + mu) * (self.fb - h * self.fx), s=(self.n, self.n))
        return out[..., mc.PAD:-mc.PAD, mc.PAD:-mc.PAD]


def rel(pred, truth):
    """Per-component relative L2 of torch [3, H, W] tensors (float64), as numpy."""
    return (((pred - truth) ** 2).sum((-2, -1)).sqrt() / (truth ** 2).sum((-2, -1)).sqrt()).cpu().numpy()


# ---------------------------------------------------------------- model and read-out

def load_model(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model = MAGICUNet().to(device)
    model.load_state_dict(state['model'])
    model.eval()
    return model, state['beta'], state.get('bin', 2)


def current(model, beta, factor, stride, b, device):
    """Predicted sheet current at 0.1 um, exactly as train_magic_unet.reconstruct."""
    j = predict_tiled(model, binned(np.asarray(b, np.float64), factor), beta, 64, stride, 32, device)
    return np.kron(j, np.ones((1, factor, factor)))


def read_height(sheet, b, device):
    b32 = torch.as_tensor(np.asarray(b, np.float32), device=device)
    b_norm = (b32 ** 2).sum((-2, -1)).sqrt()
    return mc.read_height(lambda hs: sheet.misfit(hs, b32, b_norm))


# ---------------------------------------------------------------- simulation

def render(job):
    recipe, settings = job
    item = synthesise(recipe, settings)
    return item['target'], item['input']


def simulated_items(sim, standoff, workers, limit):
    settings = dict(sim.settings, z_input=float(standoff), z_target=0.5)
    recipes = sim.recipes('validation')[:limit]
    with ProcessPoolExecutor(workers) as pool:
        items = list(pool.map(render, [(r, settings) for r in recipes], chunksize=1))
    return [(r['id'], t, b) for r, (t, b) in zip(recipes, items)]


def write_csv(path, rows):
    with open(path, 'w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--runs', nargs='+', default=['run2', 'run3'])
    p.add_argument('--magic-root', default='runs/magic_unet')
    p.add_argument('--sim-dataset', default='runs/simulated_3_to_05_v3')
    p.add_argument('--comparison-run', default='runs/combined_full_200')
    p.add_argument('--data3', default='data/data3')
    p.add_argument('--cutoff', default='runs/m1_m2/cutoff/cutoff.json')
    p.add_argument('--protocol', default='experiments/m1_m2/M1_M2_PROTOCOL.md')
    p.add_argument('--output', required=True)
    p.add_argument('--layouts', type=int, default=50)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--standoffs', type=float, nargs='+', default=list(mc.SIM_STANDOFFS))
    p.add_argument('--gate-only', action='store_true')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    started = time.time()
    log = lambda *a: print(f'[{time.time() - started:7.0f}s]', *a, flush=True)
    here = Path(__file__).resolve().parent
    meta = dict(args=vars(args),
                protocol_sha256=hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest(),
                code_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(here.glob('*.py'))},
                torch=torch.__version__)
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    k_noise = json.loads(Path(args.cutoff).read_text())['k_noise']

    models = {}
    for run in args.runs:
        folder = Path(args.magic_root) / run
        stride = json.loads((folder / 'results.json').read_text())['stride']
        model, beta, factor = load_model(folder / 'final.pt', device)
        models[run] = (model, beta, factor, stride)
        log(run, 'stride', stride, 'beta', beta, 'bin', factor)

    # 1. reproduction gate on COMSOL (raw scores only)
    pairs = discover(args.comparison_run, args.data3)
    data = {(q.name, q.z_in, q.z_tg): (q, load(q.sharp), load(q.blurred)) for q in pairs}
    currents, gate = {}, []
    for run, (model, beta, factor, stride) in models.items():
        saved = {(r['config'], float(r['z_input']), float(r['z_target']), r['component']): float(r['relative_l2'])
                 for r in csv.DictReader(open(Path(args.magic_root) / run / 'comsol.csv'))}
        for key, (q, sharp, blurred) in data.items():
            j = current(model, beta, factor, stride, blurred, device)
            currents[run, key] = j
            raw = rel(Sheet(j, device).field64(q.z_tg), torch.as_tensor(sharp, device=device))
            for c, comp in enumerate(COMPONENTS):
                ref = saved[(q.name, q.z_in, q.z_tg, comp)]
                gate.append(dict(run=run, config=q.name, z_input=q.z_in, z_target=q.z_tg, component=comp,
                                 saved=ref, rerun=float(raw[c]), diff=float(raw[c] - ref)))
    write_csv(out / 'gate.csv', gate)
    worst = max(abs(g['diff']) for g in gate)
    log('gate: worst |rerun - saved| =', worst)
    # float32 read-out against the float64 NumPy reference, on one pair and three heights
    key = ('L1_C1', 3.0, 0.5)
    j = currents[args.runs[0], key]
    b = data[key][2]
    sheet = Sheet(j, device)
    b32 = torch.as_tensor(b.astype(np.float32), device=device)
    hs = np.array([2.0, 3.0, 4.5])
    m32 = sheet.misfit(hs, b32, (b32 ** 2).sum((-2, -1)).sqrt())
    m64 = mc.height_misfit(j, b, hs)
    precision = float(np.abs(m32 - m64).max())
    log('read-out float32 vs float64 NumPy, max misfit difference', precision)
    (out / 'gate.json').write_text(json.dumps(dict(worst_abs_diff=worst, passed=bool(worst <= 1e-4),
                                                   readout_precision=precision), indent=2))
    if worst > 1e-4:
        raise SystemExit('reproduction gate failed; stopping (protocol)')
    if args.gate_only:
        return

    # 2. simulation pass 1: raw, height read-out, correction at the true height for every mu
    sim = SimulatedDataset(args.sim_dataset)
    sim_rows, mu_rows = [], []
    mu_curve = {run: [] for run in models}
    for z in args.standoffs:
        items = simulated_items(sim, z, args.workers, args.layouts)
        log(f'standoff {z}: {len(items)} layouts synthesised')
        for run, (model, beta, factor, stride) in models.items():
            errors = np.zeros((len(items), len(mc.MU_GRID), 3))
            for i, (layout, target, b) in enumerate(items):
                j = current(model, beta, factor, stride, b, device)
                sheet = Sheet(j, device)
                pred = sheet.field64(0.5)
                t = torch.as_tensor(target.astype(np.float64), device=device)
                raw = rel(pred, t)
                h, misfit, edge = read_height(sheet, b, device)
                corr = Corrector(pred.cpu().numpy(), b, z - 0.5, device)
                for m, mu in enumerate(mc.MU_GRID):
                    errors[i, m] = rel(corr(float(mu)), t)
                sim_rows.append(dict(run=run, standoff=z, layout=layout, h_star=h, misfit=misfit, edge=edge,
                                     **{f'raw_{c}': float(raw[k]) for k, c in enumerate(COMPONENTS)}))
            mean = errors.mean((0, 2))
            best = int(np.argmin(mean))
            mu_curve[run].append(float(mc.MU_GRID[best]))
            for m, mu in enumerate(mc.MU_GRID):
                mu_rows.append(dict(run=run, standoff=z, mu=float(mu), mean_relative_l2=float(mean[m]),
                                    chosen=m == best))
            for k, row in enumerate(r for r in sim_rows if r['run'] == run and r['standoff'] == z):
                row.update({f'dc_true_{c}': float(errors[k, best, ci]) for ci, c in enumerate(COMPONENTS)})
                row['mu_true'] = float(mc.MU_GRID[best])
            raw_mean = np.mean([[r[f'raw_{c}'] for c in COMPONENTS] for r in sim_rows
                                if r['run'] == run and r['standoff'] == z])
            log(f'  {run} z={z}: raw {raw_mean:.4f}, corrected {mean[best]:.4f} at mu {mc.MU_GRID[best]:.2e}')
    write_csv(out / 'sim_mu.csv', mu_rows)
    curves = {run: dict(standoffs=list(args.standoffs), mu=v) for run, v in mu_curve.items()}
    (out / 'mu_curve.json').write_text(json.dumps(curves, indent=2))

    # 3. simulation pass 2: correction at h* with mu(h*)
    for z in args.standoffs:
        items = simulated_items(sim, z, args.workers, args.layouts)
        for run, (model, beta, factor, stride) in models.items():
            rows = {r['layout']: r for r in sim_rows if r['run'] == run and r['standoff'] == z}
            for layout, target, b in items:
                row = rows[layout]
                j = current(model, beta, factor, stride, b, device)
                pred = Sheet(j, device).field64(0.5)
                mu = mc.mu_at(row['h_star'], args.standoffs, mu_curve[run])
                fixed = Corrector(pred.cpu().numpy(), b, row['h_star'] - 0.5, device)(mu)
                e = rel(fixed, torch.as_tensor(target.astype(np.float64), device=device))
                row.update({f'dc_est_{c}': float(e[k]) for k, c in enumerate(COMPONENTS)})
                row['mu_est'] = mu
        log(f'standoff {z}: pass 2 done')
    write_csv(out / 'sim_scores.csv', sim_rows)

    # 4. COMSOL
    rows, saved_pred = [], {}
    for run in models:
        for key, (q, sharp, blurred) in data.items():
            j = currents[run, key]
            sheet = Sheet(j, device)
            pred = sheet.field64(q.z_tg).cpu().numpy()
            h, misfit, edge = read_height(sheet, blurred, device)
            mu_true = mc.mu_at(q.z_in, args.standoffs, mu_curve[run])
            mu_est = mc.mu_at(h, args.standoffs, mu_curve[run])
            variants = dict(raw=pred,
                            dc_true=Corrector(pred, blurred, q.z_in - q.z_tg, device)(mu_true).cpu().numpy(),
                            dc_est=Corrector(pred, blurred, h - q.z_tg, device)(mu_est).cpu().numpy())
            for name, field in variants.items():
                four = mc.fourway(field, sharp, blurred, q.z_in - q.z_tg, k_noise)
                for c, comp in enumerate(COMPONENTS):
                    rows.append(dict(run=run, variant=name, config=q.name, shape=mc.SHAPE[q.name],
                                     z_input=q.z_in, z_target=q.z_tg, component=comp, h_star=h, misfit=misfit,
                                     edge=edge, mu_true=mu_true, mu_est=mu_est, **four[c]))
                if q.z_in == 3.0 and q.z_tg == 0.5:
                    saved_pred[f'{run}_{name}_{q.name}'] = field.astype(np.float32)
            log(f'COMSOL {run} {q.name} {q.z_in}->{q.z_tg}: h* {h:.2f}')
    write_csv(out / 'comsol.csv', rows)
    np.savez_compressed(out / 'comsol_3_to_05_predictions.npz', **saved_pred)
    log('done')


if __name__ == '__main__':
    main()
