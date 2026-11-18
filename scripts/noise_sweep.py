"""Noise sweep, B-field SSIM and timing on simulated layouts (inference only; server GPU).

Asked for in the 18 Nov 2026 meeting: how do the results change as the noise grows (1, 2, 5 % and beyond), how
does our SSIM compare with Reed et al., and how long does each method take. Nothing here is trained or chosen on
COMSOL.

What is fixed in advance:
  * layouts: the first --layouts validation layouts of runs/simulated_3_to_05_v3, rendered at --z-input -> 0.5 um
    (the same layouts and noise patterns as every earlier simulated score);
  * noise: the simulator's own model (speckle with 0.38 um correlation, sigma = level % of each component's
    max - min). Only the level changes; the noise pattern of a layout stays the same (same noise_seed). Level
    'native' keeps each layout's own 1-2.5 % and is the reproduction gate against the saved scores;
  * MAGIC-UNet: the run's final.pt and stride, height read-out and one data-consistency step with mu(h*) from
    the run's mu_curve.json, exactly as m1_rescore.py. mu was chosen at the training noise and is not re-tuned;
  * FISTA: dictionary, padding, depth and lambda from the run's results.json (reference depth and scale). For
    levels above --retune-above, lambda x 1, 3 and 10 are all scored and reported; picking the best per level on
    the same layouts favours FISTA (an oracle lambda), which makes any MAGIC-UNet win conservative;
  * metrics per component of B at 0.5 um: relative L2, SSIM (Wang et al. 2004, data range = truth max - min, as
    magic_unet.ssim) and RMSE in uT; for MAGIC-UNet also the paper's J metrics against J_eff; sigma_n in the
    paper's convention (sigma / max |B| of the clean input) is recorded per image;
  * timing: wall time per image on the GPU with torch.cuda.synchronize, split into network and correction.

Usage (from the project root on the server, frozen code folder on the path):
  python noise_sweep.py magic --run run10b --z-input 10 --output runs/noise_sweep/magic_10um
  python noise_sweep.py fista --fista-results runs/high_standoff/fista/results.json --z-input 10 \
      --layouts 5 --output runs/noise_sweep/fista_10um
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

import m12_common as mc
from m1_rescore import Corrector, Sheet, current, load_model, read_height
from magic_data import binned
from magic_unet import ssim
from simulate_fields import COMPONENTS, SimulatedDataset, synthesise
from train_magic_unet import true_sheet

LEVELS = [0, 1, 2, 5, 10, 25, 50]


def b_metrics(pred, truth, prefix):
    """relative L2, SSIM and RMSE (uT) per component of [3, H, W] arrays."""
    pred, truth = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    row = {}
    for c, name in enumerate(COMPONENTS):
        err = pred[c] - truth[c]
        row[f'{prefix}_l2_{name}'] = float(np.linalg.norm(err) / np.linalg.norm(truth[c]))
        row[f'{prefix}_ssim_{name}'] = ssim(pred[c], truth[c])
        row[f'{prefix}_rmse_{name}'] = float(np.sqrt(np.mean(err ** 2)))
    for metric in ('l2', 'ssim', 'rmse'):
        row[f'{prefix}_{metric}'] = float(np.mean([row[f'{prefix}_{metric}_{n}'] for n in COMPONENTS]))
    return row


def sigma_n(item):
    """Paper convention: noise sigma divided by max |B| of the clean input (mean over components)."""
    return float(np.mean(item['sigma_ut']) / np.abs(item['clean']).max())


class Writer:
    """CSV that is rewritten after every row, so a stopped job leaves usable results."""

    def __init__(self, path):
        self.path, self.rows = path, []

    def add(self, row):
        self.rows.append(row)
        keys = list(dict.fromkeys(k for r in self.rows for k in r))
        with open(self.path, 'w', newline='') as stream:
            w = csv.DictWriter(stream, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def saved_magic(a):
    """Saved per-layout scores at the native noise: raw (run's sim_validation.csv) and corrected (m1 sim_scores)."""
    raw = {r['layout']: [float(r[f'relative_l2_{c}']) for c in COMPONENTS]
           for r in csv.DictReader(open(Path(a.magic_root) / a.run / 'sim_validation.csv'))}
    dc = {}
    if a.m1_scores and Path(a.m1_scores).exists():
        for r in csv.DictReader(open(a.m1_scores)):
            if r['run'] == a.run and float(r['standoff']) == a.z_input:
                dc[r['layout']] = ([float(r[f'dc_est_{c}']) for c in COMPONENTS], float(r['h_star']))
    return raw, dc


def run_magic(a, sim, settings, recipes, device, out, log):
    folder = Path(a.magic_root) / a.run
    stride = json.loads((folder / 'results.json').read_text())['stride']
    model, beta, factor = load_model(folder / 'final.pt', device)
    curve = json.loads(Path(a.mu_curve).read_text())[a.run]
    log(a.run, 'stride', stride, 'bin', factor, 'mu curve', curve)
    raw_saved, dc_saved = saved_magic(a)
    writer, gate, examples = Writer(out / 'scores.csv'), [], {}
    current(model, beta, factor, stride, synthesise(recipes[0], settings)['input'], device)    # warm-up
    for level in ['native'] + list(a.levels):
        for i, recipe in enumerate(recipes):
            r = recipe if level == 'native' else dict(recipe, noise_percent=float(level))
            item = synthesise(r, settings)
            b, target = item['input'], item['target']
            sync(device)
            t0 = time.perf_counter()
            j = current(model, beta, factor, stride, b, device)
            sheet = Sheet(j, device)
            pred = sheet.field64(0.5)
            sync(device)
            t1 = time.perf_counter()
            h, misfit, edge = read_height(sheet, b, device, a.height_max)
            mu = mc.mu_at(h, curve['standoffs'], curve['mu'])
            pred_np = pred.cpu().numpy()
            fixed = Corrector(pred_np, b, h - 0.5, device)(mu).cpu().numpy()
            sync(device)
            t2 = time.perf_counter()
            row = dict(method='magic', run=a.run, z_input=a.z_input, level=level, layout=recipe['id'],
                       noise_percent=r['noise_percent'], sigma_n=sigma_n(item), h_star=h, edge=edge, mu=mu,
                       seconds_network=t1 - t0, seconds_correction=t2 - t1, seconds_total=t2 - t0)
            row.update(b_metrics(pred_np, target, 'raw'))
            row.update(b_metrics(fixed, target, 'dc'))
            truth = binned(true_sheet(recipe, settings), factor)
            jp = j[:, ::factor, ::factor]
            for c, name in enumerate(('Jx', 'Jy')):
                row[f'J_ssim_{name}'] = ssim(jp[c], truth[c])
                row[f'J_l2_{name}'] = float(np.linalg.norm(jp[c] - truth[c]) / np.linalg.norm(truth[c]))
                row[f'J_rmse_{name}'] = float(np.sqrt(np.mean((jp[c] - truth[c]) ** 2)))
            row['J_ssim'] = (row['J_ssim_Jx'] + row['J_ssim_Jy']) / 2
            row['J_l2'] = (row['J_l2_Jx'] + row['J_l2_Jy']) / 2
            writer.add(row)
            if level == 'native':
                if recipe['id'] in raw_saved:
                    d = [row[f'raw_l2_{c}'] - s for c, s in zip(COMPONENTS, raw_saved[recipe['id']])]
                    gate.append(dict(layout=recipe['id'], kind='raw', worst=float(np.max(np.abs(d)))))
                if recipe['id'] in dc_saved:
                    s, hs = dc_saved[recipe['id']]
                    d = [row[f'dc_l2_{c}'] - v for c, v in zip(COMPONENTS, s)]
                    gate.append(dict(layout=recipe['id'], kind='dc_est', worst=float(np.max(np.abs(d))),
                                     h_diff=float(h - hs)))
            if i == 0:
                examples[f'{level}_input_Bz'] = b[2].astype(np.float16)
                examples[f'{level}_target_Bz'] = target[2].astype(np.float16)
                examples[f'{level}_dc_Bz'] = fixed[2].astype(np.float16)
                examples[f'{level}_J'] = jp.astype(np.float16)
                examples['J_truth'] = truth.astype(np.float16)
        rows = [x for x in writer.rows if x['level'] == level]
        log(f'level {level}: raw {np.mean([x["raw_l2"] for x in rows]):.4f}, '
            f'dc {np.mean([x["dc_l2"] for x in rows]):.4f}, B ssim {np.mean([x["dc_ssim"] for x in rows]):.3f}, '
            f'J ssim {np.mean([x["J_ssim"] for x in rows]):.3f}, sigma_n {np.mean([x["sigma_n"] for x in rows]):.4f}, '
            f'{np.mean([x["seconds_total"] for x in rows]):.2f} s/image')
        if level == 'native':
            worst = max((g['worst'] for g in gate), default=float('nan'))
            (out / 'gate.json').write_text(json.dumps(dict(rows=gate, worst=worst, passed=bool(worst <= 1e-3)),
                                                      indent=2))
            log('gate vs saved scores: worst |diff|', worst, 'over', len(gate), 'rows')
        np.savez_compressed(out / 'examples.npz', **examples)


def load_conv_lista(frozen):
    """The frozen FISTA operator module, loaded by path so its folder does not shadow ours."""
    spec = importlib.util.spec_from_file_location('conv_lista_frozen', Path(frozen) / 'scripts' / 'conv_lista.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_fista(a, sim, settings, recipes, device, out, log):
    csc = load_conv_lista(a.frozen)
    ref = json.loads(Path(a.fista_results).read_text())
    depth, scale, pad = int(ref['reference_depth']), float(ref['reference_scale']), int(ref['pad'])
    dz = float(ref['z_delta'])
    if abs(dz - (a.z_input - 0.5)) > 1e-9:
        raise SystemExit(f'results.json is for z_delta {dz}, not {a.z_input - 0.5}')
    saved = torch.load(a.dictionary, weights_only=True)
    g, lam0 = saved['g'].to(device), saved['lam']
    log('FISTA depth', depth, 'scale', scale, 'pad', pad, 'lambda', lam0 * scale)

    @torch.no_grad()
    def fista(b):
        """conv_lista.code_ista (fast=True) as m2_fista_depth.fista_snapshots, final depth only. b: [B, H, W]."""
        bp = torch.nn.functional.pad(b, (pad, pad, pad, pad))
        n = bp.shape[-1]
        physics = csc.Operator(g, csc.transfer(n, 0.1, dz, bp.dtype, device), a.chunk)
        synth = csc.Operator(g, chunk=a.chunk)
        mask = csc.window_mask(n, pad, device)
        step = 1 / physics.bound(n)
        x = torch.zeros(bp.shape[0], g.shape[0], n, n, dtype=bp.dtype, device=device)
        y, t = x, 1.0
        for _ in range(depth):
            residual = mask * (physics.forward(y) - bp)
            update = csc.shrink(y - step * physics.adjoint(residual), lam * step)
            t_next = (1 + (1 + 4 * t * t) ** .5) / 2
            y = update + (t - 1) / t_next * (update - x)
            t, x = t_next, update
        return synth.forward(x)[..., pad:-pad, pad:-pad]

    saved_rows = {}
    if a.fista_sim and Path(a.fista_sim).exists():
        saved_rows = {r['layout']: [float(r[c]) for c in COMPONENTS] for r in csv.DictReader(open(a.fista_sim))}
    writer, gate, examples = Writer(out / 'scores.csv'), [], {}
    for level in ['native'] + list(a.levels):
        mults = a.lambda_mults if level != 'native' and float(level) > a.retune_above else [1.0]
        items = []
        for recipe in recipes:
            r = recipe if level == 'native' else dict(recipe, noise_percent=float(level))
            items.append((recipe, r, synthesise(r, settings)))
        for mult in mults:
            lam = lam0 * scale * mult
            for first in range(0, len(items), a.images_per_batch):
                part = items[first:first + a.images_per_batch]
                b = torch.tensor(np.concatenate([it['input'] for _, _, it in part]) / sim.scale,
                                 dtype=torch.float32, device=device)
                sync(device)
                t0 = time.perf_counter()
                y = fista(b)
                sync(device)
                seconds = (time.perf_counter() - t0) / len(part)
                y = y.cpu().numpy().astype(np.float64) * sim.scale
                for k, (recipe, r, item) in enumerate(part):
                    pred = y[3 * k:3 * k + 3]
                    row = dict(method='fista', depth=depth, z_input=a.z_input, level=level, layout=recipe['id'],
                               noise_percent=r['noise_percent'], sigma_n=sigma_n(item), lambda_mult=mult,
                               lam=lam, images_per_batch=len(part), seconds_total=seconds)
                    row.update(b_metrics(pred, item['target'], 'fista'))
                    writer.add(row)
                    if level == 'native' and recipe['id'] in saved_rows:
                        d = [row[f'fista_l2_{c}'] - s for c, s in zip(COMPONENTS, saved_rows[recipe['id']])]
                        gate.append(dict(layout=recipe['id'], worst=float(np.max(np.abs(d)))))
                    if first == 0 and k == 0 and mult == 1.0:
                        examples[f'{level}_fista_Bz'] = pred[2].astype(np.float16)
            rows = [x for x in writer.rows if x['level'] == level and x['lambda_mult'] == mult]
            log(f'level {level} lambda x{mult:g}: {np.mean([x["fista_l2"] for x in rows]):.4f}, '
                f'ssim {np.mean([x["fista_ssim"] for x in rows]):.3f}, '
                f'{np.mean([x["seconds_total"] for x in rows]):.1f} s/image')
        if level == 'native':
            worst = max((g_['worst'] for g_ in gate), default=float('nan'))
            (out / 'gate.json').write_text(json.dumps(dict(rows=gate, worst=worst, passed=bool(worst <= 1e-3)),
                                                      indent=2))
            log('gate vs saved sim_full.csv: worst |diff|', worst, 'over', len(gate), 'layouts')
        np.savez_compressed(out / 'examples.npz', **examples)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('method', choices=['magic', 'fista'])
    p.add_argument('--output', required=True)
    p.add_argument('--z-input', type=float, required=True)
    p.add_argument('--levels', type=float, nargs='+', default=LEVELS, help='Noise, % of each component range')
    p.add_argument('--layouts', type=int, default=20)
    p.add_argument('--sim-dataset', default='runs/simulated_3_to_05_v3')
    p.add_argument('--device', default='cuda')
    # MAGIC-UNet
    p.add_argument('--run', default='run10b')
    p.add_argument('--magic-root', default='runs/magic_unet')
    p.add_argument('--mu-curve', default='runs/high_standoff/m1/mu_curve.json')
    p.add_argument('--m1-scores', default='runs/high_standoff/m1/sim_scores.csv', help='Gate for the corrected score')
    p.add_argument('--height-max', type=float, default=20.0)
    # FISTA
    p.add_argument('--frozen', default='experiments/fista_depths')
    p.add_argument('--dictionary', default='runs/csc_lista_v2/dictionary_strong/dictionary_0.3.pt')
    p.add_argument('--fista-results', default='runs/high_standoff/fista/results.json')
    p.add_argument('--fista-sim', default='runs/high_standoff/fista/sim_full.csv', help='Gate (native level)')
    p.add_argument('--lambda-mults', type=float, nargs='+', default=[1.0, 3.0, 10.0])
    p.add_argument('--retune-above', type=float, default=2.5, help='Score all lambda multiples above this level')
    p.add_argument('--images-per-batch', type=int, default=1)
    p.add_argument('--chunk', type=int, default=8)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'args.json').write_text(json.dumps(vars(a), indent=2))
    started = time.time()
    log = lambda *x: print(f'[{time.time() - started:7.0f}s]', *x, flush=True)
    device = torch.device(a.device)
    sim = SimulatedDataset(a.sim_dataset)
    settings = dict(sim.settings, z_input=float(a.z_input), z_target=0.5)
    recipes = sim.recipes('validation')[:a.layouts]
    log(a.method, 'z_input', a.z_input, 'layouts', len(recipes), 'levels', a.levels,
        'GPU', torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu')
    (run_magic if a.method == 'magic' else run_fista)(a, sim, settings, recipes, device, out, log)
    log('done')


if __name__ == '__main__':
    main()
