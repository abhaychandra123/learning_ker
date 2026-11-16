"""Export reconstructions for the 10 -> 0.5 um figures (README_HIGH_STANDOFF.md). Runs on the server, read-only
apart from its own output folder. Nothing is trained, chosen or tuned here; every setting is read from the
finished runs, and each re-computed score is printed next to the saved one.

Two parts, run as two processes because the FISTA code is a frozen older copy with its own modules:

  magic  MAGIC-UNet run A, run B and run 3 on two held-out simulated layouts at 10 um (raw, and with the
         data-consistency step at the estimated height), plus the saved COMSOL predictions of L1_C1 and IC_4um.
  fista  FISTA at its reference depth (3,000 iterations, lambda scale 0.0003, padding 128 px) on the same two
         simulated layouts, which the scoring run scored but did not save.

The two simulated layouts are picked by a rule, not by eye: among the first 20 validation layouts (the ones FISTA
was scored on), the one with the median run B error and the one with the worst run B error at 10 um.

Maps are saved as Bz, averaged 2x2 for display (700 x 700); the middle row is saved at full resolution; scores are
computed at full resolution (1400 x 1400, 0.1 um).
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

COMPONENTS = ('Bx', 'By', 'Bz')
SIM_RUNS = ('run10b', 'run10a', 'run3')
Z_IN, Z_TG = 10.0, 0.5


def scores(pred, truth):
    """Per-component relative L2 and NRMSE (RMSE / target range) of [3, H, W] arrays."""
    pred, truth = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    err = pred - truth
    l2 = np.sqrt((err ** 2).sum((-2, -1)) / (truth ** 2).sum((-2, -1)))
    nrmse = np.sqrt((err ** 2).mean((-2, -1))) / (truth.max((-2, -1)) - truth.min((-2, -1)))
    return dict(l2=[float(v) for v in l2], nrmse=[float(v) for v in nrmse])


def line_scores(pred, truth):
    err = np.asarray(pred, np.float64) - np.asarray(truth, np.float64)
    return dict(l2=float(np.linalg.norm(err) / np.linalg.norm(truth)),
                nrmse=float(np.sqrt((err ** 2).mean()) / (truth.max() - truth.min())))


def show(a):
    """Bz, 2x2 average, float32."""
    z = np.asarray(a[2], np.float64)
    return z.reshape(z.shape[0] // 2, 2, z.shape[1] // 2, 2).mean((1, 3)).astype(np.float32)


def pick_layouts(m1, ids):
    rows = [r for r in csv.DictReader(open(m1 / 'sim_scores.csv'))
            if r['run'] == 'run10b' and float(r['standoff']) == Z_IN and r['layout'] in ids]
    err = {r['layout']: np.mean([float(r[f'dc_est_{c}']) for c in COMPONENTS]) for r in rows}
    order = sorted(err, key=err.get)
    return dict(median=order[(len(order) - 1) // 2], worst=order[-1])


def add(arrays, meta, key, field, truth, extra=None):
    arrays[key] = show(field)
    arrays[key + '__row'] = np.asarray(field[2][field.shape[-2] // 2], np.float32)
    row_truth = truth[2][truth.shape[-2] // 2]
    meta[key] = dict(scores(field, truth), line=line_scores(field[2][field.shape[-2] // 2], row_truth),
                     **(extra or {}))


def magic_part(a, out):
    sys.path.insert(0, str(Path(a.code).resolve()))
    import torch
    import m12_common as mc
    from m1_rescore import Corrector, Sheet, current, load_model, read_height, rel
    from simulate_fields import SimulatedDataset, synthesise
    device = torch.device('cuda')
    m1 = Path(a.results) / 'm1'
    mu_curve = json.loads((m1 / 'mu_curve.json').read_text())
    saved = {(r['run'], r['layout']): r for r in csv.DictReader(open(m1 / 'sim_scores.csv'))
             if float(r['standoff']) == Z_IN}
    sim = SimulatedDataset(a.sim_dataset)
    recipes = sim.recipes('validation')[:20]
    chosen = pick_layouts(m1, {r['id'] for r in recipes})
    settings = dict(sim.settings, z_input=Z_IN, z_target=Z_TG)
    arrays, meta = {}, dict(chosen=chosen, z_input=Z_IN, z_target=Z_TG, dx_show_um=0.2)
    models = {}
    for run in SIM_RUNS:
        folder = Path(a.magic_root) / run
        model, beta, factor = load_model(folder / 'final.pt', device)
        models[run] = (model, beta, factor, json.loads((folder / 'results.json').read_text())['stride'])
    for tag, layout in chosen.items():
        recipe = next(r for r in recipes if r['id'] == layout)
        item = synthesise(recipe, settings)
        target, b = item['target'], item['input']
        t = torch.as_tensor(target.astype(np.float64), device=device)
        add(arrays, meta, f'sim_{tag}_target', target, target, dict(layout=layout))
        add(arrays, meta, f'sim_{tag}_input', b, target)
        for run, (model, beta, factor, stride) in models.items():
            j = current(model, beta, factor, stride, b, device)
            sheet = Sheet(j, device)
            pred = sheet.field64(Z_TG)
            h, misfit, edge = read_height(sheet, b, device, 20.0)
            mu = mc.mu_at(h, mu_curve[run]['standoffs'], mu_curve[run]['mu'])
            fixed = Corrector(pred.cpu().numpy(), b, h - Z_TG, device)(mu)
            row = saved[run, layout]
            for name, field in (('raw', pred), ('dc_est', fixed)):
                e = rel(field, t)
                ref = [float(row[f'{name}_{c}']) for c in COMPONENTS]
                print(f'{tag} {layout} {run} {name}: rerun {np.round(e, 4)} saved {np.round(ref, 4)} '
                      f'max diff {np.abs(e - ref).max():.2e}; h* {h:.2f} saved {float(row["h_star"]):.2f}', flush=True)
                add(arrays, meta, f'sim_{tag}_{run}_{name}', field.cpu().numpy(), target, dict(h_star=float(h)))

    # COMSOL: saved predictions only
    fista = np.load(Path(a.results) / 'fista' / 'predictions.npz')
    magic = np.load(m1 / 'comsol_10_to_05_predictions.npz')
    ref_m = {(r['run'], r['variant'], r['config'], r['component']): (float(r['raw']), float(r['h_star']))
             for r in csv.DictReader(open(m1 / 'comsol.csv'))}
    ref_f = {(r['method'], r['config'], r['component']): float(r['relative_l2'])
             for r in csv.DictReader(open(Path(a.results) / 'fista' / 'comsol.csv'))}
    for config in ('L1_C1', 'IC_4um'):
        with np.load(Path(a.comparison_run) / 'comparison' / f'{config}_z{Z_IN:04.1f}_to_z00.5.npz') as z:
            sharp, blurred = z['sharp'].astype(np.float64), z['blurred'].astype(np.float64)
        add(arrays, meta, f'comsol_{config}_target', sharp, sharp)
        add(arrays, meta, f'comsol_{config}_input', blurred, sharp)
        for run in SIM_RUNS:
            for name in ('raw', 'dc_est'):
                key = f'comsol_{config}_{run}_{name}'
                add(arrays, meta, key, magic[f'{run}_{name}_{config}'], sharp,
                    dict(h_star=ref_m[run, name, config, 'Bz'][1]))
                diff = max(abs(meta[key]['l2'][i] - ref_m[run, name, config, c][0]) for i, c in enumerate(COMPONENTS))
                print(f'COMSOL {config} {run} {name}: {np.round(meta[key]["l2"], 4)} max diff from csv {diff:.2e}',
                      flush=True)
        for method in ('fista1000', 'fista3000'):
            key = f'comsol_{config}_{method}'
            add(arrays, meta, key, fista[f'{method}_{config}'], sharp)
            diff = max(abs(meta[key]['l2'][i] - ref_f[method, config, c]) for i, c in enumerate(COMPONENTS))
            print(f'COMSOL {config} {method}: {np.round(meta[key]["l2"], 4)} max diff from csv {diff:.2e}', flush=True)
    np.savez_compressed(out / 'magic_part.npz', **arrays)
    (out / 'magic_part.json').write_text(json.dumps(meta, indent=2))


def fista_part(a, out):
    sys.path.insert(0, str(Path(a.frozen).resolve() / 'scripts'))
    import torch
    import conv_lista as csc
    from simulate_fields import SimulatedDataset
    from train_conv_lista import padded, render
    device = torch.device('cuda')
    results = json.loads((Path(a.results) / 'fista' / 'results.json').read_text())
    depth, pad, dz = results['reference_depth'], results['pad'], results['z_delta']
    scale_choice = results['choice'][str(depth)]
    saved = {r['layout']: [float(r[c]) for c in COMPONENTS]
             for r in csv.DictReader(open(Path(a.results) / 'fista' / 'sim_full.csv'))}
    dataset = SimulatedDataset(a.sim_dataset)
    dataset.settings = dict(dataset.settings, z_input=Z_IN)
    recipes = dataset.recipes('validation')[:20]
    chosen = pick_layouts(Path(a.results) / 'm1', {r['id'] for r in recipes})
    stored = torch.load(a.dictionary, weights_only=True)
    g, lam = stored['g'].to(device), stored['lam'] * scale_choice
    arrays, meta = {}, dict(depth=depth, pad=pad, z_delta=dz, lam=lam, chosen=chosen)
    for tag, layout in chosen.items():
        target, blurred = render((next(r for r in recipes if r['id'] == layout), dataset.settings))
        b = padded(torch.tensor(blurred / dataset.scale, dtype=torch.float32, device=device), pad)
        with torch.no_grad():       # the loop of m2_fista_depth.fista_snapshots, final iterate only
            n = b.shape[-1]
            physics = csc.Operator(g, csc.transfer(n, 0.1, dz, b.dtype, b.device), 8)
            synth = csc.Operator(g, chunk=8)
            mask = csc.window_mask(n, pad, b.device)
            step = 1 / physics.bound(n)
            x = torch.zeros(b.shape[0], g.shape[0], n, n, dtype=b.dtype, device=b.device)
            y, t = x, 1.0
            for _ in range(depth):
                update = csc.shrink(y - step * physics.adjoint(mask * (physics.forward(y) - b)), lam * step)
                t_next = (1 + (1 + 4 * t * t) ** .5) / 2
                y = update + (t - 1) / t_next * (update - x)
                t, x = t_next, update
            field = synth.forward(x)[..., pad:-pad, pad:-pad].cpu().numpy().astype(np.float64) * dataset.scale
        key = f'sim_{tag}_fista{depth}'
        add(arrays, meta, key, field, target, dict(layout=layout))
        print(f'{tag} {layout} FISTA-{depth}: rerun {np.round(meta[key]["l2"], 4)} saved {np.round(saved[layout], 4)} '
              f'max diff {np.abs(np.array(meta[key]["l2"]) - saved[layout]).max():.2e}', flush=True)
    np.savez_compressed(out / 'fista_part.npz', **arrays)
    (out / 'fista_part.json').write_text(json.dumps(meta, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('part', choices=('magic', 'fista'))
    p.add_argument('--code', default='experiments/magic_unet_10um/scripts')
    p.add_argument('--frozen', default='experiments/fista_depths')
    p.add_argument('--results', default='runs/high_standoff')
    p.add_argument('--magic-root', default='runs/magic_unet')
    p.add_argument('--sim-dataset', default='runs/simulated_3_to_05_v3')
    p.add_argument('--comparison-run', default='runs/combined_full_200')
    p.add_argument('--dictionary', default='runs/csc_lista_v2/dictionary_strong/dictionary_0.3.pt')
    p.add_argument('--output', default='runs/high_standoff/export')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (magic_part if a.part == 'magic' else fista_part)(a, out)


if __name__ == '__main__':
    main()
