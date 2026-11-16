"""Amendment A5 of HIGH_STANDOFF_PROTOCOL.md: does COMSOL's extra decay explain run B's COMSOL score?

The 50 simulated validation layouts at 10 -> 0.5 um are rendered with the simulator's `decay_k0` setting
(decay exp(-sqrt(k^2 + k0^2) (z - 0.5 um)) above the 0.5 um plane; k0 = 0 is free space). Inference only, on the
server; existing checkpoints, strides, mu curves and FISTA settings; nothing is chosen here.

  magic  run B and run A at k0 = 0, 0.05, 0.10, 0.15: raw and data-consistency (at h*) relative L2, h*, and per
         component the amplitude <output, target> / <target, target> and the cosine similarity. k0 = 0 is the control
         against the saved scores. The same amplitude and cosine for the saved COMSOL predictions.
  fista  FISTA at its reference depth with the free-space operator, k0 = 0.10, first 5 layouts.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

COMPONENTS = ('Bx', 'By', 'Bz')
RUNS = ('run10b', 'run10a')
Z_IN, Z_TG = 10.0, 0.5


def shape_scores(pred, truth):
    """Per-component relative L2, amplitude and cosine of [3, H, W] arrays."""
    p, t = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    dot, pp, tt = (p * t).sum((-2, -1)), (p * p).sum((-2, -1)), (t * t).sum((-2, -1))
    out = {}
    for i, c in enumerate(COMPONENTS):
        out[f'l2_{c}'] = float(np.sqrt(((p[i] - t[i]) ** 2).sum() / tt[i]))
        out[f'amp_{c}'] = float(dot[i] / tt[i])
        out[f'cos_{c}'] = float(dot[i] / np.sqrt(pp[i] * tt[i]))
    return out


def write_csv(path, rows):
    with open(path, 'w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def magic_part(a, out):
    sys.path.insert(0, str(Path(a.code).resolve()))
    import torch
    import m12_common as mc
    from concurrent.futures import ProcessPoolExecutor
    from m1_rescore import Corrector, Sheet, current, load_model, read_height, render
    from simulate_fields import SimulatedDataset
    from wiener_oracle import discover, load
    device = torch.device('cuda')
    m1 = Path(a.results) / 'm1'
    mu_curve = json.loads((m1 / 'mu_curve.json').read_text())
    saved = {(r['run'], r['layout']): r for r in csv.DictReader(open(m1 / 'sim_scores.csv'))
             if float(r['standoff']) == Z_IN}
    sim = SimulatedDataset(a.sim_dataset)
    recipes = sim.recipes('validation')[:a.layouts]
    models = {}
    for run in RUNS:
        folder = Path(a.magic_root) / run
        model, beta, factor = load_model(folder / 'final.pt', device)
        models[run] = (model, beta, factor, json.loads((folder / 'results.json').read_text())['stride'])
    rows, worst = [], 0.0
    for k0 in a.k0:
        settings = dict(sim.settings, z_input=Z_IN, z_target=Z_TG, decay_k0=float(k0))
        with ProcessPoolExecutor(a.workers) as pool:
            items = list(pool.map(render, [(r, settings) for r in recipes], chunksize=1))
        for recipe, (target, b) in zip(recipes, items):
            for run, (model, beta, factor, stride) in models.items():
                j = current(model, beta, factor, stride, b, device)
                sheet = Sheet(j, device)
                pred = sheet.field64(Z_TG)
                h, misfit, edge = read_height(sheet, b, device, 20.0)
                mu = mc.mu_at(h, mu_curve[run]['standoffs'], mu_curve[run]['mu'])
                fixed = Corrector(pred.cpu().numpy(), b, h - Z_TG, device)(mu)
                for name, field in (('raw', pred), ('dc_est', fixed)):
                    s = shape_scores(field.cpu().numpy(), target)
                    if k0 == 0:
                        ref = saved[run, recipe['id']]
                        worst = max(worst, max(abs(s[f'l2_{c}'] - float(ref[f'{name}_{c}'])) for c in COMPONENTS))
                    rows.append(dict(k0=k0, run=run, variant=name, layout=recipe['id'], h_star=h, misfit=misfit,
                                     edge=edge, **s))
        for run in RUNS:
            pick = [r for r in rows if r['k0'] == k0 and r['run'] == run and r['variant'] == 'dc_est']
            print(f'k0 {k0}: {run} dc_est L2 {np.mean([[r[f"l2_{c}"] for c in COMPONENTS] for r in pick]):.4f} '
                  f'amp {np.mean([[r[f"amp_{c}"] for c in COMPONENTS] for r in pick]):.3f} '
                  f'cos {np.mean([[r[f"cos_{c}"] for c in COMPONENTS] for r in pick]):.3f} '
                  f'h* {np.mean([r["h_star"] for r in pick]):.2f}', flush=True)
        if k0 == 0:
            print('control: worst |rerun - saved| at k0 = 0:', worst, flush=True)
    write_csv(out / 'sim_k0.csv', rows)

    # the same amplitude and cosine for the saved COMSOL predictions
    magic = np.load(m1 / 'comsol_10_to_05_predictions.npz')
    fista = np.load(Path(a.results) / 'fista' / 'predictions.npz')
    comsol = []
    for q in discover(a.comparison_run, a.data3, (Z_IN,)):
        if q.z_in != Z_IN or q.z_tg != Z_TG:
            continue
        sharp = load(q.sharp)
        for label, field in ([(f'{run}_{v}', magic[f'{run}_{v}_{q.name}']) for run in RUNS for v in ('raw', 'dc_est')]
                             + [(m, fista[f'{m}_{q.name}']) for m in ('fista1000', 'fista3000')]):
            comsol.append(dict(config=q.name, method=label, **shape_scores(field, sharp)))
    write_csv(out / 'comsol_shape.csv', comsol)
    (out / 'control.json').write_text(json.dumps(dict(worst_abs_diff_k0_0=worst, passed=bool(worst <= 1e-4))))
    print('COMSOL rows', len(comsol), flush=True)


def fista_part(a, out):
    sys.path.insert(0, str(Path(a.frozen).resolve() / 'scripts'))
    import torch
    import conv_lista as csc
    from simulate_fields import SimulatedDataset
    from train_conv_lista import padded, render
    device = torch.device('cuda')
    results = json.loads((Path(a.results) / 'fista' / 'results.json').read_text())
    depth, pad, dz = results['reference_depth'], results['pad'], results['z_delta']
    dataset = SimulatedDataset(a.sim_dataset)
    stored = torch.load(a.dictionary, weights_only=True)
    g, lam = stored['g'].to(device), stored['lam'] * results['choice'][str(depth)]
    settings = dict(dataset.settings, z_input=Z_IN, decay_k0=float(a.fista_k0))
    rows = []
    for recipe in dataset.recipes('validation')[:a.fista_layouts]:
        target, blurred = render((recipe, settings))
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
        rows.append(dict(k0=a.fista_k0, layout=recipe['id'], depth=depth, **shape_scores(field, target)))
        print(f'FISTA-{depth} k0 {a.fista_k0} {recipe["id"]}: L2 '
              f'{np.mean([rows[-1][f"l2_{c}"] for c in COMPONENTS]):.4f} '
              f'amp {np.mean([rows[-1][f"amp_{c}"] for c in COMPONENTS]):.3f}', flush=True)
    write_csv(out / 'fista_k0.csv', rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('part', choices=('magic', 'fista'))
    p.add_argument('--code', default='experiments/magic_unet_10um/scripts')
    p.add_argument('--frozen', default='experiments/fista_depths')
    p.add_argument('--results', default='runs/high_standoff')
    p.add_argument('--magic-root', default='runs/magic_unet')
    p.add_argument('--sim-dataset', default='runs/simulated_3_to_05_v3')
    p.add_argument('--comparison-run', default='runs/combined_full_200')
    p.add_argument('--data3', default='data/data3')
    p.add_argument('--dictionary', default='runs/csc_lista_v2/dictionary_strong/dictionary_0.3.pt')
    p.add_argument('--k0', type=float, nargs='+', default=[0.0, 0.05, 0.10, 0.15])
    p.add_argument('--layouts', type=int, default=50)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--fista-k0', type=float, default=0.10)
    p.add_argument('--fista-layouts', type=int, default=5)
    p.add_argument('--output', default='runs/high_standoff/k0_check')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (magic_part if a.part == 'magic' else fista_part)(a, out)


if __name__ == '__main__':
    main()
