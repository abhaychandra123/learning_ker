"""Train CSC-LISTA (Eq. 20 of the ALISTA paper) on simulated fields; test once on COMSOL.

Trains on crops of the 0.1 um fields and reconstructs full images afterwards.
Dictionary, network and baseline settings are chosen on simulated validation data
only. COMSOL pairs are read after training and only for the final scores.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import csc_lista as csc
from simulate_fields import COMPONENTS, SimulatedDataset, synthesise

DX, IMAGE = 0.1, 1400
DATA3 = ('C1_C2', 'C1_C2_C3', 'C1_C2_C3_C4')


def render(job):
    recipe, settings = job
    item = synthesise(recipe, settings)
    return item['target'], item['input']


def simulated(dataset, split, workers):
    """Normalised target and input arrays, one row per component."""
    jobs = [(r, dataset.settings) for r in dataset.recipes(split)]
    with ProcessPoolExecutor(workers) as pool:
        items = list(pool.map(render, jobs, chunksize=1))
    scale = dataset.scale
    return (torch.tensor(np.concatenate([t for t, _ in items]) / scale, dtype=torch.float32),
            torch.tensor(np.concatenate([b for _, b in items]) / scale, dtype=torch.float32))


def load_csv(path):
    table = pd.read_csv(path, comment='%', header=None, usecols=[2, 3, 4]).to_numpy()
    return table.T.reshape(3, 1400, 1400) * 1e6


def comsol_pairs(run, data3):
    """The five COMSOL configurations at 3 -> 0.5 um, in uT."""
    for geometry in ('L1_C1', 'IC_4um'):
        with np.load(Path(run) / 'comparison' / f'{geometry}_z03.0_to_z00.5.npz') as z:
            yield geometry, z['sharp'].astype(np.float64), z['blurred'].astype(np.float64)
    for config in DATA3:
        stem = Path(data3) / f'Coil_{config}' / f'L1_{config}'
        yield (config, load_csv(f'{stem}_Z0p5um_600uA_0p1PR_MF_B.csv'),
               load_csv(f'{stem}_Z3p0um_600uA_0p1PR_MF_B.csv'))


def padded(x, pad):
    return torch.nn.functional.pad(x, (pad, pad, pad, pad))


def relative_l2(output, target):
    dims = (-2, -1)
    return torch.linalg.vector_norm(output - target, dim=dims) / torch.linalg.vector_norm(target, dim=dims)


def crop_loss(output, target, kind):
    """Mean squared error, or the mean per-crop relative L2 used for validation."""
    if kind == 'relative':
        return relative_l2(output, target).mean()
    return ((output - target) ** 2).mean()


@torch.no_grad()
def predict(method, inputs, pad, device, batch=1):
    out = []
    for part in inputs.split(batch):
        y = method(padded(part.to(device), pad))
        out.append(y[..., pad:-pad, pad:-pad].cpu())
    return torch.cat(out)


def error(method, inputs, targets, pad, device, batch=1):
    return float(relative_l2(predict(method, inputs, pad, device, batch), targets).mean())


class CodedSolver:
    """Classical ISTA/FISTA on the same dictionary, as a comparison."""

    def __init__(self, atoms, dz, lam, steps, pad, fast, chunk):
        self.atoms, self.dz, self.lam, self.steps = atoms, dz, lam, steps
        self.pad, self.fast, self.chunk = pad, fast, chunk

    def __call__(self, b):
        h = csc.transfer(b.shape[-1], DX, self.dz, b.dtype, b.device)
        physics = csc.Operator(self.atoms, h, self.chunk)
        x = csc.code_ista(b, physics, self.lam, self.steps, self.pad, self.fast)
        return csc.Operator(self.atoms, chunk=self.chunk).forward(x)


def paired_crops(pair, count, size, seed):
    """Fixed target and input crops at the same positions."""
    generator = torch.Generator().manual_seed(seed)
    targets, inputs = pair
    height = targets.shape[-1]
    out = []
    while len(out) < count:
        index = int(torch.randint(0, len(targets), (1,), generator=generator))
        row = int(torch.randint(0, height - size + 1, (1,), generator=generator))
        col = int(torch.randint(0, height - size + 1, (1,), generator=generator))
        target = targets[index, row:row + size, col:col + size]
        if float(target.abs().max()) >= .05:
            out.append((target, inputs[index, row:row + size, col:col + size]))
    return torch.stack([t for t, _ in out]), torch.stack([b for _, b in out])


def dictionary_stage(args, targets, val_crops, device, output):
    """Learn one dictionary per lambda and keep the validated best."""
    trial = []
    for lam in args.dict_lambda:
        print(f'dictionary lambda {lam}', flush=True)
        g, history = csc.learn_dictionary(targets, args.atoms, args.atom_size, lam,
                                         args.dict_steps, args.dict_batch, args.dict_crop,
                                         seed=args.seed, device=device)
        pad = args.atom_size // 2 + 1
        operator = csc.Operator(g)
        errors, density = [], []
        with torch.no_grad():
            for part in val_crops.split(4):
                b = padded(part.to(device), pad)
                x = csc.code_ista(b, operator, lam, 200, pad)
                y = operator.forward(x)[..., pad:-pad, pad:-pad]
                errors.append(relative_l2(y, part.to(device)).cpu())
                density.append(float((x != 0).float().mean()))
        row = dict(lam=lam, relative_l2=float(torch.cat(errors).mean()),
                   density=float(np.mean(density)))
        print(row, flush=True)
        trial.append(row)
        torch.save(dict(g=g.cpu(), lam=lam, history=history), output / f'dictionary_{lam:g}.pt')
    (output / 'dictionary_trials.json').write_text(json.dumps(trial, indent=2))
    return trial


def train(args, model, train_set, val_crops, device, output):
    targets, inputs = train_set
    # Filters start near 1e-8, so a fixed rate would swamp them; scale it to each.
    groups = [dict(params=[model.w], lr=args.lr_relative * float(model.w.pow(2).mean().sqrt())),
              dict(params=[model.theta], lr=args.lr_relative * float(model.theta.abs().mean()))]
    if model.beta is not None:
        groups.append(dict(params=[model.beta], lr=args.lr_relative * .1))
    optimiser = torch.optim.Adam(groups)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: .5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps)))
    floor = args.theta_floor * float(model.theta.min())
    generator = torch.Generator().manual_seed(args.seed)
    pad, margin, size, log = args.pad, args.margin, args.crop, []
    best = float('inf')
    started = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            score = error(model, val_crops[1], val_crops[0], pad, device, batch=args.batch)
            with torch.no_grad():
                sample = padded(val_crops[1][:args.batch].to(device), pad)
                density = float((model.codes(sample) != 0).float().mean())
            model.train()
            zero = int((model.theta <= 0).sum())
            if step == 0 or score < best:
                best = score
                torch.save(dict(state=model.state_dict(), step=step, score=score),
                           output / 'best.pt')
            memory = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else 0
            log.append(dict(step=step, validation=score, seconds=time.time() - started,
                            peak_gib=memory, density=density, zero_thresholds=zero))
            (output / 'training.json').write_text(json.dumps(log, indent=2))
            print(f'step {step} validation {score:.4f} best {best:.4f} '
                  f't {time.time() - started:.0f}s peak {memory:.1f} GiB '
                  f'density {density:.3f} zero thresholds {zero}/{model.layers}', flush=True)
        if step == args.steps:
            break
        chosen = []
        for _ in range(args.batch):
            index = int(torch.randint(0, len(targets), (1,), generator=generator))
            for _ in range(30):
                row = int(torch.randint(0, IMAGE - size + 1, (1,), generator=generator))
                col = int(torch.randint(0, IMAGE - size + 1, (1,), generator=generator))
                if float(targets[index, row:row + size, col:col + size].abs().max()) >= .05:
                    break
            chosen.append((index, row, col))
        b = padded(torch.stack([inputs[i, r:r + size, c:c + size] for i, r, c in chosen]).to(device), pad)
        target = torch.stack([targets[i, r:r + size, c:c + size] for i, r, c in chosen]).to(device)
        if args.loss == 'lasso':
            loss = model.lasso_objective(b, args.lasso_lambda)
        else:
            y = model(b)[..., pad:-pad, pad:-pad]
            keep = (slice(None), slice(margin, -margin), slice(margin, -margin))
            loss = crop_loss(y[keep], target[keep], args.loss)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()
        with torch.no_grad():
            model.theta.clamp_(min=floor)
            if model.beta is not None:
                model.beta.clamp_(0, 1)
    model.load_state_dict(torch.load(output / 'best.pt', map_location=device,
                                     weights_only=True)['state'])
    model.eval()


def comsol_scores(methods, args, scale, device, output):
    rows = []
    for name, sharp, blurred in comsol_pairs(args.comparison_run, args.data3):
        target = torch.tensor(sharp / scale, dtype=torch.float32)
        observed = torch.tensor(blurred / scale, dtype=torch.float32)
        for method, run in methods.items():
            y = predict(run, observed, args.pad, device)
            errors = relative_l2(y, target)
            for i, component in enumerate(COMPONENTS):
                span = float(target[i].max() - target[i].min())
                rmse = float(((y[i] - target[i]) ** 2).mean().sqrt())
                rows.append(dict(config=name, component=component, method=method,
                                 relative_l2=float(errors[i]), nrmse_range=rmse / span))
    with (output / 'comsol.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--comparison-run', default='runs/combined_full_200')
    p.add_argument('--data3', default='data/data3')
    p.add_argument('--stage', choices=('dictionary', 'all'), default='all')
    p.add_argument('--trial', action='store_true',
                   help='Train and score on simulated validation only; no COMSOL')
    p.add_argument('--atoms', type=int, default=32)
    p.add_argument('--atom-size', type=int, default=75)
    p.add_argument('--dict-lambda', type=float, nargs='+', default=[.01])
    p.add_argument('--dict-steps', type=int, default=600)
    p.add_argument('--dict-batch', type=int, default=8)
    p.add_argument('--dict-crop', type=int, default=256)
    p.add_argument('--dictionary', help='Reuse a saved dictionary .pt file')
    p.add_argument('--layers', type=int, default=16)
    p.add_argument('--filter-width', type=int, default=155)
    p.add_argument('--crop', type=int, default=512)
    p.add_argument('--margin', type=int, default=64)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr-relative', type=float, default=.03)
    p.add_argument('--theta-floor', type=float, default=0.0,
                   help='Lowest threshold, as a fraction of its starting value')
    p.add_argument('--skip-fista', action='store_true')
    p.add_argument('--loss', choices=('mse', 'relative', 'lasso'), default='mse')
    p.add_argument('--lasso-lambda', type=float, default=.03)
    p.add_argument('--momentum', action='store_true')
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--val-crops', type=int, default=64)
    p.add_argument('--pad', type=int, default=128)
    p.add_argument('--chunk', type=int, default=8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--z-delta', type=float, default=2.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    return p


def main():
    args = parser().parse_args()
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'args.json').write_text(json.dumps(vars(args), indent=2))
    device = torch.device(args.device)
    dataset = SimulatedDataset(args.dataset)
    print('loading simulated fields', flush=True)
    train_set = simulated(dataset, 'train', args.workers)
    val_set = simulated(dataset, 'validation', args.workers)
    print('train', train_set[0].shape, 'validation', val_set[0].shape, flush=True)
    val_crops = paired_crops(val_set, args.val_crops, args.crop, args.seed + 1)
    if args.dictionary:
        saved = torch.load(args.dictionary, weights_only=True)
        choice = dict(lam=saved['lam'], source=args.dictionary)
    else:
        trials = dictionary_stage(args, train_set[0], val_crops[0], device, output)
        if args.stage == 'dictionary':
            return
        choice = min(trials, key=lambda row: row['relative_l2'])
        saved = torch.load(output / f"dictionary_{choice['lam']:g}.pt", weights_only=True)
    g = saved['g'].to(device)
    canvas = args.crop + 2 * args.pad
    model = csc.CSCLISTA(g, DX, args.z_delta, canvas, args.pad, choice['lam'], args.layers,
                         args.filter_width, chunk=args.chunk,
                         momentum=args.momentum).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print('parameters', parameters, 'crop canvas', canvas, flush=True)

    if args.trial:
        baseline = min((error(csc.Dataless(DX, args.z_delta, mu), val_crops[1], val_crops[0],
                              args.pad, device, 8), mu) for mu in 10.0 ** np.arange(-6, 0, .5))
        print('Wiener on these validation crops', baseline, flush=True)
        train(args, model, train_set, val_crops, device, output)
        return

    methods = {}
    wiener = min((error(csc.Dataless(DX, args.z_delta, mu), val_crops[1], val_crops[0],
                        args.pad, device, 8), mu) for mu in 10.0 ** np.arange(-6, 0, .5))
    print('Wiener validation', wiener, flush=True)
    methods['wiener'] = csc.Dataless(DX, args.z_delta, wiener[1])
    solver = None
    if not args.skip_fista:
        solvers = [(error(CodedSolver(g, args.z_delta, choice['lam'] * s, 300, args.pad, True, args.chunk),
                          val_crops[1][:16], val_crops[0][:16], args.pad, device, 4),
                    choice['lam'] * s) for s in (.1, .3, 1, 3)]
        solver = min(solvers)
        print('FISTA-300 validation', solver, flush=True)
        methods['ista16'] = CodedSolver(g, args.z_delta, solver[1], 16, args.pad, False, args.chunk)
        methods['fista300'] = CodedSolver(g, args.z_delta, solver[1], 300, args.pad, True, args.chunk)

    train(args, model, train_set, val_crops, device, output)
    methods['csc_lista'] = model
    validation = {name: error(run, val_crops[1], val_crops[0], args.pad, device, 4)
                  for name, run in methods.items()}
    print('validation', validation, flush=True)
    rows = comsol_scores(methods, args, dataset.scale, device, output)
    summary = {name: float(np.mean([r['relative_l2'] for r in rows if r['method'] == name]))
               for name in methods}
    (output / 'results.json').write_text(json.dumps(dict(
        parameters=parameters, dictionary=choice, wiener=wiener, fista=solver,
        validation=validation, comsol_mean_relative_l2=summary, comsol=rows), indent=2))
    print('COMSOL mean relative L2', summary)


if __name__ == '__main__':
    main()
