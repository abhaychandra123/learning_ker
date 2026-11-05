"""Score FISTA on the CSC-LISTA dictionary at chosen iteration counts.

Each depth picks lambda the way train_csc_lista.py does for FISTA-300: on the first
16 simulated validation crops, by default from 0.1, 0.3, 1 and 3 times the dictionary's
lambda. COMSOL pairs are read afterwards and only for the final scores.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from simulate_fields import SimulatedDataset
from train_csc_lista import CodedSolver, comsol_scores, error, paired_crops, simulated


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True)
    p.add_argument('--dictionary', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--iterations', type=int, nargs='+', default=[32, 64, 300])
    p.add_argument('--lambda-scales', type=float, nargs='+', default=[.1, .3, 1, 3])
    p.add_argument('--comparison-run', default='runs/combined_full_200')
    p.add_argument('--data3', default='data/data3')
    p.add_argument('--val-crops', type=int, default=64)
    p.add_argument('--crop', type=int, default=512)
    p.add_argument('--pad', type=int, default=128)
    p.add_argument('--chunk', type=int, default=8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--z-delta', type=float, default=2.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'args.json').write_text(json.dumps(vars(args), indent=2))
    device = torch.device(args.device)
    dataset = SimulatedDataset(args.dataset)
    print('loading simulated fields', flush=True)
    val_crops = paired_crops(simulated(dataset, 'validation', args.workers),
                             args.val_crops, args.crop, args.seed + 1)
    saved = torch.load(args.dictionary, weights_only=True)
    g = saved['g'].to(device)

    methods, choices, validation = {}, {}, {}
    for steps in args.iterations:
        solvers = []
        for scale in args.lambda_scales:
            lam = saved['lam'] * scale
            solver = CodedSolver(g, args.z_delta, lam, steps, args.pad, True, args.chunk)
            score = error(solver, val_crops[1][:16], val_crops[0][:16], args.pad, device, 4)
            print(f'FISTA-{steps} lambda {lam:g} on 16 crops {score:.4f}', flush=True)
            solvers.append((score, lam))
        lam = min(solvers)[1]
        name = f'fista{steps}'
        methods[name] = CodedSolver(g, args.z_delta, lam, steps, args.pad, True, args.chunk)
        choices[name] = dict(lam=lam, trials=solvers)
        validation[name] = error(methods[name], val_crops[1], val_crops[0], args.pad, device, 4)
        print(name, 'lambda', lam, 'validation', validation[name], flush=True)

    rows = comsol_scores(methods, args, dataset.scale, device, output)
    summary = {name: float(np.mean([r['relative_l2'] for r in rows if r['method'] == name]))
               for name in methods}
    (output / 'results.json').write_text(json.dumps(dict(
        choices=choices, validation=validation, comsol_mean_relative_l2=summary, comsol=rows),
        indent=2))
    print('COMSOL mean relative L2', summary, flush=True)


if __name__ == '__main__':
    main()
