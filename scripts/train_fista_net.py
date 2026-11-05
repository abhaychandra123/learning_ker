"""Train FISTA-Net on simulated fields; score once on COMSOL.

Same data, validation crops and COMSOL pairs as train_csc_lista.py. Adam at 1e-3
for the network and the scalars, as in the authors' code, with no decay.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import fista_net as fn
from simulate_fields import SimulatedDataset
from train_csc_lista import (DX, IMAGE, comsol_scores, error, padded, paired_crops,
                              simulated)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--comparison-run', default='runs/combined_full_200')
    p.add_argument('--data3', default='data/data3')
    p.add_argument('--layers', type=int, default=7)
    p.add_argument('--crop', type=int, default=512)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--pad', type=int, default=128)
    p.add_argument('--val-crops', type=int, default=64)
    p.add_argument('--evals', type=int, default=30)
    p.add_argument('--start', choices=('paper', 'code'), default='paper')
    p.add_argument('--checkpoint', help='Score this best.pt and skip training')
    p.add_argument('--z-delta', type=float, default=2.5)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'args.json').write_text(json.dumps(vars(args), indent=2))
    device = torch.device('cuda')
    dataset = SimulatedDataset(args.dataset)
    print('loading simulated fields', flush=True)
    targets, inputs = simulated(dataset, 'train', args.workers)
    val = simulated(dataset, 'validation', args.workers)
    val_crops = paired_crops(val, args.val_crops, args.crop, args.seed + 1)
    model = fn.FISTANet(DX, args.z_delta, args.layers, start=args.start).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                         weights_only=True)['state'])
    parameters = sum(q.numel() for q in model.parameters())
    print('parameters', parameters, flush=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    run = lambda b: model(b)[0]
    pad, size = args.pad, args.crop
    steps = 0 if args.checkpoint else args.epochs * len(targets) // args.batch
    every = max(steps // args.evals, 1)
    log, best, started = [], float('inf'), time.time()
    for step in range(steps + 1):
        if step % every == 0 or step == steps:
            model.eval()
            score = error(run, val_crops[1], val_crops[0], pad, device, batch=4)
            model.train()
            if score < best:
                best = score
                torch.save(dict(state=model.state_dict(), step=step, score=score),
                           output / 'best.pt')
            mu, theta, rho = [float(v) for v in model.schedules(args.layers)]
            memory = torch.cuda.max_memory_allocated() / 2 ** 30
            log.append(dict(step=step, validation=score, seconds=time.time() - started,
                            peak_gib=memory, mu=mu, theta=theta, rho=rho))
            (output / 'training.json').write_text(json.dumps(log, indent=2))
            print(f'step {step}/{steps} validation {score:.4f} best {best:.4f} '
                  f't {time.time() - started:.0f}s peak {memory:.1f} GiB '
                  f'mu {mu:.4f} theta {theta:.4f} rho {rho:.3f}', flush=True)
        if step == steps:
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
        field, symmetry, sparsity = model(b)
        loss = fn.total_loss(field[..., pad:-pad, pad:-pad], target, symmetry, sparsity)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    model.load_state_dict(torch.load(output / 'best.pt', map_location=device,
                                     weights_only=True)['state'])
    model.eval()
    methods = dict(fista_net=run)
    validation = error(run, val_crops[1], val_crops[0], pad, device, batch=4)
    rows = comsol_scores(methods, args, dataset.scale, device, output)
    mean = float(np.mean([r['relative_l2'] for r in rows]))
    (output / 'results.json').write_text(json.dumps(dict(
        parameters=parameters, validation=validation, comsol_mean_relative_l2=mean,
        comsol=rows), indent=2))
    print('validation', validation)
    print('COMSOL mean relative L2', mean)


if __name__ == '__main__':
    main()
