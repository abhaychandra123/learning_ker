"""Train MAGIC-UNet on simulated crops, then score it on simulated full images and COMSOL.

Training follows Table IV and Supplement D of Reed et al. 2024 (arXiv 2407.14553):
RMSprop (Keras defaults: rho 0.9, epsilon 1e-7), learning rate 1e-4, 50 epochs,
mini-batch 32, mean squared error averaged over the Jx and Jy outputs, BatchNorm in
training mode, fresh additive Gaussian noise on the inputs every epoch, inputs and
outputs divided by the per-image alpha = max |B| and outputs multiplied by a global beta.
The model saved after the last epoch is the result; there is no checkpoint selection.

Evaluation (never used for any choice except the tiling stride, which is picked on
simulated validation): full 1400 x 1400 images at 0.1 um are area-averaged to the
training pitch, tiled with overlapping 64 x 64 tiles (paper Fig. 5c), the predicted
sheet current is replicated back to 0.1 um pixels and propagated to the target height
by thin-sheet Biot-Savart, and relative L2 is computed per component at 0.1 um.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from magic_data import binned
from magic_unet import (MAGICUNet, alpha, field_from_sheet, predict_tiled, psnr, relative_l2,
                        solenoidal_sheet, ssim)
from simulate_fields import COMPONENTS, SimulatedDataset, current_maps, synthesise

DX = 0.1


class Crops(torch.utils.data.Dataset):
    """Stored clean crops (memory-mapped) with their noise level in uT."""

    def __init__(self, root, split):
        root = Path(root)
        self.b = np.load(root / f'{split}_B.npy', mmap_mode='r')
        self.j = np.load(root / f'{split}_J.npy', mmap_mode='r')
        self.sigma = np.load(root / f'{split}_sigma.npy')

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        return (torch.from_numpy(np.asarray(self.b[i], np.float32)),
                torch.from_numpy(np.asarray(self.j[i], np.float32)),
                torch.tensor(self.sigma[i], dtype=torch.float32))


def unit_noise(shape, generator, corr_um=0.0, dx=DX):
    """White unit Gaussian noise, or optionally exponentially correlated (ablation only)."""
    w = torch.randn(shape, generator=generator)
    if corr_um <= 0:
        return w
    h, wd = shape[-2:]
    fy = torch.fft.fftfreq(h, dx)[:, None]
    fx = torch.fft.rfftfreq(wd, dx)[None, :]
    amp = (1 + (2 * np.pi * corr_um) ** 2 * (fx ** 2 + fy ** 2)) ** -0.75
    w = torch.fft.irfft2(torch.fft.rfft2(w) * amp, s=(h, wd))
    w = w - w.mean((-2, -1), keepdim=True)
    return w / w.std((-2, -1), keepdim=True)


def batch_tensors(b, j, sigma, beta, generator, corr_um, device):
    """Normalised noisy input and scaled target, as in the authors' pipeline."""
    a = alpha(b).clamp_min(1e-30)
    x = b / a[:, None, None, None]
    x = x + (sigma / a)[:, None, None, None] * unit_noise(x.shape, generator, corr_um)
    y = j / a[:, None, None, None] * beta
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def validation_loss(model, loader, beta, seed, corr_um, device):
    model.eval()
    g = torch.Generator().manual_seed(seed)       # same noise every epoch
    total, count = 0.0, 0
    for b, j, sigma in loader:
        x, y = batch_tensors(b, j, sigma, beta, g, corr_um, device)
        total += float(((model(x) - y) ** 2).mean()) * len(b)
        count += len(b)
    return total / count


def train(args, data, output, device):
    manifest = json.loads((Path(data) / 'manifest.json').read_text())
    beta = manifest['beta']
    train_set, val_set = Crops(data, 'train'), Crops(data, 'validation')
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True,
                                         drop_last=True, num_workers=args.workers,
                                         pin_memory=device.type == 'cuda',
                                         persistent_workers=args.workers > 0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch, shuffle=False,
                                             num_workers=args.workers)
    model = MAGICUNet().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1e-7,
                                    momentum=0.0, centered=False)
    start, history = 0, []
    latest = output / 'latest.pt'
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        start, history = state['epoch'], state['history']
        print('resumed after epoch', start, flush=True)
    noise = torch.Generator().manual_seed(args.seed * 1000 + start)
    for epoch in range(start, args.epochs):
        model.train()
        t0, total, steps = time.time(), 0.0, 0
        for b, j, sigma in loader:
            x, y = batch_tensors(b, j, sigma, beta, noise, args.noise_corr_um, device)
            loss = ((model(x) - y) ** 2).mean()        # = (MSE_Jx + MSE_Jy) / 2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss)
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        val = validation_loss(model, val_loader, beta, args.seed + 7, args.noise_corr_um, device)
        row = dict(epoch=epoch + 1, train_loss=total / steps, val_loss=val, seconds=time.time() - t0)
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch + 1,
                        history=history, torch_rng=torch.get_rng_state(), beta=beta),
                   output / 'latest.tmp')
        (output / 'latest.tmp').replace(latest)
        with (output / 'history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
    torch.save(dict(model=model.state_dict(), beta=beta, epochs=args.epochs, data=str(data),
                    bin=manifest['crops']['bin'], crop=manifest['crops']['crop']),
               output / 'final.pt')
    return model, beta, manifest['crops']['bin']


# ----------------------------------------------------------------- evaluation

def true_sheet(recipe, settings):
    ny, nx, pad = settings['ny'], settings['nx'], settings['pad']
    kx, ky = current_maps(recipe['paths_um'], recipe['width_um'], settings['current_a'],
                          (ny + 2 * pad, nx + 2 * pad), settings['dx'], pad)
    jx, jy = solenoidal_sheet(kx, ky, settings['dx'], depth=recipe['offset_um'])
    crop = (slice(pad, pad + ny), slice(pad, pad + nx))
    return np.stack([jx[crop], jy[crop]])


def reconstruct(model, beta, b, height, factor, args, stride, device):
    """Field [3, H, W] at `height` and sheet current at 0.1 um from a 0.1 um field b."""
    j = predict_tiled(model, binned(np.asarray(b, np.float64), factor), beta, args.tile, stride,
                      args.infer_batch, device)
    j = np.kron(j, np.ones((1, factor, factor)))
    return field_from_sheet(j, height, DX), j


def score_simulated(model, beta, factor, sim, recipes, stride, args, device, rows=None):
    errors = []
    for recipe in recipes:
        item = synthesise(recipe, sim.settings)
        b, j = reconstruct(model, beta, item['input'], sim.settings['z_target'], factor, args,
                           stride, device)
        e = [relative_l2(b[c], item['target'][c]) for c in range(3)]
        errors.append(np.mean(e))
        if rows is not None:
            truth = binned(true_sheet(recipe, sim.settings), factor)   # J at the network's pitch
            pred = j[:, ::factor, ::factor]
            row = dict(layout=recipe['id'], family=recipe['family'], stride=stride,
                       **{f'relative_l2_{c}': v for c, v in zip(COMPONENTS, e)})
            for i, name in enumerate(('Jx', 'Jy')):
                row[f'ssim_{name}'] = ssim(pred[i], truth[i])
                row[f'psnr_{name}'] = psnr(pred[i], truth[i])
                row[f'rmse_{name}'] = float(np.sqrt(np.mean((pred[i] - truth[i]) ** 2)))
                row[f'relative_l2_{name}'] = relative_l2(pred[i], truth[i])
            rows.append(row)
    return float(np.mean(errors))


def comsol(model, beta, factor, stride, args, device, output):
    from wiener_oracle import discover, load
    data3 = args.data3 if args.data3 and Path(args.data3).exists() else None
    rows, saved = [], {}
    lo, hi = args.z_range or (args.z_mean - 3 * args.z_std, args.z_mean + 3 * args.z_std)
    for pair in discover(args.comparison_run, data3):
        sharp, blurred = load(pair.sharp), load(pair.blurred)
        b, j = reconstruct(model, beta, blurred, pair.z_tg, factor, args, stride, device)
        for c, name in enumerate(COMPONENTS):
            rows.append(dict(config=pair.name, z_input=pair.z_in, z_target=pair.z_tg, component=name,
                             in_training_spread=bool(lo <= pair.z_in <= hi),
                             relative_l2=relative_l2(b[c], sharp[c])))
        if pair.z_in == 3.0 and pair.z_tg == 0.5:
            saved[pair.name] = dict(J=j[:, ::factor, ::factor].astype(np.float32),
                                    B=b.astype(np.float32))
    with (output / 'comsol.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / 'comsol_3_to_05_predictions.npz',
                        **{f'{k}_{a}': v[a] for k, v in saved.items() for a in v})
    return rows


def evaluate(model, beta, factor, args, device, output):
    sim = SimulatedDataset(args.sim_dataset)
    recipes = sim.recipes('validation')[:args.val_layouts]
    pick = recipes[:args.stride_layouts]
    strides = {s: score_simulated(model, beta, factor, sim, pick, s, args, device)
               for s in args.strides}
    stride = min(strides, key=strides.get)
    print('stride on simulated validation', strides, '->', stride, flush=True)
    rows = []
    sim_mean = score_simulated(model, beta, factor, sim, recipes, stride, args, device, rows)
    with (output / 'sim_validation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(stride_trials=strides, stride=stride,
                  sim_validation_relative_l2=sim_mean,
                  sim_validation_layouts=len(recipes),
                  sim_J={k: float(np.mean([r[k] for r in rows]))
                         for k in rows[0] if k.startswith(('ssim', 'psnr', 'rmse', 'relative_l2_J'))})
    if not args.no_comsol:
        crow = comsol(model, beta, factor, stride, args, device, output)
        groups = {}
        for r in crow:
            groups.setdefault((r['z_input'], r['z_target']), []).append(r)
        result['comsol_mean_relative_l2'] = {
            f'{zi:g}->{zt:g}': dict(mean=float(np.mean([r['relative_l2'] for r in g])),
                                   configs=sorted({r['config'] for r in g}),
                                   in_training_spread=g[0]['in_training_spread'])
            for (zi, zt), g in sorted(groups.items())}
    (output / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data', help='Folder made by magic_data.py')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--noise-corr-um', type=float, default=0.0,
                   help='Correlated training noise; ablation only (paper: white noise)')
    p.add_argument('--max-steps', type=int, default=0, help='Cap steps per epoch (smoke tests)')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--evaluate', metavar='CHECKPOINT', help='Skip training; score this checkpoint')
    p.add_argument('--sim-dataset', default='runs/simulated_3_to_05_v3')
    p.add_argument('--val-layouts', type=int, default=50)
    p.add_argument('--stride-layouts', type=int, default=10)
    p.add_argument('--strides', type=int, nargs='+', default=[16, 32, 48])
    p.add_argument('--tile', type=int, default=64)
    p.add_argument('--infer-batch', type=int, default=32)
    p.add_argument('--comparison-run', default='runs/combined_full_200')
    p.add_argument('--data3', default='data/data3')
    p.add_argument('--no-comsol', action='store_true')
    p.add_argument('--z-mean', type=float, default=3.0, help='Training standoff mean (labels only)')
    p.add_argument('--z-std', type=float, default=0.6, help='Training standoff std (labels only)')
    p.add_argument('--z-range', type=float, nargs=2, metavar=('LO', 'HI'),
                   help='Uniform training standoff range (labels only); overrides mean +- 3 std')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    return p


def main():
    args = parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    if args.evaluate:
        output.mkdir(parents=True, exist_ok=False)
        state = torch.load(args.evaluate, map_location=device, weights_only=False)
        model = MAGICUNet().to(device)
        model.load_state_dict(state['model'])
        beta, factor = state['beta'], state.get('bin', 2)
    else:
        if not args.data:
            raise SystemExit('--data is required for training')
        resuming = (output / 'latest.pt').exists()
        output.mkdir(parents=True, exist_ok=resuming)
        if not resuming:
            (output / 'args.json').write_text(json.dumps(vars(args), indent=2))
        model, beta, factor = train(args, args.data, output, device)
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    evaluate(model, beta, factor, args, device, output)


if __name__ == '__main__':
    main()
