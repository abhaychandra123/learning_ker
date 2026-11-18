"""Training crops for MAGIC-UNet from the current-layout simulator.

Follows the paper's recipe (Reed et al., Phys. Rev. Applied 23, 034035 (2025): Sec. V A, Table V, App. A) where it applies at our scale:

- random current layouts (here: the simulator's spirals, meanders and mirrored arrays,
  each drawn with a random quarter turn and mirror, which gives the paper's
  8 dihedral augmentations physically, at the source);
- standoff drawn per image from a normal distribution, mean 3 um, standard deviation
  0.6 um (the paper's 50 +- 10 um, same 20% spread), truncated at 3 standard deviations;
  the network is never told the standoff (run 3, --z-uniform: uniform over a range instead);
- about 2% of crops show fields from wires that lie outside the crop;
- clean fields are stored; the trainer adds fresh white Gaussian noise every epoch;
- per-image scale alpha = max |B|, and a global output scale beta stored in the manifest.

Adaptations (see README_MAGIC_UNET.md): fields and currents are simulated at 0.1 um,
then area-averaged 2 x 2 to 0.2 um pixels (like camera binning), and 64 x 64 crops
(12.8 um) are cut from the 140 um layouts. The target is the equivalent
divergence-free sheet current at the zero-offset plane, J_eff = J exp(-k d), so that B
at any height h is forward(J_eff, h). The noise level is the simulator's own setting
(1-2.5% of the mean component range), applied to each crop's own range, so that sigma / max|B|
stays at a few per cent as in the paper, where sigma_n is relative to each image's own max|B|.
(Run 1 took the range of the whole 140 um layout. Crops far from every wire then had sigma
up to 165 times their own max|B|, which corrupted the BatchNorm running statistics.)
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import shutil

import numpy as np

from magic_unet import solenoidal_sheet
from simulate_fields import DEFAULTS, current_maps, draw_recipe, fields_from_current, sample_seed

KIND = 'magic_unet_crops'
CURRENT_FRACTION = 0.05     # a crop "contains current" if max|J| >= 5% of the layout's max


def binned(a, factor):
    """Area average over factor x factor pixels; a: [..., H, W]."""
    if factor == 1:
        return a
    *lead, h, w = a.shape
    return a.reshape(*lead, h // factor, factor, w // factor, factor).mean((-3, -1))


def crop_sigma(b, percent):
    """White-noise sigma (uT) of one crop b [3, H, W]: percent of its own mean component range."""
    return percent / 100 * float(np.mean(b.max((1, 2)) - b.min((1, 2))))


def layout_crops(job):
    """Clean crops of one layout: B [n,3,c,c] uT, J [n,2,c,c] A/m, z [n], sigma [n] uT."""
    label, seed, s, cfg = job
    recipe = draw_recipe(seed, label, s)
    rng = np.random.default_rng(sample_seed(seed, label + ':magic'))
    ny, nx, pad, dx = s['ny'], s['nx'], s['pad'], s['dx']
    shape = (ny + 2 * pad, nx + 2 * pad)
    kx, ky = current_maps(recipe['paths_um'], recipe['width_um'], s['current_a'], shape, dx, pad)
    d = recipe['offset_um']
    crop = (slice(pad, pad + ny), slice(pad, pad + nx))
    jx, jy = solenoidal_sheet(kx, ky, dx, depth=d)
    j = binned(np.stack([jx[crop], jy[crop]]), cfg['bin'])
    side = j.shape[-1]
    jmag = np.hypot(j[0], j[1])
    jmax = jmag.max()
    zs = []
    while len(zs) < cfg['standoffs']:
        if cfg.get('z_uniform'):                  # run 3: any standoff in [lo, hi]
            zs.append(float(rng.uniform(*cfg['z_uniform'])))
            continue
        z = rng.normal(cfg['z_mean'], cfg['z_std'])
        if abs(z - cfg['z_mean']) <= cfg['z_trunc'] * cfg['z_std']:
            zs.append(float(z))
    fields = fields_from_current(kx, ky, dx, [z + d for z in zs])
    size = cfg['crop']
    out_b, out_j, out_z, out_sigma = [], [], [], []
    for z, f in zip(zs, fields):
        b = binned(np.stack([c[crop] for c in f]), cfg['bin'])
        for _ in range(cfg['crops_per_standoff']):
            empty_ok = rng.random() < cfg['empty_fraction']
            for _ in range(50):
                r, c = (int(v) for v in rng.integers(0, side - size + 1, 2))
                if empty_ok or jmag[r:r + size, c:c + size].max() >= CURRENT_FRACTION * jmax:
                    break
            out_b.append(b[:, r:r + size, c:c + size])
            out_j.append(j[:, r:r + size, c:c + size])
            out_z.append(z)
            out_sigma.append(crop_sigma(out_b[-1], float(rng.choice(s['noise_percent']))))
    return (np.stack(out_b), np.stack(out_j), np.asarray(out_z), np.asarray(out_sigma),
            dict(label=label, family=recipe['family'], width_um=recipe['width_um'], offset_um=d))


def build_split(output, split, count, seed, settings, cfg, workers, dtype):
    per_layout = cfg['standoffs'] * cfg['crops_per_standoff']
    if count % per_layout:
        raise SystemExit(f'{split}: {count} crops is not a multiple of {per_layout} per layout')
    layouts = count // per_layout
    size = cfg['crop']
    from numpy.lib.format import open_memmap
    arrays = dict(
        B=open_memmap(output / f'{split}_B.npy', 'w+', dtype, (count, 3, size, size)),
        J=open_memmap(output / f'{split}_J.npy', 'w+', dtype, (count, 2, size, size)))
    z = np.zeros(count, np.float32)
    sigma = np.zeros(count, np.float32)
    layout_index = np.zeros(count, np.int32)
    info = []
    jobs = [(f'magic_{split}_{i:06d}', seed, settings, cfg) for i in range(layouts)]
    with ProcessPoolExecutor(workers) as pool:
        for i, (b, j, zz, ss, meta) in enumerate(pool.map(layout_crops, jobs, chunksize=1)):
            sl = slice(i * per_layout, (i + 1) * per_layout)
            arrays['B'][sl] = b.astype(dtype)
            arrays['J'][sl] = j.astype(dtype)
            z[sl], sigma[sl], layout_index[sl] = zz, ss, i
            info.append(meta)
            if (i + 1) % 50 == 0 or i + 1 == layouts:
                print(f'{split}: {i + 1}/{layouts} layouts', flush=True)
    for a in arrays.values():
        a.flush()
    np.save(output / f'{split}_z.npy', z)
    np.save(output / f'{split}_sigma.npy', sigma)
    np.save(output / f'{split}_layout.npy', layout_index)
    return info


def resigma(data, seed=0):
    """Rewrite stored {split}_sigma.npy with per-crop noise (crop_sigma), keeping the crops.

    Used once to repair run 1's data: B and J are unchanged; the noise percent is drawn
    afresh per crop from the simulator's set, because run 1 did not store it.
    """
    data = Path(data)
    manifest = json.loads((data / 'manifest.json').read_text())
    choices = manifest['settings']['noise_percent']
    for split in ('train', 'validation'):
        b = np.load(data / f'{split}_B.npy', mmap_mode='r')
        rng = np.random.default_rng(sample_seed(seed, f'resigma:{split}'))
        percent = rng.choice(choices, len(b)).astype(np.float32)
        sigma = np.zeros(len(b), np.float32)
        for s in range(0, len(b), 4096):
            blk = np.asarray(b[s:s + 4096], np.float32)
            rng_ = blk.max((2, 3)) - blk.min((2, 3))
            sigma[s:s + 4096] = percent[s:s + 4096] / 100 * rng_.mean(1)
        old = data / f'{split}_sigma.npy'
        if not (data / f'{split}_sigma_run1.npy').exists():
            old.rename(data / f'{split}_sigma_run1.npy')
        np.save(old, sigma)
        np.save(data / f'{split}_noise_percent.npy', percent)
        a = np.sqrt((np.asarray(b[:4096], np.float32) ** 2).sum(1)).max((1, 2))
        print(split, 'sigma/alpha max over first 4096:', float((sigma[:4096] / np.maximum(a, 1e-30)).max()))
    manifest['noise'] = dict(rule='percent of each crop\'s own mean component range',
                             resigma_seed=seed, code_sha256=code_hashes()['magic_data.py'])
    (data / 'manifest.json').write_text(json.dumps(manifest, indent=1))


def output_scale(output, split='train', sample=4096):
    """beta: a power of ten that makes max|J / alpha| of order one (paper: beta ~ 1e8)."""
    b = np.load(output / f'{split}_B.npy', mmap_mode='r')
    j = np.load(output / f'{split}_J.npy', mmap_mode='r')
    idx = np.linspace(0, len(b) - 1, min(sample, len(b))).astype(int)
    a = np.sqrt((b[idx].astype(np.float64) ** 2).sum(1)).max((1, 2))
    m = np.sqrt((j[idx].astype(np.float64) ** 2).sum(1)).max((1, 2))
    ratio = np.median(m[a > 0] / a[a > 0])
    return float(10.0 ** np.round(-np.log10(ratio)))


def code_hashes():
    here = Path(__file__).parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in ('magic_data.py', 'magic_unet.py', 'simulate_fields.py')}


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--output', help='New folder')
    p.add_argument('--resigma', metavar='DATA', help='Only rewrite the noise levels of an existing dataset')
    p.add_argument('--train-crops', type=int, default=153600, help='Paper: 153,600 pairs')
    p.add_argument('--val-crops', type=int, default=4096, help='Paper: 4,096 pairs')
    p.add_argument('--crop', type=int, default=64, help='Crop side in binned pixels (paper: 64)')
    p.add_argument('--bin', type=int, default=2, help='Area-average factor from the 0.1 um grid')
    p.add_argument('--standoffs', type=int, default=4, help='Standoffs drawn per layout')
    p.add_argument('--crops-per-standoff', type=int, default=16)
    p.add_argument('--z-mean', type=float, default=3.0)
    p.add_argument('--z-std', type=float, default=0.6)
    p.add_argument('--z-trunc', type=float, default=3.0, help='Truncation in standard deviations')
    p.add_argument('--z-uniform', type=float, nargs=2, metavar=('LO', 'HI'),
                   help='Draw the standoff uniformly in [LO, HI] um instead of the normal')
    p.add_argument('--empty-fraction', type=float, default=0.02)
    p.add_argument('--dtype', choices=('float16', 'float32'), default='float16')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=8)
    return p


def main():
    args = parser().parse_args()
    if args.resigma:
        return resigma(args.resigma, args.seed)
    if not args.output:
        raise SystemExit('--output is required')
    output = Path(args.output)
    settings = dict(DEFAULTS)
    cfg = dict(crop=args.crop, bin=args.bin, standoffs=args.standoffs, crops_per_standoff=args.crops_per_standoff,
               z_mean=args.z_mean, z_std=args.z_std, z_trunc=args.z_trunc, z_uniform=args.z_uniform,
               empty_fraction=args.empty_fraction)
    if args.crop > settings['ny'] // args.bin or args.crop % 8 or settings['ny'] % args.bin:
        raise SystemExit('crop must fit the image and be divisible by 8 (three poolings)')
    item = np.dtype(args.dtype).itemsize
    need = (args.train_crops + args.val_crops) * 5 * args.crop ** 2 * item
    output.mkdir(parents=True, exist_ok=False)
    free = shutil.disk_usage(output).free
    print(f'needs {need / 1e9:.1f} GB, {free / 1e9:.1f} GB free', flush=True)
    if need > 0.9 * free:
        output.rmdir()
        raise SystemExit('not enough disk space')
    layouts = {}
    for split, count in (('train', args.train_crops), ('validation', args.val_crops)):
        layouts[split] = build_split(output, split, count, args.seed, settings, cfg,
                                     args.workers, args.dtype)
    beta = output_scale(output)
    manifest = dict(kind=KIND, seed=args.seed, settings=settings, crops=cfg, dtype=args.dtype,
                    counts=dict(train=args.train_crops, validation=args.val_crops),
                    beta=beta, layouts=layouts,
                    noise=dict(rule='percent of each crop\'s own mean component range'), target='equivalent sheet current J exp(-k d), A/m',
                    code_sha256=code_hashes(), numpy=np.__version__,
                    python=platform.python_version())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print('beta', beta, 'written', output, flush=True)


if __name__ == '__main__':
    main()
