"""Simulated QDM training pairs computed from random wire current layouts.

One recipe holds one or more closed current paths. Bx, By and Bz at both heights
all derive from that single current map, using thin-sheet Biot-Savart relations:

    E(k, z) = (mu0 / 2) * exp(-k * zr) * exp(-sqrt(k^2 + k0^2) * (z - zr))
    Bx(k) = E * Kx(k)
    By(k) = E * Ky(k)
    Bz(k) = E * 1j * (ky * Kx(k) - kx * Ky(k)) / k

Here zr is the 0.5 um reference plane plus the source offset d. The default k0 = 0
gives ideal free-space decay, exp(-k * (z + d)). A nonzero k0 is empirical and
mimics data3 COMSOL exports; it breaks harmonicity, so it is not real physics.

Kx and Ky are sheet current density along array columns and rows. Those
component labels are the COMSOL export's frame, which is rotated 90 degrees
from the image axes; see README_SIMULATED_FIELDS.md. Fields are in microtesla
and lengths in micrometres. NumPy only. COMSOL data is never read here.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from synthetic_data import digest, write_json

COMPONENTS = ('Bx', 'By', 'Bz')
KIND = 'simulated_current_layout'
MU0_UT = 4e-7 * np.pi * 1e6     # uT m / A
IMAGE_UM = 140.0
OUTSIDE_UM = 35.0               # return wiring sits this far outside the image
MIN_GAP_UM = 1.0

DEFAULTS = dict(
    nx=1400, ny=1400, dx=0.1, pad=700,
    z_input=3.0, z_target=0.5,
    current_a=600e-6,
    width_um=(1.6, 3.2),
    offset_um=(0.0, 0.2),
    noise_percent=(1.0, 1.5, 2.0, 2.5),
    noise_corr_um=0.38,     # exponential speckle correlation from mirrored coils
    decay_k0=0.0,           # rad/um; nonzero only mimics COMSOL, not physics
    decay_ref_um=0.5,
    psf_sigma_um=0.0,
)


def sample_seed(seed, label):
    text = f'{seed}:{label}'.encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'little')


def spiral_path(rng, width, box=IMAGE_UM):
    """Two-arm rectangular spiral: legs shrink inwards, turn, grow back out."""
    px = width + rng.uniform(2.0, 8.0)
    py = px if rng.random() < 0.7 else width + rng.uniform(2.0, 8.0)
    legs = int(rng.integers(4, 10))
    x, y = 0.0, 0.0
    points = [(x, y)]
    moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for i, n in enumerate(list(range(legs, 0, -1)) + [1]):
        dx, dy = moves[i % 4]
        x, y = x + dx * n * px, y + dy * n * py
        points.append((x, y))
    turn = legs % 4
    for i, n in enumerate(range(1, legs - 1)):
        dx, dy = moves[(turn - 1 - i) % 4]
        x, y = x + dx * n * px, y + dy * n * py
        points.append((x, y))
    points = np.array(points)
    low, high = points.min(0), points.max(0)
    size = high - low
    if size.max() > box - 10:
        return None
    origin = rng.uniform(5.0, box - 5.0 - size + 1e-9) - low
    points = points + origin
    # The first leg runs right from the inlet lead and the last point exits
    # left, so both leads meet the same outside rail.
    start, end = points[0], points[-1]
    rail = -OUTSIDE_UM
    return [(rail, end[1]), (rail, start[1])] + [tuple(p) for p in points] + [
        (rail, end[1])]


def meander_path(rng, width):
    """Serpentine with uneven teeth, closed by a rail outside the image."""
    ax0, bx1 = rng.uniform(2.0, 25.0), rng.uniform(115.0, 138.0)
    y_low, y_high = 5.0, IMAGE_UM - 5.0
    x = ax0
    y = rng.uniform(y_low, y_high)
    points = [(-OUTSIDE_UM, y), (x, y)]
    high = True
    while True:
        target = (rng.uniform(y + 8, y_high) if high and y + 8 < y_high
                  else rng.uniform(y_low, y - 8) if y - 8 > y_low
                  else None)
        if target is None:
            high = not high
            continue
        y = target
        points.append((x, y))
        step = rng.uniform(width + 3.0, 45.0)
        if x + step > bx1:
            break
        x += step
        points.append((x, y))
        high = not high
    last = points[-1][1]
    rail = -OUTSIDE_UM
    points += [(IMAGE_UM + OUTSIDE_UM, last),
               (IMAGE_UM + OUTSIDE_UM, rail), (rail, rail)]
    points.append(points[0])
    return points


def array_paths(rng, width):
    """Two to four mirror copies of one spiral in the image quadrants."""
    path = None
    while path is None:
        path = spiral_path(rng, width, box=IMAGE_UM / 2 - 5)
    base = np.array(path)
    count = int(rng.integers(2, 5))
    paths = []
    for flip_rows, flip_cols in ((0, 0), (1, 0), (0, 1), (1, 1))[:count]:
        copy = base.copy()
        if flip_rows:
            copy[:, 1] = IMAGE_UM - copy[:, 1]
        if flip_cols:
            copy[:, 0] = IMAGE_UM - copy[:, 0]
        paths.append(copy)
    return paths


def transform(points, turns, mirror):
    """Quarter turn and mirror about the image centre. Keeps paths Manhattan."""
    p = np.asarray(points, dtype=np.float64) - IMAGE_UM / 2
    if mirror:
        p[:, 1] *= -1
    for _ in range(turns):
        p = np.column_stack((-p[:, 1], p[:, 0]))
    return p + IMAGE_UM / 2


def merge_collinear(points):
    """Drop joints inside straight runs; drawn ends would double the current."""
    p = np.asarray(points, dtype=np.float64)
    step = np.sign(np.diff(p, axis=0))
    keep = np.ones(len(p), dtype=bool)
    keep[1:-1] = (step[:-1] != step[1:]).any(1)
    return p[keep]


def segments_clear(points, clearance):
    """True when non-adjacent axis-aligned segments keep `clearance` apart."""
    p = np.asarray(points)
    a, b = p[:-1], p[1:]
    low, high = np.minimum(a, b), np.maximum(a, b)
    n = len(a)
    closed = np.array_equal(p[0], p[-1])
    if np.any((a != b).sum(1) != 1):
        return False
    step = np.sign(b - a)
    if np.any((step[:-1] * step[1:]).sum(1) < 0):
        return False
    for i in range(n):
        for j in range(i + 2, n):
            if closed and i == 0 and j == n - 1:
                continue
            gap = np.maximum(low[j] - high[i], low[i] - high[j]).max()
            if gap < clearance:
                return False
    return True


def paths_clear(paths, clearance):
    """Each path is clear of itself, and every pair of paths stays apart."""
    if not all(segments_clear(p, clearance) for p in paths):
        return False
    boxes = []
    for p in paths:
        p = np.asarray(p)
        boxes.append((np.minimum(p[:-1], p[1:]), np.maximum(p[:-1], p[1:])))
    for i, (low_a, high_a) in enumerate(boxes):
        for low_b, high_b in boxes[i + 1:]:
            gap = np.maximum(low_b[None] - high_a[:, None],
                             low_a[:, None] - high_b[None]).max(2)
            if gap.min() < clearance:
                return False
    return True


def draw_recipe(seed, label, settings):
    """Deterministic recipe: reject and redraw until the layout is valid."""
    rng = np.random.default_rng(sample_seed(seed, label))
    family = str(rng.choice(('spiral', 'meander', 'array'), p=(.35, .35, .3)))
    width = float(rng.uniform(*settings['width_um']))
    while True:
        if family == 'array':
            paths = array_paths(rng, width)
        else:
            make = spiral_path if family == 'spiral' else meander_path
            paths = [make(rng, width)]
            if paths[0] is None:
                continue
        turns, mirror = int(rng.integers(4)), bool(rng.integers(2))
        paths = [merge_collinear(transform(p, turns, mirror)) for p in paths]
        if paths_clear(paths, width + MIN_GAP_UM):
            break
    if rng.random() < 0.5:
        paths = [p[::-1] for p in paths]
    return dict(
        id=label, family=family, width_um=width,
        offset_um=float(rng.uniform(*settings['offset_um'])),
        noise_percent=float(rng.choice(settings['noise_percent'])),
        noise_seed=sample_seed(seed, f'{label}:noise'),
        paths_um=[[[round(float(x), 4), round(float(y), 4)] for x, y in p]
                  for p in paths],
    )


def coverage(low, high, size):
    """Pixel-centred box coverage. Returns the first index and the weights."""
    first = max(int(np.floor(low + 0.5)), 0)
    last = min(int(np.ceil(high + 0.5)), size)
    i = np.arange(first, last)
    return first, np.clip(np.minimum(i + .5, high) - np.maximum(i - .5, low), 0, 1)


def rasterize(path, width, current, shape, dx, pad):
    """Sheet current [A/m] on the padded canvas. Pixel i sits at i * dx."""
    ny, nx = shape
    kx, ky = np.zeros(shape), np.zeros(shape)
    half = width / 2 / dx
    sheet = current / (width * 1e-6)
    p = np.asarray(path) / dx + pad
    if p.min() < half or p[:, 0].max() > nx - half or p[:, 1].max() > ny - half:
        raise ValueError('Current path leaves the padded canvas')
    for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
        horizontal = y0 == y1
        if horizontal:
            lo, hi = sorted((x0, x1))
            cols = coverage(lo - half, hi + half, nx)
            rows = coverage(y0 - half, y0 + half, ny)
            target, sign = kx, np.sign(x1 - x0)
        else:
            lo, hi = sorted((y0, y1))
            cols = coverage(x0 - half, x0 + half, nx)
            rows = coverage(lo - half, hi + half, ny)
            target, sign = ky, np.sign(y1 - y0)
        block = np.outer(rows[1], cols[1])
        target[rows[0]:rows[0] + block.shape[0],
               cols[0]:cols[0] + block.shape[1]] += sign * sheet * block
    # Both wires cover a corner square; share the current between them there.
    corner = (kx != 0) & (ky != 0)
    kx[corner] /= 2
    ky[corner] /= 2
    return kx, ky


def current_maps(paths, width, current, shape, dx, pad):
    """Summed sheet current from every path of one recipe."""
    kx, ky = np.zeros(shape), np.zeros(shape)
    for path in paths:
        px, py = rasterize(path, width, current, shape, dx, pad)
        kx += px
        ky += py
    return kx, ky


def fields_from_current(kx, ky, dx, heights, k0=0.0, z_ref=0.0):
    """Divergence-free [Bx, By, Bz] in uT at each height on the padded canvas.

    Above z_ref, decay uses the empirical rate sqrt(k^2 + k0^2).
    """
    ny, nx = kx.shape
    wx = 2 * np.pi * np.fft.rfftfreq(nx, dx)[None, :]
    wy = 2 * np.pi * np.fft.fftfreq(ny, dx)[:, None]
    k = np.hypot(wx, wy)
    safe = k.copy()
    safe[0, 0] = 1.0
    fx, fy = np.fft.rfft2(kx), np.fft.rfft2(ky)
    # Drawn corners leak a little current; remove the divergent part so the
    # in-plane fields describe the same conserved current as Bz.
    longitudinal = (wx * fx + wy * fy) / safe ** 2
    fx, fy = fx - wx * longitudinal, fy - wy * longitudinal
    fz = 1j * (wy * fx - wx * fy) / safe
    fx[0, 0] = fy[0, 0] = fz[0, 0] = 0
    result = []
    for z in heights:
        if z < z_ref:
            raise ValueError('Height lies below the decay reference plane')
        decay = 0.5 * MU0_UT * np.exp(-k * z_ref
                                      - np.sqrt(k ** 2 + k0 ** 2) * (z - z_ref))
        result.append([np.fft.irfft2(f * decay, s=(ny, nx))
                       for f in (fx, fy, fz)])
    return result


def gaussian_blur(fields, sigma_um, dx):
    """Optional optical point spread, applied to one [3, ny, nx] stack."""
    ny, nx = fields.shape[1:]
    fx = np.fft.rfftfreq(nx, dx)[None, :]
    fy = np.fft.fftfreq(ny, dx)[:, None]
    h = np.exp(-2 * (np.pi * sigma_um) ** 2 * (fx ** 2 + fy ** 2))
    return np.stack([np.fft.irfft2(np.fft.rfft2(f) * h, s=(ny, nx))
                     for f in fields])


def correlated_noise(white, corr_um, dx):
    """Unit-variance noise with exponential spatial correlation length corr_um."""
    ny, nx = white.shape
    fx = np.fft.rfftfreq(nx, dx)[None, :]
    fy = np.fft.fftfreq(ny, dx)[:, None]
    f2 = fx ** 2 + fy ** 2
    amplitude = (1 + (2 * np.pi * corr_um) ** 2 * f2) ** -0.75
    field = np.fft.irfft2(np.fft.rfft2(white) * amplitude, s=(ny, nx))
    return (field - field.mean()) / field.std()


def synthesise(recipe, settings):
    """Physical uT float32 arrays: target, clean input and observed input."""
    ny, nx, pad = settings['ny'], settings['nx'], settings['pad']
    dx = settings['dx']
    shape = (ny + 2 * pad, nx + 2 * pad)
    # Version 1 datasets hold one `path_um` and predate decay and speckle.
    paths = recipe.get('paths_um') or [recipe['path_um']]
    kx, ky = current_maps(paths, recipe['width_um'], settings['current_a'],
                          shape, dx, pad)
    d = recipe['offset_um']
    k0 = settings.get('decay_k0', 0.0)
    z_ref = settings['decay_ref_um'] + d if k0 > 0 else 0.0
    high, low = fields_from_current(
        kx, ky, dx, (settings['z_input'] + d, settings['z_target'] + d), k0, z_ref)
    crop = (slice(pad, pad + ny), slice(pad, pad + nx))
    clean = np.stack([f[crop] for f in high])
    target = np.stack([f[crop] for f in low])
    observed = clean
    if settings['psf_sigma_um'] > 0:
        observed = gaussian_blur(observed, settings['psf_sigma_um'], dx)
    rng = np.random.default_rng(recipe['noise_seed'])
    spread = observed.max((1, 2)) - observed.min((1, 2))
    sigma = recipe['noise_percent'] / 100 * spread
    unit = rng.standard_normal(observed.shape)
    corr = settings.get('noise_corr_um', 0.0)
    if corr > 0:
        unit = np.stack([correlated_noise(u, corr, dx) for u in unit])
    observed = observed + sigma[:, None, None] * unit
    return dict(target=target.astype(np.float32), clean=clean.astype(np.float32),
                input=observed.astype(np.float32), sigma_ut=sigma)


def summarise(job):
    recipe, settings = job
    item = synthesise(recipe, settings)
    row = dict(id=recipe['id'], family=recipe['family'],
               width_um=recipe['width_um'], offset_um=recipe['offset_um'],
               noise_percent=recipe['noise_percent'])
    for name, array in (('target', item['target']), ('input', item['input'])):
        for i, c in enumerate(COMPONENTS):
            row[f'{name}_{c}_peak_ut'] = float(np.abs(array[i]).max())
            row[f'{name}_{c}_rms_ut'] = float(np.sqrt(np.mean(array[i] ** 2)))
    return row


def build(output, train, validation, seed, settings, workers):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', dict(state='preparing'))
    recipes = dict(
        train=[draw_recipe(seed, f'train_{i:05d}', settings) for i in range(train)],
        validation=[draw_recipe(seed, f'validation_{i:05d}', settings)
                    for i in range(validation)])
    jobs = [(r, settings) for split in recipes.values() for r in split]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(summarise, jobs, chunksize=1))
    with (output / 'inventory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Scale comes from simulated training targets only, so COMSOL test data
    # cannot leak into normalisation.
    keys = [f'target_{c}_peak_ut' for c in COMPONENTS]
    scale = max(row[k] for row in rows if row['id'].startswith('train')
                for k in keys) if train else None
    source = Path(__file__)
    write_json(output / 'manifest.json', dict(
        kind=KIND, seed=seed, settings=settings, normalization_ut=scale,
        splits=recipes, code_sha256={source.name: digest(source)},
        numpy=np.__version__, python=platform.python_version()))
    write_json(output / 'status.json', dict(state='complete', samples=len(jobs)))
    return output


class SimulatedDataset:
    """Regenerates one sample at a time from a completed compact dataset."""

    def __init__(self, root):
        self.root = Path(root)
        status = json.loads((self.root / 'status.json').read_text())
        if status.get('state') != 'complete':
            raise ValueError(f'Incomplete dataset: {status}')
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest.get('kind') != KIND:
            raise ValueError('Not a simulated current-layout dataset')
        self.settings = self.manifest['settings']
        self.scale = self.manifest['normalization_ut']

    def recipes(self, split):
        return self.manifest['splits'][split]

    def iter_components(self, split, normalized=True):
        """Yield one scalar component pair; arrays are 2-D float32."""
        scale = self.scale if normalized else 1.0
        for recipe in self.recipes(split):
            item = synthesise(recipe, self.settings)
            for i, component in enumerate(COMPONENTS):
                yield dict(sharp=item['target'][i] / scale,
                           clean=item['clean'][i] / scale,
                           blurred=item['input'][i] / scale,
                           component=component, recipe=recipe['id'])


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--output', required=True)
    p.add_argument('--train', type=int, default=200)
    p.add_argument('--validation', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--z-input', type=float, default=DEFAULTS['z_input'])
    p.add_argument('--z-target', type=float, default=DEFAULTS['z_target'])
    p.add_argument('--noise-percent', type=float, nargs='+',
                   default=list(DEFAULTS['noise_percent']))
    p.add_argument('--noise-corr-um', type=float, default=DEFAULTS['noise_corr_um'],
                   help='Speckle correlation length; 0 gives white noise')
    p.add_argument('--decay-k0', type=float, default=DEFAULTS['decay_k0'],
                   help='Empirical COMSOL-like decay in rad/um; default 0 is free space')
    p.add_argument('--psf-sigma-um', type=float, default=0.0,
                   help='Gaussian blur on the input only; off for COMSOL work')
    p.add_argument('--workers', type=int, default=4)
    return p


def main():
    args = parser().parse_args()
    if not 0 < args.z_target < args.z_input:
        raise ValueError('Require 0 < z-target < z-input')
    if args.decay_k0 > 0 and args.z_target < DEFAULTS['decay_ref_um']:
        raise ValueError('z-target lies below the decay reference plane')
    settings = dict(DEFAULTS, z_input=args.z_input, z_target=args.z_target,
                    noise_percent=tuple(args.noise_percent),
                    noise_corr_um=args.noise_corr_um, decay_k0=args.decay_k0,
                    psf_sigma_um=args.psf_sigma_um)
    out = build(args.output, args.train, args.validation, args.seed, settings,
                args.workers)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
