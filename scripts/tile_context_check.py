"""Share of a tile's field that comes from currents outside the tile, and speckle after pixel averaging.

Run before the 10 um MAGIC-UNet runs (HIGH_STANDOFF_PROTOCOL.md), with no network and no training:
    python3 scripts/tile_context_check.py
For each (standoff, tile width): take a simulated layout, keep only the current inside one tile, compute its
field, and compare with the field of the whole layout inside that tile. NumPy only, about a minute on the Mac.
"""
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_fields import DEFAULTS, current_maps, draw_recipe, fields_from_current
from simulate_fields import MU0_UT


def binned(a, factor):
    *lead, h, w = a.shape
    return a.reshape(*lead, h // factor, factor, w // factor, factor).mean((-3, -1))


def solenoidal_sheet(kx, ky, dx, depth=0.0):
    """Divergence-free part of a sheet current, upward-continued by `depth` um.

    With depth = d this is the equivalent sheet at the zero-offset plane: it gives the
    same field above that plane as the true sheet d below it. Same projection as
    simulate_fields.fields_from_current.
    """
    ny, nx = kx.shape
    wx = 2 * np.pi * np.fft.rfftfreq(nx, dx)[None, :]
    wy = 2 * np.pi * np.fft.fftfreq(ny, dx)[:, None]
    k = np.hypot(wx, wy)
    k2 = k ** 2
    k2[0, 0] = 1.0
    fx, fy = np.fft.rfft2(kx), np.fft.rfft2(ky)
    longitudinal = (wx * fx + wy * fy) / k2
    decay = np.exp(-k * depth)
    fx, fy = (fx - wx * longitudinal) * decay, (fy - wy * longitudinal) * decay
    fx[0, 0] = fy[0, 0] = 0
    return np.fft.irfft2(fx, s=(ny, nx)), np.fft.irfft2(fy, s=(ny, nx))


def biot_savart_sheet(jx, jy, dx, height):
    """Thin-sheet Biot-Savart for ANY sheet current (no divergence-free projection).

    Same formulas and COMSOL component frame as simulate_fields.fields_from_current,
    which additionally projects the current; for a divergence-free current both agree.
    Without the projection the field is linear in the current, so a current cropped to
    the image gives exactly the field of the in-image current.
    """
    ny, nx = jx.shape
    wx = 2 * np.pi * np.fft.rfftfreq(nx, dx)[None, :]
    wy = 2 * np.pi * np.fft.fftfreq(ny, dx)[:, None]
    k = np.hypot(wx, wy)
    safe = k.copy()
    safe[0, 0] = 1.0
    fx, fy = np.fft.rfft2(jx), np.fft.rfft2(jy)
    fz = 1j * (wy * fx - wx * fy) / safe
    decay = 0.5 * MU0_UT * np.exp(-k * height)
    return [np.fft.irfft2(f * decay, s=(ny, nx)) for f in (fx, fy, fz)]


def field_from_sheet(j, height, dx=0.1, pad=700):
    """[3, H, W] field in uT at `height` um above a [2, H, W] sheet current in A/m.

    Zero padding outside the image, then thin-sheet Biot-Savart in the simulator's
    component frame, so signs match the training data and COMSOL.
    """
    jx = np.pad(np.asarray(j[0], np.float64), pad)
    jy = np.pad(np.asarray(j[1], np.float64), pad)
    f = biot_savart_sheet(jx, jy, dx, height)
    crop = (slice(pad, pad + j.shape[-2]), slice(pad, pad + j.shape[-1]))
    return np.stack([c[crop] for c in f])




from simulate_fields import correlated_noise
s = dict(DEFAULTS)
ny, nx, pad, dx = s['ny'], s['nx'], s['pad'], s['dx']
t0 = time.time()
rng = np.random.default_rng(0)
cases = [(3.0, 12.8), (8.0, 12.8), (10.0, 12.8), (10.0, 25.6), (10.0, 44.8)]
out = {c: [] for c in cases}
for i in range(4):
    recipe = draw_recipe(0, f'context_check_{i:03d}', s)
    kx, ky = current_maps(recipe['paths_um'], recipe['width_um'], s['current_a'], (ny + 2 * pad, nx + 2 * pad), dx, pad)
    jx, jy = solenoidal_sheet(kx, ky, dx, depth=recipe['offset_um'])
    jmag = np.hypot(jx, jy)
    full = {z: biot_savart_sheet(jx, jy, dx, z) for z in sorted({c[0] for c in cases})}
    for z, w in cases:
        n = int(round(w / dx))
        for _ in range(3):
            for _ in range(50):
                r, c = (int(v) for v in rng.integers(pad, pad + ny - n + 1, 2))
                if jmag[r:r + n, c:c + n].max() >= 0.05 * jmag.max():
                    break
            mx, my = np.zeros_like(jx), np.zeros_like(jy)
            mx[r:r + n, c:c + n], my[r:r + n, c:c + n] = jx[r:r + n, c:c + n], jy[r:r + n, c:c + n]
            inside = biot_savart_sheet(mx, my, dx, z)
            num = sum(np.sum((full[z][k][r:r + n, c:c + n] - inside[k][r:r + n, c:c + n]) ** 2) for k in range(3))
            den = sum(np.sum(full[z][k][r:r + n, c:c + n] ** 2) for k in range(3))
            out[(z, w)].append(np.sqrt(num / den))
for (z, w), v in out.items():
    print(f'standoff {z:4.1f} um, tile {w:4.1f} um ({w / z:.1f} standoffs): field from OUTSIDE the tile = {100 * np.median(v):.0f}% of the tile field (median of {len(v)}, range {100 * min(v):.0f}-{100 * max(v):.0f}%)')
u = correlated_noise(np.random.default_rng(1).standard_normal((1400, 1400)), s['noise_corr_um'], dx)
for b in (1, 2, 4, 7):
    print(f'speckle std after {b}x{b} averaging: {binned(u, b).std():.2f} of the 0.1 um value')
print(f'{time.time() - t0:.0f} s')
