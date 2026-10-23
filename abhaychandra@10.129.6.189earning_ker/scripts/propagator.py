"""
Fourier machinery for the standoff propagator.

Above all sources the field obeys Laplace's equation, so a plane at height z1
determines every plane above it exactly:

    B(k, z2) = B(k, z1) * exp(-2*pi*k*dz),   k = hypot(fx, fy) in cycles/um

with the same operator for Bx, By and Bz. That factor is the "blur kernel";
in real space it is the 2-D Poisson kernel

    K(r) = dz / (2*pi*(r^2 + dz^2)^(3/2))

whose mass outside radius R is dz/sqrt(R^2+dz^2) - 20% still outside R = 5*dz.
That long tail is why kernel estimation belongs in Fourier space and not in a
small real-space convolution window.

This module currently holds what the visualisation needs. The masking,
padding and kernel-estimation pieces land here in Phase 1.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def k_grid(shape, dx, dy=None):
    """Radial spatial frequency |k| in cycles/um, matching np.fft.fft2 layout."""
    ny, nx = shape
    dy = dx if dy is None else dy
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    return np.hypot(fy[:, None], fx[None, :])


def analytic_transfer(k, dz):
    """The exact propagator exp(-2*pi*k*dz). Zero free parameters."""
    return np.exp(-TWO_PI * np.asarray(k) * dz)


def propagate(field, dz, dx):
    """Upward continuation of a plane by dz um (dz > 0 moves away from source).

    Naive: no padding or masking, so the outer ~3*dz/dx pixels are corrupted by
    FFT wraparound. Phase 1 adds the padding and interior mask.
    """
    k = k_grid(field.shape, dx)
    spec = np.fft.fft2(field) * analytic_transfer(k, dz)
    return np.real(np.fft.ifft2(spec))


def hann2d(shape):
    """Separable Hann window - suppresses edge leakage in diagnostic spectra."""
    ny, nx = shape
    return np.hanning(ny)[:, None] * np.hanning(nx)[None, :]


def amplitude_spectrum(field, window=True):
    """Per-mode amplitude |FFT| / N, same layout as k_grid."""
    f = field.astype(np.float64)
    f = f - f.mean()
    if window:
        f = f * hann2d(f.shape)
    return np.abs(np.fft.fft2(f)) / f.size


def radial_average(values, k, dk=None, k_max=None):
    """Average `values` into radial |k| bins. Returns (k_centres, mean)."""
    if dk is None:
        dk = float(k[k > 0].min())
    if k_max is None:
        k_max = k.max()

    nbins = int(np.ceil(k_max / dk))
    idx = np.minimum((k.ravel() / dk).astype(int), nbins - 1)
    w = values.ravel().astype(np.float64)

    total = np.bincount(idx, weights=w, minlength=nbins)
    count = np.bincount(idx, minlength=nbins)
    mean = np.divide(total, count, out=np.full(nbins, np.nan), where=count > 0)
    centres = (np.arange(nbins) + 0.5) * dk
    return centres, mean


def radial_spectrum(field, dx, window=True, dk=None):
    """Convenience: radially averaged amplitude spectrum of one plane."""
    k = k_grid(field.shape, dx)
    return radial_average(amplitude_spectrum(field, window=window), k, dk=dk)
