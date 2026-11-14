"""Is our free-space generator correct? Check scripts/simulate_fields.fields_from_current against direct Biot-Savart.

Source: a square loop, centre-line half-side 10 um, 2 um wide sheet, 600 uA, in the plane z = 0.
Reference: exact Biot-Savart line integrals (closed-form straight-segment formula) over 81 nested square filaments
across the width; no FFT, no padding, no COMSOL. Test: the repository's Fourier generator on a 0.1 um grid.
Usage: python3 simulator_vs_biot_savart.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulate_fields import fields_from_current          # the generator used for all training data

I, A, W = 600e-6, 10.0, 2.0          # current (A), centre-line half-side (um), width (um)
mu0 = 4e-7*np.pi
dx, n = 0.1, 2048                     # generator grid: 204.8 um canvas, loop in the middle
c = np.arange(n)*dx - n*dx/2 + dx/2   # pixel centres at +-0.05, +-0.15, ... (wire edges fall on pixel edges)
Y, X = np.meshgrid(c, c, indexing="ij")
r = np.maximum(np.abs(X), np.abs(Y))
band = (r > A - W/2) & (r < A + W/2)
K0 = I/(W*1e-6)                       # A/m
side = np.abs(X) >= np.abs(Y)         # vertical sides; counter-clockwise current
kx = np.where(band & ~side, -np.sign(Y)*K0, 0.0)
ky = np.where(band & side, np.sign(X)*K0, 0.0)

# evaluation points: every 1 um over a 60 x 60 um window, at pixel centres of the generator grid
ev = np.arange(-29.95, 30.0, 1.0)
EY, EX = np.meshgrid(ev, ev, indexing="ij")
iy = np.searchsorted(c, ev - 1e-9)


def biot_savart(z, nfil=81):
    """Exact field of nested square filaments (uT), components x, y, z."""
    B = np.zeros((3,) + EX.shape)
    P = np.stack([EX, EY, np.full_like(EX, z)], -1)
    for a in A - W/2 + W*(np.arange(nfil) + 0.5)/nfil:          # midpoint rule across the width
        corners = [(a, -a), (a, a), (-a, a), (-a, -a), (a, -a)]   # counter-clockwise
        for (x1, y1), (x2, y2) in zip(corners[:-1], corners[1:]):
            R1 = P - np.array([x1, y1, 0.0]); R2 = P - np.array([x2, y2, 0.0])
            n1, n2 = np.linalg.norm(R1, axis=-1), np.linalg.norm(R2, axis=-1)
            f = (n1 + n2)/(n1*n2*(n1*n2 + np.sum(R1*R2, -1)))
            B += np.moveaxis(np.cross(R1, R2)*f[..., None], -1, 0)
    return B*mu0*(I/nfil)/(4*np.pi)*1e6*1e6                  # 1/um -> 1/m, T -> uT


print("Square loop, 2 um wire, 600 uA. Generator (Fourier, 0.1 um pixels) vs exact Biot-Savart (81 filaments).")
for z in (0.5, 1.0, 3.0, 10.0, 20.0):
    out = fields_from_current(kx, ky, dx, [z])[0]           # repository convention, uT
    gen = np.array([out[1], -out[0], -out[2]])               # -> physical (Bx, By, Bz)
    gen = gen[:, iy][:, :, iy]
    ref = biot_savart(z)
    errs = [np.linalg.norm(gen[i] - ref[i])/np.linalg.norm(ref[i]) for i in range(3)]
    peak = np.max(np.abs(ref))
    print(f"  z = {z:4.1f} um: relative L2 error Bx/By/Bz = " + "/".join(f"{100*e:.3f}%" for e in errs)
          + f"   (peak |B| {peak:.2f} uT)")
