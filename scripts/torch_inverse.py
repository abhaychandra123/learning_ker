"""Batched PyTorch version of the repository's padded real convolution.

Inputs have shape [batch, 1, height, width]. FFTs use full precision (no AMP).
The norm bound applies to Crop * FFT-convolution * Pad, including boundaries.
"""
import math
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as TF


class PaddedFFT(nn.Module):
    def __init__(self, transfer, shape, pad, dtype=torch.float32):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("FFT operator supports float32 or float64 only")
        if len(shape) != 2 or min(shape) < 1 or pad < 0 or int(pad) != pad:
            raise ValueError("invalid image shape or padding")
        self.shape, self.pad = tuple(shape), int(pad)
        h = np.asarray(transfer, dtype=np.complex128)
        if h.shape != tuple(n+2*self.pad for n in self.shape) or not np.isfinite(h).all():
            raise ValueError("transfer must be finite and match the padded shape")
        self.bound = float(np.max(np.abs(h)**2))
        if not math.isfinite(self.bound) or self.bound <= 0:
            raise ValueError("transfer norm bound must be positive and finite")
        # Store real/imaginary parts as real buffers: nn.Module.to(dtype) then
        # preserves both parts instead of accidentally casting complex to real.
        self.register_buffer("transfer_real", torch.as_tensor(h.real.copy(), dtype=dtype))
        self.register_buffer("transfer_imag", torch.as_tensor(h.imag.copy(), dtype=dtype))

    def _apply_transfer(self, x, adjoint):
        if x.ndim != 4 or x.shape[1] != 1 or tuple(x.shape[-2:]) != self.shape:
            raise ValueError(f"expected [batch,1,{self.shape[0]},{self.shape[1]}]")
        if x.dtype != self.transfer_real.dtype or x.device != self.transfer_real.device:
            raise ValueError("input and operator must share dtype and device")
        p = self.pad
        padded = TF.pad(x, (p, p, p, p)) if p else x
        h = torch.complex(self.transfer_real, self.transfer_imag)
        if adjoint:
            h = h.conj()
        out = torch.fft.ifft2(torch.fft.fft2(padded) * h).real
        return out[..., p:p+self.shape[0], p:p+self.shape[1]]

    def forward(self, x):
        return self._apply_transfer(x, False)

    def adjoint(self, x):
        return self._apply_transfer(x, True)


def shrink(x, threshold):
    return torch.sign(x) * torch.relu(torch.abs(x)-threshold)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(x, b, truth, operator, lam, scale=1.):
    """Per-image physical-unit metrics; x,b,truth are divided by scale.

    lam is in these normalized units: lambda_uT / scale.
    """
    residual = operator(x)-b
    grad = operator.adjoint(residual)
    step = .99/operator.bound
    pg = (x-shrink(x-step*grad, step*lam))/step
    data = .5*residual.square().flatten(1).sum(1)*scale**2
    l1 = lam*x.abs().flatten(1).sum(1)*scale**2
    rmse = (x-truth).square().flatten(1).mean(1).sqrt()*scale
    ranges = (truth.flatten(1).max(1).values-truth.flatten(1).min(1).values)*scale
    rms_truth = truth.square().flatten(1).mean(1).sqrt()*scale
    pg_rms = pg.square().flatten(1).mean(1).sqrt()*scale
    rows = []
    for i in range(len(x)):
        rows.append({"data": float(data[i]), "l1": float(l1[i]),
                     "objective": float(data[i]+l1[i]), "rmse_uT": float(rmse[i]),
                     "nrmse_range": float(rmse[i]/ranges[i]) if ranges[i] > 0 else None,
                     "relative_l2": float(rmse[i]/rms_truth[i]) if rms_truth[i] > 0 else None,
                     "pg_rms_uT": float(pg_rms[i])})
    return rows


@torch.no_grad()
def solve_torch(b, operator, lam, *, method, n_iter, initial, truth,
                scale=1., record_every=10):
    """ISTA/FISTA on one device, with equal step, units and diagnostics.

    Wall time includes tensor setup and metric evaluation. solver_seconds
    accumulates synchronized update times, excluding evaluation/I/O.
    """
    if method not in ("ista", "fista") or initial not in ("zero", "blurred"):
        raise ValueError("invalid method or initialization")
    if n_iter < 0 or int(n_iter) != n_iter or record_every < 1:
        raise ValueError("invalid iteration settings")
    if lam < 0 or not math.isfinite(lam) or scale <= 0 or not math.isfinite(scale):
        raise ValueError("invalid regularization or scale")
    if truth.shape != b.shape or not torch.isfinite(b).all() or not torch.isfinite(truth).all():
        raise ValueError("input/truth must be finite and have equal shape")
    wall_start = perf_counter()
    x = b.clone() if initial == "blurred" else torch.zeros_like(b)
    y, t = x.clone(), 1.
    step = .99/operator.bound
    elapsed, history = 0., []

    def record(k):
        history.append({"iteration": k, "solver_seconds": elapsed,
                        "components": evaluate(x, b, truth, operator, lam, scale)})

    record(0)
    for k in range(1, n_iter+1):
        synchronize(b.device)
        start = perf_counter()
        point = x if method == "ista" else y
        new_x = shrink(point-step*operator.adjoint(operator(point)-b), step*lam)
        if method == "fista":
            next_t = (1+math.sqrt(1+4*t*t))/2
            y = new_x+(t-1)/next_t*(new_x-x)
            t = next_t
        x = new_x
        synchronize(b.device)
        elapsed += perf_counter()-start
        if k % record_every == 0 or k == n_iter:
            record(k)
    synchronize(b.device)
    return x, history, perf_counter()-wall_start
