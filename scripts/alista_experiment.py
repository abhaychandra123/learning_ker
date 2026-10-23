"""Shared ALISTA experiment utilities and dense ISTA/FISTA baselines."""
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from alista import soft_threshold


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes():
    root = Path(__file__).resolve().parent
    names = ("alista.py", "alista_weights.py", "alista_data.py", "alista_experiment.py",
             "train_alista.py", "compare_alista.py")
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def configure_runtime(device, threads, seed):
    if threads < 1:
        raise ValueError("CPU thread count must be positive")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("requested device is unavailable")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device(device)


def as_tensor(array, model):
    return torch.as_tensor(array, dtype=model.D.dtype, device=model.D.device)


@torch.no_grad()
def nmse(prediction, truth):
    """Ratio of aggregate energies, Eq. in Section 5.1, NOT mean sample ratios."""
    residual = prediction.double()-truth.double()
    numerator = float(residual.square().sum())
    denominator = float(truth.double().square().sum())
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("NMSE requires finite predictions and nonzero total truth energy")
    ratio = numerator/denominator
    # JSON null for the exact-zero error case (dB = -infinity).
    return {"nmse": ratio, "nmse_db": 10.*math.log10(ratio) if ratio > 0 else None}


def baseline_iterates(b, d, lam, *, method="ista", steps=16, step_size=None):
    """Zero-start fixed-budget diagnostics; not a convergence stopping test."""
    if method not in ("ista", "fista") or not isinstance(steps, int) or steps < 0:
        raise ValueError("invalid baseline method or iteration count")
    if not math.isfinite(lam) or lam < 0:
        raise ValueError("lambda must be finite and nonnegative")
    if step_size is None:
        step_size = 1./(1.001*float(torch.linalg.matrix_norm(d.double(), ord=2)**2))
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("step size must be finite and positive")
    x = b.new_zeros((b.shape[0], d.shape[1]))
    y, t = x, 1.
    yield x
    for _ in range(steps):
        point = x if method == "ista" else y
        nx = soft_threshold(point + step_size*((b-point @ d.T) @ d), lam*step_size)
        if method == "fista":
            nt = .5*(1.+math.sqrt(1.+4.*t*t))
            y = nx + ((t-1.)/nt)*(nx-x)
            t = nt
        x = nx
        yield x


def last_iterate(iterator):
    for x in iterator:
        pass
    return x


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark(call, device, repeats=3):
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be positive")
    call()
    synchronize(device)
    times = []
    for _ in range(repeats):
        synchronize(device)
        start = perf_counter()
        call()
        synchronize(device)
        times.append(perf_counter()-start)
    return {"median_seconds": float(np.median(times)), "samples_seconds": times}
