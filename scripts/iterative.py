"""Comparable ISTA/FISTA for 0.5 ||A x-b||² + lambda ||x||_1.

The padded forward operator is the one in deblur.py. Solver timing excludes
metric evaluation. Iteration 0 is the initial image, before any update.
"""
from time import perf_counter

import numpy as np

from deblur import forward_operator, adjoint_operator, soft_threshold


def validate_problem(blurred, K_hat, lam, pad):
    b = np.asarray(blurred, dtype=np.float64)
    h = np.asarray(K_hat, dtype=np.complex128)
    if b.ndim != 2 or min(b.shape) == 0 or not np.isfinite(b).all():
        raise ValueError("blurred must be a finite, nonempty 2D array")
    if not isinstance(pad, (int, np.integer)) or pad < 0:
        raise ValueError("pad must be a nonnegative integer")
    if h.shape != tuple(n + 2 * pad for n in b.shape) or not np.isfinite(h).all():
        raise ValueError("K_hat must be finite and match the padded image shape")
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lambda must be finite and nonnegative")
    bound = float(np.max(np.abs(h) ** 2))
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("K_hat must have a finite, positive squared-norm bound")
    return b, h, bound


def metrics(x, b, h, lam, pad, step, truth=None):
    """Objective and proximal-gradient stationarity, in the original units.

    pg_rms = ||(x-prox(x-step*grad f(x)))/step||_2 / sqrt(N).
    This is zero at an L1 optimum; it is not relative reconstruction error.
    """
    residual = forward_operator(x, h, pad) - b
    data = 0.5 * float(np.sum(residual ** 2))
    penalty = float(lam * np.sum(np.abs(x)))
    gradient = adjoint_operator(residual, h, pad)
    prox = soft_threshold(x - step * gradient, step * lam)
    row = {
        "data": data, "l1": penalty, "objective": data + penalty,
        "pg_rms": float(np.sqrt(np.mean(((x - prox) / step) ** 2))),
    }
    if truth is not None:
        rmse = float(np.sqrt(np.mean((x - truth) ** 2)))
        value_range = float(np.ptp(truth))
        signal_rms = float(np.sqrt(np.mean(truth ** 2)))
        row.update(rmse=rmse,
                   nrmse_range=rmse / value_range if value_range > 0 else None,
                   relative_l2=rmse / signal_rms if signal_rms > 0 else None)
    return row


def solve_l1(blurred, K_hat, lam, *, method, n_iter=100, pad=32,
             initial="blurred", record_every=10, truth=None, pg_tol=None):
    """ISTA or standard (non-monotone) Beck--Teboulle FISTA.

    Both methods use the SAME step 0.99/max|H|² and initial iterate. pg_tol,
    when supplied, is an absolute RMS proximal-gradient tolerance, checked
    only at recorded iterations. Ground truth is used ONLY for diagnostics.
    FISTA returns x_k, never the extrapolated y_(k+1).
    """
    b, h, bound = validate_problem(blurred, K_hat, lam, pad)
    if method not in ("ista", "fista"):
        raise ValueError("method must be ista or fista")
    if not isinstance(n_iter, (int, np.integer)) or n_iter < 0:
        raise ValueError("n_iter must be a nonnegative integer")
    if not isinstance(record_every, (int, np.integer)) or record_every < 1:
        raise ValueError("record_every must be a positive integer")
    if pg_tol is not None and (not np.isfinite(pg_tol) or pg_tol < 0):
        raise ValueError("pg_tol must be finite and nonnegative")
    if initial not in ("blurred", "zero"):
        raise ValueError("initial must be blurred or zero")
    if truth is not None:
        truth = np.asarray(truth, dtype=np.float64)
        if truth.shape != b.shape or not np.isfinite(truth).all():
            raise ValueError("truth must be finite and match blurred shape")
    step = 0.99 / bound
    x = b.copy() if initial == "blurred" else np.zeros_like(b)
    y = x.copy()
    t = 1.0
    elapsed = 0.0
    history = []

    def record(iteration):
        row = metrics(x, b, h, lam, pad, step, truth)
        if not all(np.isfinite(v) for v in row.values() if v is not None):
            raise FloatingPointError("nonfinite solver metrics")
        row.update(iteration=iteration, solver_seconds=elapsed)
        history.append(row)
        return pg_tol is not None and row["pg_rms"] <= pg_tol

    if record(0):
        return x, history
    for iteration in range(1, n_iter + 1):
        start = perf_counter()
        point = x if method == "ista" else y
        residual = forward_operator(point, h, pad) - b
        gradient = adjoint_operator(residual, h, pad)
        new_x = soft_threshold(point - step * gradient, lam * step)
        if method == "fista":
            new_t = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            y = new_x + ((t - 1.0) / new_t) * (new_x - x)
            t = new_t
        x = new_x
        elapsed += perf_counter() - start
        if iteration % record_every == 0 or iteration == n_iter:
            if record(iteration):
                break
    return x, history
