"""Comparable ISTA/FISTA for 0.5 ||A x-b||² + lambda ||x||_1.

The padded forward operator is the one in deblur.py. Solver timing excludes
metric evaluation. Iteration 0 is the initial image, before any update.
"""
from time import perf_counter

DEFAULT_EPSILON = 1e-3
SAFETY_MAX_ITERATIONS = 100_000

import numpy as np

from deblur import forward_operator, adjoint_operator, soft_threshold


def console_progress(label):
    """Report recorded solver progress with immediate terminal flushing."""
    started = perf_counter()

    def report(row):
        status = f"; {row['stop_reason']}" if row['stop_reason'] else ""
        print(f"    {label}: {row['iteration']} updates | "
              f"relative PG={row['relative_pg']:.6g} "
              f"(target <= {row['epsilon']:g}) | "
              f"elapsed {perf_counter()-started:.1f}s{status}", flush=True)
    return report


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


def validate_stopping(epsilon, n_iter, max_iterations):
    if not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must be finite and strictly between 0 and 1")
    if n_iter is not None and (not isinstance(n_iter, (int, np.integer)) or n_iter < 0):
        raise ValueError("n_iter must be a nonnegative integer or None")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")


def solve_l1(blurred, K_hat, lam, *, method, epsilon=DEFAULT_EPSILON,
             pad=32, initial="blurred", record_every=10, truth=None,
             n_iter=None, max_iterations=SAFETY_MAX_ITERATIONS, progress=None):
    """ISTA/FISTA until RMS(G(x))/RMS(G(x0)) <= epsilon.

    G(x) = (x - soft(x-step*A*(Ax-b), step*lam))/step. Check every
    update at x (never FISTA's extrapolated y), independent of recording.
    The internal safety limit/stagnation exit is NOT reported as convergence.
    Explicit n_iter is retained only for fixed-depth reference tests and LISTA
    initialization comparisons; user-facing CLIs expose epsilon instead.
    Optional progress(row) receives each recorded diagnostic, including start/end.
    """
    b, h, bound = validate_problem(blurred, K_hat, lam, pad)
    validate_stopping(epsilon, n_iter, max_iterations)
    if method not in ("ista", "fista"):
        raise ValueError("method must be ista or fista")
    if not isinstance(record_every, (int, np.integer)) or record_every < 1:
        raise ValueError("record_every must be a positive integer")
    if initial not in ("blurred", "zero"):
        raise ValueError("initial must be blurred or zero")
    if truth is not None:
        truth = np.asarray(truth, dtype=np.float64)
        if truth.shape != b.shape or not np.isfinite(truth).all():
            raise ValueError("truth must be finite and match blurred shape")
    step = 0.99 / bound
    x = b.copy() if initial == "blurred" else np.zeros_like(b)
    y, t = x.copy(), 1.0
    elapsed, history, unchanged = 0.0, [], 0
    adaptive = n_iter is None

    def pg_rms():
        gradient = adjoint_operator(forward_operator(x, h, pad)-b, h, pad)
        value = float(np.sqrt(np.mean(((x-soft_threshold(x-step*gradient, step*lam))/step)**2)))
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite proximal-gradient residual")
        return value

    start = perf_counter()
    initial_pg = pg_rms()
    elapsed += perf_counter()-start
    relative = 0.0 if initial_pg == 0 else 1.0

    def record(iteration, reason=None):
        row = metrics(x, b, h, lam, pad, step, truth)
        if not all(np.isfinite(v) for v in row.values() if v is not None):
            raise FloatingPointError("nonfinite solver metrics")
        row.update(iteration=iteration, solver_seconds=elapsed,
                   relative_pg=relative, initial_pg_rms=initial_pg,
                   epsilon=epsilon, converged=bool(adaptive and relative <= epsilon),
                   stop_reason=reason, safety_max_iterations=max_iterations)
        history.append(row)
        if progress is not None:
            progress(dict(row))

    reason = "converged" if adaptive and relative <= epsilon else ("fixed_budget" if n_iter == 0 else None)
    record(0, reason)
    if reason:
        return x, history
    limit = max_iterations if adaptive else n_iter
    for iteration in range(1, limit+1):
        start = perf_counter()
        point = x if method == "ista" else y
        gradient = adjoint_operator(forward_operator(point, h, pad)-b, h, pad)
        new_x = soft_threshold(point-step*gradient, lam*step)
        unchanged = unchanged+1 if np.array_equal(new_x, x) else 0
        if method == "fista":
            new_t = (1+np.sqrt(1+4*t*t))/2
            y = new_x+(t-1)/new_t*(new_x-x)
            t = new_t
        x = new_x
        current_pg = pg_rms()
        relative = current_pg/initial_pg if initial_pg else current_pg
        elapsed += perf_counter()-start
        reason = None
        if adaptive and relative <= epsilon:
            reason = "converged"
        elif adaptive and unchanged >= 20:
            reason = "numerical_stagnation"
        elif iteration == limit:
            reason = "safety_limit" if adaptive else "fixed_budget"
        if iteration % record_every == 0 or reason:
            record(iteration, reason)
        if reason:
            break
    return x, history
