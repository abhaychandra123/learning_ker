"""Data-free ALISTA weights: Liu et al. (ICLR 2019), Eq. (16), App. E.1.

Matrices use the paper's convention D, W: [measurements, coefficients].
No regularization, training pairs, or convolutional approximation is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def validate_dictionary(dictionary):
    raw = np.asarray(dictionary)
    if raw.ndim != 2 or min(raw.shape) < 1 or np.iscomplexobj(raw):
        raise ValueError("D must be a nonempty real matrix [measurements, coefficients]")
    d = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(d).all():
        raise ValueError("D must be finite")
    if not np.allclose(np.linalg.norm(d, axis=0), 1., rtol=1e-6, atol=1e-8):
        raise ValueError("D columns must have unit L2 norm; normalize explicitly with matching coefficient units")
    return d


def dictionary_hash(d):
    d = np.ascontiguousarray(d, dtype='<f8')
    return hashlib.sha256(str(d.shape).encode() + d.tobytes()).hexdigest()


def project_columns(w, d):
    """Orthogonal projection onto d_j.T w_j = 1 (also correct off unit norm).

    Appendix E.1's printed projection has a typo: the correction direction
    must be d_j, as in the authors' matlabs/CalculateW.m, not w_j.
    """
    return w + d * ((1. - np.sum(d*w, axis=0)) / np.sum(d*d, axis=0))


def weight_diagnostics(d, w):
    """Feasibility and KKT stationarity without an M-by-M Gram allocation."""
    if w.shape != d.shape or not np.isfinite(w).all():
        raise ValueError("W must be finite and have D's shape")
    gradient = (d @ d.T) @ w  # gradient of half the objective
    tangent = gradient - d * (np.sum(d*gradient, axis=0)/np.sum(d*d, axis=0))
    objective = float(np.sum(w*gradient))
    return {
        "objective": objective,
        "off_diagonal_energy": max(0., objective-float(np.sum(np.sum(d*w, axis=0)**2))),
        "max_diagonal_error": float(np.max(np.abs(np.sum(d*w, axis=0)-1.))),
        "relative_stationarity": float(np.linalg.norm(tangent)/max(1., np.linalg.norm(gradient))),
        "weight_frobenius_norm": float(np.linalg.norm(w)),
    }


def analytic_weights(dictionary, *, method="pgd", tolerance=1e-9,
                     max_iterations=10000, max_workspace_gib=1.):
    """Solve Eq. (16) in float64, returning W and convergence diagnostics.

    PGD implements Algorithm 1 with a safe 1/||D||_2^2 step and an actual
    constrained stationarity check, rather than a small objective change.
    'svd' independently solves the same quadratic program, with no ridge:
      u_j = (D D.T)^dagger d_j; w_j = u_j/(d_j.T u_j).
    Numerical rank is reported; no near-zero singular value is inverted.
    A nonconverged result is returned with converged=False, never relabeled.
    """
    raw = np.asarray(dictionary)
    if raw.ndim != 2 or min(raw.shape) < 1:
        raise ValueError("D must be a nonempty matrix")
    n, m = raw.shape
    if not np.isfinite(max_workspace_gib) or max_workspace_gib <= 0:
        raise ValueError("max_workspace_gib must be positive and finite")
    # Conservative working-array estimate, not a platform peak-memory promise.
    if 8*(8*n*m + 4*n*n) > max_workspace_gib*2**30:
        raise ValueError("dense ALISTA workspace exceeds budget; do not materialize the full image operator")
    d = validate_dictionary(raw)
    if method not in ("pgd", "svd"):
        raise ValueError("method must be pgd or svd")
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("tolerance must lie strictly between 0 and 1")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    start = perf_counter()
    gram = d @ d.T
    spectral = float(np.linalg.eigvalsh(gram)[-1])
    initial = weight_diagnostics(d, d)
    rank = None
    if method == "svd":
        u, singular, vh = np.linalg.svd(d, full_matrices=False)
        cutoff = max(d.shape)*np.finfo(np.float64).eps*singular[0]
        keep = singular > cutoff
        rank = int(np.count_nonzero(keep))
        unconstrained = (u[:, keep]/singular[keep]) @ vh[keep]
        denominators = np.sum(d*unconstrained, axis=0)
        if np.any(denominators <= 0):
            raise FloatingPointError("nonpositive normalization in analytic solution")
        w = unconstrained/denominators
        iterations = 0
    else:
        w = project_columns(d.copy(), d)
        for iterations in range(1, max_iterations+1):
            w = project_columns(w - (gram @ w)/spectral, d)
            if iterations == 1 or iterations % 10 == 0 or iterations == max_iterations:
                if weight_diagnostics(d, w)["relative_stationarity"] <= tolerance:
                    break
    final = weight_diagnostics(d, w)
    converged = final["relative_stationarity"] <= tolerance and final["max_diagonal_error"] <= 1e-8
    return w, dict(final, method=method, converged=bool(converged),
                   stop_reason="stationarity" if converged else "tolerance_not_met",
                   iterations=iterations, tolerance=tolerance,
                   numerical_rank=rank, spectral_norm_squared=spectral,
                   initial_objective=initial["objective"], seconds=perf_counter()-start,
                   dictionary_sha256=dictionary_hash(d))


def load_weights(path):
    with np.load(path, allow_pickle=False) as saved:
        d = validate_dictionary(saved["D"])
        if np.iscomplexobj(saved["W"]):
            raise ValueError("W must be real")
        w = np.asarray(saved["W"], dtype=np.float64)
        metadata = json.loads(str(saved["metadata"]))
    if metadata.get("format") != "alista-weights-v1":
        raise ValueError("unsupported analytic weight format")
    if metadata["dictionary_sha256"] != dictionary_hash(d):
        raise ValueError("dictionary checksum mismatch")
    checks = weight_diagnostics(d, w)
    if (not metadata.get("converged") or checks["max_diagonal_error"] > 1e-8
            or checks["relative_stationarity"] > max(1e-8, 2*metadata["tolerance"])):
        raise ValueError("analytic weights do not satisfy feasibility/stationarity tolerances")
    return d, w, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", type=Path, help="unit-column D.npy; otherwise generate Section 5.1 Gaussian D")
    parser.add_argument("--measurements", type=int, default=250)
    parser.add_argument("--coefficients", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=("pgd", "svd"), default="pgd")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--max-workspace-gib", type=float, default=1.)
    parser.add_argument("--output", type=Path, required=True, help="new .npz file")
    args = parser.parse_args(argv)
    if args.output.suffix != ".npz" or args.output.exists():
        parser.error("output must be a new .npz file")
    if min(args.measurements, args.coefficients) < 1 or args.seed < 0:
        parser.error("dimensions must be positive and seed nonnegative")
    if not np.isfinite(args.max_workspace_gib) or args.max_workspace_gib <= 0:
        parser.error("max-workspace-gib must be finite and positive")
    if args.dictionary:
        d = np.load(args.dictionary, allow_pickle=False, mmap_mode="r")
    else:
        if 8*(8*args.measurements*args.coefficients + 4*args.measurements**2) > args.max_workspace_gib*2**30:
            parser.error("requested dictionary exceeds dense workspace budget")
        d = np.random.default_rng(args.seed).normal(size=(args.measurements, args.coefficients))
        d /= np.linalg.norm(d, axis=0)
    w, report = analytic_weights(d, method=args.method, tolerance=args.tolerance,
                                 max_iterations=args.max_iterations,
                                 max_workspace_gib=args.max_workspace_gib)
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["converged"]:
        raise RuntimeError("analytic weight solve did not converge; no weight artifact written")
    report.update(format="alista-weights-v1", seed=args.seed,
                  dictionary_source=str(args.dictionary) if args.dictionary else "normalized Gaussian, Section 5.1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, D=d, W=w, metadata=json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
