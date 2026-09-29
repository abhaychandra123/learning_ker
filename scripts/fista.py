"""Separate FISTA entry point; uses the same learned operator as deblur.py.

Example: python scripts/fista.py --z 3 --comp Bz --epsilon 1e-3 --output runs/fista_z3
Outputs are kept in a new run directory; existing directories are refused.
Reference-assisted kernel fitting is explicit: this is a same-sample experiment.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

import fields as F
from deblur import learn_kernel_joint
from iterative import solve_l1, console_progress


def deblur_fista(blurred, K_hat, lambda_ista, n_iter=None, pad=32, **kwargs):
    return solve_l1(blurred, K_hat, lambda_ista, method="fista",
                    n_iter=n_iter, pad=pad, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--comp", choices=F.SIGNED, default=None)
    parser.add_argument("--epsilon", type=float, default=1e-3,
                        help="relative proximal-gradient tolerance (default: 0.001)")
    parser.add_argument("--lambda-ista", "--lambda_ista", type=float, default=1e-4)
    parser.add_argument("--pad", type=int, default=32)
    parser.add_argument("--initial", choices=("blurred", "zero"), default="blurred")
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.z <= 0.5 or args.z not in F.z_values():
        parser.error("z must be an available primary-cache height above 0.5")
    if args.pad < 0 or args.record_every < 1:
        parser.error("pad must be nonnegative and record-every positive")
    if not np.isfinite(args.epsilon) or not 0 < args.epsilon < 1:
        parser.error("epsilon must be finite and strictly between 0 and 1")
    if not np.isfinite(args.lambda_ista) or args.lambda_ista < 0:
        parser.error("lambda-ista must be finite and nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    sharps = [F.load(0.5, c) for c in F.SIGNED]
    blurreds = [F.load(args.z, c) for c in F.SIGNED]
    start = perf_counter()
    h = learn_kernel_joint(sharps, blurreds, pad=args.pad)
    kernel_seconds = perf_counter() - start
    np.save(args.output / "kernel.npy", h)
    results = {}
    for c, sharp, blurred in zip(F.SIGNED, sharps, blurreds):
        if args.comp is not None and c != args.comp:
            continue
        x, history = deblur_fista(
            blurred, h, args.lambda_ista, pad=args.pad, epsilon=args.epsilon,
            initial=args.initial, record_every=args.record_every, truth=sharp,
            progress=console_progress(f"FISTA {c}"))
        np.save(args.output / f"reconstruction_{c}.npy", x)
        results[c] = history
        print(c, json.dumps(history[-1]))
    summary = [{"component": c, "iterations": history[-1]["iteration"],
                "converged": history[-1]["converged"], "stop_reason": history[-1]["stop_reason"],
                "epsilon": args.epsilon, "relative_pg": history[-1]["relative_pg"],
                "nrmse_range": history[-1]["nrmse_range"]} for c, history in results.items()]
    with (args.output/"summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(results), 2, figsize=(12, 4*len(results)),
                             constrained_layout=True, squeeze=False)
    for row, (component, history) in enumerate(results.items()):
        last = history[-1]
        label = f"{component}: {last['iteration']} updates; {last['stop_reason']}"
        for col, (metric, ylabel) in enumerate([
                ("objective", "L1 objective"), ("relative_pg", "PG RMS / initial PG RMS")]):
            ax = axes[row, col]
            ax.plot([h["iteration"] for h in history], [h[metric] for h in history], "o-", ms=3)
            ax.scatter([last["iteration"]], [last[metric]], s=35, zorder=3)
            ax.set(title=label, xlabel="Completed updates", ylabel=ylabel)
            ax.set_yscale("symlog", linthresh=1e-10)
            if metric == "relative_pg":
                ax.axhline(args.epsilon, color="black", ls="--", label=f"epsilon={args.epsilon:g}")
                ax.legend()
            ax.grid(alpha=.3)
    fig.suptitle("FISTA convergence: actual update counts per component")
    fig.savefig(args.output/"convergence.png", dpi=120)
    plt.close(fig)
    sources = [F.CACHE / "grid.json"] + [
        F.CACHE / f"{F.z_tag(z)}_{c}.npy"
        for z in (0.5, args.z) for c in F.SIGNED]
    sources += [Path(__file__), Path(__file__).with_name("iterative.py"),
                Path(__file__).with_name("deblur.py"), Path(F.__file__)]
    manifest = {
        "algorithm": "FISTA", "objective": "0.5 sum((A x-b)^2) + lambda sum(abs(x))",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "reference_z": 0.5, "units": "uT", "dtype": "float64",
        "evaluation": "same-sample; reference and target used to fit kernel",
        "norm_bound": float(np.max(abs(h) ** 2)), "step_fraction": 0.99,
        "kernel_seconds": kernel_seconds,
        "timing": "solver_seconds includes convergence checks; excludes reporting diagnostics, kernel fitting, I/O and plotting",
        "python": platform.python_version(), "numpy": np.__version__,
        "sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=F.ROOT,
                                       capture_output=True, text=True).stdout.strip(),
        "history": results,
        "summary": summary,
    }
    (args.output / "metrics.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
