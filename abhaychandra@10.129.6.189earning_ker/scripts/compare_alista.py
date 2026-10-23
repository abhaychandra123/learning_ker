"""Evaluate ALISTA on unseen sparse codes; tune ISTA/FISTA only on validation.

One or more compatible checkpoints can be compared (e.g. soft vs support
selection and the learned-ISTA control). Never compares these sparse-code
scores against the existing QDM image scores, which are a different task.
"""
import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from alista import ALISTA, load_checkpoint
from alista_data import SparseProblem, make_rng
from alista_experiment import (as_tensor, baseline_iterates, benchmark, configure_runtime,
                               file_hash, last_iterate, nmse, source_hashes, write_json)
from alista_weights import dictionary_hash


@torch.no_grad()
def select_baseline_lambdas(b, truth, d, lambdas, budgets, step_size):
    """Independently select each method/budget on validation, never test truth."""
    if (not lambdas or any(not math.isfinite(v) or v < 0 for v in lambdas)
            or not budgets or any(not isinstance(v, int) or v < 1 for v in budgets)):
        raise ValueError("need finite nonnegative lambdas and positive integer budgets")
    best, records = {}, []
    for method in ("ista", "fista"):
        for lam in lambdas:
            for k, x in enumerate(baseline_iterates(b, d, lam, method=method,
                                                    steps=max(budgets), step_size=step_size)):
                if k in budgets:
                    score = nmse(x, truth)["nmse"]
                    records.append(dict(method=method, iterations=k, lam=lam, validation_nmse=score))
                    key = (method, k)
                    if key not in best or score < best[key][1]:
                        best[key] = (lam, score)
    return best, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new evaluation directory")
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--long-iterations", type=int, default=500)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0., .001, .01, .05, .1, .2, .4])
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be a new directory")
    if min(args.test_samples, args.long_iterations, args.benchmark_repeats) < 1:
        parser.error("sample counts, iteration counts and benchmark repeats must be positive")
    if any(not math.isfinite(v) or v < 0 for v in args.lambdas):
        parser.error("lambdas must be finite and nonnegative")
    device = configure_runtime(args.device, args.threads, 0)
    loaded = [load_checkpoint(path, device) for path in args.checkpoint]
    first, metadata = loaded[0]
    if args.long_iterations < first.layers:
        parser.error("long-iterations must be at least the model depth")
    for model, other in loaded[1:]:
        if (model.layers != first.layers or model.D.dtype != first.D.dtype
                or not torch.equal(model.D, first.D)
                or other["dictionary_sha256"] != metadata["dictionary_sha256"]
                or other["problem"] != metadata["problem"] or other["seed"] != metadata["seed"]
                or other["validation_sha256"] != metadata["validation_sha256"]):
            raise ValueError("checkpoints must share dictionary, depth, distribution, seed, and validation data")
    validation_path = args.checkpoint[0].parent/"validation.npz"
    if file_hash(validation_path) != metadata["validation_sha256"]:
        raise ValueError("validation artifact checksum mismatch")
    with np.load(validation_path, allow_pickle=False) as saved:
        d = saved["D"]
        b_val, x_val = as_tensor(saved["b"], first), as_tensor(saved["x"], first)
    if dictionary_hash(d) != metadata["dictionary_sha256"] or not torch.equal(as_tensor(d, first), first.D):
        raise ValueError("validation dictionary does not match checkpoint")
    problem = SparseProblem(d, probability=metadata["problem"]["probability"],
                            snr_db=math.inf if metadata["problem"]["snr_db"] is None else metadata["problem"]["snr_db"])
    step_size = 1./(1.001*float(torch.linalg.matrix_norm(first.D.double(), ord=2)**2))
    budgets = sorted(set([first.layers, args.long_iterations]))
    selected, tuning = select_baseline_lambdas(b_val, x_val, first.D, args.lambdas, budgets, step_size)
    # Test samples are generated only after baseline selection is finished.
    b_array, x_array = problem.sample(args.test_samples, make_rng(metadata["seed"], "test"))
    b, truth = as_tensor(b_array, first), as_tensor(x_array, first)
    rows, curves, predictions = [], [], {}
    with torch.no_grad():
        for index, (model, info) in enumerate(loaded):
            uses_support = model.support_percent > 0 and model.max_support_percent > 0
            label = f"{model.mode}_{'ss' if uses_support else 'soft'}_{index}"
            for k, x in enumerate(model.iterates(b)):
                curves.append(dict(method=label, iteration=k, **nmse(x, truth)))
            predictions[label] = x.cpu().numpy()
            rows.append(dict(method=label, iterations=model.layers, **nmse(x, truth),
                             trainable_parameters=sum(p.numel() for p in model.parameters()), trained=True,
                             fixed_buffer_bytes=sum(v.numel()*v.element_size() for v in model.buffers()),
                             training_seconds=info["training_seconds"],
                             analytic_preparation_seconds=info["analytic"]["seconds"] if model.mode == "alista" else 0.,
                             timing=benchmark(lambda: model(b), device, args.benchmark_repeats)))
            untrained = ALISTA(model.D.cpu().numpy(), model.W.cpu().numpy(),
                               dtype=model.D.dtype, **model.configuration()).to(device).eval()
            for k, init_x in enumerate(untrained.iterates(b)):
                curves.append(dict(method=label+"_untrained", iteration=k, **nmse(init_x, truth)))
            rows.append(dict(method=label+"_untrained", iterations=model.layers,
                             **nmse(init_x, truth), trained=False,
                             trainable_parameters=sum(p.numel() for p in untrained.parameters()),
                             timing=benchmark(lambda: untrained(b), device, args.benchmark_repeats)))
        for method in ("ista", "fista"):
            for budget in budgets:
                lam, validation_score = selected[(method, budget)]
                label = f"{method}_{budget}"
                for k, x in enumerate(baseline_iterates(b, first.D, lam, method=method,
                                                       steps=budget, step_size=step_size)):
                    if budget == first.layers:
                        curves.append(dict(method=label, iteration=k, **nmse(x, truth)))
                predictions[label] = x.cpu().numpy()
                def infer():
                    return last_iterate(baseline_iterates(b, first.D, lam, method=method,
                                                         steps=budget, step_size=step_size))
                rows.append(dict(method=label, iterations=budget, lam=lam,
                                 validation_nmse=validation_score, **nmse(x, truth),
                                 trainable_parameters=0,
                                 timing=benchmark(infer, device, args.benchmark_repeats)))
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/"test_predictions.npz", b=b_array, truth=x_array, **predictions)
    write_json(args.output/"metrics.json", dict(
        scope="independent synthetic sparse-code test set; fixed dictionary; not a QDM result",
        checkpoints=[dict(path=str(p.resolve()), sha256=file_hash(p)) for p in args.checkpoint],
        problem=metadata["problem"], dictionary_sha256=metadata["dictionary_sha256"],
        seed=metadata["seed"], test_samples=args.test_samples,
        test_sha256=file_hash(args.output/"test_predictions.npz"),
        baseline_protocol="fixed budgets, zero start, lambda selected independently on validation per method/budget",
        device=str(device), threads=args.threads, torch=str(torch.__version__), sources=source_hashes(), results=rows))
    write_json(args.output/"baseline_selection.json", tuning)
    with (args.output/"curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "iteration", "nmse", "nmse_db"])
        writer.writeheader()
        writer.writerows(curves)
    with (args.output/"comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "iterations", "test_nmse_db", "lambda", "trainable_parameters", "median_batch_seconds"])
        for row in rows:
            writer.writerow([row["method"], row["iterations"], row["nmse_db"], row.get("lam"),
                             row["trainable_parameters"], row["timing"]["median_seconds"]])
            db_text = "-inf" if row["nmse_db"] is None else f"{row['nmse_db']:.4f}"
            print(f"{row['method']:28s} test NMSE {db_text} dB")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in dict.fromkeys(row["method"] for row in curves):
        points = [r for r in curves if r["method"] == label]
        ax.plot([r["iteration"] for r in points], [r["nmse_db"] for r in points],
                label=label, linestyle="--" if label.endswith("untrained") else "-")
    ax.set(xlabel="Layer / iteration", ylabel="Test NMSE (dB; lower is better)",
           title="Sparse recovery on unseen examples, fixed dictionary")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output/"comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
