"""Train the fixed-dictionary ALISTA sparse recovery model, separately from QDM.

Layerwise schedule: newest layer only, then refine the active prefix twice.
Fresh training examples, fixed independent validation, no test-set access.
"""
import argparse
import math
import os
from pathlib import Path
import platform
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from alista import ALISTA, save_checkpoint
from alista_data import SparseProblem, make_rng
from alista_experiment import (as_tensor, configure_runtime, file_hash, nmse,
                               source_hashes, synchronize, write_json)
from alista_weights import load_weights


def training_stages(layers, lr, refinement_factors=(.2, .02), gradient_decay=.3):
    """Mirror the structure of authors' utils/train.py setup_sc_training.

    The authors multiply gradients (before Adam), not Adam learning rates,
    by gradient_decay**age for previously trained layers. Preserve that detail.
    """
    if not isinstance(layers, int) or layers < 1:
        raise ValueError("layers must be positive")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    if (not math.isfinite(gradient_decay) or not 0 < gradient_decay <= 1
            or any(not math.isfinite(f) or f <= 0 for f in refinement_factors)):
        raise ValueError("invalid training schedule")
    for depth in range(1, layers+1):
        yield dict(depth=depth, name="new_layer", lr=lr,
                   active=[depth-1], multipliers=[1.])
        for factor in refinement_factors:
            yield dict(depth=depth, name=f"refine_{factor:g}", lr=lr*factor,
                       active=list(range(depth)),
                       multipliers=[gradient_decay**(depth-1-k) for k in range(depth)])


def train_model(model, problem, validation, *, seed=0, batch_size=64,
                steps_per_stage=2000, validate_every=10, patience=2000,
                lr=5e-4, refinement_factors=(.2, .02), gradient_decay=.3,
                callback=None):
    """Return validation history and stage results, restore best stage weights.

    Restoring each stage's best validation state and nonnegative threshold
    projection are explicit robustness choices; see ALISTA.md for differences
    from the legacy TensorFlow training script. No test observations are used.
    """
    integers = (batch_size, steps_per_stage, validate_every, patience)
    if any(not isinstance(v, int) or v < 1 for v in integers):
        raise ValueError("batch size, stage steps, validation interval, patience must be positive integers")
    stages = list(training_stages(model.layers, lr, refinement_factors, gradient_decay))
    b_val, x_val = (as_tensor(a, model) for a in validation)
    if x_val.shape != (b_val.shape[0], model.D.shape[1]):
        raise ValueError("validation coefficient shape mismatch")
    if not torch.isfinite(b_val).all() or not torch.isfinite(x_val).all():
        raise ValueError("validation must be finite")
    # Do not allow a sampler to train against a different measurement operator.
    if not np.array_equal(problem.D.astype(np.float32 if model.D.dtype == torch.float32 else np.float64),
                          model.D.detach().cpu().numpy()):
        raise ValueError("training problem and model dictionaries differ")
    rng = make_rng(seed, "train")
    history, summaries = [], []
    original_flags = [p.requires_grad for p in model.parameters()]
    synchronize(model.D.device)
    start = perf_counter()
    try:
        for stage in stages:
            for p in model.parameters():
                p.requires_grad_(False)
            parameters, multipliers = [], []
            for k, multiplier in zip(stage["active"], stage["multipliers"]):
                for p in (model.steps[k], model.thresholds[k]):
                    p.requires_grad_(True)
                    parameters.append(p)
                    multipliers.append(multiplier)
            optimizer = torch.optim.Adam(parameters, lr=stage["lr"])
            best, best_step, best_parameters = math.inf, 0, None
            last_training_loss = None
            for step in range(steps_per_stage+1):
                if step:
                    model.train()
                    b, x = (as_tensor(a, model) for a in problem.sample(batch_size, rng))
                    optimizer.zero_grad(set_to_none=True)
                    loss = .5*(model(b, depth=stage["depth"])-x).square().sum()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"nonfinite loss in {stage} step {step}")
                    loss.backward()
                    for p, multiplier in zip(parameters, multipliers):
                        if p.grad is None or not torch.isfinite(p.grad).all():
                            raise FloatingPointError("nonfinite or missing scalar gradient")
                        p.grad.mul_(multiplier)
                    optimizer.step()
                    model.project_parameters()
                    if any(not torch.isfinite(p).all() for p in parameters):
                        raise FloatingPointError("nonfinite learned scalar")
                    last_training_loss = float(loss.detach())
                if step % validate_every == 0 or step == steps_per_stage:
                    model.eval()
                    with torch.no_grad():
                        scores = nmse(model(b_val, depth=stage["depth"]), x_val)
                    if scores["nmse"] < best:
                        best, best_step = scores["nmse"], step
                        best_parameters = [p.detach().clone() for p in parameters]
                    row = dict(depth=stage["depth"], stage=stage["name"], step=step,
                               validation_nmse=scores["nmse"], validation_nmse_db=scores["nmse_db"],
                               training_half_squared_error=last_training_loss)
                    history.append(row)
                    if callback:
                        callback(row)
                    if step-best_step >= patience:
                        break
            with torch.no_grad():
                for p, best_p in zip(parameters, best_parameters):
                    p.copy_(best_p)
            summaries.append(dict(stage, steps=step, best_step=best_step,
                                  best_validation_nmse=best,
                                  stop_reason="patience" if step-best_step >= patience else "stage_budget"))
    finally:
        for p, flag in zip(model.parameters(), original_flags):
            p.requires_grad_(flag)
    synchronize(model.D.device)
    model.eval()
    return history, summaries, perf_counter()-start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new experiment directory")
    parser.add_argument("--mode", choices=("alista", "learned-ista"), default="alista")
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--initial-lambda", type=float, default=.4)
    parser.add_argument("--support-percent", type=float, default=0.)
    parser.add_argument("--max-support-percent", type=float, default=13.)
    parser.add_argument("--probability", type=float, default=.1)
    parser.add_argument("--snr-db", type=float, default=math.inf)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument("--steps-per-stage", type=int, default=2000)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--refinement-factors", type=float, nargs="*", default=[.2, .02])
    parser.add_argument("--gradient-decay", type=float, default=.3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    if args.seed < 0 or min(args.batch_size, args.validation_samples, args.steps_per_stage,
                           args.validate_every, args.patience) < 1:
        parser.error("sample counts and budgets must be positive; seed nonnegative")
    device = configure_runtime(args.device, args.threads, args.seed)
    d, w, analytic_metadata = load_weights(args.weights)
    problem = SparseProblem(d, probability=args.probability, snr_db=args.snr_db)
    model = ALISTA(d, w if args.mode == "alista" else d, layers=args.layers,
                   initial_lambda=args.initial_lambda, support_percent=args.support_percent,
                   max_support_percent=args.max_support_percent, mode=args.mode).to(device)
    list(training_stages(args.layers, args.lr, args.refinement_factors, args.gradient_decay))
    validation = problem.sample(args.validation_samples, make_rng(args.seed, "validation"))
    # Fail on unusable zero-energy validation before creating an experiment.
    nmse(torch.zeros_like(as_tensor(validation[1], model)), as_tensor(validation[1], model))
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/"validation.npz", D=d, b=validation[0], x=validation[1])
    config = vars(args).copy()
    config.update(weights=str(args.weights.resolve()), output=str(args.output.resolve()),
                  snr_db=None if args.snr_db == math.inf else args.snr_db)
    metadata = dict(config=config, problem=problem.configuration(), analytic=analytic_metadata,
                    dictionary_sha256=analytic_metadata["dictionary_sha256"], seed=args.seed,
                    validation_sha256=file_hash(args.output/"validation.npz"),
                    sources=source_hashes(), python=platform.python_version(),
                    numpy=str(np.__version__), torch=str(torch.__version__),
                    scope="Section 5.1 sparse recovery; not QDM field reconstruction")
    write_json(args.output/"configuration.json", metadata)
    with (args.output/"training.jsonl").open("w") as log:
        def progress(row):
            import json
            log.write(json.dumps(row, allow_nan=False)+"\n")
            log.flush()
            if row["step"] == 0 or row["step"] % max(args.validate_every, 500) == 0:
                print(f"depth {row['depth']:2d} {row['stage']:12s} step {row['step']:5d} "
                      f"validation NMSE {row['validation_nmse_db']} dB", flush=True)
        _, summaries, seconds = train_model(
            model, problem, validation, seed=args.seed, batch_size=args.batch_size,
            steps_per_stage=args.steps_per_stage, validate_every=args.validate_every,
            patience=args.patience, lr=args.lr, refinement_factors=args.refinement_factors,
            gradient_decay=args.gradient_decay, callback=progress)
    metadata.update(training_seconds=seconds, stages=summaries,
                    trainable_parameters=sum(p.numel() for p in model.parameters()))
    save_checkpoint(args.output/"model.pt", model, metadata)
    write_json(args.output/"training_summary.json", metadata)
    print(f"Saved {args.output/'model.pt'} ({metadata['trainable_parameters']} learned scalars)")


if __name__ == "__main__":
    main()
