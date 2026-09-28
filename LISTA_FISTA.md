# LISTA, ISTA and FISTA: implementation and experiment

This extends the original padded inverse operator. ISTA/FISTA now stop by epsilon in both the standalone entry points and the comparison pipeline. The agreed experiment is the primary coil, z=3 → 0.5 µm, fitting all three component images. LISTA is supervised against the sharp COMSOL images and learns convolutional corrections to the exact padded physics operator, plus a threshold. The supplied notebook runs training on Kaggle; no full-data LISTA training result is claimed from local checks.

## Theory and the exact variant

Gregor and LeCun's [original LISTA paper](https://icml.cc/Conferences/2010/papers/449.pdf), equations 2–5 and Algorithm 3, unfolds shrinkage iterations into a network with learned input/recurrent maps and thresholds. Their supervision approximates sparse-optimization solutions. Our chosen sharp-field supervision is a different training target; this is a structured convolutional LISTA reconstruction variant, not a reproduction of that paper's experiments.

Let the original field units be µT and define

```
J(x) = 0.5 ||A x - b||² + lambda ||x||₁
A x = Crop(real(IFFT(H * FFT(Pad(x)))))
A* r = Crop(real(IFFT(conj(H) * FFT(Pad(r)))))
```

`H` is the unregularised joint kernel learned from Bx/By/Bz at both heights, exactly as in `deblur.py`. Padding is 32 pixels by default. Keeping the intermediate crop is essential: `A*A` is not generally one circular convolution with `|H|²` when padding and cropping are present.

The step is `alpha = 0.99 / max(abs(H)²)`. The maximum is an upper bound for `||A||²`; crop and zero-padding cannot increase the norm. With the half-squared data term, the gradient is `A*(Ax-b)`; there is no factor of two.

```
ISTA:
  x_next = soft(x - alpha*A*(A*x-b), alpha*lambda)

FISTA:
  x_next = soft(y - alpha*A*(A*y-b), alpha*lambda)
  t_next = (1 + sqrt(1 + 4*t²))/2
  y_next = x_next + ((t-1)/t_next)*(x_next-x)
```

FISTA starts with `y=x0, t=1` and returns `x`, not its extrapolated `y`. This is the constant-step method in Beck and Teboulle's [FISTA paper, §4](https://www.ceremade.dauphine.fr/~carlier/FISTA). Its objective-gap rate is O(1/k²), versus O(1/k) for ISTA under the stated convex assumptions. This is not a guarantee of lower reconstruction error at every iteration, and standard FISTA need not have a monotone objective.

Our learned maps are:

```
W_b = alpha*A* + C_b
W_x = I - alpha*A*A + C_x
x_next = soft(W_b*b + W_x*x, theta)
```

`C_b` and `C_x` are bias-free, single-input/single-output, zero-padded 9×9 convolutional linear maps. The weights are tied across the 8 default layers, as is a scalar nonnegative threshold. There are 163 trainable parameters with these defaults (2×81 weights + 1 threshold). The fixed `H` is a buffer, not a trainable parameter. PyTorch's Conv2d convention is cross-correlation; learned coefficients need no analytic kernel reversal.

Corrections start at zero and `theta=alpha*lambda`, so before training every layer equals ISTA, including boundaries. The corrections' small spatial support does **not** truncate the fixed physical operator's long-range response. After learning, the effective maps need not be adjoints, positive, contractive or minimizers of J. There is no LISTA convergence guarantee and running more layers than trained is not supported. Only the final layer is supervised; intermediate layers are diagnostic outputs.

All methods default to `x0=b` to match existing repository ISTA. `--initial zero` is available and applies consistently to all three; original LISTA usually starts from zero. This deviation is explicit.

## Epsilon-based stopping (29 September 2026)

The CLI and Kaggle configuration ask for `--epsilon` / `EPSILON`, not a baseline iteration count. Default **1e-3 = 0.001 = 10E-4**. This is a dimensionless relative optimality tolerance, not an image-error target or an absolute change in the objective.

For each component, at the initial image and after every update, compute:

```
G(x) = (x - soft(x - alpha*A*(A*x-b), alpha*lambda)) / alpha
relative_pg = RMS(G(x)) / RMS(G(x0))
stop when relative_pg <= epsilon
```

An initially zero residual is already converged. The GPU batch stops only when **every component** satisfies the criterion. Check the actual proximal iterate `x`, not FISTA's extrapolated `y`. Ground truth is only used for reporting reconstruction errors. Because both residuals scale together, changing between physical and normalized field units leaves this stopping criterion unchanged. The proximal-gradient mapping measures stationarity; see [Parikh and Boyd's proximal algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html).

`record_every` controls saved history frequency, not stopping checks. Final history is always saved, including when the solver stops between reporting intervals. Metrics include epsilon, relative residual, actual iterations, `converged`, and `stop_reason`. Actual counts are also shown in reconstruction titles, convergence legends/endpoints, timing labels, and `comparison_table.csv` (with LISTA trained layers in a separate column). Standalone ISTA/FISTA save per-component CSV/JSON summaries and convergence plots showing each component’s own count and stopping reason. A 100,000-update internal safety limit or 20 consecutive exactly unchanged iterates while epsilon remains unmet returns `converged=false`, with `safety_limit` or `numerical_stagnation`; neither is silently presented as convergence. Nonfinite calculations raise an error.

LISTA retains its configured trained depth, explicitly confirmed by the user. Its training epoch count also remains a training setting. The model is not repeatedly extended to force the L1 residual below epsilon. Fixed-depth ISTA/FISTA comparisons are still automatically derived from LISTA's layer count. Low-level `n_iter` remains only for these fixed-depth numerical/reference uses, while normal solver calls default to convergence stopping.

Comparisons are named `ista_to_epsilon` / `fista_to_epsilon`, alongside `lista` and same-depth diagnostics such as `ista_8`. The adaptive latency benchmark reruns the full epsilon-stopped algorithm with its stopping checks, not a cheaper fixed-count replay. Recorded `solver_seconds` also includes stopping checks, excluding extra reporting diagnostics.

## Units and loss

All six field arrays are loaded using the existing float32-T-to-µT scaling. Kernel fitting uses NumPy float64. Training/inference use PyTorch float32 without AMP or TF32. Define a single scale `q=max(abs(all sharp and blurred fields))`, then use `u=x/q`, `v=b/q` and **lambda_normalized=lambda_uT/q**. Thus

```
J(q*u)/q² = 0.5 ||A*u-v||² + (lambda_uT/q) ||u||₁.
```

Scaling both field arrays but keeping lambda unchanged would change the optimization problem. Diagnostics restore physical units: objectives multiply by q², fields/RMSE/proximal-gradient residuals by q. The tests check this scaling against the original NumPy operators.

Training minimises mean squared error `mean((LISTA(v)-sharp/q)²)` over all pixels and all three component images. It is not the L1 objective J. Adam's learning rate defaults to 1e−4; gradient norm is clipped at 1; the threshold is projected to nonnegative values after each update. Threshold parametrisation avoids freezing an extremely small initial value behind a softplus derivative. The reported best checkpoint is selected by **post-update training MSE**, not validation or test error.

Defaults of 8 layers, 9×9 filters, 200 epochs and learning rate 1e−4 are explicit starting configurations, not theoretically optimal hyperparameters. They are editable in the notebook. The physical lambda=1e−4 and pad=32 match the original ISTA defaults.

## Data and comparison protocol

There are three whole-image training examples: Bx, By and Bz from the same coil. Each is processed as one channel, with one shared model. There is no random patch split, artificial resampling, geometric augmentation or independent test set. Training preserves complete images to avoid silently changing the forward operator at patch boundaries.

The reference and target images are used to fit H for all methods, and the reference images additionally supervise LISTA. This is the agreed **same-sample fitting experiment**. Any inference about another height, coil or unseen measurement requires a separate data protocol and training experiment.

Two T4s are used through DataParallel to split the three-image training batch 2+1. MSE is computed after gathering predictions, so both chunks receive correct sample weighting. Both GPUs do not automatically imply twice the speed with so few images. All timing comparisons run on the **same single GPU**, with the same float32 dtype and three-image batch. See the official [DataParallel documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html).

Outputs compare:

- ISTA/FISTA at the same number of updates as LISTA's trained depth.
- ISTA/FISTA run until the relative proximal-gradient residual reaches epsilon (default 1e-3); actual update counts and stop reasons are recorded.
- L1 objective and proximal-gradient RMS, measuring optimization progress.
- Physical RMSE, range-normalised RMSE and relative L2 error, measuring reconstruction agreement.
- Warmed, synchronized median inference latency, separately from kernel fitting and training time.

A LISTA layer performs the physics gradient **plus** learned convolutions; it costs more than an ISTA iteration. Equal depth alone is not equal computation. No general speedup follows from layer counts alone. Training time includes post-update evaluation and best-weight copying; inference latency excludes reporting diagnostics and disk writes, but includes the convergence checks needed by adaptive ISTA/FISTA. CUDA timing explicitly synchronizes before and after each measured call.

The proximal-gradient mapping is `(x-soft(x-alpha*grad J_data,alpha*lambda))/alpha`. Its RMS is zero at an L1 optimum and is reported in physical units. It does not require a ground-truth solution or invent a reference optimum. For LISTA it is only a diagnostic because LISTA optimizes supervised MSE.

## Files and use

- `scripts/fista.py`: standalone NumPy FISTA, with saved reconstruction, kernel and JSON metadata.
- `scripts/iterative.py`: shared comparable NumPy ISTA/FISTA and objective/stationarity diagnostics.
- `scripts/torch_inverse.py`: batched differentiable padded operator and GPU ISTA/FISTA.
- `scripts/lista.py`: `ConvLISTA`, checkpoint loading, and standalone training CLI.
- `scripts/compare_lista.py`: whole-image supervised training, comparison and provenance.
- `scripts/plot_comparison.py`: rebuild plots from saved arrays/JSON.
- `tests/test_iterative.py`, `tests/test_lista.py`: numerical and learning checks.
- `notebooks/ISTA_LISTA_FISTA_Kaggle.ipynb`: upload this notebook to Kaggle.
- `kaggle_lista_bundle.zip`: add this generated bundle as a Kaggle dataset. It contains code, tests, metadata and only six necessary `.npy` arrays; no raw CSV upload is required.
- `scripts/prepare_kaggle.py`: rebuild the notebook and bundle after code changes.

The portable bundle retains original `grid.json` for provenance, although it only includes reference .5 and target 3 components. Other heights and Bnorm listed in that original metadata are deliberately not packaged.

Example local commands, using a suitable PyTorch environment:

```bash
python -m unittest discover -s tests -v
python scripts/fista.py --z 3 --comp Bz --epsilon 1e-3 --output runs/fista_z3
python scripts/lista.py --data-root . --z 3 --device cuda --gpus 2 \
  --layers 8 --kernel-size 9 --epochs 200 --lr 1e-4 \
  --epsilon 1e-3 --pad 32 --lambda-ista 1e-4 \
  --checkpoint-layers --output runs/compare_z3
```

Existing output directories are refused to prevent overwriting prior runs. `--checkpoint-layers` trades recomputation for lower training memory without changing the forward calculation. It is enabled in the notebook. Full-image training remains GPU work; the notebook leaves configuration and the training cell visible for you to run.

Kaggle's installed PyTorch/CUDA pairing is used, not replaced with a CPU wheel. NumPy, pandas and Matplotlib are also required. The notebook logs all actual versions/device names; the reproducibility controls disable TF32/AMP and request deterministic algorithms. As PyTorch's [reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) explain, identical seeds do not guarantee bitwise equality across different versions/devices.

## Saved artifacts

Each comparison directory contains:

| File | Contents |
|---|---|
| `lista_best_training.pt` | CPU state dict, physics buffers, exact original norm bound/step configuration, scaling, architecture, selected epoch and loss. |
| `kernel.npy` | The NumPy complex128 empirical transfer before conversion to float32 buffers. |
| `reconstructions.npz` | Sharp reference, observation, original untrained LISTA, trained LISTA, ISTA and FISTA arrays, all in µT. |
| `metrics.json` | Per-iteration objectives/errors, training history, same-depth/epsilon-stopped scores and latency samples, parameter changes, configuration, environment and SHA-256 input/source hashes. |
| `reconstructions.png` | Full-image reference/observation/three-method comparison; common percentile display scale per component. |
| `convergence.png` | Objective, reference NRMSE, physical stationarity, and relative stationarity with the epsilon line versus iteration/layer. |
| `training_and_latency.png` | Training loss and inference latencies. |

To reload a checkpoint without fitting H or retraining:

```python
import torch
from lista import load_checkpoint
model, metadata = load_checkpoint('run/lista_best_training.pt', device='cuda:0')
# observed_uT: torch.float32 [batch,1,1400,1400], on cuda:0
with torch.no_grad():
    reconstructed_uT = model(observed_uT / metadata['scale_uT']) * metadata['scale_uT']
```

The loader restores the original double-fit norm bound rather than recomputing it from rounded float32 buffers. This preserves the exact forward step of the saved model. Checkpoint loads use `weights_only=True`.

## Verification and limitations

Local checks pass for: padded adjoint against an explicit matrix transpose; FFT operator/autograd against NumPy; ISTA equivalence to existing code; FISTA against an explicit-matrix recurrence and an exact L1 solution; all-layer LISTA initialization equivalence; learned weight/threshold gradients against finite differences; physical-unit normalization; checkpointed versus ordinary gradients; training loss/weight updates; and checkpoint round-trip output equality.

The two-CUDA-device gradient-weighting check is included and runs in the Kaggle notebook; it is skipped locally because CUDA is unavailable. Convergence tests additionally check known solutions, NumPy/PyTorch stopping parity, unit scaling, logging independence, stricter tolerances, initial optima, and honest safety-limit reporting. Saved historical Kaggle runs in `notebooks/runs_results` use fixed 200/500-update baselines; they are preserved and are not results of the new epsilon-stopped protocol. Local synthetic checks verify implementation, not full-data performance.
