# ALISTA: implementation and experiment guide

This implements **dense ALISTA sparse recovery** from Liu, Chen, Wang and Yin,
*ALISTA: Analytic Weights Are As Good As Learned Weights in LISTA*, ICLR 2019.
The mathematical targets are Eq. (15), Eq. (16), and Algorithm 1 / Appendix E.1.
The synthetic experiment follows Section 5.1's signal distribution.

This is a separate, runnable benchmark. It does **not** silently replace the
project's padded magnetic-field operator with a dense matrix, local kernel, or
circular convolution. Convolutional ALISTA (Section 3), Robust ALISTA (Section 4),
and HyperLISTA are not implemented in this change. The existing ConvLISTA code
and QDM comparisons keep their existing definitions.

## Mathematical contract

Let `D` have shape `[N measurements, M coefficients]` and unit-norm columns.
Observations satisfy `b = D x + noise`. This implementation requires explicit
column normalization: normalizing an existing physical operator also changes
coefficient units, and must not happen invisibly.

Stage 1 solves, without training examples:

\[
\min_W \|W^T D\|_F^2,\qquad d_j^T w_j=1\quad\text{for every column }j.
\]

The default projected gradient solver uses `G = D D.T` and updates
`W = project(W - G W / ||D||_2²)`. Projection adds
`d_j (1 - d_j.T w_j)/(d_j.T d_j)` to each column. **The printed projection in
Appendix E.1 uses the wrong correction direction; the authors' MATLAB code uses
`d_j`, which is also the orthogonal projection derived from the constraint.**
The omitted factor two in the paper's displayed gradient is absorbed into the
step size; we minimize the same objective.

The solver checks feasibility and relative constrained stationarity:

\[
\frac{\|G W-D\,\operatorname{diag}(
  d_j^T(GW)_j/\|d_j\|^2)\|_F}{\max(1,\|G W\|_F)}.
\]

An optional independent SVD solver solves the same quadratic problem via
`u_j = (D D.T)^dagger d_j`, `w_j = u_j/(d_j.T u_j)`. It reports numerical rank
and uses the standard machine-precision singular-value cutoff. No ridge term is
added. Poorly conditioned inputs can fail the stationarity tolerance; the CLI
then refuses to write a successful weight artifact. The working-memory guard
estimates dense arrays; it is not a guarantee about platform peak memory.

Stage 2 fixes `D` and `W` and learns only:

\[
x_{k+1}=\operatorname{soft}_{\theta_k}
\left[x_k+\gamma_k W^T(b-Dx_k)\right],\qquad x_0=0.
\]

There are **2K trainable scalars**, i.e. 32 at 16 layers. `D` and `W` are saved
buffers, not trainable parameters. Runtime observations and coefficients are
batch-major, `[batch, N]` and `[batch, M]`. Matrix storage is still `2NM` numbers;
few learned parameters do not eliminate fixed-operator or activation memory.
The model does not extrapolate beyond its configured depth or claim convergence
for arbitrary trained scalars.

## Modules

| File | Responsibility |
| --- | --- |
| `scripts/alista_weights.py` | Eq. (16), PGD / SVD solutions, diagnostics, weight artifact CLI |
| `scripts/alista.py` | Eq. (15), optional support selection, model save/load |
| `scripts/alista_data.py` | Bernoulli-Gaussian coefficients, measurement noise, separate random streams |
| `scripts/alista_experiment.py` | Metrics, timing, dense ISTA/FISTA, provenance utilities |
| `scripts/train_alista.py` | Layerwise scalar training and validation selection |
| `scripts/compare_alista.py` | Validation-only baseline tuning, independent test evaluation and plots |
| `tests/test_alista.py` | Independent equation, optimization, gradient, data and CLI checks |

## Run the pipeline

From the repository root, use Python with the existing `requirements-lista.txt`
dependencies installed. CPU is the default; `--device cuda` enables one CUDA GPU.
No TensorFlow or MATLAB is required. Generated results belong under `runs/`.

```bash
python scripts/alista_weights.py --output runs/alista/weights.npz

python scripts/train_alista.py \
  --weights runs/alista/weights.npz \
  --output runs/alista/soft

python scripts/compare_alista.py \
  --checkpoint runs/alista/soft/model.pt \
  --output runs/alista/soft_test
```

Default problem: `D: 250 x 500`, unit-normalized Gaussian columns, nonzero
probability 0.1, standard Gaussian nonzero values, 16 layers, zero initial code,
1,000 validation examples and 1,000 independent test examples. The dictionary
is fixed throughout the experiment. The initial scalars follow the released
implementation: `gamma=1`, `theta=0.4/(1.001 ||D||_2²)`.

Training uses fresh batches of 64 examples. The schedule trains the newest
layer, then refines all layers up to that depth at learning-rate factors 0.2 and
0.02. Earlier-layer gradients receive the released implementation's age factor
0.3 per depth. The initial learning rate is 0.0005. Nonnegative thresholds are
projected after updates. Step sizes are learned without an extra bound.

**These are equations-faithful experiments, not a claim to reproduce the
paper's numerical tables.** Deliberate training/numerical choices are:

- PyTorch and NumPy replace TensorFlow 1.12 / MATLAB; RNG sequences differ.
- Default training is bounded at 2,000 updates per stage. The authors' maximum
  was 200,000; use `--steps-per-stage 200000 --patience 2000` for that budget and
  the ALISTA README's patience setting. Budget alone does not ensure identical
  training results.
- We restore each stage's best observed validation parameters, including the
  initial stage state. The released training loop proceeds with its final state.
- We enforce the paper's nonnegative soft threshold. The released support
  selection helper permits a negative learned threshold.
- PGD uses a spectral step and constrained optimality tolerance, rather than
  the MATLAB script's fixed 0.1 step and objective-change stopping test.
- Parameter states, dictionaries, source hashes, validation data and settings
  are saved so the actual experiment can be audited.

For a short pipeline check, reduce stage updates without changing the dictionary
or model dimensions:

```bash
python scripts/train_alista.py \
  --weights runs/alista/weights.npz --output runs/alista/short \
  --steps-per-stage 200 --validate-every 20 --patience 200
```

Each command refuses to overwrite its output. A training directory contains
`model.pt`, `validation.npz`, `configuration.json`, `training.jsonl`, and
`training_summary.json`. The checkpoint is self-contained for inference, but
comparison also requires its adjacent validation artifact. It is an inference
checkpoint, not an optimizer-resume checkpoint.

```python
from alista import load_checkpoint  # scripts/ on Python's import path
model, metadata = load_checkpoint("runs/alista/soft/model.pt")
# b must be a torch float32 tensor [batch, measurements] on the model's device.
prediction = model(b)
```

For another normalized dictionary, supply `--dictionary path/to/D.npy` to the
weight solver. This CLI still generates synthetic sparse coefficients for
training; it does not load the QDM field arrays as coefficient targets.

## Support selection and a useful control

Plain soft thresholding is the default Eq. (15) model. The authors' released
ALISTA recipe also uses support selection, which exempts sufficiently large
entries from shrinkage. Enable its published schedule explicitly:

```bash
python scripts/train_alista.py \
  --weights runs/alista/weights.npz --output runs/alista/support \
  --support-percent 1.2 --max-support-percent 13
```

For layer `k` starting at one, the selected percentage is `min(1.2*k, 13)`.
Selection uses strict comparisons to both the threshold and the nearest
percentile cutoff, separately per example. It preserves TensorFlow 1.12's
descending-sort, ties-to-even rank rounding, including the halfway cases that
differ from modern NumPy/PyTorch ascending percentiles. Ties at the percentile
are not selected. This is not a promise to select exactly a given number of entries.

The `--mode learned-ista` control learns the same number of scalars but uses
`W=D` and starts with the ISTA step size. It is explicitly labeled as a control,
**not ALISTA**, because it does not use the analytic minimizer.

```bash
python scripts/train_alista.py \
  --weights runs/alista/weights.npz --output runs/alista/control \
  --mode learned-ista

python scripts/compare_alista.py \
  --checkpoint runs/alista/soft/model.pt runs/alista/support/model.pt \
               runs/alista/control/model.pt \
  --output runs/alista/comparison
```

Comparisons require identical dictionary, depth, distribution, seed and
validation data. The evaluator includes each untrained initialization, ISTA and
FISTA at the network depth, and longer fixed-budget baselines (default 500).
ISTA/FISTA lambda is chosen separately for each method and budget from the
declared grid **using validation only**. The grid is saved; it is not exhaustive
hyperparameter optimization. The long baseline is not reported as converged.

Test NMSE is `sum ||prediction-truth||² / sum ||truth||²`, then `10 log10` for
decibels. This is the paper's aggregate-energy definition, not the QDM range
NRMSE. Lower dB is better. Exact zero error has JSON `nmse_db: null` (negative
infinity). Timings are synchronized median inference times for the whole test
batch, after warm-up, excluding metrics and disk I/O. Fixed-buffer bytes,
training time, and analytic preparation time are separately reported where
applicable; no measured peak-memory claim is made.

Outputs: `metrics.json`, `comparison.csv`, `curves.csv`, `comparison.png`,
`baseline_selection.json`, and `test_predictions.npz`. Seeds and source hashes
are recorded. Train/validation/test random streams also have a namespace
distinct from dictionary generation. Test data are not passed into training.

For noisy experiments set `--snr-db 20`, `30`, or `40` when training a separate
model. Default `inf` is noiseless. As in the released generator, Gaussian noise
standard deviation is the clean measurement's per-example population standard
deviation times `10**(-SNR/20)`. This is an expected noise scale, not an enforced
exact realized SNR for every example.

## QDM boundary and remaining work

Our magnetic task currently reconstructs a field directly with
`A = Crop * FFT-convolution * Pad`; the field pixels need not be sparse. The
paper's Section 5.1 instead predicts sparse coefficients under a normalized
dense dictionary. These are distinct experiments and must stay labeled.

A 1400-by-1400 field has 1,960,000 pixels; one square float32 matrix on that
space would require about 15.4 TB. The new CLI guards dense working memory and
does not attempt that allocation. If a square operator is invertible and W is
unrestricted, Eq. (16) yields `W.T = D^-1`, potentially amplifying noise. Simply
calling the existing adjoint an analytic weight would miss ALISTA's defining
optimization.

A field experiment therefore still needs a justified sparse representation
and an analytic-operator construction that respects the current crop/pad
boundaries. The paper's compact convolutional dictionary is a possible design,
but identifying it with our long-range magnetic blur requires additional work.
The present synthetic results must not be presented as magnetic reconstruction
performance or evidence of generalization to other dictionaries/heights.

## Verification and sources

### Completed local validation, 2026-10-15

The full suite ran **56 tests: 55 passed, one skipped** (the existing two-CUDA-GPU
test; this machine has no such devices). All 22 new ALISTA tests passed. The
tested environment was Python 3.13.5, PyTorch 2.14.1, NumPy 2.3.1 on CPU, in
`/tmp/learning-ker-alista-venv`. This temporary environment is not a portable
project dependency installation; the dependency list remains `requirements-lista.txt`.

A bounded experiment used the paper's 250-by-500 dimensions and 16 layers,
seed 0, noiseless measurements, 200 updates per stage, validation every 20
updates, 1,000 validation examples and 1,000 independent test examples. There
were three stages per layer. Results are local under
`runs/alista_validation/comparison_verified/` (the repository ignores `runs/`).

| Method | Layers / iterations | Test NMSE, dB |
| --- | ---: | ---: |
| ALISTA, ordinary soft threshold | 16 | -28.0063 |
| ALISTA, support selection | 16 | -54.6781 |
| Learned-ISTA control | 16 | -12.1351 |
| ISTA | 16 | -6.2607 |
| FISTA | 16 | -10.9823 |
| ISTA | 500 | -22.2125 |
| FISTA | 500 | -55.3376 |

This is one seed and one noiseless fixed dictionary, with a short training
budget. It validates the pipeline, not a reproduction of the paper's reported
table or a QDM performance claim. Support-selection and control training ran
concurrently; their recorded wall times should not be used for training-speed
comparisons against the separately run soft-threshold model. See the JSON for
inference timing samples, baseline lambda grid/selection, and provenance.

The original `comparison/` directory preceded the final TensorFlow percentile
edge-case correction. All saved predictions were checked bit-for-bit against
`comparison_verified/`: this 500-coordinate, 1.2%-increment, 13%-cap schedule
does not hit the affected halfway ranks. The regression test also covers those
halfway cases. Training checkpoints retain the actual source hashes from
training; the verified comparison records the final evaluation source hashes.

```bash
python -m unittest discover -s tests -p 'test_alista.py' -v
python -m unittest discover -s tests -v
```

Tests cover independent KKT solves, PGD/SVD agreement including rank deficiency,
projection feasibility, the square inverse case, layer-by-layer NumPy parity,
finite-difference scalar gradients, support selection, fixed-buffer integrity,
checkpoint round trips, baseline correctness, noise scaling, independent
splits, validation selection, and the complete three-command pipeline.

- [Paper and bibliographic record](https://openreview.net/forum?id=B1lnzn0ctQ)
- [Authors' repository and experiment commands](https://github.com/VITA-Group/ALISTA)
- [Analytic optimization reference](https://github.com/VITA-Group/ALISTA/blob/master/matlabs/CalculateW.m)
- [Network and scalar initialization reference](https://github.com/VITA-Group/ALISTA/blob/master/models/ALISTA.py)
- [Support-selection reference](https://github.com/VITA-Group/ALISTA/blob/master/utils/tf.py)
- [Original TensorFlow 1.12 percentile semantics](https://github.com/tensorflow/tensorflow/blob/v1.12.0/tensorflow/contrib/distributions/python/ops/sample_stats.py)
- [Training schedule and noise reference](https://github.com/VITA-Group/ALISTA/blob/master/utils/train.py)

The provided paper was read locally from
`/Users/abhay/Downloads/1304_alista_analytic_weights_are_as.pdf`. This is an
independent implementation of the equations and documented algorithms; no
authors' source files are vendored.
