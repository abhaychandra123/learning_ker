# Shared ConvLISTA on the combined COMSOL dataset

Prepared 22 October 2026. The user selected **one shared model** trained on
`data/` and `data/data2/`. The original single-pair experiment remains available
unchanged. No upload, HPC training, or new scientific score is implied by this
implementation.

This README describes the implemented experiment, why it differs from the
previous one, how to run it, and where to record its results. **No full-resolution
combined-data training results exist yet.** Small synthetic checks below validate
the implementation; their scores are not research results.

## What changed from the previous experiment?

The core reconstruction method remains physics-guided convolutional LISTA,
compared against ISTA and FISTA using the same padded Fourier forward/adjoint
operators. This is an expansion of the existing experiment, not a switch to a
different reconstruction family. The important difference is that one small
network must now serve several geometry/height-dependent inverse problems.

| Aspect | Previous single-pair pipeline | New combined pipeline |
|---|---|---|
| Input data | Original L1_C1 cache | Original L1_C1 CSVs plus IC_4um and IC_2um CSVs |
| Height problem | One selected target height → 0.5 µm, usually 3 → 0.5 | 17 explicitly recorded height pairs, anchored within each geometry |
| Supervision | Three whole images: Bx, By, Bz | 51 whole component pairs across three geometries |
| Network sharing | Corrections tied across layers and components of one pair | Same corrections tied across layers, components, and **all pairs** |
| Learned parameter count | 163 with 9×9 filters | Still 163 with 9×9 filters |
| Fourier operator | One frozen kernel fitted jointly to that pair's components | One frozen kernel **per pair**, fitted jointly to its components |
| Normalization | One scale for the six fields in the selected pair | One scale for all input/reference fields in the training split |
| Learned threshold | Direct normalized shrinkage threshold for one fixed step | Shared normalized threshold coefficient multiplied by each pair's step |
| Training update | One full three-image batch per epoch | Accumulate gradients over all 51 component pairs, then one update per epoch |
| GPU training | Supports one or two GPUs | One GPU; default one component per batch |
| Layer count | Archived experiments used 8 and 16; old CLI defaults to 8 | Defaults to 16; configurable |
| Optimizer and epochs | Adam, lr 1e-4, 200 epochs, norm clip 1 | Same defaults |
| Classical stopping | Archived Kaggle scores used 200/500 fixed updates; current old code uses epsilon | Epsilon-based stopping plus fixed trained-depth diagnostics |
| Recovery after timeout | Original ConvLISTA CLI has no optimizer resume | Atomic latest checkpoint every epoch, including Adam state |
| Model selection | Best post-update training MSE, including epoch 0 | Same by default; validation MSE only in optional holdout mode |
| Scientific claim | Same-sample reconstruction | Same-sample reconstruction by default; separately labeled optional geometry holdout |
| Outputs | Single-pair comparison files and figures | Per-pair/component results, overall summaries, inventory and a combined report |

The old modules `lista.py`, `compare_lista.py`, `torch_inverse.py`, and the legacy
cache were not changed for this extension. The new runner reuses the existing
kernel fitter, padded operator, classical solvers, metric definitions and timing
helpers. Reproducing the old single-pair GPU run remains a separate later task.

### What stays mathematically the same

For each pair, the classical problem remains

```text
minimize_x  0.5 ||A x - b||² + lambda ||x||₁
A(x) = Crop(real(IFFT(H * FFT(Pad(x)))))
A*(r) uses conj(H) with the same padding and cropping
step = 0.99 / max(abs(H)²)
```

Signed Bx/By/Bz are the linear components; Bnorm is excluded. Padding remains
32 by default, kernels use unshifted FFT layout, and updates retain the crop
between A and A*. LISTA still learns against sharp-field MSE, rather than the
classical L1 objective. Finishing optimization of that objective does not
necessarily minimize error against the reference image.

These are fixed-height reconstruction experiments. Unknown-standoff estimation,
adaptive depth, optical deconvolution, new noise augmentation and demonstrated
physical super-resolution have **not** been added here.

## Dataset and pairing

| Geometry | Available heights (µm) | Default sharp reference | Reconstruction pairs |
|---|---|---|---:|
| L1_C1 | 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 | 0.5 µm | 10 |
| IC_4um | 0.5, 3, 4, 5, 6, 7, 10 | 0.5 µm | 6 |
| IC_2um | 1, 2 | 1 µm | 1 |

The default `anchor` pairing reconstructs each geometry's lowest available
plane from each higher plane: **17 pairs × Bx/By/Bz = 51 component pairs**.
IC_2um is explicitly 2 → 1 µm, not 2 → 0.5 µm. No pairing crosses geometries.
The prepared dataset uses separate geometry folders; neither original CSVs nor
the old `cache/` are modified.

`--pairing all` is an optional different experiment: every lower/higher pair
within a geometry (55 + 21 + 1 = 77 pairs / 231 component pairs). The job template
defaults to anchor. Pairing policy is saved in the dataset manifest.

Every source CSV is checked for units, finite values, a complete uniform raster,
and matching coordinates across that geometry's heights. Signed fields are saved
as `[3, y, x]` float32 **µT** arrays, with metadata and SHA-256 checksums. This is
distinct from the legacy cache's float32 **tesla** format. Conversion from the raw
float64 CSV can differ slightly from first rounding to float32 tesla. All source
paths and hashes are retained. Manifest and cache paths are portable.

## Shared model and comparison protocol

The model retains two bias-free 9×9 convolutional corrections tied across layers,
and one nonnegative scalar: **163 learned parameters total**, shared across all
geometries, heights and signed components. Each pair supplies its own frozen
padded FFT operator and `step = .99 / max(abs(H)**2)`:

```text
x_next = shrink(x - step A*(A x - b) + C_b b + C_x x, step * theta)
```

The shared `theta` is in normalized field units. Unlike the old single-operator
model, the step is applied to the threshold at evaluation time, allowing one
scalar to work with different operator bounds. Zero corrections and
`theta = lambda_uT / scale` reproduce ISTA for every pair. Learned updates have
no established convergence guarantee; use only the trained layer count.

Default run:

- 16 layers, 200 epochs, Adam learning rate 1e-4; physical lambda 1e-4 µT;
  padding 32 pixels; warm start from the blurred field.
- One global scale: maximum absolute value of the input/reference fields used
  for **training only**. Physical units are restored for scores and arrays.
- Equal weight per component pair in normalized MSE. One epoch visits all
  training pairs; gradients accumulate to **one optimizer update per epoch**.
  Gradient norm is clipped to 1. No cropping or augmentation.
- One GPU. `--batch-size 1` streams components to bound activation memory;
  optional 2 or 3. Activation checkpointing is enabled in the HPC script.
- Post-update MSE selects the best checkpoint, including epoch 0. Every epoch
  writes a separate latest checkpoint with optimizer state for restarting.
- Baselines use exactly the same pair operator, normalization, initialization,
  precision, device and component batch. Report ISTA/FISTA at 16 updates and
  at relative proximal-gradient epsilon 1e-3. Convergence requires every
  component in the current batch to pass; actual counts are saved.
- Timings are synchronized warmed medians, exclude file loading/reporting, and
  include adaptive stopping checks. Batch timings are not per-image timings.
  The sum of batch medians excludes dataset I/O and is labeled accordingly.

By default, each pair's Fourier kernel is jointly fitted to Bx/By/Bz using that
pair's references, and all pairs train the shared network. Results are therefore
**same-sample reconstructions**, not unseen-geometry accuracy. This preserves the
original experiment's scientific scope while expanding its data.

An explicit optional holdout mode is available: use `--operator analytic` with
`--validation-geometry` and/or `--test-geometry`. Entire geometries are excluded
from training. Analytic kernels depend on grid spacing and height difference,
not reference fields. Training scale is computed only from training geometries;
checkpoint selection uses validation MSE when present, otherwise training MSE.
Test data are scored only after checkpoint selection. Joint reference-fitted
kernels are rejected for holdout runs to prevent target leakage. With only three
geometry groups, such a split is a small demonstration, not broad generalization
evidence. It is a different operator experiment from the default joint-fit run.

### Why bounded batches do not change the intended training loss

A "batch" here is a few **complete field images**, not a crop of a field. The
default batch contains one 1400×1400 signed-component image. The code visits the
three components of a pair, releases that pair's device data/operator, then moves
to the next pair. All images remain at full spatial resolution. The report may
subsample previews, but metrics and saved reconstructions use the full arrays.

For N training pairs, there are 3N supervised component examples. Each component
has weight `1/(3N)`. For a batch of m components, its mean pixel MSE contributes
`m/(3N)` to the accumulated loss. In simplified form:

```text
zero gradients
for every training pair:
    load its fields and frozen operator
    for each component batch:
        loss = mean((LISTA(input, operator) - reference)²)
        backward(loss * components_in_batch / total_training_components)
clip the accumulated gradient norm to 1
take one Adam step
project the threshold coefficient to be nonnegative
evaluate post-update losses and save the checkpoint
```

Thus changing batch size from 1 to 3 does not deliberately change the objective
or number of optimizer updates. A numerical test checks accumulated-gradient
agreement. Floating-point summation order can still cause small differences.
Batch size is part of the saved run configuration and cannot change during a
resume. There is no shuffling, random augmentation, dropout, or stochastic data
selection in this implementation.

The weighting is **per component pair, not per geometry**. In the default setup,
L1_C1 contributes 30/51 examples, IC_4um contributes 18/51, and IC_2um contributes
3/51. The sharp reference is reused for each target height within its geometry.
These examples are correlated; 51 component pairs are not 51 independent source
geometries. Report geometry-specific results as well as an overall mean.

### Exact normalization and threshold behavior

Let `q` be the largest absolute µT value in all training input/reference planes.
Both inputs and targets are divided by the **same q**; there is no per-image
mean subtraction, min–max rescaling, per-component scale, or independent
normalization of the sharp and blurred fields. The L1 baseline coefficient becomes
`lambda_normalized = lambda_uT / q`. Predictions and residual metrics are restored
to physical units using q.

This global normalization is a deliberate change from separately normalizing
each old pair. It preserves relative field amplitudes across the combined data.
The old model's learned scalar was the final shrinkage threshold `tau`. With one
fixed operator step `s`, the new model's threshold `s * theta` is an equivalent
parameterization (`tau = s * theta`). With multiple operators, their steps differ;
sharing theta provides an ISTA-consistent initialization for every pair. A single
fixed tau could not generally do that. This is a small but real architectural
adaptation, so the new model should not be described as an unchanged old
checkpoint run on more files.

### What is loaded and saved during a run

1. Validate the prepared manifest, each cached field's shape/finite values and
   checksum, and the within-geometry pairing rules.
2. Assign whole geometries to train/validation/test when requested; otherwise
   assign every pair to training.
3. Calculate the training-only scale and prepare one frozen kernel per pair.
4. Train the shared correction weights using the streamed gradient accumulation
   above. Reevaluate the updated weights before choosing the best checkpoint.
5. Load the selected best weights and run every comparison method on every
   declared pair. Test fields do not participate in selection.
6. Save component-level metrics, stopping records, optional full arrays, summary
   tables, figures, and an HTML report.

Kernel fitting and field loading are outside the inference benchmark. The
training timer covers epoch work including post-update loss evaluation; the
initial kernel-preparation stage is separate and checkpoint/report I/O is not a
precise total job-walltime measurement. Use scheduler accounting for total job
time and the saved CUDA peak statistics for device-memory use.

For classical baselines, the default one-component batch means each component
stops independently. With a three-component batch, all three must pass before
that batch stops. This can change reported iteration counts relative to the
old three-image batch without changing the tolerance. Stationarity, reference
error and iteration count remain separately reported.

## Files

| File | Purpose |
|---|---|
| `scripts/combined_data.py` | Discover, validate, convert, and enumerate pairs |
| `scripts/combined_lista.py` | Shared ConvLISTA and model loader |
| `scripts/compare_combined.py` | Training, resume, baselines, timing and metrics |
| `scripts/plot_combined.py` | Regenerate plots and an HTML report from saved results |
| `hpc/check_cuda.py` | GPU FFT/autograd check; defaults to Slurm, explicit `--allow-direct` for the professor's server |
| `hpc/combined.sbatch` | Prajna one-GPU preparation/training/comparison job |
| `hpc/run_combined_direct.sh` | Same experiment on a direct GPU server without Slurm |
| `hpc/combined_dataset_inventory.json` | Verified inventory of the existing raw CSVs, grids, heights and source hashes |
| `tests/test_combined.py` | Data integrity, numerical, split, streaming and restart tests |

The inventory is a record of the files inspected on 22 October 2026, not a
replacement for the live manifest or checksum checks. Rebuild the prepared
dataset in a fresh directory if raw data change.

## Local inventory and preparation

Use Python 3.11+ with `requirements-lista.txt`. The cluster needs a Linux
environment with a CUDA-enabled PyTorch compatible with its driver; do not copy
a Mac environment. The exact site environment is still to be identified.

From the project root:

```bash
python scripts/combined_data.py --raw-root data \
  --output runs/combined_dataset --inventory-only

# Optional local conversion; the HPC job can perform this instead.
# Output must not already exist.
python scripts/combined_data.py --raw-root data \
  --output runs/combined_dataset --pairing anchor

python -m unittest discover -s tests -v
```

Conversion takes place once and can run on CPU. Run numerical workloads and
training on allocated compute nodes at the HPC. The prepared signed-field cache
is about 470 MB (20 × 3 × 1400 × 1400 × 4 bytes), in addition to the raw CSVs.
Kernel arrays and saved reconstructions add several GB per full comparison run.
Use a verified writable location with sufficient space; consult `HPC_USAGE_GUIDE.md`.

## Professor's GPU server: direct execution

The user switched to `abhaychandra@10.129.6.189`. Their observed host is
`user-ESC4000A-E10`, Linux x86_64, with an RTX A6000 (49140 MiB reported), NVIDIA
driver 590.48.01, system Python 3.10.12, and about 89 GB available on the shared
`/home` filesystem at inspection. No Slurm/PBS executable appeared on PATH.
The GPU was idle in that snapshot; this is not a persistent reservation. Use a
separate Python 3.11+ project environment. Its installation and actual GPU
functionality on this server still need verification.

`run_combined_direct.sh` changes only the launch procedure: it uses no scheduler
variables, runs the explicit direct-mode GPU check, saves an environment listing,
prepares data if needed and calls the **same** combined comparison command with
the same defaults. It does not change the model, pairs, scaling, loss, solver
stopping or report definitions. Unlike Slurm, it imposes no walltime limit.

After uploading to `~/learning_ker` and configuring `~/qdm-env`, first check:

```bash
cd "$HOME/learning_ker"
"$HOME/qdm-env/bin/python" hpc/check_cuda.py --allow-direct
```

Then start a two-epoch combined-data pilot that survives SSH disconnection:

```bash
cd "$HOME/learning_ker"
export QDM_PYTHON="$HOME/qdm-env/bin/python"
export QDM_DATASET="$HOME/learning_ker/runs/combined_dataset"
export QDM_RUN="$HOME/learning_ker/runs/combined_pilot"
mkdir -p logs

QDM_LOG="logs/pilot-$(date +%Y%m%d-%H%M%S).log"
nohup bash hpc/run_combined_direct.sh --epochs 2 > "$QDM_LOG" 2>&1 < /dev/null &
echo "$!" > logs/combined.pid
echo "PID: $(cat logs/combined.pid); log: $QDM_LOG"
```

The two-epoch pilot also runs all classical comparisons and can take substantial
time. Starting a background process does not establish success: inspect its log
and `runs/combined_pilot/status.json`. `status.json` is first created after data
and operator preparation. Check GPU use with `nvidia-smi`; follow the reported log
with `tail -f PATH_TO_LOG` (Ctrl+C exits the log viewer, not the training process).
Do not launch overlapping jobs against the same run or preparation directory.

After the pilot finishes successfully, the same environment variables and
`QDM_RUN` can continue its training to 200 epochs:

```bash
QDM_LOG="logs/resume-$(date +%Y%m%d-%H%M%S).log"
nohup bash hpc/run_combined_direct.sh --resume --epochs 200 > "$QDM_LOG" 2>&1 < /dev/null &
echo "$!" > logs/combined.pid
echo "PID: $(cat logs/combined.pid); log: $QDM_LOG"
```

After reconnecting, re-enter the project directory and set the three QDM variables
before launching a new job. Keep the logs and environment listing with the results.

## Prajna submission (alternative)

First transfer the updated source, `data/` (including `data/data2/`), and this
guide. The old primary `cache/` may be transferred for future reproduction but
is not read by the new combined experiment. Submit from the uploaded project
root. Set the three paths below to **real, verified cluster paths**, not these
placeholders. No remote environment installation has been performed.

```bash
export QDM_PYTHON=/ABSOLUTE/PATH/TO/LINUX_ENV/bin/python
export QDM_DATASET=/ABSOLUTE/WRITABLE/PATH/combined_dataset
export QDM_RUN=/ABSOLUTE/WRITABLE/PATH/combined_pilot

# Scheduler acceptance check (does not test Python/data).
sbatch --test-only --time=01:00:00 hpc/combined.sbatch --epochs 2

# A bounded pilot on the NEW combined dataset, not the old experiment.
sbatch --time=01:00:00 hpc/combined.sbatch --epochs 2
```

The pilot includes all pairs and epsilon-stopped baselines and may exceed an
hour; this is a scheduler bound, not a runtime estimate. The one-day template
limit is likewise a starting value, not a measured requirement. The job checks
CUDA, prepares the dataset if absent, trains, compares, and writes reports.
If an interrupted conversion leaves a directory without `manifest.json`, use a
fresh dataset path rather than accidentally treating it as complete.

After a successful pilot and memory/runtime inspection, continue the same run
to 200 epochs with the same settings and paths:

```bash
sbatch hpc/combined.sbatch --resume --epochs 200
```

Alternatively, choose a fresh `QDM_RUN` and omit `--resume` to start a new run.
Do not launch concurrent jobs against the same dataset-preparation or run folder.
Job output logs and environment package lists are saved in the submission
directory; preserve them with the run. Monitor with `squeue --me` and inspect
`sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS` after completion.

Resume is available once `last.pt` has been written (after epoch 0 evaluation).
It restores the latest model, Adam state and best-checkpoint selection. Work
within an interrupted epoch is repeated. Data, source hashes and settings must
match; only the requested total epoch count and filesystem locations may change.
Resume uses the latest optimizer state, not the best model, so it continues
training correctly. A restarted comparison currently recomputes its baselines
and replaces its own report files. Never run two jobs in one output directory.

Optional holdout example, in a **fresh** QDM_RUN:

```bash
sbatch hpc/combined.sbatch --operator analytic \
  --validation-geometry IC_4um --test-geometry IC_2um --epochs 200
```

This trains only on L1_C1, validates on IC_4um, and tests on IC_2um. It is not the
default requested all-data fitting run. Do not retune against its test scores.

## Outputs and completion checks

- `setup.json`: pairing/splits, units, scale, source/data/operator hashes and
  initial environment. `physics/`: frozen per-pair kernels.
- `last.pt`: latest training/optimizer state; `best.pt`: selected inference
  weights and model configuration. Load with `load_shared_checkpoint`; provide
  the pair's `PaddedFFT` operator when calling the model.
- `training.json`: post-update training/validation losses and cumulative time.
- `comparison.csv`: all methods per geometry, pair and component; range NRMSE is
  stored as a **fraction**. Multiply by 100 for percent.
- `summary.csv`: unweighted mean errors by method/split; NRMSE already in percent.
- `comparison/*.json`: per-batch convergence histories and metrics.
- `comparison/*.npz`: full-resolution reference, input and predictions when
  `--save-reconstructions` is enabled (enabled in the HPC job).
- `index.html`, `figures/`: training, error heatmaps, and reconstruction previews.
- `metrics.json`: aggregation/timing definitions, peak allocated/reserved CUDA
  memory, and nonconverged batches. `status.json` must say `complete` for a fully
  converged comparison. A safety-limit/stagnation exit is saved and makes the
  command fail visibly rather than being mislabeled as convergence.

Reloadable model example (inside a configured Python session):

```python
import sys
sys.path.insert(0, 'scripts')
import numpy as np
from combined_lista import load_shared_checkpoint
from torch_inverse import PaddedFFT

model, saved = load_shared_checkpoint('RUN/best.pt', device='cuda')
# The pair's input tensor b must be [batch, 1, ny, nx], float32 CUDA,
# with µT fields divided by saved['scale_uT'].
operator = PaddedFFT(np.load('RUN/physics/PAIR_ID.npy'), (ny, nx), pad).cuda()
prediction_uT = model(b, operator) * saved['scale_uT']
```

Regenerate a saved report without training:

```bash
python scripts/plot_combined.py --output /ABSOLUTE/PATH/TO/RUN
```

## Validation completed before HPC execution

On 22 October 2026:

- Converted and checked all **20 actual CSVs** in `data/` and `data/data2/`.
  Every plane was 1400×1400 with approximately 0.1 µm sampling. Within each
  geometry, coordinates aligned across heights. Different geometry origins were
  retained. All prepared-array checksums were verified. The default manifest
  contained exactly 17 height pairs and 51 component pairs.
- Ran the full repository test suite in a temporary CPU environment:
  **63 tests run, 62 passed, one existing two-CUDA-GPU test skipped**.
  The seven new combined-experiment tests all passed.
- Checked shared-model initialization against fixed-depth ISTA for operators
  with different norm bounds, activation-checkpoint gradient agreement,
  geometry/pair validation, duplicate-height rejection, grid/hash mismatch
  rejection, and training-only normalization.
- Ran tiny complete training/comparison/report experiments. An epoch-boundary
  resume matched uninterrupted training weights exactly on CPU; selected-model
  reload and streamed versus three-component gradients also passed.
- Ran a separate synthetic analytic-operator train/validation/test experiment
  to completion. Deliberately setting a one-update classical limit produced
  saved diagnostic output and the expected nonconverged failure status.
- Checked Python compilation and batch-script shell syntax, and visually
  inspected the generated error report.

The temporary validation environment used Python 3.13 and PyTorch 2.14.1 on the
Mac CPU. This is a test environment, **not** a prescribed Prajna package version.
Full-resolution shared-model training, actual CUDA operation of this pipeline,
peak A100 memory, runtime, and scheduler acceptance of the new production job
remain to be measured. Existing A100 inventory diagnostics are not a training
benchmark. No new reconstruction-quality claim follows from the synthetic tests.

## Archived results for reference only

These are the earlier **single-coil, z=3 → 0.5 µm Kaggle results**, not results of
the new combined experiment. Both LISTA models trained for 200 epochs using two
Tesla T4 GPUs; inference timings used one GPU and a three-image batch. Range
NRMSE is shown in percent, lower is better.

| Archived method | Bx NRMSE | By NRMSE | Bz NRMSE | Median three-image inference |
|---|---:|---:|---:|---:|
| LISTA, 8 layers | 2.059% | 1.799% | 2.962% | 0.240 s |
| LISTA, 16 layers | 1.988% | 1.751% | 2.861% | 0.460 s |
| ISTA, 16 iterations | 2.920% | 2.475% | 3.815% | 0.418 s |
| FISTA, 16 iterations | 2.310% | 1.998% | 3.109% | 0.431 s |
| ISTA, 500 iterations | 1.595% | 1.443% | 2.145% | 13.074 s |
| FISTA, 500 iterations | 2.484% | 2.443% | 2.026% | 13.486 s |

Sources: `notebooks/runs_results/comparison_table.csv` and
`notebooks/runs_results/comparison_table-2.csv`, with their corresponding metrics
JSONs. Their fixed-iteration protocol predates the epsilon-stopping change.

Do not compare an overall 51-example error directly with the old three-example
error as if they describe the same task. For a closer comparison, extract the
new run's L1_C1 z=3 → 0.5 pair, disclose that its shared model trained on additional
data, and retain the normalization/stopping differences. A100 timing versus T4
timing also measures a hardware change. Shared-model training costs must be
reported separately from inference costs.

## Results to fill after running on HPC

**Status: not run yet.** Leave the following entries unfilled until the saved
run artifacts exist. Copy values from the named files; do not replace missing
or nonconverged results with successful-looking numbers.

### Run identity and configuration

| Item | Recorded value |
|---|---|
| Date and Slurm job ID(s) | — |
| Cluster, GPU model/count | — |
| Run directory | — |
| Dataset manifest hash | — |
| Source hashes / source snapshot | — |
| Python, PyTorch and CUDA runtime | — |
| Pairing policy and pair count | — |
| Operator mode: joint or analytic | — |
| Train / validation / test geometries | — |
| Layers, filter size and parameter count | — |
| Requested/completed epochs and best epoch | — |
| Learning rate, seed and batch size | — |
| Scale q (µT) and normalized lambda | — |
| Physical lambda, padding and stopping epsilon | — |
| Checkpoint activation enabled / resumed from epoch | — |
| Final status and nonconverged batches | — |

### Overall method comparison

Use `summary.csv`; mean range NRMSE here is already a percentage. State the
split and number of component pairs. Rename fixed-depth rows if using a depth
other than 16. The timing column is the **sum of warmed batch medians**, not total
job duration.

| Method | Split / count | Mean range NRMSE (%) | Mean RMSE (µT) | Summed batch inference (s) | Nonconverged component rows |
|---|---|---|---|---|---|
| Shared LISTA | — | — | — | — | Not an iterative convergence claim |
| ISTA, 16 updates | — | — | — | — | Fixed-depth diagnostic |
| FISTA, 16 updates | — | — | — | — | Fixed-depth diagnostic |
| ISTA to epsilon | — | — | — | — | — |
| FISTA to epsilon | — | — | — | — | — |

### Geometry and height breakdown

Use `comparison.csv` to retain separate Bx/By/Bz scores and actual stopping
counts. Each metric row identifies geometry, sharp height and blurred height.
Multiply its `nrmse_range` by 100 when reporting percentages. Add one row per
method/pair/component or link the complete CSV rather than reporting only a
favorable subset.

| Geometry / blurred → sharp height | Component | Method | Range NRMSE (%) | RMSE (µT) | Updates/layers | Stop reason |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

### Runtime, memory and interpretation

| Item | Recorded value |
|---|---|
| Total scheduler walltime (including preparation/comparison) | — |
| Training time reported at final epoch | — |
| Peak CUDA allocated / reserved memory | — |
| Host peak RSS from scheduler | — |
| Resume/retry history and any lost partial epoch | — |
| Location of HTML report, plots, checkpoints and logs | — |

After the run, discuss whether the shared model improves over same-depth
classical solvers on each geometry, how it compares with converged baselines,
whether particular components/heights degrade, and the cost of sharing one
model. Keep same-sample and held-out scores distinct. Record unexpected errors,
failed convergence, or worse reconstructions as findings, not omissions.
