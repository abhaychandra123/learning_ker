# Project context for a new coding chat

## Combined-data extension (22 October 2026)

The user requested local code preparation before transferring to Prajna; they
will execute transfer/submission commands themselves. They explicitly selected
**one shared ConvLISTA** trained across `data/` plus `data/data2/`. Reproducing the
old single-pair Kaggle experiment is deferred.

Later on 22 October, the user switched the execution target to their professor's
server, `abhaychandra@10.129.6.189` (`user-ESC4000A-E10`, RTX A6000 48 GB,
driver 590.48.01, system Python 3.10.12). No scheduler executable was visible.
`hpc/run_combined_direct.sh` now launches the same experiment without Slurm;
`hpc/check_cuda.py --allow-direct` explicitly permits its GPU check on this server.
The user will perform the transfer/environment setup/run commands themselves.
They prefer the server's Python 3.10 with standard `venv` over Miniforge.
The SHA-256 helper in `scripts/kernel_crops.py` now uses compatible streaming
hashing instead of the Python 3.11-only `hashlib.file_digest`; sync this file to
the server before running. Numerical behavior and digest values are unchanged.
The new server's project environment and actual GPU calculation are not yet
verified. Prajna's Python 3.11 environment and torch 2.6.0+cu124 imports succeeded,
but its GPU test was queued, and a second submission hit AssocGrpSubmitJobsLimit;
no successful GPU-check output has been supplied. Do not infer its final job state.

Read [README_COMBINED_EXPERIMENT.md](README_COMBINED_EXPERIMENT.md) for the detailed
old/new comparison, scientific scope, commands, validation and blank results
tables to populate after the HPC run. New entry points are
`scripts/combined_data.py`, `scripts/combined_lista.py`,
`scripts/compare_combined.py`, `scripts/plot_combined.py`, and
`hpc/combined.sbatch` / `hpc/check_cuda.py`. The original pipeline is unchanged.

The new converter keeps geometry-specific float32 **µT** `[3,y,x]` caches in a
fresh output directory, never the primary tesla cache. Default pairing uses the
lowest plane per geometry: 10 L1_C1 pairs anchored at 0.5 µm, six IC_4um pairs
anchored at 0.5 µm, and IC_2um 2→1 µm. All 20 real CSVs were converted, coordinate
alignment checked, and output hashes verified: 17 height pairs / 51 signed
component pairs. `hpc/combined_dataset_inventory.json` records the raw-file audit.

One shared set of 163 parameters uses the appropriate frozen operator per pair.
The threshold coefficient is multiplied by each operator's step, preserving
ISTA initialization across different operator bounds. A global training-only
scale and streamed full-image batches accumulate gradients for one optimizer
update per epoch. The default joint-kernel fit and model selection use all pairs
as same-sample training reconstructions. Optional geometry holdouts require
analytic operators; reference-fitted joint operators are rejected for holdouts.
Epoch checkpoints include optimizer state and support strict data/code/config
checked resume; only total epochs/locations may change. This new runner is
single-GPU. The old ConvLISTA CLI's no-resume limitation still applies to it.

Validation: 63 repository tests ran on CPU, 62 passed and one existing two-GPU
test skipped; all seven new tests passed. Separate synthetic holdout and
nonconvergence-status integrations passed. Shell syntax, Python compilation,
and generated-report visual QA passed. No full-resolution combined training or
HPC execution was performed; new scientific scores are not yet available.

---

Last checked: 29 September 2026 against base revision `da5a140` plus the working-tree epsilon-stopping changes. This is the short entry point; read the relevant implementation before editing. Update this file when the experiment, interfaces, or results change.

## What we are trying to do

Reconstruct magnetic-field maps close to a simulated micro-coil from smoother maps measured farther above it. COMSOL exports supply the reference data. Earlier work studied analytic upward propagation and empirical Fourier blur kernels; the current experiment compares ISTA, FISTA, and supervised convolutional LISTA for reconstruction from z=3 µm to z=0.5 µm.

The immediate scientific question is how reconstruction quality, optimization progress, and inference time compare. The existing experiment fits one coil's three component images. Generalization to another geometry or height is not established. No particular next feature is prescribed here: the user's current request supplies it.

## Research direction confirmed on 8 October 2026

The user wants a 4–6 month advanced research program on joint effective-standoff estimation and near-plane magnetic-field reconstruction, with evaluation across unseen source geometries, FOVs, pixel spacings, and noise. The primary output remains the magnetic field at a declared reference plane; current density is an auxiliary physical representation. Optical deconvolution and arbitrary multilayer separation are later extensions.

The current goal is deblurring and physically validated super-resolution of QDM magnetic-field images across unknown standoffs, different FOVs, and geometries, using a learned unrolled network with variable/adaptive parameters. Begin from the existing LISTA/FISTA machinery and investigate conditioning or joint refinement of the propagation distance and learned reconstruction updates; the exact parameterization remains to be selected through controlled experiments. At fixed pixel count, a narrower physical FOV gives finer sampling (`pixel_spacing = FOV_extent / pixel_count`); actual recoverable resolution also depends on standoff, optics, and noise. Do not equate changing tensor size or interpolating an image with demonstrated physical super-resolution.

The user confirmed access to an extensive existing COMSOL database, good GPUs, the real QDM, and the original experimental dataset underlying [Bathla et al.](https://arxiv.org/html/2506.05491v1), because they work in the same research group. The external database/files have not been inspected; the experimental archive will be obtained from a hard disk later and its calibration metadata are not yet known. Do not infer that the repository's current few geometry groups describe the full available database, or that the experimental archive already provides independently calibrated absolute heights or sharp reference-plane targets.

The planning guide is authored in `output/pdf/qdm_research_guide/qdm_research_guide.tex` with chapter sources, original figures, experiment protocols, architecture comparisons, and dataset-audit instructions. Its methods are proposals, not implemented or validated new results. The existing numerical experiment and archived results below retain their original scope. A next implementation task still needs a concrete experiment selection and access to the relevant data.

## Read these in order

1. This file for current scope, conventions, and entry points.
2. [LISTA_FISTA.md](LISTA_FISTA.md) for the current mathematical formulation and experiment protocol.
3. Relevant source files and tests for the requested change.
4. [CODEBASE_UNDERSTANDING.md](CODEBASE_UNDERSTANDING.md) when detailed original-data, kernel, boundary, or historical-result context is needed. It audits revision `d845403`, before LISTA/FISTA were added. Statements that there is no network, notebook, dependency manifest, test suite, or saved numerical output describe that earlier revision.

Prefer implementation for actual behavior and saved run metadata for reported results. `info.md`, `res.md`, `kernel_learning_summary.md`, `ista_deblurring_summary.md`, and presentation text contain historical claims; do not assume they describe the current solver.

## Code map

| Area | Files and responsibility |
|---|---|
| Data | `scripts/convert.py`: primary CSVs to cache. `scripts/fields.py`: grid, units, cached loads. `scripts/plot_ic.py`: separate CSV plotting, including `data/data2`. |
| Forward physics | `scripts/propagator.py`: analytic Fourier propagation. `scripts/kernel.py`, `scripts/kernel_regularised.py`: unregularised and Wiener joint-fit experiments. `scripts/mesh_noise.py`: propagation residual analysis. |
| Original inverse | `scripts/deblur.py`: padded joint kernel fitting, forward/adjoint operators, original L1 ISTA, and historical plotting entry point. |
| Comparable classical solvers | `scripts/iterative.py`: shared NumPy ISTA/FISTA and diagnostics. `scripts/fista.py`: standalone CLI and saved outputs. |
| GPU operator and solvers | `scripts/torch_inverse.py`: batched differentiable padded FFT operator, ISTA/FISTA, physical-unit metrics. |
| Learned inverse | `scripts/lista.py`: `ConvLISTA`, checkpoint loader, and training CLI delegation. `scripts/compare_lista.py`: data loading, normalization, training, comparison, timing, and provenance. |
| Experiment delivery | `scripts/prepare_kaggle.py`: generates the notebook and ZIP bundle. `notebooks/ISTA_LISTA_FISTA_Kaggle.ipynb`: Kaggle workflow. `scripts/plot_comparison.py`: plots complete saved runs. |
| IC_4um crop experiment | `scripts/kernel_crops.py`: equal-sized paired crops, shared FFT kernel per target, spatial/height holdouts, saved arrays, and HTML visualization report. `tests/test_kernel_crops.py`: numerical and alignment checks. |
| Checks | `tests/test_iterative.py`, `tests/test_lista.py`: numerical equivalence, adjoints, gradients, training, scaling, checkpoint, and multi-GPU checks. |
| Reports | `make_ppt.py` and `presentations/`: presentation generation. Scientific behavior lives in `scripts/`. |

Scripts use sibling imports rather than an installed Python package. Run documented commands from the repository root.

## Contracts to preserve

- Primary cache: 1400×1400 grids at 0.1 µm spacing, 11 heights (0.5, 1, 2, …, 10 µm), four components. Arrays are indexed `[y, x]`; plots use `origin='lower'`.
- Cache arrays store float32 tesla. `fields.load()` defaults to µT. Coordinates/heights are µm. The comparison loader reads the cache directly and matches that unit conversion.
- Fit the common inverse operator using signed `Bx`, `By`, `Bz`. `Bnorm` is a nonlinear magnitude, not a fourth independent linear component.
- `H` is in unshifted FFT layout. Inversion uses `A(x) = Crop(real(IFFT(H * FFT(Pad(x)))))`; its adjoint uses `conj(H)` with the same padding/cropping. Preserve the intermediate crop when composing `A* A`.
- ISTA/FISTA minimize `0.5 * ||A(x)-b||² + lambda * ||x||₁`, with `step = 0.99 / max(abs(H)²)`. Defaults: pad=32, lambda=1e-4 in µT units, initialization `x0=b`.
- ISTA/FISTA now stop by `--epsilon` (default 1e-3, literally 10E-4), using per-component proximal-gradient RMS divided by its initial RMS. Check every update at the proximal iterate; the GPU batch requires every component to pass. Ground truth and `record_every` do not control stopping. Record actual iterations, residual, and stop reason. Show counts in reconstruction plots, convergence legends/endpoints, timing labels, and comparison CSVs. Standalone ISTA/FISTA also save per-component CSV/JSON summaries and convergence plots. The internal 100,000-update guard/stagnation exit is reported as nonconverged. `n_iter` remains only as a low-level fixed-depth reference/testing option.
- The user explicitly chose to keep LISTA's trained layer count and training epochs as architecture/training settings. The notebook now exposes `EPSILON` instead of `BASELINE_ITERATIONS`; same-depth diagnostic counts are derived automatically from LISTA depth. Adaptive timing includes stopping checks.
- LISTA adds two bias-free 9×9 convolutional corrections to the fixed physics update and learns a nonnegative scalar threshold. Parameters are tied across layers (163 total with 9×9 filters). Zero corrections and the initial threshold reproduce ISTA at every layer.
- LISTA learns from sharp-field MSE, not the L1 objective. Learned updates have no established convergence guarantee. Do not run more layers than the model was trained for without defining a new experiment.
- Training normalizes all six input/reference fields by one scale `q`; use `lambda_normalized = lambda_uT / q`. Restore physical units for metrics. PyTorch tensors are `[batch, 1, height, width]`.
- All three whole component images fit `H`; those same references supervise LISTA and select its best checkpoint by training MSE. Scores are same-sample reconstruction scores, with no independent validation/test split.
- Range NRMSE is RMSE divided by the reference's max-minus-min. JSON stores a fraction; multiply by 100 for percent. Relative L2, L1 objective, and reconstruction error measure different things.
- Training may use two GPUs; inference comparisons use the same single GPU, dtype, and three-image batch. Equal layers/iterations do not imply equal compute.
- Cache filenames encode only height and component. Do not mix another geometry into the existing cache; `data/data2` is separate. The user confirmed on 29 September 2026 that its `IC_2um` and `IC_4um` files represent different geometries. IC_4um has heights 0.5, 3, 4, 5, 6, 7, 10 µm; IC_2um has 1 and 2 µm. Do not construct pairs across these groups.

## Implemented crop experiment

The user requested and we implemented a simple direct crop experiment for IC_4um only; [CROP_KERNEL_PLAN.md](CROP_KERNEL_PLAN.md) records its method, holdouts, results, and commands. Use z=0.5 µm as reference for targets 3, 4, 5, 6, 7, 10 µm. The completed run uses a 5×5 grid of aligned 280×280 crops across heights and Bx/By/Bz, with one shared unregularized Fourier least-squares kernel per target fitted across all 25 crops and components. Separate spatial-holdout models fit 20 crops and score five excluded crops. A height-holdout model uses only .5→3 and 3→4 fits to predict z=5 via `H_anchor * H_step**2`; its Bx/By/Bz range NRMSE is 4.371/5.886/5.643%, versus 6.269/9.199/10.086% for the analytic baseline applied to the same crops. Direct crop FFTs have periodic boundaries and visible seams. The user explicitly chose this simple approach; retain it unless asked to change it.

Open `runs/ic4_crop_kernel/index.html` for all 34 figures. Crop arrays, kernels, predictions, and metrics are saved under that run directory (ignored by Git). Run `python3 scripts/kernel_crops.py --output <fresh_directory>` to reproduce, or add `--replot` with an existing run directory to regenerate figures. At the crop implementation checkpoint, seven new crop tests and six NumPy iterative tests passed; PyTorch was absent from the default Python environment. During the later epsilon-stopping change, a temporary test environment with PyTorch was created at `/tmp/learning-ker-convergence-venv`: all 26 locally runnable tests passed, with one two-CUDA-GPU test skipped. An end-to-end synthetic comparison, standalone FISTA CLI, notebook code-cell compilation, ZIP hash verification, and extracted-bundle tests also passed. CUDA and a new full-data Kaggle run remain unverified locally.

## Existing results

`notebooks/runs_results/` contains metrics, comparison CSVs, and figures from two Kaggle runs:

| Metadata | LISTA depth | Training epochs | Long ISTA/FISTA baseline |
|---|---:|---:|---:|
| `metrics.json` | 8 | 200 | 200 iterations |
| `metrics-2.json` | 16 | 200 | 500 iterations |

Both record z=3 → 0.5 µm, two Tesla T4s for training, 9×9 filters, pad=32, lambda=1e-4, and learning rate=1e-4. All nine recorded source hashes matched the scripts at the initial context refresh, before the subsequent epsilon-stopping changes. These archived runs retain their original fixed-iteration protocol. The experiments were not rerun during the refresh.

The stored scores show trained LISTA improving reference NRMSE over the same-depth classical solvers for all three components. Longer baselines can reconstruct more accurately; there is no universal LISTA accuracy/speed win. Use the per-method/component metrics and timing in each JSON for precise comparisons.

The checked-in results directory does not include the run checkpoints, `kernel.npy`, or `reconstructions.npz`. The training pipeline can generate them, but the archived JSON/PNGs alone cannot reload a trained model or rebuild every plot with `plot_comparison.py`.

## Commands and validation

Dependencies are in `requirements-lista.txt`; retain Kaggle's compatible CUDA PyTorch installation. Presentation generation additionally uses `python-pptx` and Pillow.

```bash
# Numerical/unit checks; the two-CUDA-device check needs suitable hardware.
python -m unittest discover -s tests -v

# Standalone classical baseline; choose a fresh output directory.
python scripts/fista.py --z 3 --comp Bz --epsilon 1e-3 --output runs/fista_z3

# Full-resolution GPU experiment, not a routine quick smoke test.
python scripts/lista.py --data-root . --z 3 --device cuda --gpus 2 \
  --layers 8 --kernel-size 9 --epochs 200 --lr 1e-4 \
  --epsilon 1e-3 --pad 32 --lambda-ista 1e-4 \
  --checkpoint-layers --output runs/compare_z3

# Regenerates/overwrites the notebook and portable ZIP after relevant changes.
python scripts/prepare_kaggle.py
```

The new FISTA/comparison CLIs refuse existing output directories. Older experiment scripts can overwrite figures. Preserve archived results and use fresh run locations. Update the notebook generator when changing generated notebook behavior; rebuild the bundle when preparing updated code for Kaggle.

For solver/operator changes, run the relevant numerical tests and then the suite when feasible. Report actual checks and hardware skips; do not infer a fresh pass from historical documentation. For a new scientific claim, specify the data split, operator fitting, units, normalization, stopping budget, and timing protocol. Full training is not required just to orient a new chat.

## Suggested new-chat prompt

> Read PROJECT_CONTEXT.md, then inspect the source and tests relevant to my task. Use LISTA_FISTA.md for the current experiment and CODEBASE_UNDERSTANDING.md for historical background. My task is: [describe change]. After implementing it, run appropriate checks and update PROJECT_CONTEXT.md if the project state changed.

This file supplies orientation; a chat still needs repository access (or the relevant source files) to make reliable code changes.

Standalone FISTA visual reports: `scripts/fista_report.py` renders saved runs; `fista.py` calls it automatically or via `--replot --output <existing-run>`. `runs/fista_z3/index.html` contains the user's completed z=3 reconstruction (Bx/By/Bz: 346/356/330 converged updates), kernel/input/forward/profile/spectrum plots, and a separate forward height holdout fitted only on .5→1 and predicting z=2 via three padded steps. This holdout does not establish inverse reconstruction generalization. `report_metrics.json` and `report_provenance.json` document the added diagnostics; original solver arrays/metrics are retained. Cache hashes are checked before pairing existing reconstructions with source fields.

Patch-based lambda selection: `scripts/select_lambda.py` implements the professor's manual homogeneous-region variance estimate (`ddof=1`) and Morozov full-image squared-residual target. See `LAMBDA_SELECTION.md` for interactive rectangle selection, coordinate/reused-ROI modes, FISTA/ISTA search, separate solver/discrepancy tolerances, tighter-solver verification, outputs and assumptions. It is a primary-cache, one-component-at-a-time, same-sample kernel experiment. No scientific λ has been selected yet: the user must choose a representative patch. Numerical estimator/search tests and a complete synthetic CLI/report run validate implementation; interactive window selection has not been manually exercised.

## ALISTA implementation (2026-10-15)

See `ALISTA.md` for the mathematical contract, sources, commands, and scope.
`scripts/alista_weights.py` solves the paper's Eq. (16) using projected gradient
descent (Appendix E.1), with an independent SVD option and explicit feasibility /
stationarity diagnostics. `scripts/alista.py` implements Eq. (15): frozen D and W,
zero initial coefficients, and one learned step and threshold per layer (32
scalars at 16 layers). Optional support selection follows the authors' released
schedule. The learned-ISTA mode uses W=D as an explicitly named control.

`alista_data.py`, `alista_experiment.py`, `train_alista.py`, and
`compare_alista.py` separate data generation, metrics/baselines, training, and
evaluation. Training uses fresh Bernoulli-Gaussian examples and fixed validation
data. Testing uses an independent stream; baseline lambda selection uses only
validation. Checkpoints, source hashes, analytic diagnostics, scalar training
logs, predictions, CSVs and plots are saved. Each command refuses existing output.

This implements the **dense sparse-code experiment**, not a full-resolution QDM
ALISTA model. The magnetic crop/pad operator is unchanged. A justified sparse
representation and tractable analytic operator respecting those boundaries are
still needed before a QDM comparison. Do not call the existing adjoint alone an
ALISTA analytic weight, compare sparse-code NMSE directly to QDM range NRMSE, or
claim the paper's numerical results have been reproduced. Conv-ALISTA,
Robust-ALISTA and HyperLISTA remain separate future work.

Validation completed on CPU: 56 tests run, 55 passed, one existing two-GPU test
skipped; all 22 new ALISTA tests passed. The bounded Section 5.1 experiment is
under `runs/alista_validation/`: 250-by-500 D, 16 layers, seed 0, noiseless,
200 updates per stage and 1,000 independent test examples. Final results are in
`comparison_verified/metrics.json`: plain ALISTA -28.0063 dB, support-selection
ALISTA -54.6781 dB, learned-ISTA control -12.1351 dB, ISTA16 -6.2607 dB,
FISTA16 -10.9823 dB, ISTA500 -22.2125 dB, FISTA500 -55.3376 dB. Baseline
lambdas were selected on validation only. These are single-seed synthetic
results, not QDM scores or replicated paper tables. `ALISTA.md` records training
protocol differences and the final legacy-percentile edge-case verification.

## Synthetic QDM data pipeline (2026-10-23)

`README_SYNTHETIC_DATASET.md` documents the new, separate dataset generation
pipeline. `scripts/synthetic_data.py` prepares portable compact recipes from
either a prepared COMSOL manifest or a completed combined comparison's saved
COMSOL arrays. It propagates full lowest-plane fields analytically, applies
physical vector quarter-turns, and adds deterministic Gaussian input noise.
Noise sigma is a percentage of each clean propagated component's max–min range;
1%/1.5% are experimental settings, not estimates inferred from COMSOL residuals.
Default padding is 256 zero-extension pixels; the highest-plane boundary audit
records periodic/default/double-padding discrepancies without attributing them
solely to mesh noise. Original data and comparison runners are unchanged.

`runs/synthetic_v1` contains 360 vector recipes / 1,080 scalar component pairs
from three references, using 10 heights, four rotations, and 0%/1%/1.5% noise.
This local dataset is all-train; variants are not independent geometries.
Train-only normalization, geometry-level optional holdouts, portable checksums,
component-at-a-time loading, fixed evaluation noise and optional fresh training
noise are implemented. `scripts/plot_synthetic.py` produced nine preview figures
and measured noise statistics in `runs/synthetic_v1_preview`; six components
were materialized in `runs/synthetic_v1_smoke`. Eight NumPy CPU tests passed,
the shell wrapper passed syntax checking, and real-data preparation/inspection/
preview/export completed. No PyTorch or GPU check ran in this local Python.

`hpc/generate_synthetic_direct.sh` is for the user to run on the remote machine.
No remote commands or training were executed. The synthetic manifest is a new
schema, not an input to the unchanged `compare_combined.py`. Next training work
must explicitly connect its loader and known analytical padded operator to the
shared-model trainer, select the split before training and keep synthetic versus
actual COMSOL evaluation results separate. The previous 200-epoch model saw all
three geometries, so it does not establish held-out performance on these data.

The synthetic preview now lists every target height and the exact count
30 geometry/height combinations × 4 rotations × 3 noise levels = 360 vector
pairs (1,080 components). `plot_synthetic.py --comparison-run <combined-run>`
adds verified actual COMSOL/analytical/residual figures, with `--refresh` for an
existing preview. The local report covers L1_C1 at 1/6/10 µm, IC_4um at 3/6/10 µm
and IC_2um at 2 µm, all three components (21 figures). Actual and synthetic
references/grids are checked for equality; metrics and archive hashes are saved
in `comsol_checks.csv` and `comsol_provenance.json`. These are noiseless forward
comparisons, not reconstruction or noise-estimation results. Highest-plane
metrics were checked against the existing boundary audit.

## Standalone reconstruction command (2026-10-23)

`scripts/reconstruct_combined.py` and `README_RECONSTRUCTION.md` provide a
reusable inference/report CLI: `--run`, `--geometry`, `--standoff`, optional
`--target-standoff`, `--component`, `--device`, and fresh `--output`. It loads
`best.pt` and a checksum-verified saved pair operator, uses checkpoint
normalization, processes one component at a time, and writes physical arrays,
RMSE/range-NRMSE, figures and HTML. `--include-saved-baselines` adds archived
ISTA/FISTA predictions with convergence metadata; it does not rerun solvers.
`--input`/`--reference` accept calibrated NPY fields on the selected geometry
grid. New inputs never borrow a COMSOL reference automatically. Explicit
`--operator analytic` supports new heights with supplied inputs, with the
trained experiment's padding and a changed-operator caveat in the report.

Local `runs/combined_full_200` lacks checkpoints AND has an empty physics folder.
The explicit `--saved-results` mode reports archived predictions without either;
it never implies fresh inference. A full 10→0.5 µm L1_C1 report was generated in
`runs/reconstruction_L1_C1_10um_preview`, with six figures, exact archived LISTA
arrays and all 15 solver/component errors agreeing with original CSV metrics
within 0.0001 percentage points. Three new tests passed; one actual PyTorch
checkpoint/normalization test was skipped because local Python has no torch.
Fresh inference/GPU execution remains to be checked by the user on the training
machine. No remote commands or retraining were performed.

On 2026-10-24, reconstruction reports gained `--color-scale independent|shared|both`
(default both, independent panels first) and `--replot <completed-output>`.
Replot updates only figures/HTML/display diagnostics, retaining numerical arrays,
metrics, inference provenance and status. Each independent map has its own µT
colourbar; shared views remain expandable. The final panel is signed LISTA-minus-
reference error. `plot_diagnostics.json` adds physical min/max and relative L2
error to explain limitations of full-field range NRMSE. The user's fresh local
output `runs/reconstruction_local_10um` was redrawn without inference. Its L1_C1
Bx 10→0.5 result has range NRMSE 4.273% but relative L2 error about 90.619%; field
range is about -39.65 to +36.06 µT versus reference -186.43 to +187.39 µT. This is
poor fine-detail/amplitude recovery, not merely a colour-scale issue. Three CLI
tests still pass, one torch test skipped in the agent's Python; output figures
were visually checked. Preserve unrelated existing repository changes.

## Synthetic augmentation training integration (2026-10-25, India time)

`README_SYNTHETIC_EXPERIMENT.md` tracks the hypothesis that increased synthetic
coverage improves actual COMSOL reconstruction. `train_synthetic.py` connects
`SyntheticDataset` to the existing `SharedConvLISTA` and `PaddedFFT`, with
rotation-aware analytical operators, manifest padding, global normalization,
one-component streaming, and one accumulated Adam update per epoch. Original
model/solver/trainer files and datasets are unchanged. Epoch-boundary resume
checks data/code/config hashes; primary evaluation uses the final checkpoint,
not one selected on actual COMSOL scores.

The default paired experiment trains fresh seed-0 models for 200 updates:
51 synthetic components matching original heights (no rotation/noise), versus
all 1,080 synthetic components. Both use analytical pad-256 physics, the same
215.14413452148438 µT scale, 16 layers, 9x9 kernels, lr=1e-4 and lambda=1e-4.
Both evaluate the same 51 actual COMSOL components only after training.
`synthetic_experiment.py` validates selection and archive/reference identity;
`compare_synthetic_experiment.py` produces paired/per-geometry error deltas.
`hpc/run_synthetic_experiment.sh` checks CUDA/tests, runs the augmented arm then
control sequentially and writes the comparison. Historical fitted-pad-32
results are contextual only. New actual evaluations share geometries/targets
with training and cannot establish unseen-geometry generalization or prove
that limited data was the only failure cause. Augmentation also changes noise,
height, rotation and geometry weighting. No calibrated confidence is claimed.

Local verification: all 17 actual archives checked; references exactly match;
control=51, full=1080. All 14 synthetic dataset/training tests passed on CPU in
an isolated cached-wheel PyTorch 2.14.1 / Python 3.13 environment, including
rotated rectangular-grid physics, exact resume equivalence and unchanged
training weights when actual evaluation inputs change. Shell syntax and Python
compilation passed. Tiny training fixtures are not a full-size GPU run. SSH to
10.129.6.189 was blocked by the execution environment, so no remote dataset,
training launch, PID or result had been confirmed at implementation time. README
contains exact copy, launch, monitoring and resume commands.

Subsequent user-provided terminal output confirms successful remote inspection
on `10.129.6.189`: 3 geometries, 360 vector / 1,080 scalar pairs, all training.
The user launched `hpc/launch_synthetic_remote.sh`; background launcher PID is
66593 and log is `~/learning_ker/logs/synthetic_data_hypothesis_v1.launch.log`.
Later user logs confirm epoch 6 completed (MSE 0.0033977439, elapsed 8721.7 s)
and epoch 7 reached 780/1080 gradient components by roughly 540 s. Full loss
evaluation took 619.4 s. Projected unchanged runtime is about 22.8 min/epoch,
76 hours for 200 epochs excluding control/final evaluation. User's NVIDIA
snapshot shows RTX A6000, training Python PID 66663, 776 MiB process VRAM and
0% sampled GPU utilization. This suggests loader stalls but is not sustained
utilization or peak VRAM evidence. Local trace confirmed nine CPU propagations
for three unique inputs across three noise recipes: the one-entry cache evicts
components between recipes. Also every epoch repeats all 1080 samples for loss
measurement and uses gradient checkpointing. No training source/contract was
changed during diagnosis. See README runtime investigation; optimize caching,
prefetch/evaluation cadence before blaming GPU capacity. Minibatch updates
would be a separate protocol change. Hypothesis outcome remains pending.

## Full-field training optimization and crop audit (2026-10-25)

User authorized optimizing the running synthetic experiment and asked about
cropping. `README_SYNTHETIC_OPTIMIZATION.md` records the implemented new path;
legacy Python sources remain unchanged to preserve checkpoint hash validation.
`synthetic_fast.py` builds fixed normalized inputs once with a 3-component
clean-propagation cache, shares rotated targets, caches on CUDA when budget
permits (about 8.75 GB inputs+targets), or uses pinned CPU cache/prefetch. It
batches consecutive identical operators, uses real analytical FFTs with the
same full image/padding, and benchmarks batch 1/2/4/8 and checkpoint on/off on
CUDA after cache allocation. Batch losses retain per-component weights and
one Adam update per full epoch. Precision remains float32 with no AMP/TF32.

`train_synthetic_fast.py` measures post-update loss every 5 epochs and at final,
logs pre-update loss each gradient pass, checkpoints each epoch, and supports
SIGTERM stopping at an epoch boundary. `--migrate-from` only accepts audited
legacy trainer/hash/data/settings, restores model+Adam+history and continues
from the next epoch in a new directory. To keep arms comparable, the control
replays K original scalar/full-FFT/checkpointed updates before adopting the
same measured optimized settings, where K is the migrated epoch. Final actual
COMSOL evaluation still uses the existing full-FFT evaluator. Rounding can
differ from legacy arithmetic; numerical equivalence was tested, not bitwise
identity between full and real FFTs.

`hpc/optimize_synthetic_remote.sh USER@HOST` copies new files, runs new CPU tests
on the server, then `switch_synthetic_run.py` validates source/data and exact
launcher/trainer identities, stops only the old experiment, copies its latest
completed checkpoint and starts `hpc/run_synthetic_fast.sh`. Only incomplete
epoch work may be lost. Old outputs remain intact; new root is
`runs/synthetic_data_hypothesis_fast_v2`. The helper refuses existing v2 output
before copying files. It is NOT executed by this agent because remote SSH is
blocked. At implementation time, no remote speedup or new GPU memory/epoch
timing had been measured; subsequent user-provided measurements are below.

ROI audit (`audit_synthetic_roi.py`, `runs/synthetic_roi_audit_v1`) finds 99%+
per-component squared-field-energy reference boxes: IC_2um 1253x1315, IC_4um
1236x1313, L1_C1 514x641. At highest propagated plane: 1285x1316, 1365x1374,
844x860 respectively. A reference box +128px halo covers the full IC images;
for L1_C1 it yields 768x771 and adds ~0.277/0.674/0.941% forward relative-L2
error inside the reference ROI (Bx/By/Bz). Not an inverse-error guarantee.
Keep current experiment full-field; any L1 crop study is separate and must
check nonlocal/boundary effects. Cropping every geometry tightly is unsupported.

Seven new optimization tests passed on CPU, including input identity, cache
reuse, real/full FFT adjoints+gradients, batch weighting, migration with Adam,
matched legacy prefix, sparse-evaluation weight invariance, exact optimized
resume and rejection of invalid contracts. Full 1400x1400 CPU operator checks
on all 9 geometry/components at high standoff/90-degree rotation showed maximum
normalized real/full FFT difference <1.94e-7 and physical RMSE <4.26e-6 uT.
At that validation stage, CUDA caching/prefetch/autotuning and actual optimized
throughput were pending; subsequent remote measurements are below.

Remote optimization switch update (user-provided terminal output): all seven
optimization tests passed on the server in 2.871 s. The switch preserved the
completed epoch-7 checkpoint and started optimized launcher PID 67736.
Log: /home/abhaychandra/learning_ker/runs/synthetic_data_hypothesis_fast_v2/launch.log.
At the switch, the next update was epoch 8, with epoch 200 still the final target.
Do not rerun the migration/launch helper or infer a GPU benchmark result from
the successful CPU tests and launcher PID alone.

Remote performance update (25 October 2026, user-provided logs): cache build
completed in 95.9 s; all eight CUDA batch/checkpoint trials accepted. Fastest
candidate batch 8, activation checkpointing off: 5.3757 components/s and
17,231,754,752 bytes peak allocated memory (~16.05 GiB), not total framebuffer
usage or a full-epoch peak. Epochs 8, 9, 10 completed with gradient passes
203.2 / 203.7 / 203.6 s (mean 203.5 s). Epoch-10 full loss evaluation took
34.3 s; total epochs 203.3 / 203.7 / 237.9 s. Epoch 11 started. Epoch-10
pre-update MSE 0.00345068960, post-update 0.00343422336. None on epochs 8/9
means intentionally skipped post-update evaluation, not failed training.

Presentation evidence is in README_SYNTHETIC_OPTIMIZATION.md: ~3.67x gradient
speedup against the extrapolated 747.7 s legacy pass, ~18.06x full loss-pass
speedup against measured 619.4 s. With evaluation every five updates, estimated
average epoch is 203.5 + 34.3/5 = 210.36 s (3.51 min), ~6.50x faster than the
projected legacy 1367.1 s (22.8 min). This combines implementation and evaluation
cadence gains, not a measured full-run ratio or isolated per-change ablation.
Same-cadence gradient+evaluation stage comparison implies ~5.75x. From epoch 10,
190 remaining augmented updates +38 evaluations project ~11.1 h, excluding
control/final actual evaluation and assuming unchanged throughput. Full images,
all1080 pairs, model/Adam migration and one update/epoch preserved. Sustained
GPU utilization, explicit cache placement and CPU fallback/prefetch remain
unconfirmed by this excerpt. No reconstruction-quality improvement claim yet.

## Completed synthetic experiment results (2026-10-25)

Supersedes the pending-result status above. User supplied final remote
comparison/results.json and summary.csv for synthetic_data_hypothesis_fast_v2;
both arms completed 200 updates and actual-COMSOL evaluation. These are pasted
remote results, not independently downloaded/verified remote artifacts.
Control final training MSE 0.004416467621922493; logged cumulative training time
2434.3 s and final actual evaluation 30.1 s. Cached control epochs ~9.7 s and
loss passes ~1.6 s. These timing counters are not total experiment wall time.

Across 51 evaluation components, control -> augmented means:
NRMSE fraction 0.03740098333312365 -> 0.03711002305367689 (0.778% relative
reduction, 0.029096 percentage points); relative L2 0.6506458469165415 ->
0.6466198099165708 (0.619% relative reduction); RMSE 13.601660091477898 ->
13.531605461164997 uT (0.515% relative reduction); amplitude slope
0.6189114830874668 -> 0.5663313069944927 (farther from ideal 1).
Only 16/51 component cases improve NRMSE; other 35 could include ties.
IC_2um NRMSE improves 17.39%, 3/3 cases, slope moves toward1;
IC_4um NRMSE worsens0.67%, 4/18 improve, slope moves farther below1;
L1_C1 NRMSE improves0.21%, 9/30 improve, slope moves farther below1.
Overall gain largely driven by one IC_2um vector case.

Automatic directional rule passes, but aggregate gains <1%, mixed case outcomes
and worse mean amplitude recovery do not establish substantial improvement or
support data scarcity as sole cause. One seed, shared source geometries and
changed augmentation/geometry weighting preclude broad causal/generalization
claims. 21.18x recipes are not 21.18x independent geometries. No calibrated
confidence. README_SYNTHETIC_EXPERIMENT.md now records final tables, interpretation
and presentation wording; optimization README links the outcome separately
from ~6.5x estimated compute speedup. Proposed next diagnostics: paired errors
by height/component, matched-scale reconstructions and amplitude/residual checks,
then targeted forward-mismatch/fit/augmentation-weighting tests. No new training
or model changes performed for this result documentation.

## Local synthetic reconstruction and classical report (2026-10-26)

User ran the prepared hpc/reconstruct_synthetic_local.sh on Mac after the prior
session disconnected. Local runs/reconstruction_synthetic_L1_C1_10um is complete:
new augmented final checkpoint SHA256 bd8d704b5c1eda121f0dc2559c6805cd037e15fe68f167e29064bc04b7016eee,
epoch200, CPU, L1_C1 10->0.5um all components, analytical pad256. Entry point
scripts/reconstruct_synthetic.py reuses reconstruct_combined.render, not its
legacy checkpoint loader. Initial report omitted saved ISTA/FISTA, prompting
user correction. Added --include-saved-baselines and --replot to attach the
historical ISTA/FISTA-16 and to-epsilon arrays with archive/input/grid checks,
iteration/convergence metadata and explicit joint-pad32 versus analytic-pad256
labels. Helper now includes baselines for future fresh reports. Replot preserves
the new model's reconstruction_uT.npy and original inference details. Seven
reconstruction tests passed. README_RECONSTRUCTION.md documents the new path.
