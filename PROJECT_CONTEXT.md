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
