# Synthetic magnetic-field dataset — version 1

**Training integration added 25 October 2026:** see
[README_SYNTHETIC_EXPERIMENT.md](README_SYNTHETIC_EXPERIMENT.md) for the new
`train_synthetic.py` entry point, 51-versus-1,080 controlled augmentation
experiment, validation evidence and remote launch commands. The generation
pipeline described below still only prepares data; the original
`compare_combined.py` still expects the original COMSOL manifest format.

This pipeline uses the lowest available COMSOL field from each geometry to generate additional paired inputs for magnetic-field reconstruction. It implements analytical upward propagation, physical quarter-turn rotations and reproducible Gaussian noise. It prepares data; it does **not** run a new LISTA/ISTA/FISTA experiment.

## What was generated locally

On 23 October 2026, the pipeline was run from `runs/combined_full_200` with 10 propagation distances per geometry, rotations 0°/90°/180°/270°, noise levels 0%/1%/1.5%, one realization and seed 0.

| Item | Result |
|---|---:|
| COMSOL reference geometries | 3 |
| Vector input/target pairs | 360 |
| Scalar component pairs (Bx, By, Bz) | 1,080 |
| Split | All training; no held-out score |
| Reference image dimensions | 1,400 × 1,400 |
| Compact dataset size | About 70.7 MB / 67.5 MiB |
| Estimated full materialization, uncompressed | 25.4 GB / 23.7 GiB |
| Shared training normalization scale | 215.1441345 µT |

The dataset is `runs/synthetic_v1`. Its nine representative figures and measured noise statistics are in `runs/synthetic_v1_preview/index.html`. Six scalar pairs were exported to `runs/synthetic_v1_smoke` to exercise materialization, covering all three components at 0% and 1% noise for the first geometry/height/rotation.

The preview also lists every generated target height and the count calculation:
**3 geometries × 10 heights × 4 rotations × 3 noise levels = 360 vector pairs**,
then **360 × 3 components = 1,080 scalar pairs**. There are 30 distinct
geometry/height propagations before rotation and noise augmentation.

### Actual COMSOL comparison gallery

The local preview includes actual higher-plane COMSOL fields alongside noiseless
analytical predictions and signed residuals. Seven standoffs are shown:
L1_C1 at 1, 6 and 10 µm; IC_4um at 3, 6 and 10 µm; IC_2um at 2 µm.
Selection is the lowest, upper-middle and highest matching available height per
geometry, independent of measured error. Bz is visible by default; expand each
case to show Bx/By. There are 21 comparison figures in total.

The reference arrays and grid must match the synthetic dataset exactly. Actual
fields come from the comparison archive's `blurred` COMSOL arrays, not solver
reconstructions. No Gaussian noise is added in these comparisons. The actual
and predicted maps use the same height-specific colour scale; residuals have
a separate labelled scale. NRMSE uses the actual higher-plane COMSOL range.
`comsol_checks.csv` records full/interior metrics and `comsol_provenance.json`
records source hashes. These scores measure forward discrepancy, not inverse
reconstruction quality or a Gaussian-noise level.

To include these figures when creating a preview, add
`--comparison-run runs/combined_full_200`. To update the existing preview in
place, explicitly use:

```bash
"$QDM_PYTHON" scripts/plot_synthetic.py \
  --dataset runs/synthetic_v1 --output runs/synthetic_v1_preview \
  --comparison-run runs/combined_full_200 --refresh
```

`--refresh` only accepts an existing synthetic preview directory. It rewrites
the generated report/figures without modifying the dataset or source run.

**These are 1,080 augmented component pairs, not 1,080 independent physical configurations.** All variants descend from only three reference geometries. Their Bx/By/Bz components are related as well.

## Relationship to the previous experiment

| Previous combined experiment | This dataset pipeline |
|---|---|
| Actual COMSOL fields at 17 height pairs, giving 51 component pairs | Analytical fields at extra heights, rotations and noise settings |
| Lowest available plane used as reconstruction target | Same reference convention |
| Full images and bounded component batches | Full images; one component generated at a time |
| Recorded inventory, source hashes and physical units | Same principles, with explicit generation recipes and seeds |
| An operator constructed by the comparison runner, including fitted-kernel modes | Known analytical operator only; no fitting to synthetic targets |
| Existing shared-model training and classical comparison | Generation and inspection only; training integration is a separate next step |

Nothing is written into the previous run, original data, COMSOL cache, or comparison metrics. A synthetic manifest has its own schema and must not be passed directly to `compare_combined.py`, which expects actual COMSOL plane pairs. The existing 200-epoch model has already seen all three geometries; it cannot provide an unseen-geometry test on these references.

## Exact generation process

1. Read each geometry's lowest reference plane and its actual highest COMSOL plane from an existing prepared dataset or downloaded comparison archive.
2. Verify shape, finite values, units and source provenance. Copy only the lowest reference into the new portable dataset.
3. Assign each geometry to train, validation or test **before** creating variants. Every component, height, rotation and noise realization inherits that split.
4. Compute one shared scale `q = max(abs(training reference fields))`, excluding validation/test references. Save physical fields in µT; the loader can optionally divide inputs and targets by `q` without clipping.
5. Propagate the entire reference component using the frequency-domain transfer

   `H(fx, fy, Δz) = exp(-2π sqrt(fx² + fy²) Δz)`.

   Frequencies are in cycles/µm and Δz is in µm. The same scalar operator applies to each component. FFT calculations use float64; returned and saved fields use float32.
6. Apply the selected physical rotation to both target and propagated input.
7. Add independent, zero-mean Gaussian noise to the **propagated input only**. Keep the target and noiseless propagated field alongside it.

The references already contain COMSOL discretization artifacts. “Clean” here means **no added synthetic noise**, not an ideal noise-free magnetic field. This version does not generate new current layouts, crops, arbitrary-angle interpolation or reflections.

### Heights

With `--height-count 10 --min-dz-um 0.5`, each geometry receives 10 uniformly spaced propagation distances between 0.5 µm and its highest available plane minus its reference height. No generated height exceeds the existing height range.

| Geometry | Reference height | Generated absolute heights |
|---|---:|---|
| L1_C1 | 0.5 µm | 1, 2, …, 10 µm |
| IC_4um | 0.5 µm | 1, 2, …, 10 µm |
| IC_2um | 1 µm | 10 evenly spaced values from 1.5 to 2 µm |

Thus the L1 height grid matches its existing heights, while IC_4um and IC_2um also receive interpolated heights. New heights still share the same underlying reference. If `height-count=1`, only the minimum distance is used. Duplicate distances are removed when the available span collapses to that minimum.

### Meaning of 1% and 1.5% noise

For each propagated component, before normalization:

```text
sigma_uT = (noise_percent / 100) × (max(clean) − min(clean))
input = clean + Normal(0, sigma_uT²)
```

For example, if a propagated component spans 100 µT, 1% means σ = 1 µT and 1.5% means σ = 1.5 µT. These are percentages of the **propagated component's range**, not of its RMS, peak value, reference range, or per-pixel value. The added-noise range NRMSE, measured against `clean`, should be approximately the requested percentage. That is not the reconstruction NRMSE against the sharp target.

Noise is independent across pixels and components. Sigma changes with component and height. A constant field has zero range and therefore receives zero noise under this convention. A fixed sensor-noise floor, spatial correlation or common-mode vector noise is not implemented in v1.

The levels 0%, 1% and 1.5% are controlled experimental settings. The residual between analytical propagation and COMSOL also includes boundary effects, finite field of view and other numerical/model discrepancies; it is not a direct measurement of independent Gaussian sensor noise. Synthetic validation alone cannot establish performance on measured data.

PCG64 seeds are derived deterministically from the global seed, recipe ID, output component and epoch. Repeated loading with the same settings reproduces the fields. Record the NumPy version for reproducibility across machines. `--realizations N` creates N independent noisy variants at each nonzero level; the zero-noise variant is not redundantly repeated.

### Rotations and boundaries

Rows increase in physical +y, matching plots with `origin='lower'`. A 90° active counterclockwise rotation about +z transforms the field as:

```text
Bx_new = −spatial_rotate(By)
By_new =  spatial_rotate(Bx)
Bz_new =  spatial_rotate(Bz)
```

With this row convention, the spatial operation is `np.rot90(field, k=-1)`. Merely rotating each component's pixels independently would give the wrong vector transformation. The code handles all four quarter-turns and swaps grid dimensions/pixel spacings when needed.

The default forward operator is `Crop × FFT convolution × Zero-pad`, with **256 pixels per edge**. At 0.1 µm/pixel this is 25.6 µm per edge. `--pad 0` explicitly selects periodic propagation. Positive padding reduces periodic-image interaction but assumes zero field outside the supplied window; it does not recover the unknown exterior. Full images are propagated before any rotation. No crop approximation is introduced.

Preparation writes `boundary_audit.json`, comparing the highest actual COMSOL plane against propagation using padding 0, the selected padding, and twice that padding (1 pixel is used for the extra check when selected padding is 0). It records all components, full-image and interior errors, plus differences from the selected-padding prediction. The default interior margin is 128 pixels per edge. Interior and full-image NRMSE use their respective truth ranges, so consult RMSE too.

In the local run, Bz full-image range NRMSE at the highest existing plane was:

| Geometry | Periodic, pad 0 | Pad 256 | Pad 512 |
|---|---:|---:|---:|
| IC_2um, 1 → 2 µm | 1.2241% | 1.1623% | 1.1623% |
| IC_4um, 0.5 → 10 µm | 7.8308% | 6.5735% | 6.5858% |
| L1_C1, 0.5 → 10 µm | 5.2003% | 4.7113% | 4.7119% |

These are **forward-model discrepancies**, not chosen noise levels or reconstruction scores. The large-height mismatch remains even when padding changes little. Keep actual COMSOL fields as a separate realism check. Audit rows for a held-out test geometry are evaluation-only: do not select padding or other settings using those rows.

## Running on the professor's machine

Copy these new files into the corresponding locations in `~/learning_ker`:

```text
scripts/synthetic_data.py
scripts/plot_synthetic.py
hpc/generate_synthetic_direct.sh
tests/test_synthetic_data.py
README_SYNTHETIC_DATASET.md
```

They also use the existing `scripts/propagator.py`. Generation needs Python 3.10+ and NumPy; previews additionally need Matplotlib. The existing Python 3.11 environment is sufficient. No GPU, scheduler, pandas or new package installation is required for generation.

Run from the project folder on that machine:

```bash
cd "$HOME/learning_ker"
export QDM_PYTHON="$HOME/qdm-env/bin/python"

"$QDM_PYTHON" -m unittest discover -s tests -p 'test_synthetic_data.py' -v

bash hpc/generate_synthetic_direct.sh \
  --source-manifest runs/combined_dataset/manifest.json \
  --output runs/synthetic_v1 \
  --height-count 10 \
  --noise-percent 0 1 1.5 \
  --rotations 0 90 180 270 \
  --pad 256 --interior-margin 128 --seed 0

"$QDM_PYTHON" scripts/synthetic_data.py inspect --dataset runs/synthetic_v1

"$QDM_PYTHON" scripts/plot_synthetic.py \
  --dataset runs/synthetic_v1 --output runs/synthetic_v1_preview
```

If only the downloaded comparison outputs are available, replace the `--source-manifest ...` argument with `--source-run runs/combined_full_200`. That mode reads `setup.json` and the highest-height comparison NPZ for each geometry, specifically its `sharp` and `blurred` COMSOL arrays, not its reconstructions. It does not require a model checkpoint. The two source options are mutually exclusive.

All output directories must be new. Existing directories are refused rather than overwritten. The already generated local `runs/synthetic_v1` therefore does not need to be regenerated. Use a new name for a changed configuration. There is no interrupted-preparation resume in v1; a failed/incomplete dataset is rejected by the loader.

The wrapper sets a four-thread CPU budget. Override it with `QDM_THREADS` if needed. It generates references, audit and recipes; it does not eagerly write 1,080 full arrays. `--rotations` defaults to 0° if omitted; the command above explicitly enables all four rotations.

### Optional geometry holdout

To reserve IC_4um for validation and IC_2um for testing, add:

```text
--validation-geometry IC_4um --test-geometry IC_2um
```

Use a fresh output directory and train a **new** model. This gives 120 vector recipes / 360 component pairs in each split for the configuration above. The normalization scale is then determined only by L1_C1. Do not randomly split recipes or components: that would put variants of the same reference in both training and evaluation. IC_2um and IC_4um are related layouts, so describe this as a geometry-label holdout rather than proof of independent topology generalization. Choose the split to match the intended claim before training.

## Outputs and bounded loading

```text
synthetic_v1/
  manifest.json         Recipes, geometry splits, normalization, environment and provenance
  inventory.csv         One row per vector recipe
  boundary_audit.json   Analytical-versus-COMSOL and padding diagnostics
  status.json           complete / preparing / failed
  references/           One float32 [3, ny, nx] NPY per geometry, in µT
```

References use relative paths and SHA-256 checksums. The compact folder can be moved without its original source archives; original absolute paths remain only as provenance. Source metadata, source arrays/archives and generation code hashes are recorded. `inspect` checks the reference hashes and recipe consistency, not just the status flag.

The loader memory-maps references, computes FFTs one scalar component at a time, and caches at most one clean propagated component. It never constructs a dense convolution matrix or loads every variant into RAM. The cache is bounded, so a complete iteration can repeat FFT work; this deliberately favors predictable memory use.

Example from the project root:

```python
import sys
sys.path.insert(0, 'scripts')
from synthetic_data import SyntheticDataset

dataset = SyntheticDataset('runs/synthetic_v1')
for item in dataset.iter_components('train', epoch=0, normalized=True):
    target = item['sharp']     # 2-D float32, divided by q
    clean = item['clean']      # analytically propagated, divided by q
    observed = item['blurred'] # propagated + noise, divided by q
    metadata = item['metadata']
    # Consume one component, or explicitly collect a small bounded batch.
```

For fresh training noise, pass the training epoch number. The target, height and clean field remain unchanged. Validation/test data reject nonzero epochs and remain fixed. Reinitializing the dataset every batch is unnecessary and repeats checksum work. Do not wrap the iterator in `list(...)` for the full-resolution dataset.

When integrating a trainer, use each recipe's **known** analytical operator and its rotated grid. Do not estimate a kernel from the noisy pair. The existing PyTorch operator can be constructed as follows:

```python
from synthetic_data import transfer
from torch_inverse import PaddedFFT

m = item['metadata']
g, op = m['grid'], m['operator']
shape = (g['ny'], g['nx'])
h = transfer(shape, g['dx'], g['dy'], op['dz_um'], op['pad_pixels'])
operator = PaddedFFT(h, shape, op['pad_pixels'])
# Tensor convention: [batch, 1, ny, nx], same dtype/device as operator.
```

If a future solver uses a regularizer expressed in physical µT, its normalized lambda is `lambda_uT / q`, as in the existing solver contract. Inputs and targets must use the same q; denormalize before reporting physical RMSE. This README does not select lambda, network depth or a training schedule.

### Optional materialization

Export a small sample first:

```bash
"$QDM_PYTHON" scripts/synthetic_data.py export \
  --dataset runs/synthetic_v1 --output runs/synthetic_v1_smoke \
  --split train --limit 6
```

Each NPZ has `sharp`, `clean` and `blurred`, all in **physical µT**. Its separate export manifest records component, recipe, sigma, operator, grid, normalization scale and file hash. `--limit` counts scalar component pairs. Omit it to export the selected split completely; `--split all` includes all splits. Export uses epoch 0. Compressed size depends on the fields, especially noise; the 25.4 GB inventory estimate is the uncompressed size of all three float32 arrays across all pairs.

## Verification and next experiment

Eight CPU tests passed locally. They check an analytically known Fourier mode, zero-distance identity, padded adjoint consistency, rotation/propagation consistency on an anisotropic non-square grid, vector signs/positions, Gaussian statistics and exact seed repeatability, train-only normalization, split integrity, portable loading, checksums, source modes, bounded export and invalid settings. The shell wrapper passed syntax checking. The real-data preparation, inspection, preview and six-component export also completed. Across the nine preview components, measured added-noise range NRMSE was 1.4993%–1.5021% for the requested 1.5%. PyTorch is not installed in the local test Python, so no new GPU/training run or PyTorch equivalence test is claimed.

Before the next training comparison, select the evaluation split, connect this loader to the shared-model trainer using the known operator, and compare on fixed synthetic evaluation data and actual COMSOL fields. Keep synthetic scores separate from the prior 51-component COMSOL results. Add the resulting learning curves, reconstruction grids, stopping diagnostics and measured timings here after that experiment runs.
