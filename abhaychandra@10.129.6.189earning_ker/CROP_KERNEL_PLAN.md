# Simple crop-pair kernel experiment: IC_4um

Implemented and run on 29 September 2026 using the user's requested direct crop experiment. This replaces the earlier full-context/masked-solver proposal. The subsequent request added held-out tests and a complete visualization report.

## Goal

Create multiple equal-sized sharp/blurred crop pairs from each IC_4um standoff pair. Use all crops and Bx, By, Bz jointly to learn one blur kernel per target standoff, then inspect the fitted kernels and predictions.

## 1. Load IC_4um only

Read the seven `data/data2/IC_4um_*.csv` exports. Use z=0.5 µm as the sharp reference and z=3, 4, 5, 6, 7, 10 µm as the six blurred targets. IC_2um is a different geometry and is outside this experiment.

Check that coordinates and array shapes match across these files. Keep fields in µT and arrays in `[y, x]` order. Use Bx, By, Bz; exclude Bnorm from the fit.

## 2. Create matching equal-sized crops

Start with **280×280 pixels** (28×28 µm). This divides each 1400×1400 image into a **5×5 grid of 25 non-overlapping crops**, using the entire image without resizing or dropping edges.

Use the exact same crop coordinates for the reference, every target, and every component. For example, crop `(row=1, col=2)` of reference Bx is paired with crop `(row=1, col=2)` of target Bx.

For each target height this produces:

- 25 spatial pairs, each containing Bx, By, Bz.
- Equivalently, 75 scalar-image pairs entering the joint kernel fit.
- Across six target heights: 150 spatial pairs / 450 scalar-image pairs.

Save the reference crops once, the target crops by height, and a small manifest containing crop coordinates, source filenames, component order, units, and heights. Suggested array layout: `[25, 3, 280, 280]` per plane. Save a crop-grid overview so alignment is easy to inspect. Crop size can be configurable, requiring exact divisibility for this first version.

## 3. Fit one kernel per standoff pair

Use the existing joint Fourier least-squares approach, adding the crop index to the sum:

```text
S[p,c] = FFT2(sharp_crop[p,c])
B[z,p,c] = FFT2(blurred_crop[z,p,c])

H[z] = sum over p,c of conj(S[p,c]) * B[z,p,c]
       / sum over p,c of abs(S[p,c])**2
```

This produces six 280×280 Fourier transfer arrays, one for each 0.5 → target pair. Fit one shared kernel across all crops and components for a given target; do not fit a separate kernel per crop or combine different target heights.

Use float64 calculations, with a documented small-denominator mask setting unsupported frequencies to zero. Start without Wiener regularization, independent crop normalization, windowing, or padding. Use the same direct crop FFT convention for fitting and prediction.

## 4. Apply and inspect

For each crop and component, predict:

```text
predicted_blurred_crop = real(IFFT2(H[z] * FFT2(sharp_crop)))
```

Save the Fourier transfer and centered real-space kernel for each target. Save crop predictions, per-crop/component RMSE and range NRMSE, and an aggregate per-component summary. Calculate aggregate metrics from all crop pixels together; mark zero-range NRMSE undefined. Include sharp/actual-blurred/predicted/error panels and reassemble predicted crops into a full-size mosaic for inspection.

These are fitting errors on the crops used to learn the kernel. Direct crop FFTs assume periodic crop boundaries, so edge artifacts may appear; inspect them as part of this initial experiment.

## 5. Implementation

Implemented in `scripts/kernel_crops.py`, handling IC_4um loading, aligned cropping, joint fitting, and saved diagnostics. Outputs are in `runs/ic4_crop_kernel/`; original data/cache/results are preserved. Runs and Python bytecode are ignored by Git.

Seven new tests in `tests/test_kernel_crops.py` passed: crop alignment/reassembly, known-kernel recovery and unseen-crop prediction, dense least-squares agreement, exclusion of held-out targets, height composition, zero-signal behavior, and complete coordinate validation. All six original NumPy iterative tests passed too. The LISTA test module could not import because PyTorch is absent from the current Python environment.

The delivered run includes float64 µT crop arrays, six direct-fit kernels plus separate holdout kernels, saved prediction arrays, metrics CSV/JSON, source hashes, and **34 figures** in [the HTML report](runs/ic4_crop_kernel/index.html). The initial crop atlases include every crop/component/target. Kernel plots include impulse responses, radial transfer amplitudes, and center-row profiles. Prediction mosaics include actual/learned/analytic fields and error maps. Impulse heatmaps use a labeled symmetric log color scale to expose weak tails.

## 6. Holdout tests added for this run

- **Spatial holdout:** fit separate kernels on 20 crops and score only the five excluded crop IDs `[0, 6, 12, 18, 24]`. Use the same split across all components/heights. These are separate models from the six all-crop fits. Both learned and analytic predictions use crop-wise circular FFTs.
- **Height holdout:** learn an anchor kernel from 0.5→3 µm and a 1 µm step kernel from 3→4 µm, using all 25 crops. Predict z=5 from the z=0.5 reference with `H_anchor * H_step**2`, i.e. 0.5→3→4→5. The fitting function receives only z=0.5, 3, and 4. The z=5 field is used afterward for scoring. Its direct fitted model elsewhere in the report is independent of this holdout prediction.

Height-holdout pooled range NRMSE:

| Method | Bx | By | Bz |
|---|---:|---:|---:|
| Composed learned kernels | 4.371% | 5.886% | 5.643% |
| Analytic propagation on the same crops | 6.269% | 9.199% | 10.086% |

Crop seams are visible in the predictions. These results describe this direct-crop model and this geometry. They do not compare against a full-plane analytic calculation or establish that cropping improves over full-image learning.

## Commands

```bash
# New experiment; output directory must not exist.
python3 scripts/kernel_crops.py --output runs/ic4_crop_kernel_new

# Regenerate all visualizations from saved arrays, without CSV loading/fitting.
python3 scripts/kernel_crops.py --output runs/ic4_crop_kernel --replot

# Numerical tests for this experiment.
python3 -m unittest discover -s tests -p 'test_kernel_crops.py' -v
```

`manifest.json` records the code hash at fitting time; `plot_provenance.json` records the plotting script hash on regeneration. The original run's plots were subsequently reformatted for readability without changing its learned arrays or metrics.
