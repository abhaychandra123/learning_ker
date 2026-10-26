# Why did more synthetic data help so little?

Next authorized study: [README_GEOMETRY_CAPACITY.md](README_GEOMETRY_CAPACITY.md)
defines a clean L1_C1-only 2×2 comparison of depth (16/32) and parameter sharing
(tied/untied), with 163 to 5,216 parameters. Five tests and a one-update full-size
local smoke run completed. Full GPU launch awaits the user's Mac terminal
because SSH from this session is blocked. This is a separate experiment.

26 October 2026. Read-only local diagnostics of the completed augmented epoch-200
checkpoint, SHA256 `bd8d704b5c1eda121f0dc2559c6805cd037e15fe68f167e29064bc04b7016eee`.
No retraining, checkpoint edits or tuning against the final evaluation were done.
The formal 51-versus-1,080-pair outcome remains in README_SYNTHETIC_EXPERIMENT.md.

## Main finding

The model remains poor on clean inputs produced with its own forward operator,
and longer classical solves with that same operator improve substantially over
its 16-layer output. Therefore a shortage of augmented examples or a domain gap
alone does not explain the large-standoff failure. Model fitting, shared-parameter
tradeoffs, iteration budget and inverse-problem conditioning need investigation.
The tests do not yet isolate capacity from optimization or prove a code defect.

## Measurements

All probes use full 1,400 × 1,400 Bz fields, analytical pad 256, float32 CPU
inference, and the original global normalization. Clean synthetic inputs are
regenerated using the original NumPy propagator, with no noise or rotation.
Real FFT inference matches the earlier full-FFT local 10 µm Bz report's relative
L2 error to within 1.35e-10. The original model weights remain unchanged.

| Geometry / input → target height | Trained relative L2 error, actual input | Trained relative L2 error, clean synthetic input | Forward discrepancy |
|---|---:|---:|---:|
| L1_C1, 1 → 0.5 µm | 40.51% | 20.77% | 16.56% |
| L1_C1, 6 → 0.5 µm | 78.31% | 78.01% | 16.46% |
| L1_C1, 10 → 0.5 µm | 95.28% | 94.98% | 39.47% |
| IC_2um, 2 → 1 µm | 35.79% | 27.32% | 19.35% |
| IC_4um, 10 → 0.5 µm | 87.20% | 87.01% | 36.39% |

Forward discrepancy is `||A(sharp) - actual_input|| / ||actual_input||`,
not reconstruction error and not a percentage of full-scale field amplitude.
The trained reconstruction error uses the sharp target in the denominator.
These are selected Bz probes, not averages over the complete experiment.
The noiseless/unrotated L1_C1 10 µm recipe exists in the training manifest.

### Same-operator iteration comparison: L1_C1 Bz, 10 → 0.5 µm

| Method | Actual-input relative L2 error | Clean-input relative L2 error |
|---|---:|---:|
| Untrained initialization = ISTA, 16 iterations | 97.46% | 97.21% |
| Trained shared model, 16 layers | 95.28% | 94.98% |
| FISTA, 16 iterations | 96.42% | 96.12% |
| FISTA, 64 iterations | 89.05% | 88.57% |
| FISTA, 256 iterations | 71.43% | 70.54% |

All use the same analytical operator, padding, input, target and blurred
initialization. ISTA/FISTA use the original physical lambda 1e-4 µT; the trained
model retains its learned threshold (~0.00756 µT). This compares attained
reconstruction errors, not identical regularized objectives or equal compute.
FISTA is run to fixed budgets, not asserted to have converged. Its 71% error
still leaves substantial error; this is evidence of improvement available
beyond the current output, not a solved reconstruction or a recoverability bound.

The learned corrections help some high-standoff cases but hurt some easy cases.
At L1_C1 1 µm, actual-input error rises from 25.85% for initial ISTA-16 to 40.51%
for the trained model; clean-input error rises from 14.62% to 20.77%.
At IC_2um 2 µm, actual-input error rises from 24.41% to 35.79%.
These initial-ISTA comparisons are distinct from the trained 51-pair control.
They suggest investigating shared-height/geometry tradeoffs rather than assuming
that the larger dataset improved every part of the learned inverse mapping.

![Selected-case diagnostics](runs/synthetic_fit_diagnostic_v1/fit_diagnostic.png)

## Why pair count alone is an incomplete explanation

- 1,080 component pairs still originate from three source geometries. Rotation,
  standoff and noise variation broaden conditions, not independent structures.
- Both arms received 200 full-dataset Adam updates. More samples per update did
  not give more optimization steps. Duplicating every sample equally would
  leave the mean objective/gradient unchanged apart from numerical effects.
- The model has 163 parameters: two single-channel 9 × 9 convolution kernels
  and a threshold, tied across 16 layers, heights, components and geometries.
  A small parameter count alone does not prove insufficient capacity; the
  observed fit must be tested directly.
- Geometry weights changed: IC_2um went from 3/51 (5.88%) to 360/1080 (33.33%);
  L1_C1 from 30/51 (58.82%) to 33.33%; IC_4um from 35.29% to 33.33%.
  The formal experiment's largest relative gain was IC_2um. Reweighting is a
  plausible contributor, not a demonstrated causal explanation.
- Under the implemented continuation law, a spatial wavelength L is attenuated
  by `exp(-2*pi*dz/L)`. At dz=9.5 µm, a 5 µm wavelength retains only about
  6.54e-6 of its original amplitude, while a 10 µm wavelength retains ~0.00256.
  Direct inversion amplifies noise and mismatch. Learned priors can help but
  cannot guarantee recovery of unsupported detail. These are ideal spectral
  examples, not measured per-feature SNR or exact finite-window singular values.

Background primary research: Ye et al. (2022),
https://doi.org/10.3389/feart.2022.897055, describes instability/noise amplification
in downward continuation. Gregor and LeCun (2010),
https://icml.cc/Conferences/2010/papers/449.pdf, introduces learned sparse-code
approximations. The current constrained, operator-conditioned correction model
must be assessed on its own evidence; published LISTA gains do not establish
that this implementation will attain comparable acceleration.

## Next experiments, in order

1. **Single-pair fit diagnostic.** Start with one clean, easy L1_C1 Bz pair,
   then a separate clean 10 µm pair. Compare fresh initialization and a separate
   fine-tune copy of the final model. Track loss, relative L2, amplitude slope,
   parameter changes, gradient norms, clipping and threshold clamping over
   more updates. Use a small predeclared learning-rate sweep. Do not overwrite
   the completed run or interpret this training-only fit as generalization.
   Failure to fit after a bounded trial still requires distinguishing optimizer
   problems from representational limits and numerical conditioning.
2. **Separate depth from fitting.** Train small diagnostic models with 16/32/64
   layers under controlled budgets, and compare tied versus layer-specific
   parameters. Merely repeating the trained 16-layer weights at greater depth
   is not an equivalent trained model. Use matched-operator FISTA curves as
   attainable reference points, not as guarantees for a neural model.
3. **Test parameter-sharing and weighting.** If one-pair fitting succeeds, add
   heights and geometries progressively. Record per-group losses/gradients.
   Compare preserved original geometry weights against equal weights. This
   tests whether improvements on one group trade off against another.
4. **Locate the forward mismatch.** Map `A(sharp)-actual_input` and compare
   spectra, borders and interior. Check standoff metadata, grids, registration,
   finite field of view/zero exterior assumptions and COMSOL boundary settings.
   A measured discrepancy is not by itself proof of a broken analytical law.
5. **Only then expand data or change the loss.** If clean inputs fit but actual
   inputs do not, improve forward realism. If fitting works but independent
   geometry tests fail, acquire genuinely new geometries. ROI/relative-error
   weighting can be tested as explicit objective changes, retaining full-field
   physics; do not silently crop or change the completed protocol.

## Reproduction and files

```bash
MPLCONFIGDIR=/tmp/qdm-synthetic-inference-mpl \
  /tmp/qdm-synthetic-torch-env/bin/python scripts/diagnose_synthetic_fit.py \
  --output runs/synthetic_fit_diagnostic_new
MPLCONFIGDIR=/tmp/qdm-synthetic-inference-mpl \
  /tmp/qdm-synthetic-torch-env/bin/python scripts/plot_synthetic_fit_diagnostic.py \
  runs/synthetic_fit_diagnostic_new
```

Raw results: `runs/synthetic_fit_diagnostic_v1/diagnostics.json`, 36 reconstruction
rows, five forward checks, checkpoint/archive/code provenance. The measured
local diagnostic took 272.2 seconds, excluding plotting. A post-run wording
correction distinguishes classical lambda from the learned neural threshold;
numerical results are unchanged. No new training experiment has been launched.
