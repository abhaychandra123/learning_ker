# M1 and M2 protocol (fixed before any new COMSOL score)

10 November 2026. Written before running anything that reads a COMSOL target. Its SHA-256 is recorded in
every output folder (`protocol_sha256.txt`). Any later change gets a dated "Amendment" section below, with
the reason; nothing above that line is edited after the first COMSOL score.

Where this comes from: `FABLE_REVIEW.md` §4.2, milestones M1 and M2. The review's 24.2% for "MAGIC-UNet + one
Tikhonov correction" came from trying three μ values on COMSOL test data. M1 replaces that with μ chosen on
simulation only.

## Common rules

- Code: new scripts only, copied to a fresh server folder `experiments/m1_m2/scripts` with their hashes.
  Frozen model code is imported from the experiment folder each model was trained with. Nothing is written into
  existing run folders. New outputs go to `runs/m1_m2/...`.
- COMSOL test pairs: `wiener_oracle.discover(runs/combined_full_200, data/data3)`, i.e. the 17 local pairs plus
  data3 C1+C2, C1+C2+C3, C1–C4 at 3 → 0.5 µm. 20 pairs in total.
- Metric: relative L2 per component over the full 1400 × 1400 image, as in every earlier table.
- Tables are always split by wire shape: **C1 family** (L1_C1, C1_C2, C1_C2_C3, C1_C2_C3_C4), **IC**
  (IC_4um, IC_2um), and the 15-pair mean at 3 → 0.5 µm where it exists.
- Reproduction gate: before any new quantity is computed from a model, its re-run inference must reproduce the
  saved `comsol.csv` of that model to within 1e-4 absolute relative L2 on every row. A model that fails the gate is
  reported as failed and not used.
- Simulated data: validation recipes of `runs/simulated_3_to_05_v3` (50 layouts, never used for training by
  any model), regenerated with `simulate_fields.synthesise` with `z_input` changed to the standoff under test and
  `z_target` = 0.5 µm. All other settings, including the correlated input noise, as stored in its manifest.

## M1. Re-score MAGIC-UNet run 2 and run 3 (inference only)

Models: `runs/magic_unet/run2/final.pt` and `run3/final.pt`. Code: `experiments/magic_unet_run3/scripts`
(identical `magic_unet.py`, `simulate_fields.py`, `wiener_oracle.py` to run 2's copy; checked by hash). Tile
stride: each run's own `stride` from its `results.json` (run 2: 32, run 3: 16). It is not re-chosen.

### M1a. Height read-out

For a predicted sheet current J (replicated to 0.1 µm) and an input field b at unknown height:

- misfit(h) = mean over Bx, By, Bz of ‖forward(J, h) − b‖ / ‖b‖, with `forward` = `magic_unet.field_from_sheet`
  (thin-sheet Biot–Savart, 700 px zero padding).
- Coarse grid h = 0.5, 0.6, …, 12.0 µm, then a fine grid of step 0.01 µm over ±0.1 µm around the coarse minimum.
- h* = argmin. A result within 0.1 µm of 0.5 or 12.0 is flagged "edge".
- Ground truth for simulated images is `z_input` (not `z_input + d`): the network's target is
  J_eff = J·e^(−kd), and forward(J_eff, z) equals the true field at height z above the zero-offset plane.
- Simulated standoffs: 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10 µm, 50 layouts each.

### M1b. Data-consistency correction

For a predicted field x at 0.5 µm and the input b at height h (true or estimated), per component, on a 128 px
zero-padded canvas (the same padding as the old Tikhonov baseline):

  X ← X + H/(H² + μ) · (B − H X),   H = exp(−k (h − 0.5)),

then crop to the image. This is exactly the step used in the review's diagnostic.

- μ grid: 10^(−4), 10^(−3.75), …, 10^0 (17 values).
- μ(h) is chosen per simulated standoff h as the value with the lowest mean full-image relative L2 (over the 50
  layouts and three components) after correction at the true height.
- For a height not on the simulated list, log10 μ is linearly interpolated in h between neighbours (and held
  constant outside 1–10 µm).
- Each run gets its own μ(h) curve.

### M1c. What is reported, for each run, on simulation and on every COMSOL pair

1. raw output;
2. correction at the true height;
3. correction at the estimated height h*, with μ(h*);
4. h* and h* − z_input.

### M1 decision rules (from the review)

- (a) passes if the correction at the true height lowers the simulated error at 3 µm, and then the COMSOL error.
  It fails as "COMSOL accident" if it helps COMSOL but hurts simulation.
- (b) passes if run 3's |h* − z_input| ≤ 15% of z_input on simulation over 1–10 µm, and run 3's IC_4um error falls
  clearly when the estimated-height correction is used. It fails if run 2's h* sticks near 3 µm for 2 and 4 µm
  inputs, or run 3's h* sticks to the top of its range on the IC coil.

## M2. Repair the measuring stick

### M2a. Noise cutoff (from targets only, before any prediction is scored)

- For each of the five 0.5 µm targets (L1_C1, IC_4um, C1_C2, C1_C2_C3, C1_C2_C3_C4): 128 px zero padding, FFT,
  least-squares projection of (Bx, By, Bz) onto fields one current sheet can produce:
  Bx = (k_y/k)φ, By = −(k_x/k)φ, Bz = iφ, in the simulator's component frame. The residual is the
  Maxwell-inconsistent part.
- Pool the five targets and sum, per radial bin of width 0.1 rad/µm, the total and inconsistent energy.
- The cutoff k_n is the lower edge of the first bin above 1.0 rad/µm whose inconsistent share exceeds 0.5. The
  region below 1.0 rad/µm is excluded because edge truncation, not mesh noise, dominates there.
- k_n is fixed once and used for every method.

### M2b. Four-way score for every saved prediction

For each pair and component:

1. raw relative L2;
2. relative L2 against the target low-passed at k_n (sharp cut, 128 px padding), with the prediction low-passed
   the same way;
3. in-band error ‖(P − T)·1[k < k_c]‖ / ‖T‖;
4. prior-band error ‖(P − T)·1[k_c ≤ k < k_n]‖ / ‖T‖,

where k_c is the first-crossing SNR = 1 frequency of `information_budget.py` for that pair and component (0.05
rad/µm bins, 128 px padding, free-space operator). k_c uses the truth, so (3) and (4) are a diagnostic
decomposition, not something a method could use. The energy share of the target in each band is reported next to
the errors.

### M2c. Predictions of the Group B models

Re-run, with each model's frozen code and `args.json`, on the five 3 → 0.5 µm configurations and save
full-image predictions:

| Model | Run folder | Frozen code |
|---|---|---|
| CSC-LISTA strict 32 | `runs/csc_lista_strict/layers_32` | `experiments/csc_lista_v4` |
| CSC-LISTA strict 64 | `runs/csc_lista_strict/layers_64` | `experiments/csc_lista_v4` |
| CSC-LISTA streamed 64 | `runs/csc_lista_stream/layers_64` | `experiments/csc_lista_stream` |
| CSC-LISTA relative 64, stored / streamed | `runs/csc_lista_relative/layers_64_{stored,stream}` | `experiments/csc_lista_relative` |
| LFISTA-style 32 | `runs/csc_lista_lfista/layers_32` | `experiments/csc_lista_lfista` |
| CSC-LISTA plain 16 | `runs/csc_lista_v2/full_plain` | `experiments/csc_lista_v2` |
| FISTA-Net run 2 | `runs/fista_net/run_2` | `experiments/fista_net_code_start` |

FISTA-300 predictions already exist (`runs/fista_depths_lambda/fista300_comsol_predictions.npz`) and are
re-checked against `runs/fista_depths_lambda/comsol.csv`. Each model passes the reproduction gate first.

### M2d. FISTA beyond 300 iterations

- Same dictionary, operator, padding and validation crops as `runs/fista_depths_lambda`.
- λ grid: the same scales 2.29e-6, 0.001, 0.01, 0.03, 0.1 × the dictionary's λ (0.3). If a depth's best scale is
  the largest (0.1), the scale 0.3 is added for that depth; if it is 0.001, the scale 0.0003 is added. This is decided
  on simulation only.
- One FISTA run per λ to 3,000 iterations, with snapshots at 300, 1,000 and 3,000. FISTA's iterate at step s does
  not depend on the total, so each depth's λ is chosen from the same trajectories on the first 16 validation crops.
- Reproduction gate: the 300-iteration snapshot must reproduce the stored `fista300` trials of
  `runs/fista_depths_lambda/results.json` within 1e-4.
- COMSOL: each depth with its chosen λ, all five configurations, predictions saved.

### M2e. Seeds of MAGIC-UNet run 2

- `experiments/magic_unet_run2/scripts/train_magic_unet.py`, called directly (not through
  `hpc/run_magic_unet.sh`), with run 2's `args.json` exactly, except `--seed 1` / `--seed 2` and new output folders
  `runs/m1_m2/magic_seed1`, `runs/m1_m2/magic_seed2`.
- Before launch: `runs/magic_unet/data/manifest.json` must contain `"noise"` (the per-crop noise fix).
- The crops were generated with seed 0, so the seeds vary initialisation, batch order and noise draws, not the
  training layouts.
- Reported as three individual values (seed 0 = run 2) and their range; no standard deviation from three runs.
- The seed spread sets the smallest COMSOL difference that is discussed as a difference.

## Amendments

**A1 (10 November 2026, before any M1 COMSOL score; after M2a).** M2a gave k_n = 2.9 rad/µm (pooled; per target
2.7–2.9). Two clarifications, no rule changed:

- One COMSOL pair has a 1 µm target (IC_2um 2 → 1). For it, the M1b correction uses H = exp(−k (h − 1)) and μ is
  read from the μ(h) curve at the input height (true or estimated). The curve itself is selected only at
  target 0.5 µm. This pair is reported but not in any 3 → 0.5 µm mean.
- The share threshold 0.5 in M2a: a band of pure, component-independent noise has an inconsistent share of about
  2/3 (two of three directions are rejected), so a share of 0.5 means noise carries about 75% of that band's energy.

**A2 (10 November 2026, after the M2d selection trials on 16 crops, before any FISTA-1000 or FISTA-3000 64-crop
validation or COMSOL score was read).** The FISTA *reference depth* for every comparison with learned methods is the
depth (300, 1,000 or 3,000) with the lowest mean relative L2 on all 64 validation crops, each depth at its selected λ.
FISTA-300 stays in the tables as the historical baseline.

**A3 (10 November 2026, before either seed finished training).** The headline MAGIC-UNet result is now the
estimated-height correction (h*), not the raw output, so the seed spread must be measured on it too. After training,
`m1_rescore.py` is run on each seed's `final.pt` exactly as in M1 (own stride from its `results.json`, own
reproduction gate against its `comsol.csv`, own μ(h) curve), except that the simulated standoffs are restricted to
2, 2.5, 3, 3.5 and 4 µm to save time. μ is held constant outside that range by the existing M1b rule. Only the
3 → 0.5 µm COMSOL rows of these re-scores are used for the seed spread. Rule (b) note: "≤ 15%" was not specified as
per-image or per-median; results report both and do not choose between them after the fact.

**A4 (10 November 2026, while both seeds were at epoch 19 of 50, before any seed result existed; after the FISTA-1000
COMSOL score was known).** M2e said the seed spread sets the smallest difference discussed, but not which spread. Fixed
now:

- Seed 0 is run 2 as scored in M1 (`runs/m1_m2/m1/comsol.csv`). Its 3 → 0.5 µm μ(h*) does not depend on the wider list
  of simulated standoffs used there: μ is chosen per standoff from that standoff's 50 layouts only (deterministic
  renders, fixed noise seeds), and every 3 → 0.5 µm h* of run 2 lies in 2.93–3.10 µm, so μ(h*) is interpolated between
  the 2.5, 3 and 3.5 µm values, which are in both lists.
- Spreads are reported as the range (max − min) of the three seeds, separately for each variant (raw, corrected at
  h*) and each subset (15 pairs, C1 family, IC_4um).
- **Claim against FISTA-1000 (deterministic given depth and λ).** "Run 2, variant V, beats FISTA-1000 on subset S" is
  written only if all three seeds of V score below FISTA-1000 on S. If one seed does not, the result is written as "not
  distinguishable on this test set", whatever the seed mean.
- **Difference between two seeded variants or methods.** A difference on subset S is discussed only if it exceeds the
  range of the three seeds of the same variant on S.
- These rules apply in the same way to the C1 family, where the current margin is 2.8 points, and to IC_4um, where it
  is 0.8 points.
