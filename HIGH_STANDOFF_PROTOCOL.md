# High-standoff protocol: MAGIC-UNet at 10 → 0.5 µm

Written 14 November 2026, before either run was launched and before any 10 µm score of the new runs existed.
Everything below is fixed now. Later changes go in a dated "Amendments" section at the end, each written before the
number it affects is read.

## Question

Does MAGIC-UNet, trained on a narrow standoff band centred on 10 µm, reconstruct the 0.5 µm field from a 10 µm
measurement, and does a tile scaled to the standoff (0.7 µm pixels) do better than run 2's unchanged 0.2 µm pixels?

## The two runs

Both use the unmodified network and the paper's training recipe (RMSprop, learning rate 1e-4, 50 epochs, batch 32,
MSE, last-epoch model), free-space simulated data only (`simulate_fields.py`, no COMSOL, `decay_k0 = 0`), seed 0,
per-crop white noise (1, 1.5, 2 or 2.5% of each crop's own mean component range), and the equivalent-sheet target.
The standoff is never given to the network.

| | Run A (`run10a`, control) | Run B (`run10b`) |
|---|---|---|
| Standoff | N(10, 2) µm, cut at ±3σ (4–16 µm) | same |
| Pixels | 0.2 µm (2×2 average of the 0.1 µm grid) | 0.7 µm (7×7 average) |
| Tile | 64×64 = 12.8 µm | 64×64 = 44.8 µm |
| Training crops | 4,800 layouts × 4 standoffs × 8 crops = 153,600 | same layouts, same standoffs |
| Validation crops (training curves only) | 128 layouts × 4 × 8 = 4,096 | same |
| Data folder | `runs/magic_unet/data10a` | `runs/magic_unet/data10b` |

Run A differs from run 2 in the standoff and in the layout split (run 2: 2,400 layouts × 4 × 16). Run B differs from
run A in the pixel size only. Both pixel sizes and the 4,800-layout split are our choices, not the paper's.

Measured before launch, with no training (14 November): the share of a tile's field that comes from currents outside
the tile is 50% for run 2 (3 µm, 12.8 µm tile), 78% for run A's setting and 52% for run B's; the error of a perfect
current is 2.2% at 0.2 µm pixels and 4.6% at 0.7 µm (3 layouts).

## Evaluation data

1. **Primary: free-space simulation.** The first 50 validation layouts of `runs/simulated_3_to_05_v3` (never used
   for training; their own noise seeds and speckle), rendered at input height 10 µm, target 0.5 µm. The same layouts
   at 6, 8, 12 and 14 µm are reported to show the range. Scores are at full resolution (1400², 0.1 µm) against the
   exact simulated target.
2. **Secondary: COMSOL 10 → 0.5 µm, 15 component pairs** (L1_C1, IC_4um, and data3's three configurations). Nothing
   is chosen on COMSOL. It is reported with the statement that COMSOL differs from free-space physics by 64–140% at
   10 µm for a reason not yet known (`README_SIMULATED_FIELDS.md`), so these scores partly measure that mismatch.

## What is chosen, and on what

- **Tile stride** per run: from 16, 32, 48 on the first 10 of the 50 simulated layouts at 10 µm (the trainer's
  existing rule).
- **Height read-out:** as M1a of `M1_M2_PROTOCOL.md`, with the coarse grid extended to 0.5–20 µm.
- **Data-consistency step:** as M1b, same 17-value μ grid; μ(h) chosen per simulated standoff at 6, 8, 10, 12 and
  14 µm on the 50 layouts, correction at the true height (in-sample for μ, as in M1: a small optimism, stated).
- **Winner of A against B:** the lower mean relative L2 of the 0.5 µm field over the 50 simulated layouts at 10 µm
  (three components), for the network with the data-consistency step at its own estimated height h*. A difference
  under 1.0 point is reported as a tie (one seed each). The raw network score is reported next to it.
- **Baseline:** FISTA with the same learned dictionary as before, operator exp(−k·9.5 µm). λ from the same
  five-value grid on the first 16 simulated validation crops at 10 µm, extended if the best value is at a grid edge.
  Iteration count: the lowest 64-crop simulated validation error among 300, 1,000 and 3,000. Padding and crop size
  are fixed in an amendment before FISTA is run, using simulated data only.
- Run 2 and run 3 are scored on the same sets, inference only, with their own strides.

## What is reported

For each method on each set: raw relative L2; for MAGIC-UNet also corrected at the true height and at h*, the
read-out h* and its error; on COMSOL also the low-passed score (noise edge 2.9 rad/µm, fixed in M2a). The
information floor at 10 µm (56% L1_C1, 73.5% IC_4um on COMSOL) is quoted for context.

## Claims this can and cannot support

- It can show whether a narrow-band model works at 10 µm on free-space data, whether the wider tile helps, and how
  both compare with FISTA there.
- One seed per run. COMSOL at 10 µm is not a clean test. Only two wire shapes exist in COMSOL. Nothing here is
  experimental QDM data.

## Amendments

### A1. FISTA and scoring details (14 November 2026, about 07:30 server time)

Written while both runs were still generating data, before any 10 µm score of run A, run B or FISTA existed. The
only numbers seen so far are plumbing checks: run 3's existing checkpoint on COMSOL at 10 → 0.5 µm (77.8% raw on
the 15 pairs), and FISTA after 3 iterations on 16 crops (about 90%, in a test that the scripts run). No choice
below used them.

- **FISTA padding.** The smallest of 128, 256, 384 and 512 px whose best 16-crop simulated error (best over the λ
  grid, 300 iterations) is within 0.5 points of the best of the four. Reason: at 10 µm the blur kernel is about
  15–20 µm wide and the 3 µm study's 12.8 µm padding may be too small.
- **FISTA crops.** 512 px, the same 64 crop positions as the 3 µm study (positions depend only on the targets).
- **λ extension.** While the best scale is the largest tried, add 0.3, then 1, then 3. While it is the smallest
  tried (0.001, or the 2.29e-6 value that means almost no penalty), add 0.0003, 0.0001, 0.00003, 0.00001.
- **FISTA on the primary set.** FISTA at its reference depth is scored on full simulated images at 10 µm: all 50
  layouts if the reference depth is 300 or 1,000 iterations, the first 20 if it is 3,000 (cost). In the second
  case the MAGIC-UNet against FISTA comparison is made on those 20 layouts, and MAGIC-UNet is still reported on 50.
- **No stored-trial gate at 10 µm.** M2d's gate compared with stored 3 µm trials; none exist at 10 µm. The code path
  is M2d's, which passed that gate exactly; the new options default to the old behaviour.
- **Data-consistency padding** stays at 128 px, as in M1b.
- **Reproduction gate of the rescoring.** Pairs that a run's own evaluation never scored (run 2 and run 3 on data3
  at 10 µm) have no saved score to compare with and are skipped by the gate; the count is recorded.
- **Order on the server** (`hpc/run_high_standoff_scoring.sh`): padding test, then after both runs finish the
  height read-out and correction for run A, run B, run 2 and run 3 (simulated standoffs 6, 8, 10, 12, 14 µm; COMSOL
  pairs at 10 µm only), then FISTA.

### A2. FISTA depth sampled more finely (14 November 2026, about 08:00 server time)

Written before FISTA was launched at 10 µm; no FISTA score at 10 µm exists apart from the plumbing checks in A1.

- **Depths.** The iterate is saved at 300, 500, 700, 1,000, 1,500, 2,000 and 3,000 iterations (one run per λ; the
  iterate at step s does not depend on the total), not only at 300, 1,000 and 3,000. λ is chosen per depth as before.
  The reference depth is the one with the lowest 64-crop simulated validation error among these seven. Reason: with
  three samples the true best depth could lie anywhere between two of them, which would leave the baseline slightly
  weaker than it can be.
- **If the reference depth is 3,000**, the minimum is not bracketed and the report says so.
- **Full simulated images:** all 50 layouts if the reference depth is 1,000 or less, the first 20 if it is above.
- **Optional 3 µm recheck** (`QDM_RECHECK_3UM=1`): M2d's 3 → 0.5 µm FISTA with its own settings and gate, at the same
  seven depths. If a depth other than 1,000 has the lowest 64-crop validation error, that depth replaces FISTA-1000
  as the 3 µm baseline in every comparison, whatever its COMSOL score.

### A3. FISTA depth from a 25-step validation curve (14 November 2026, about 08:15 server time)

Replaces the depth rule of A2. Written before FISTA was launched at 10 µm; still no FISTA score at 10 µm apart from
the plumbing checks in A1.

- **Curve.** The simulated validation error is recorded every 25 iterations from 300 to 3,000 (the fields are not
  kept, so this costs about 1%). λ is chosen per depth on the first 16 crops, with the A1 extension rule applied
  until no depth has its best λ at an end of the grid.
- **Reference depth.** The minimum of the 64-crop curve. The report also gives the range of depths within 0.1 point
  of that minimum, and says so if the minimum is at 3,000 (not bracketed).
- **Why not a single exact step.** The curve is expected to be flat near its minimum, and the step is chosen on 64
  simulated crops; resolving it below 25 steps would fit that sample, not the method. It is never chosen on COMSOL.
- **COMSOL** is scored at the reference depth and, for continuity with M2d, at 300, 1,000 and 3,000.
- **Full simulated images:** all 50 layouts if the reference depth is 1,500 or less, the first 20 if it is above.
- **Optional 3 µm recheck** (`QDM_RECHECK_3UM=1`): M2d's 3 → 0.5 µm FISTA with its own settings and gate and the same
  25-step curve. If the reference depth differs from 1,000, FISTA at that depth replaces FISTA-1000 as the 3 µm
  baseline in every comparison, whatever its COMSOL score.

### A4. Server out of memory; run B resumed and scoring restarted (16 November 2026, about 01:15 server time)

Written before run B was resumed and before any score of run B, of the height read-out and correction, or of FISTA
at 10 µm existed. Numbers seen so far: run A's own evaluation (raw network: 54.8% on the 50 simulated layouts at
10 µm, 68.2% on the 15 COMSOL pairs at 10 µm) and the padding test (A1 rule; 128 px chosen). No choice below used
them.

- **What happened.** On 15 November, from shortly after 00:05, the server ran short of memory. The system's
  out-of-memory service (systemd-oomd) killed processes, and the machine was rebooted at 10:57. Run A had finished
  at 19:13 on 14 November and is unaffected. Run B was killed during epoch 50, with 49 epochs saved. The rescoring,
  FISTA at 10 µm and the 3 µm recheck were killed before they wrote any output. What used the memory is not
  established (the system logs need admin rights).
- **Run B** is resumed from its epoch-49 checkpoint with the trainer's existing resume path. Weights, optimiser
  state and shuffling state are restored; the training-noise generator is re-seeded for epoch 50, so that epoch's
  noise draws differ from an uninterrupted run. Nothing else changes.
- **One-line fix to the resume path** (16 November, 01:05, before run B resumed): the first resume attempt stopped
  at once because the saved random state had been loaded onto the GPU. `train_magic_unet.py` now moves it back to
  the CPU (`state['torch_rng'].cpu()`). Training and evaluation code are otherwise unchanged; run A never resumed
  and is unaffected.
- **Scoring** is re-run with the same script, settings and rules (A1, A3). The padding test is not repeated.
- **Diagnostic only:** memory use is logged every 5 minutes to `logs/memwatch_10um.log`.

### A5. Check: does COMSOL's extra decay explain run B's COMSOL score? (16 November 2026, about 08:00 server time)

Written after all protocol results were read (run B 15.8% on simulation and 59.3% on COMSOL; FISTA 48.8% and 54.3%)
and before any number of this check exists. It is an added diagnostic, not part of the pre-registered comparison, and
it changes no earlier result.

- **Why.** On COMSOL L1_C1, run B's Bz output has the right pattern (cosine similarity 0.95 with the target on the
  figure's 0.2 µm grid) but 40% of the amplitude. `README_COMSOL_MODEL.md` found that COMSOL's field decays as
  exp(−√(k² + k₀²)·z) with k₀ ≈ 0.10 rad/µm above the sample, so its 10 µm field is weaker than free-space physics
  gives. Run B was trained on free-space fields.
- **What is run.** The same 50 simulated validation layouts at 10 → 0.5 µm, rendered with the simulator's existing
  `decay_k0` setting at k₀ = 0, 0.05, 0.10 and 0.15 rad/µm (reference plane 0.5 µm, so the target is unchanged and
  only the 10 µm input changes). k₀ = 0.10 is the headline value; the others show the trend. Inference only: run B
  and run A with their existing checkpoints, strides and μ curves; FISTA at its reference depth (3,000 iterations,
  λ scale 0.0003, 128 px) with its free-space operator, at k₀ = 0.10 on the first 5 layouts only (cost). Nothing is
  re-chosen.
- **What is reported.** Relative L2 (raw, and with the data-consistency step at h*), h*, and per component the
  amplitude ⟨output, target⟩ / ⟨target, target⟩ and the cosine similarity, at full resolution. The same amplitude and
  cosine are computed for the saved COMSOL predictions.
- **Control.** k₀ = 0 must reproduce the saved simulated scores.
- **Reading, fixed now.** At k₀ = 0.10: if run B's error is 45% or more, the extra decay accounts for most of the gap
  between its simulated score (16%) and its COMSOL score (60% on the C1 family); if it is 25% or less, it does not;
  in between, it accounts for part. If FISTA's error there is lower than run B's on the same 5 layouts, the COMSOL
  ordering is reproduced by the decay alone.
- **Limits.** The simulator's `decay_k0` models the extra decay only, not COMSOL's box walls or mesh noise, and k₀ in
  COMSOL is 0.07–0.17 depending on the estimate and the mesh. A positive result shows the decay is sufficient to
  produce a gap of this size, not that nothing else contributes.

### A6. FISTA-Net and CSC-LISTA at 10 µm; FISTA curve to 10,000 iterations (16 November 2026, about 09:30 server time)

Written after all results above were read and before any of these three jobs was launched. Asked for on
16 November: the method comparison at 10 µm should show one MAGIC-UNet (run B with the data-consistency step) next
to FISTA, FISTA-Net and CSC-LISTA, and the FISTA depth with the lowest error should be found, not capped at 3,000.

- **CSC-LISTA at 10 → 0.5 µm.** The run that is in the 3 µm scoreboard (strict, 64 layers, 33.8%), with the same
  frozen code (`experiments/csc_lista_v4`) and the same arguments: same dictionary, filter width 155 px, 1,000 steps,
  batch 8, MSE loss, 512 px crops, 128 px padding, best checkpoint by the 64-crop simulated validation error. Two
  things change: the operator is exp(−k·9.5 µm), and the 200 training and 50 validation layouts of
  `runs/simulated_3_to_05_v3` are rendered at input height 10 µm.
- **FISTA-Net at 10 → 0.5 µm.** Run 2 of the 3 µm scoreboard (33.3%), same frozen code
  (`experiments/fista_net_code_start`) and arguments: 7 layers, 9,000 steps (360 epochs), batch 24, 90 evaluations,
  best checkpoint by validation. The same two changes.
- **The only code change** is one environment variable (`QDM_Z_INPUT`) in each copy of `train_conv_lista.py`, which
  sets the input height for rendering and for the COMSOL files. Unset, the code is unchanged. Copies:
  `experiments/high_standoff_methods`.
- **Known limits, stated before the runs.** Both methods train on 200 layouts and 512 px (51 µm) crops, as at 3 µm;
  MAGIC-UNet used 4,800 layouts. CSC-LISTA's learned filters are 15.5 µm wide, narrower than the 10 µm blur. Nothing
  is re-tuned for 10 µm, because the request is the same published methods with only the standoff changed. Both
  are given the true standoff, like FISTA.
- **Scoring.** As FISTA: full images of the first 20 simulated validation layouts at 10 → 0.5 µm, and the 15 COMSOL
  pairs at 10 → 0.5 µm. Nothing is chosen on COMSOL.
- **FISTA curve.** The 64-crop validation curve at 10 µm is extended from 3,000 to 10,000 iterations, every 25
  iterations, with λ chosen per depth on the first 16 crops among the scales 0.001, 0.0003 and 0.0001 (at 3,000
  iterations 0.0003 was best and both neighbours were worse). Curve only: no COMSOL, no full images.
- **Rule for the curve, fixed now.** The depth with the lowest 64-crop error is reported. FISTA is re-scored at that
  depth (COMSOL and the 20 simulated layouts) only if its validation error there is at least 0.5 point lower than at
  3,000 iterations; otherwise FISTA at 3,000 iterations stays the baseline. If the best λ at the new depth is the
  smallest of the three, that is reported as a grid end.
- **Order on the GPU** (`hpc/run_high_standoff_methods.sh`): CSC-LISTA, the FISTA curve, FISTA-Net.
