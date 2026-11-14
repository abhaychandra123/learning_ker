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

None yet.
