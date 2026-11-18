# MAGIC-UNet for our standoff reconstruction

7 November 2026. Port of MAGIC-UNet (Reed et al., "Machine learning for improved current-density
reconstruction from two-dimensional vector magnetic images", Phys. Rev. Applied 23, 034035 (2025),
doi:10.1103/PhysRevApplied.23.034035; preprint arXiv 2407.14553) to our 3 → 0.5 µm
problem. **Run 2 (8 November 2026) scored 27.0% on COMSOL 3 → 0.5 µm without being told the
standoff, against FISTA-300's 30.5%.** Run 1 was invalid (BatchNorm bug); see Results.

## What they did

| Item | Paper (and released files) |
|---|---|
| Task | Vector field (Bx, By, Bz) at one plane → planar current density (Jx, Jy). Not the near-plane field. |
| Network | U-Net, 64×64 input, 3 levels of 2×2 average pooling, 64→128→256→512 channels, 3 × [conv3×3, BatchNorm, ReLU] per level, then **two separate decoders**, one for Jx and one for Jy. First (64-channel) skip connection omitted. |
| Data | 153,600 training and 4,096 validation pairs, from random wires: Class I curves (COMSOL-simulated current), Class II straight segments; 2 mm field of view (31 µm pixels); 14 µm thick current layer; widths 9–320 µm; 2–311 mA. Fields from Fourier-space Biot–Savart. |
| Standoff | Drawn per image from a normal distribution, 50 ± 10 µm (500 ± 50 µm for a separate second model). **The network is not told the standoff** (paper Sec. III, Table V, Sec. V C; checked against the published PDF 18 Nov 2026). |
| Augmentation | 90° rotation, horizontal and vertical flips, each with 50% probability (8 variants). About 2% of images have wires only outside the field of view. |
| Noise | Additive Gaussian, redrawn every epoch. σ_n = σ / max\|B\|. Training σ_n is not stated. |
| Scaling | Each input divided by its own scalar α_i; outputs divided by the same α_i and multiplied by a global β (~1e8) (App. A). The paper does not say what α_i is; we use max over pixels of \|B\|. |
| Training | RMSprop, learning rate 1e-4, 50 epochs, mini-batch 32, MSE; TensorFlow 2.10; one RTX A6000; 11 hours. |
| Larger images | Overlapping 64×64 tiles; overlaps divided equally between tiles, duplicates discarded (Fig. 5c). |
| Results | SSIM 0.99 → 0.84 for σ_n 0 → 1, versus Fourier method 0.71 → 0.32; RMSE 3–4× lower. Weak on correlated experimental noise, multiple planes and 500 µm standoff. |

## Four sources, and which one we follow

1. **The paper** (text, Fig. 1b, Table IV, Table V, Appendix A; the published version has no supplement).
2. **The released trained model**, `MAGIC_UNet_64x64_50um/checkpoint/keras_metadata.pb`
   (Harvard Dataverse doi:10.7910/DVN/SD6PVP). Its full layer graph is saved in
   `tests/data/magic_unet_released_layers.json`. It agrees with the paper and settles the details:
   pooling stride 2 (the paper text says "stride of 1"); nearest upsampling followed by a 3×3 conv
   with no BN; concatenation order [encoder, upsampled]; BN + ReLU applied after the concatenation;
   final conv3×3 to 2 channels then conv3×3 to 1 channel, both linear; He-normal (truncated)
   initialisation, zero biases, BN momentum 0.99 and epsilon 1e-3. 15,505,194 trainable parameters.
   Variable bytes match the released checkpoint to 0.15%.
3. **The released training code** (`unet_model.py`, `unet_trainer.py`, `train.json`). It does **not**
   match the paper or the released model: a different U-Net (tanh, no BN, 128→2048 channels, one
   joint output), lr 1e-5, 25 epochs, batch 16, noise σ 0.003, ×U(0.99, 1.01) multiplicative noise,
   and BN called with `training=False`. It reads like an earlier version. **Ignored.**
4. **Our adaptations**, below.

So: architecture from source 2, training recipe from source 1.

## Our adaptations (each one is ours, not the paper's)

| Adaptation | Why |
|---|---|
| Our current-layout simulator (`scripts/simulate_fields.py`: spirals, meanders, mirrored arrays) instead of their wire classes | Their Class I needs COMSOL LiveLink; ours match the COMSOL test coils. Each layout gets a random quarter turn and mirror, which gives the paper's 8 augmentations physically, at the source (vector components stay consistent). |
| Standoff N(3 µm, 0.6 µm), truncated at ±3σ (1.2–4.8 µm) | Same 20% spread as their 50 ± 10 µm, centred on our 3 → 0.5 µm benchmark. Unknown to the network, as in the paper. |
| 0.2 µm pixels (0.1 µm simulation, area-averaged 2×2), 64×64 crops = 12.8 µm | Paper's image size. Measured pipeline floor with a perfect current: 2.93% at 0.2 µm vs 2.85% at 0.1 µm. **Caveat:** the tile is only ~4× the standoff (their 50 µm model ~40×), so most of every tile's field comes from wires outside it. Their separate 500 µm model is also ~4× (2 mm view), and there the paper says both methods were too poor to score (Sec. III C). |
| Target = equivalent sheet current J_eff = J·e^(−k·d) at the zero-offset plane, divergence-free | Our source offset d (0–0.2 µm) is unknown at test time. Predicting the true J and assuming the mean offset costs 5.1% at 0.5 µm even with a perfect J; J_eff makes B(h) = forward(J_eff, h) exact. |
| Training noise: the paper's white Gaussian, with σ per layout taken from our simulator's setting (1, 1.5, 2 or 2.5% of each component's range) | The paper does not state its training σ_n. Never derived from COMSOL. Correlated noise (`--noise-corr-um`) is an ablation only; it is the authors' own suggested fix. |
| β = power of ten making max\|J/α\| of order 1 (0.1 here, from training data) | Paper: "β ~ 1e8" for SI units; ours are µT and A/m. |
| Near-plane field: replicate the predicted J to 0.1 µm, then thin-sheet Biot–Savart (no divergence projection) to 0.5 µm | Our goal is the field, not J. Plain Biot–Savart is linear, so currents outside the image only cost their own field (2.9% floor). Re-projecting a cropped current spreads the error over the whole image (7.3%). |
| Tiling stride chosen from {16, 32, 48} on simulated full-image validation | Paper does not give its overlap. Never chosen on COMSOL. |
| PyTorch; RMSprop with Keras defaults (alpha 0.9, eps 1e-7) | Same optimiser maths as TF 2.10. The last-epoch model is the result, as in their trainer. |

## Pipeline floor (perfect current, simulated validation layouts)

| Setup | Error at 0.5 µm |
|---|---:|
| True J at 0.1 µm, assume mean offset 0.1 µm | 5.14% |
| J_eff, full canvas, no cropping | 0.00% |
| J_eff cropped to the 140 µm image, 0.1 µm | 2.85% (0.91% excluding a 10 µm border) |
| J_eff cropped, 0.2 µm, replicated to 0.1 µm (**used**) | 2.93% (1.07% excluding a 10 µm border) |

Field-to-field methods (FISTA, CSC-LISTA) do not pay this floor; MAGIC-UNet does.

## Files

- `scripts/magic_unet.py`: network, Keras initialisation, tiled inference, Biot–Savart, metrics.
- `scripts/magic_data.py`: training crops (memory-mapped `.npy`, float16 by default).
- `scripts/train_magic_unet.py`: training, then evaluation on simulated full images and COMSOL.
- `tests/test_magic_unet.py`: layer-by-layer match with the released model, parameter count,
  checkpoint size, branch independence, initialisation, tiling, physics, data.
- `hpc/run_magic_unet.sh`: server run (data, then training; re-run to resume).

## Run

```bash
python -m unittest tests.test_magic_unet
QDM_PYTHON=/path/to/venv/bin/python QDM_RUN=run2 bash hpc/run_magic_unet.sh
```

The full dataset is about 6.5 GB (float16) and takes ~15–20 min with 12 workers.
Training is expected near the paper's 11 hours on the A6000.

## Evaluation outputs

`results.json`: stride trials, simulated full-image validation (field relative L2 at 0.5 µm,
and the paper's J metrics: SSIM, PSNR, RMSE per Jx/Jy against J_eff), and COMSOL relative L2
for every local pair (plus data3 3 → 0.5 µm on the server), each marked inside or outside the
training standoff spread. `comsol.csv`, `sim_validation.csv`,
`comsol_3_to_05_predictions.npz` (predicted J at 0.2 µm and B at 0.1 µm).

Compare at 3 → 0.5 µm: FISTA-300 30.5% and FISTA-Net 33.3% (five configurations, server). Local
runs have only L1_C1 and IC_4um; label them as two-configuration scores. Pairs at 2 and 4 µm are
inside the training spread (±1.7σ): a direct test of the unknown-standoff claim.

## Server run (launched 7 November 2026, 09:22 server time)

Frozen code: `~/learning_ker/experiments/magic_unet` (13/13 tests pass there). Outputs:
`~/learning_ker/runs/magic_unet/{data,run}`; logs `runs/magic_unet/data.log`, `run.log` and
`logs/magic_unet.launch.log`. Launched with `setsid nohup`, 24 data workers. Data generation runs
at about 25 layouts a minute (≈1.7 h for 2,464 layouts), then 50 epochs of training. If it stops,
re-run the same command; training resumes from `run/latest.pt`.

## Results

**Run 1 (finished 7 November 2026, 14:25): invalid because of a BatchNorm bug.** Training took
50 × 286 s ≈ 4 h. The scores in `run/results.json` are 74% on simulated validation and 82% on
COMSOL 3 → 0.5 µm (five configurations). The predicted field is about 3.5× too small, which is no
better than copying the input (75%).

The cause is the BatchNorm running statistics, not the weights:

| Validation crops (2,048) | Relative L2 of J | Best-fit scale |
|---|---:|---:|
| Eval mode, stored running statistics | 93.8% | 5.8 |
| Batch statistics (train-mode BN) | 17.0% | 1.00 |
| Running statistics recalibrated on clean training batches | 16.8% | 1.00 |

About 0.5% of training crops lie far from every wire. They have a tiny α but the layout's full
noise σ, so σ/α > 1, up to 165. Those batches corrupt the running variance. Recalibrating
including them gives 98.9%; recalibrating without them gives 16.8%. The data-design cause is that
σ is set per layout, whereas the paper's σ_n is relative to each image's own max|B|.

Not re-scored on full images or COMSOL. Diagnostic script: `logs/magic_diag.py` on the server.

**Run 2 (launched 7 November 2026, about 17:00 server time): fixed data.** Noise is now the
simulator's percent of each crop's own mean component range (`crop_sigma` in `magic_data.py`). The
stored crops were kept: `magic_data.py --resigma` rewrote only the noise levels, drawing the percent
afresh per crop because run 1 did not store it. Run 1's levels are kept as `*_sigma_run1.npy`.
Over a 20,000-crop sample, σ/α is now median 1.4%, 99th percentile 3.3%, maximum 4.2%
(run 1: median 3.0%, maximum 165). Frozen code: `experiments/magic_unet_run2`, 13/13 tests pass.
Outputs: `runs/magic_unet/run2`, log `runs/magic_unet/run2.log`. About 4 h of training, then scoring.

**Run 2 results (finished 8 November 2026, about 01:00).** Training loss reached 0.00083 and
validation loss 0.00095 at epoch 50, with no train/validation gap. Stride 32 was chosen on simulated
validation. Field error at 0.5 µm on 50 full simulated images: 15.2%. J against J_eff: SSIM
0.81/0.89 (Jx/Jy), relative L2 26%/24%.

COMSOL, 3 → 0.5 µm, 15 pairs, same measure as the CSC-LISTA table (per-component relative L2 over
the full image, averaged):

| Method | Standoff given? | L1_C1 | IC_4um | C1+C2 | C1+C2+C3 | C1–C4 | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| **MAGIC-UNet run 2** | **No** | **26.3%** | 29.6% | **26.6%** | **26.4%** | **26.0%** | **27.0%** |
| FISTA-300, lambda 0.009 | Yes, 3 µm | 31.2% | **27.6%** | 31.4% | 31.2% | 31.1% | 30.5% |
| FISTA-Net run 2 | Yes, 3 µm | 33.1% | 34.6% | 33.2% | 33.1% | 32.7% | 33.3% |

Other standoffs, one network, standoff never given (two configurations at most; trained range
1.2–4.8 µm):

| Input → 0.5 µm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean relative L2 | 59.2% | 31.7% | 27.0% | 36.8% | 51.8% | 66.5% | 78.4% | 93.8% |
| Inside training spread | no | yes | yes | yes | no | no | no | no |

IC_2um 2 → 1 µm: 26.6%. Inside the trained range, one network handles an unknown standoff.
Outside it, the error rises quickly, including at 1 µm, which is closer than training.

**Run 3 (launched 8 November 2026): any standoff, one network.** It changes exactly one thing
from run 2: the training standoff is drawn uniformly from 1–10 µm (`magic_data.py --z-uniform 1 10`)
instead of N(3, 0.6). That range is exactly the span of the COMSOL input standoffs. The network is
still never told the standoff. Everything else is identical to run 2: the architecture, 64×64
tiles at 0.2 µm, 153,600 + 4,096 crops, 4 standoffs × 16 crops per layout, per-crop noise, the
paper's training recipe, stride chosen on the same 3 → 0.5 µm simulated validation, and the same
COMSOL scoring. The paper trained one model per standoff range (50 µm and 500 µm), so a single
1–10 µm model is our adaptation. Code: `experiments/magic_unet_run3` (14/14 tests). Data:
`runs/magic_unet/data_any`. Outputs: `runs/magic_unet/run3`, log `run3.log`. Launch variables:
`QDM_DATA=data_any QDM_DATA_ARGS="--z-uniform 1 10" QDM_TRAIN_ARGS="--z-range 1 10" QDM_RUN=run3`.

**Run 3 results (finished 9 November 2026).** Final training loss 0.020 and validation loss 0.026
(run 2: 0.0008 / 0.0009; wider standoffs are harder). Stride 16. Simulated validation at 3 → 0.5 µm:
22.7% (run 2: 15.2%). COMSOL, per configuration, standoff never given:

| Pair | Run 2 (3 ± 0.6 µm) | Run 3 (1–10 µm) | Lost-band floor |
|---|---:|---:|---:|
| L1_C1 1 → 0.5 | 59.2% | 51.3% | 17.5% |
| L1_C1 2 → 0.5 | 31.7% | 36.3% | 21.0% |
| L1_C1 3 → 0.5 | 26.3% | 29.8% | 33.4% |
| L1_C1 4 → 0.5 | 34.9% | 33.4% | 39.8% |
| L1_C1 5 → 0.5 | 53.4% | 37.7% | 41.3% |
| L1_C1 6 → 0.5 | 67.3% | 45.9% | 42.6% |
| L1_C1 7 → 0.5 | 78.8% | 53.2% | 44.8% |
| L1_C1 8 → 0.5 | 87.1% | 64.3% | 47.4% |
| L1_C1 9 → 0.5 | 92.5% | 70.0% | 53.6% |
| L1_C1 10 → 0.5 | 95.1% | 79.3% | 56.0% |
| IC_4um 3 → 0.5 | 29.6% | 52.0% | 35.6% |
| IC_4um 4 → 0.5 | 38.7% | 54.5% | 42.6% |
| IC_4um 5 → 0.5 | 50.2% | 51.5% | 52.4% |
| IC_4um 6 → 0.5 | 65.7% | 54.3% | 58.2% |
| IC_4um 7 → 0.5 | 78.0% | 54.6% | 65.4% |
| IC_4um 10 → 0.5 | 92.6% | 80.1% | 73.5% |
| IC_2um 2 → 1 | 26.6% | 46.2% | 21.2% |
| C1+C2 / +C3 / +C4, 3 → 0.5 | 26.6 / 26.4 / 26.0% | 30.1 / 31.1 / 30.8% | – |
| **Mean 3 → 0.5, five configurations** | **27.0%** | **34.8%** | 34.5% |

The single 1–10 µm model is much better than run 2 outside run 2's range and somewhat worse inside it.
On the L1/C1 shape at 3 µm it scores about 30%, roughly level with FISTA-300 (31.2%). The IC coil
fails at 3–4 µm (52–55%; IC_4um 3 → 0.5 is By 68.8%, Bz 55.9%, Bx 31.3%), and the IC pair 2 → 1 is
also worse. One training run per setting; no seed repeats.

**Why run 3 fails on the IC coil (diagnosed 9 November 2026, saved predictions only, no retraining).**
The error is concentrated at the IC coil's pairs of closely spaced parallel wires (top-middle and
bottom-middle of the image). There run 3 predicts about twice the true current (max |J| 1,195
against run 2's 560 A/m). Its error is at large scales: in the 0–0.5 rad/µm band it is 40% for By
(run 2: 14%). Elsewhere the two runs agree. To see which standoff each run's current implies, we
forwarded the predicted J to heights 1.5–8 µm and compared with the measured 3 µm input, in
14 µm windows. Run 2 implies 2.2–4.5 µm everywhere. Run 3 implies about 3 µm in most windows but
7.5–8 µm (the top of the grid) around the top-middle pair and 5–6 µm around the bottom-middle
pair. On L1_C1 both runs imply about 3 µm everywhere.

So the cause is the source–height ambiguity, decided tile by tile. In the 3 µm input a closely
spaced pair blurs into one weak, broad feature, which a 12.8 µm tile cannot tell apart from wires
much farther away. Run 3's training range allows up to 10 µm, so in those tiles it assumes a far
source and amplifies. Run 2's prior only allows about 3 µm, so it cannot. Different tiles of one
image end up implying different standoffs, although an image has only one. This also explains why
IC_4um stays at 52–55% across 3–7 µm. The IC 2 → 1 µm pair was not checked. Scripts:
`logs/ic_diag.py`, `logs/implied_height.py` on the server.

**Earlier.** Locally only the unit tests and a plumbing smoke test ran (4 layouts, a few minutes): training
loss fell, but the model overfits 4 layouts, so its scores mean nothing. A mid-size pilot belongs on the
server; it was started locally by mistake and stopped.

## High-standoff runs at 10 → 0.5 µm (launched 14 November 2026, 07:04 server time)

Rules fixed before launch: [HIGH_STANDOFF_PROTOCOL.md](HIGH_STANDOFF_PROTOCOL.md) (hash in
`runs/magic_unet/protocol_10um_sha256.txt` on the server). Results, figures and the full account (16 November):
[final_evaluation/02_source_docs/p9_high_standoff/README_HIGH_STANDOFF.md](final_evaluation/02_source_docs/p9_high_standoff/README_HIGH_STANDOFF.md).
In short, at 10 → 0.5 µm: on simulation MAGIC-UNet (run B, 44.8 µm tile, with the data-consistency step) 15.8%,
FISTA 48.8%, CSC-LISTA with its 3 µm settings 91.8%, and run A (12.8 µm tile) 58.2%; on COMSOL FISTA 54.3%,
MAGIC-UNet 59.3%, CSC-LISTA 98.1% (original exports); FISTA-Net with its 3 µm settings 91.4% on simulation, 91.1% on
COMSOL. On the one COMSOL coil re-solved at tolerance 1e-8 (L1_C1, 17 November) the order reverses: MAGIC-UNet 26.2%,
FISTA 47.2% (one shape, one seed). Not like-for-like: only
MAGIC-UNet was adapted to the 10 µm scale (0.7 µm pixels); CSC-LISTA and FISTA-Net kept their 3 µm settings, so their
10 µm scores show that those configurations do not transfer, not that the architectures cannot work (README_HIGH_STANDOFF.md §7.3).

| | Run A, `run10a` (control) | Run B, `run10b` |
|---|---|---|
| Standoff | N(10, 2) µm, cut at 4–16 µm, never given | same |
| Pixels, tile | 0.2 µm, 64×64 = 12.8 µm | 0.7 µm (7×7 average), 64×64 = 44.8 µm |
| Crops | 4,800 layouts × 4 standoffs × 8 = 153,600; validation 128 layouts, 4,096 | same layouts and standoffs |
| Data | `runs/magic_unet/data10a` | `runs/magic_unet/data10b` |

Everything else is run 2's recipe: same network, paper training recipe, free-space simulator, per-crop noise, seed 0.
Run B waits for `runs/magic_unet/run10a.done` before training, so the two never share the GPU. Frozen code:
`experiments/magic_unet_10um` (17/17 tests; network, simulator and data code byte-identical to run 3's copy). Logs:
`logs/magic_unet_run10{a,b}.launch.log`, `runs/magic_unet/run10{a,b}.log`.

Why these two (measured 14 November, no training): the share of a tile's field that comes from currents outside the
tile is 50% for run 2 (3 µm, 12.8 µm tile), 78% at run A's setting and 52% at run B's. A perfect current costs 2.2%
at 0.2 µm pixels and 4.6% at 0.7 µm. Run 3 on simulation still scores 36.1% at 8 µm with 12.8 µm tiles and breaks at
10 µm (56.8%), the edge of its training range, so the band is centred on 10 µm.

Code changes for these runs: `train_magic_unet.py` gained `--sim-z-input` (score the simulated validation layouts at
another input height), `--data3-heights` and `--save-z-input`; `wiener_oracle.discover` takes the data3 heights;
`hpc/run_magic_unet.sh` puts `QDM_DATA_ARGS`/`QDM_TRAIN_ARGS` last so they override defaults, and can queue behind
another run (`QDM_WAIT_FOR`). Launch variables:
`QDM_DATA=data10a QDM_RUN=run10a QDM_DATA_ARGS="--bin 2 --standoffs 4 --crops-per-standoff 8 --z-mean 10 --z-std 2"
QDM_TRAIN_ARGS="--z-mean 10 --z-std 2 --sim-z-input 10 --data3-heights 3 10 --save-z-input 10"`; run B uses
`--bin 7`, `data10b`, `run10b` and `QDM_WAIT_FOR=$HOME/learning_ker/runs/magic_unet/run10a.done`.

Scoring scripts for 10 µm (prepared 14 November; rules in amendment A1 of the protocol):

- `scripts/m1_rescore.py`: `--height-max 20` (read-out grid), `--data3-heights 10`, `--comsol-z-inputs 10`,
  `--save-z-input 10`; defaults reproduce M1.
- `scripts/m2_fista_depth.py`: `--z-input 10 --z-delta 9.5`, `--skip-gate`, `--ladder` (λ extension), `--pad-test`
  (padding chosen on simulation), `--curve-every 25` (validation error every 25 iterations; the reference depth is
  the minimum of that curve, not one of three fixed depths), `--sim-layouts` (full simulated images at the reference
  depth); defaults reproduce M2d.
- `QDM_RECHECK_3UM=1` adds the 3 → 0.5 µm FISTA again with the same 25-step curve, to check that 1,000 iterations
  was the right baseline depth there (amendment A3).
- `hpc/run_high_standoff_scoring.sh` queues them: padding test, wait for both runs, rescoring, FISTA. Outputs go to
  `runs/high_standoff/{fista_pad_test,m1,fista}`.

## Real QDM data (17 November 2026)

Full write-up: `final_evaluation/02_source_docs/p10_experimental/README_EXPERIMENTAL.md`. Six measured B_NV maps of the
four-coil chip (1 µm pixels), converted to Bx, By, Bz with an NV axis fitted from the designed coil; runs 2, 3 and
10b with the correction at their own height read-out, nothing chosen on these data.

- **Run 10b (0.7 µm pixels, 44.8 µm tile) recovers a clean spiral current in all six cases**, including two coils at
  once and coils at 9–12 µm; current correlation with the fitted design 0.63 / 0.60 / 0.48 / 0.39 / 0.50 (cases 1–5).
- **Runs 2 and 3 (0.2 µm pixels) turn the measurement noise into spurious currents**: the noise is correlated over the
  1 µm camera pixels, which at 0.2 µm looks like wire-scale structure. Their height read-out is far too low (1.2–2.0 µm
  at a fitted 3–4 µm).
- **Height read-out stays inside each run's training range**: run 10b reads 3.7–4.9 µm at fitted 3.0–4.3 µm, and
  7–8 µm at fitted 8.6–11.7 µm.
- On the 0.5 µm field against the fitted design, the corrected network is not better than Tikhonov (case 1 in band:
  70% against 60%).
- Control: on the clean field of the fitted coil, run 2 reads 3.19 µm (fit 3.37), 14% field error, current correlation
  0.985; the pipeline is right and the gap is the data.
