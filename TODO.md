# To do later

Started 17 November 2026. Everything that has been proposed, promised or left open in this project and is not done
yet, from the documents and from every chat in this folder. Each item says why it matters and where it came from.
Tick items off here when they are done, and move finished results into the usual documents.

Sources: `FABLE_REVIEW.md` (the review, milestones M1–M9), `FINAL_EVALUATION_PLAN.md`, `README_HIGH_STANDOFF.md`
(in `final_evaluation/02_source_docs/p9_high_standoff/`), `README_COMSOL_MODEL.md`, `final_evaluation/README.md`, and the
chats "Final evaluation preparation", "fable milestoes", "syn fix" and this one.

---

## 0. Before the evaluation on 20 November

- [x] **FISTA-Net at 10 µm.** Finished 17 November: best checkpoint step 500 (validation 129%), 91.4% on simulation,
  91.1% on COMSOL. Added to the like-for-like table, every 10 µm table and figure, and booklet chapter 28. *(This chat)*
- [ ] **Presentation:** you are building it (30 minutes with questions, .pptx in the style of the 4 November deck).
  The material is in `final_evaluation/01_journey/JOURNEY.md`, section "For the presentation".
- [ ] **Report:** `final_evaluation/06_report/` is still empty. Needed: the IITB template and the page limit.
  *(Final evaluation plan)*
- [ ] **SSIM, noise sweep and timing (asked for in the 18 November meeting).** *MAGIC-UNet sweep, re-solved SSIM and MAGIC-UNet timing done 18 November 14:30 (README_NOISE_SSIM.md, SCOREBOARD, JOURNEY phase 11); FISTA at 3 µm done 16:33 (MAGIC-UNet better up to 2 % noise; FISTA with λ re-tuned wins from 5 %); FISTA at 10 µm still running.* Launched 18 November 13:35 server
  time, inference only (`hpc/run_noise_sweep.sh`, outputs `runs/noise_sweep/`): MAGIC-UNet run B and run 2 on 20
  simulated layouts at 0–100 % noise (J SSIM in the paper's σ_n convention, B L2/SSIM/RMSE); every method on the
  re-solved L1_C1 with SSIM; FISTA on 5 layouts at 0–50 % (λ × 1, 3, 10 above 2.5 %). COMSOL is reported only on the
  re-solved coil from now on. Write-up: `final_evaluation/02_source_docs/p11_noise_ssim/README_NOISE_SSIM.md`.
  *(18 November meeting)*
- [x] **Timing table.** Measured 18 November inside the noise sweep: MAGIC-UNet 0.67 s (10 µm) / 0.79 s (3 µm) per image with height read-out and correction, FISTA-3000 about 290 s, FISTA-925 91 s. *(Final evaluation plan)*
- [x] **Figures for phase 7 at 3 µm, one colour scale:** `final_evaluation/04_figures/p7_m1_m2/comsol3_*_maps.png`
  (17 November; `scripts/export_3um_figures.py`). A seed bar chart is still missing. *(final_evaluation/README.md)*
- [x] **Check the training standoff of Reed et al.** Done 18 November from the published PDF: 50 ± 10 µm per image, never
  given to the network (Sec. III, Table V, Sec. V C). The published version has no "Supplement D". Not new for us.
  *(Review §1.4, final_evaluation/README.md)*
- [x] **Experimental data and the new COMSOL export.** Done by the "syn fix" chat on 17 November: real-data results in
  `final_evaluation/02_source_docs/p10_experimental/README_EXPERIMENTAL.md`, the COMSOL re-solve in `README_COMSOL_MODEL.md` §9,
  both added to JOURNEY, the scoreboard and the booklet. *(syn fix)*
- [x] **FISTA-Net on the deep real-data maps (cases 3 and 6).** Rerun 17 November with the final checkpoint, which is
  still step 500, so the real-data rows were already final (the rerun reproduced them). *(syn fix, 17 November)*
- [x] **Score the methods on the re-solved COMSOL L1_C1** (amendment A7, finished 17 November). 10 µm, original →
  re-solved: MAGIC-UNet 60.4 → 26.2%, FISTA 49.8 → 47.2%, FISTA-Net 90.6 → 90.6%, CSC-LISTA 99.6 → 98.0%, Tikhonov
  98.7 → 98.1% (the last three with 3 µm settings). 3 µm: MAGIC-UNet 23.9 → 22.0%, FISTA-925 26.7 → 25.8%. On this one
  coil the 10 µm order reverses. FISTA-Net's reproduction gate failed by 0.03–0.04 point (reported with the flag).
  README_HIGH_STANDOFF §8, JOURNEY, SCOREBOARD, booklet §28.8. *(This chat)*
- [ ] **Send the group a follow-up** with the full re-solved table (draft in `reports/group_update_resolve/followup.md`).

- [ ] **Noise-aware training and correction.** The networks were trained only at σ_n ≈ 0.015–0.04 and fall faster than
  the paper's model at high noise (J SSIM 0.61 at σ_n = 1 at 10 µm, 0.02 at 3 µm; paper 0.84). Train with a wider noise
  range, and choose the correction's μ at the measured noise (at 3 µm with 25 % noise the fixed μ amplifies noise:
  355 % against 124 % raw). *(Noise sweep, 18 November)*
- [ ] **COMSOL only on re-solved data from now on**: needs the rest of COMSOL re-solved (see §4) before any COMSOL
  statement covers more than L1_C1. *(18 November)*

## 1. A fair comparison at 10 µm

At 10 µm only MAGIC-UNet was adapted to the scale (0.7 µm pixels); CSC-LISTA and FISTA-Net kept their 3 µm settings.
Their 10 µm scores show that the 3 µm configurations do not transfer, not that the architectures cannot work.
Details: `README_HIGH_STANDOFF.md` §7.3. *(This chat, 17 November)*

- [ ] **Scale-matched FISTA-Net at 10 µm.** 0.7 µm pixels (its CNN then sees about 15 µm instead of 2.1 µm), the
  smoothing constant of its starting estimate and the crop size re-chosen for the new pixel size, more stages if memory
  allows. The cleaner test of the two, because it has no dictionary. Label it as our variant.
- [ ] **Scale-matched CSC-LISTA at 10 µm.** 0.7 µm pixels (155 px filters then span 108 µm), a dictionary learned again
  at that pixel size, the starting penalty taken from FISTA's 10 µm choice (scale 0.0003), more layers.
- [ ] **Same protocol as the MAGIC-UNet runs.** Write the settings down before launch, score on the same 20 simulated
  layouts and the 15 COMSOL pairs, and keep the unadapted results next to the adapted ones.
- [ ] **FISTA on all 50 simulated layouts at 10 µm,** not only the first 20, so every method is compared on the same set.
- [ ] **A data-consistency step on top of FISTA,** at 3 and 10 µm. Never run; it would show whether the step helps any
  method or only MAGIC-UNet. *(M1/M2 results)*

## 2. The method the review recommends (method A) and its baselines

*(Review §4.1, booklet chapter 29)*

- [ ] **Method A, first version (M5).** Stream-function output, so the current conserves charge by construction; input
  resampled so the standoff is a fixed number of pixels; a MAGIC-UNet-style prior inside a few unrolled data-consistency
  stages; an outer search over height that **reconstructs at each candidate height** and reports an interval; tiles at
  least 8 standoffs wide; three seeds.
- [ ] **Unrolled version and ablations (M6).** 4–8 stages; switch off one ingredient at a time (data consistency,
  current against field output, vector against scalar input, known against estimated height, tile size, noise type).
- [ ] **Baselines a reviewer will ask for.** Fourier inversion with a Hann or Tikhonov filter; total variation or
  L1-curl on the stream function, at a grid of heights; an untrained network with the exact physics as its loss
  (Dubois et al.); MAGIC-UNet exactly as published; FISTA; a known-height version of method A.
- [ ] **Scale normalisation on its own.** One network trained at a normalised standoff, applied to every standoff by
  resampling. Run B's 0.7 µm pixels are a rough version of this; test it properly across 3–14 µm.
- [ ] **Classical identifiability test (M4).** Total variation on the stream function at a grid of heights, on simulated
  data with known noise and on COMSOL. Expect a one-sided valley with a resolution near 1/k_c. If there is no edge,
  the height must be calibrated externally, and the paper becomes "reconstruction given a calibrated standoff, with its
  sensitivity".
- [ ] **Report the current as the primary output,** with current-domain metrics (edge position, width, integrated
  current) next to the near-field error. *(Review §1.1)*
- [ ] **Report which frequencies came from the data and which from the prior,** per reconstruction. *(Review §3.2)*
- [ ] **Diffusion prior on the stream function (candidate B).** Only after method A, or for a machine-learning venue.

## 3. Fix and widen the simulation pipeline

*(Review §4.1 part 5, M7; README_COMSOL_MODEL.md; this chat)*

- [ ] **Wider layouts.** Curved and diagonal wires, pads, widths over a decade, two current levels, two layers.
  Today every layout is one of three Manhattan families with widths 1.6–3.2 µm and one current of 600 µA, so a network
  may simply recall the simulator's shapes (the review's risk 2).
- [ ] **Finite conductor thickness, NV-layer thickness and optical blur.** All three act like extra standoff and bias
  the height read-out.
- [ ] **Sensor noise like a real QDM** (photon shot noise), white and correlated, at several levels. So far the only
  noise has been synthetic speckle and COMSOL mesh noise.
- [ ] **Exact coil geometry from the DXF** as an option, and generous padding at large standoffs: the FFT canvas wraps
  the far field around, which already costs 2% in Bz at 20 µm.
- [ ] **Keep free-space physics as the default** (decided 14 November). `decay_k0` stays a test switch only.
- [ ] **Held-out simulated families** that no model was trained on, as a generalisation test.

## 4. COMSOL test data that obeys physics

*(Review M3, README_COMSOL_MODEL.md "Options")*

- [x] **Re-solve of L1_C1 at a tight tolerance** (Ritaban, 17 November, relative tolerance 1e-8; the direct solver
  fails on this model). k₀ disappears; free space matches COMSOL to 3–5% at 10 µm (`README_COMSOL_MODEL.md` §9).
- [x] **Ampère's law in the re-solve** (17 November): a single-plane test (`ampere_single_plane.py`) gives
  B_z / [i k·B_t / k] = 1.00 in every band at 3 and 10 µm, where the original export followed the screened law. The air
  current was numerical (`README_COMSOL_MODEL.md` §10).
- [ ] **Re-solve the rest at the same tolerance:** the other L1_C1 heights, data3 (C1+C2, C1+C2+C3, C1–C4) and the IC
  model. Every COMSOL file except L1_C1 at 0.5, 3 and 10 µm still carries the artefact, and the 3 µm benchmark inputs
  were off by 10–23%. Then rescore every method: only that makes the 10 µm reversal on L1_C1 (MAGIC-UNet 26.2%, FISTA 47.2%) more than one coil.
- [ ] **A calibration case** with an exact answer (one straight wire or one loop) in the same box and mesh.
- [ ] **The IC coil's model report** (`interconnects_F.mph`), which we never received.
- [ ] **At least ten new geometries** the simulator has never drawn, several widths, five or more heights. Today there
  are only two wire shapes in COMSOL, and the four C1 configurations share one solve.
- [ ] **Re-score MAGIC-UNet and FISTA at 10 µm on a corrected solve.** It would settle whether MAGIC-UNet's COMSOL loss is
  COMSOL's physics.

## 5. Real QDM data

*(Review M8)*

- [x] **First analysis of the six experimental maps** ("syn fix", 17 November): every method compared against the fitted
  design; no learned method beats Tikhonov or FISTA on the near field; MAGIC-UNet run 10b recovers the wire layout.
  `final_evaluation/02_source_docs/p10_experimental/README_EXPERIMENTAL.md`.
- [ ] **Ask Ritaban** for the NV axis in the image frame, the standoff and coil depths (are cases 3 and 6 a buried
  layer?), the current per case, the pixel size, and what differs between cases 1 and 4. Then re-run the real-data
  scores with those instead of the fitted values.
- [ ] **Train on real-QDM-like inputs:** noise correlated over 1 µm pixels, NV-projected inputs, background offsets and
  gradients, a camera seam. On the real maps, the 0.2 µm-pixel networks turned correlated noise into false currents.
- [ ] **Score the current against the fabrication mask** (wire positions, widths, integrated current) instead of a
  fitted drawing whose leads do not match the device.
- [ ] **Check case 6's fitted orientation** (MAGIC-UNet draws a spiral offset from the fitted outline).
- [ ] **A measurement with an independently calibrated standoff.** Image a wire of known width and current, fit its
  standoff from the line profile, then image a second structure in the same session. Compare estimated against
  calibrated standoff, the current map against the mask, and the integrated current against the driven current.
- [ ] **The Bathla et al. experimental dataset,** if the group can share it.

## 6. Open questions in the existing results (cheap checks)

- [ ] **MAGIC-UNet's unexplained 18 points on COMSOL at 10 µm.** Run FISTA at k₀ = 0.15 too (MAGIC-UNet's numbers
  there match COMSOL closely), and include COMSOL's box walls in the check. *(README_HIGH_STANDOFF.md §8)*
- [ ] **Tile-boundary artefact** on run B's worst simulated layout (a lead wire drawn too strong, ending at a straight
  edge). Not investigated. *(§7.1)*
- [ ] **Run B's earlier epochs.** Its validation loss stops improving after about epoch 20 while training loss keeps
  falling; the last-epoch model was used, as in the paper. Score epoch 20 as a diagnostic. *(§4)*
- [ ] **Seeds for run B** (and run A), as for run 2 at 3 µm. One seed each so far.
- [ ] **CSC-LISTA "worse with 8,000 layouts"**: three seeds of the stored and streamed runs, 300 steps each. The claim
  "data volume is not the limit" is not supported until then. *(Review §2.3)*
- [ ] **Does simulated validation predict COMSOL?** Score every saved model on 50 full simulated images, on v3 and on v2
  (which has k₀), and compare rank correlations with COMSOL. *(Review §2.3)*
- [ ] **Wiener oracle with data3** on the server; marked pending in `README_WIENER_ORACLE.md`. *(Final evaluation plan)*
- [ ] **The geometry-capacity results** exist on the server but are not in any document. *(Final evaluation plan)*

## 7. Statistics and protocol

*(Review §4.3–4.4)*

- [ ] **A frozen test set no method has seen,** built from the new COMSOL geometries, with the final protocol written
  first. The current 15 pairs have been used for more than 22 variants.
- [ ] **Three or more seeds** for every learned method in the final comparison, and bootstrap intervals over geometries.
- [ ] **Score against a noise-free or low-passed target** as standard, next to the raw score.

## 8. Uncertainty

*(Review §3.4, M9)*

- [ ] **Per-pixel spread over seeds** (ensemble).
- [ ] **A height interval:** misfit against hypothesised height, with its coverage checked on simulated data.
- [ ] **Conformal intervals** calibrated on simulation, with coverage measured on COMSOL and real data.

## 9. Documents and housekeeping

- [ ] `README_SIMULATED_FIELDS.md` intro: check whether "no model has been trained on it" is still there. *(Plan)*
- [ ] `PROJECT_CONTEXT.md`: one section still says IC_2um and IC_4um are different geometries; they are the same coil.
- [x] `RESEARCH_BEYOND_ISTA.md` §4: "fixed ~50 µm standoff" for MAGIC-UNet corrected to 50 ± 10 µm (18 November).
- [ ] `FABLE_REVIEW.md`: "geometry capacity: no result on record", once those results are written up.
- [ ] Server clean-up: about 68 MB in `runs/high_standoff/export/`, and the crop sets of the 10 µm runs (12 GB) once
  nothing needs them. Ask before deleting.
- [ ] Publication target, if the project continues: the review suggests Physical Review Applied or IEEE Transactions
  on Computational Imaging, with a real measurement; ICCP 2027 as the conference option.
