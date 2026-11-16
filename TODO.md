# To do later

Started 17 November 2026. Everything that has been proposed, promised or left open in this project and is not done
yet, from the documents and from every chat in this folder. Each item says why it matters and where it came from.
Tick items off here when they are done, and move finished results into the usual documents.

Sources: `FABLE_REVIEW.md` (the review, milestones M1–M9), `FINAL_EVALUATION_PLAN.md`, `README_HIGH_STANDOFF.md`
(in `final_evaluation/02_source_docs/p9_high_standoff/`), `README_COMSOL_MODEL.md`, `final_evaluation/README.md`, and the
chats "Final evaluation preparation", "fable milestoes", "syn fix" and this one.

---

## 0. Before the evaluation on 20 November

- [ ] **FISTA-Net at 10 µm.** Training finishes about 16:30 on 17 November. Score it with
  `scripts/score_high_standoff_methods.py fista_net`, then add it to the like-for-like table and every 10 µm table,
  figure and booklet chapter 28. *(This chat)*
- [ ] **Report and presentation.** `final_evaluation/06_report/` and `07_deck/` are still empty. Needed: the IITB
  template, the page limit and the talk length. *(Final evaluation plan)*
- [ ] **Timing table.** Inference time per method on the same GPU: FISTA at its best depth against one MAGIC-UNet pass
  plus the correction. Evaluators will ask; it was never measured for the final methods. *(Final evaluation plan)*
- [ ] **Figures for phase 7 at 3 µm.** Corrected MAGIC-UNet next to FISTA and the truth on one colour scale, and a seed
  bar chart. The predictions exist on the server. *(final_evaluation/README.md)*
- [ ] **Check Supplement D of Reed et al.** Did they train over 50 ± 10 µm, or at a fixed ~50 µm? It decides how much of
  "works without being told the standoff" is new. `RESEARCH_BEYOND_ISTA.md` §4 still says "fixed".
  *(Review §1.4, final_evaluation/README.md)*
- [ ] **Experimental data and the new COMSOL export.** Being analysed in the "syn fix" chat, which was asked to put its
  results into every document. Check that it did before the evaluation. *(syn fix)*

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

- [ ] **Ask Hari for one re-solve** of L1_C1 with a direct solver (or a much tighter tolerance), exported at 0.5, 3 and
  10 µm, or for curl B in the air domain. If k₀ disappears, COMSOL above 4 µm becomes a valid test.
- [ ] **No-re-solve check:** integrate B·dl around a closed square in a vertical plane in the air of the existing
  solution. Physics gives zero.
- [ ] **A calibration case** with an exact answer (one straight wire or one loop) in the same box and mesh.
- [ ] **The IC coil's model report** (`interconnects_F.mph`), which we never received.
- [ ] **At least ten new geometries** the simulator has never drawn, several widths, five or more heights. Today there
  are only two wire shapes in COMSOL, and the four C1 configurations share one solve.
- [ ] **Re-score MAGIC-UNet and FISTA at 10 µm on a corrected solve.** It would settle whether MAGIC-UNet's COMSOL loss is
  COMSOL's physics.

## 5. Real QDM data

*(Review M8; the "syn fix" chat is working on the experimental maps now)*

- [ ] **Finish the experimental-data analysis** started in "syn fix" (six B-field maps in `Experimental data/`,
  `scripts/experimental/`): fitted standoff, MAGIC-UNet's height read-out, agreement with a fitted-coil reference.
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
- [ ] `RESEARCH_BEYOND_ISTA.md` §4: "fixed ~50 µm standoff" for MAGIC-UNet, after Supplement D is checked.
- [ ] `FABLE_REVIEW.md`: "geometry capacity: no result on record", once those results are written up.
- [ ] Server clean-up: about 68 MB in `runs/high_standoff/export/`, and the crop sets of the 10 µm runs (12 GB) once
  nothing needs them. Ask before deleting.
- [ ] Publication target, if the project continues: the review suggests Physical Review Applied or IEEE Transactions
  on Computational Imaging, with a real measurement; ICCP 2027 as the conference option.
