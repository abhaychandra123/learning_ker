# Final evaluation plan (20 November 2026)

Written 12 November 2026, from every document in this repository plus a read-only pass over the server
(`abhaychandra@10.129.6.189`, `~/learning_ker`). Nothing was trained and nothing was committed. Today's new results
are marked **[new]**.

Eight days are left. The final report and the final deck have not been started.

**Status 17 November:** the COMSOL anomaly is explained (solver tolerance, §5.3), the 10 µm phase is done, and the
real QDM data have been analysed with every method (§5.4). The live checklist is `TODO.md` §0; the story is
`final_evaluation/01_journey/JOURNEY.md`.

---

## 1. The problem in one paragraph

A quantum diamond microscope (QDM) measures the magnetic field (Bx, By, Bz) of a micro-coil on a plane some
micrometres above it. Height blurs the image: every spatial frequency k is damped by `exp(-k·z)`, so fine wire
detail is lost exponentially fast. The task is to undo that blur: reconstruct the field close to the coil (0.5 µm)
from a measurement made farther away (3 µm on the main benchmark, up to 10 µm). Dividing by `exp(-k·Δz)` blows up noise,
so every method is really a choice of prior. COMSOL simulations of real coil designs serve as ground truth. The full
research question (`booklet/chapters/17_question.tex`) adds that the standoff is not known exactly in a real
experiment, and that a source at depth h and a sharper source at depth h + δ give the same field.

---

## 2. Storyline, phase by phase

Each phase ends with the number it delivered and the reason we moved on. These double as presentation sections.

| # | Dates | Phase | What we did | Result | Why we moved on |
|---|---|---|---|---|---|
| 0 | 2–10 Sep | Forward physics | Learned the blur kernel from COMSOL L1_C1 pairs; compared with the analytic `exp(-k·Δz)` | Joint learned kernel 0.87–0.98% error vs analytic 0.72% (`res.md`) | The physics is known; the hard part is the inverse |
| 1 | 23–30 Sep | Classical and unrolled inverse | ISTA, FISTA and in-house ConvLISTA (163 parameters), 3 → 0.5 µm on L1_C1, Kaggle T4s; ε-stopping; IC_4um crop-kernel test | LISTA-16 42–48% vs ISTA-500 34–36% relative L2. Trained and scored on the same pair | No generalisation test; operator fitted on the test pair |
| 2 | 7–15 Oct | Foundations | Morozov λ selection; research guide (8 Oct) defines the 4–6 month question; dense ALISTA reproduced on synthetic sparse codes | ALISTA −28.0 dB, −54.7 dB with support selection (`ALISTA.md`) | Synthetic sparse codes are not QDM data |
| 3 | 22–26 Oct | Scale up on the professor's server | One shared ConvLISTA over all 17 COMSOL pairs; synthetic augmentation (51 → 1,080 components); failure diagnosis; capacity test (tied/untied, 16/32 layers) | 65.5% mean; augmentation 65.1 → 64.7%; capacity 67.6 → 59.6% **[new: results found on server]** | Same geometries in train and test; high standoffs fail even on clean input |
| 4 | 31 Oct–6 Nov | Honest restart at one standoff | Professor: "ConvLISTA has no paper behind it." New current-layout simulator (`simulate_fields.py`), COMSOL held out as test only, published methods only: CSC-LISTA (ALISTA Eq. 20), FISTA-Net, FISTA with a learned dictionary, Tikhonov | CSC-LISTA-64 33.8%, FISTA-Net 33.3%, FISTA-300 30.5%, Tikhonov 49.3% on 15 pairs | The best learned field-domain model still lost to FISTA |
| 5 | 5–7 Nov | Understanding the limits | Wiener oracle, information budget, the exact height ambiguity, literature survey | Wiener oracle 34.4%; learned field-domain methods sit next to it | Field-domain priors behave almost linearly; a stronger prior is needed |
| 6 | 7–9 Nov | Current-domain prior | Exact port of MAGIC-UNet (Reed et al. 2024): field → sheet current → Biot–Savart; standoff not given | Run 2 (standoff 3 ± 0.6 µm) 27.0%; run 3 (1–10 µm) 34.8%, fails on IC_4um (52%) | First method below FISTA-300, but untested against a fully converged baseline |
| 7 | 10–11 Nov | Review and fixed protocol | Critical review (`FABLE_REVIEW.md`); M1/M2 under a protocol hashed before scoring (`M1_M2_PROTOCOL.md`) | Noise edge 2.9 rad/µm: 15.7% of the target norm is mesh noise no method can score. FISTA-1000 26.7% is the real baseline. MAGIC-UNet + one data-consistency step 24.3%. Three seeds **[new]** | Claims are now defensible; next is high standoff |
| 8 | 12–20 Nov | High standoff and write-up | COMSOL model details requested from Hari (`COMSOL_REQUEST.md`); high-standoff runs; report and deck | Pending | — |

**The one-sentence story.** Fixed-standoff field deblurring with unrolled networks plateaued at the level of the best
linear filter. Moving the prior into the current domain and adding one physics data-consistency step beats a fully
tuned classical solver (24.3% vs 26.7%), and the same network reads the standoff from the image on simulated data. It
fails on one of the two real coil shapes, and the test data themselves carry a floor no method can cross.

**Honest lessons worth a slide.** Train = test in phases 1 and 3; FISTA-300 was an early-stopped baseline that made
learned models look better; 1–3 point rankings on one seed were noise; only two independent wire shapes exist in
COMSOL; COMSOL decays faster than free space above about 4 µm, and Hari's files (13 Nov) show this is a numerical
property of the solve (COMSOL fails Ampère's law in the air; `README_COMSOL_MODEL.md`), not real physics.

---

## 3. Final scoreboard, 3 → 0.5 µm, 15 COMSOL pairs

All COMSOL-unseen during training and selection. Relative L2 against the raw COMSOL target. "Low-passed" removes the
mesh-noise band above 2.9 rad/µm from the target. Sources: `M1_M2_RESULTS.md` §M2b and §M2e.

| Method | Standoff given? | Raw | C1 family | IC_4um | Low-passed |
|---|---|---:|---:|---:|---:|
| MAGIC-UNet run 2 + data consistency at estimated height, 3 seeds | No | **24.3 / 24.5 / 24.5%** | 23.8 / 23.7 / 22.9% | 26.4 / 27.5 / 30.9% | 18.5% |
| FISTA-1000, learned dictionary, λ 0.009 | Yes | 26.7% | 26.6% | 27.2% | 21.9% |
| MAGIC-UNet run 2, raw, 3 seeds | No | 27.0 / 27.2 / 26.7% | 26.3 / 26.5 / 25.2% | 29.6 / 29.8 / 32.6% | 22.0% |
| MAGIC-UNet run 3 + data consistency (trained 1–10 µm) | No | 28.9% | 24.4% | 46.9% | 23.9% |
| FISTA-300 (historical baseline) | Yes | 30.5% | 31.2% | 27.6% | 26.4% |
| FISTA-Net run 2 | Yes | 33.3% | 33.0% | 34.6% | 29.5% |
| CSC-LISTA strict, 64 layers | Yes | 33.8% | 34.5% | 30.7% | 30.2% |
| MAGIC-UNet run 3, raw | No | 34.8% | 30.4% | 52.0% | 30.9% |
| Tikhonov, μ from simulation | Yes | 49.3% | 49.1% | 50.2% | 47.4% |

**Verdict under rule A4 [new]:** corrected MAGIC-UNet beats FISTA-1000 on all 15 pairs (margin 2.2–2.4 points, seed
range 0.21) and on the C1 family, for every seed. On IC_4um the two are **not distinguishable** (one seed of three
below). Raw MAGIC-UNet ties FISTA-1000. Say exactly this; do not say "beats on both shapes".

**Height read-out (run 3, simulation):** within 15% of the true standoff for 93% of images at 1.5–8 µm. On COMSOL,
C1 shapes read 2.97–3.10 µm at a true 3 µm; IC_4um reads 3.86 µm.

---

## 4. What lives where (union of Mac and server)

| Item | Mac | Server |
|---|---|---|
| Documents (READMEs, reviews, protocol, booklet) | **Current**, source of truth | Copies from October, stale |
| Raw COMSOL CSVs: L1_C1 (11 heights), data2 (IC_2um, IC_4um), data3 (3 configurations × 20 heights, 0.5–12.5 µm) | Deleted (`data/` gone, `cache/*.npy` deleted) | `data/`, 18 GB |
| Checkpoints and predictions: MAGIC-UNet runs 1–3, seeds, M1/M2, CSC-LISTA, FISTA-Net, capacity | Small copies only (CSV, JSON, a few npz) | Full |
| MAGIC-UNet training crops (`runs/magic_unet/data`, `data_any`) | No | 6.1 GB each |
| Old runs only on the Mac: `alista_validation`, `wiener_standoff_local`, reconstruction reports, synthetic diagnostics | Yes | No |

Any figure that needs raw fields (COMSOL maps, error maps) must now be made on the server, or the needed arrays copied
back. Booklet figure data is cached in `booklet/data/*.npz`.

**Server disk [new]:** `/home` is at 98%, 13 GB free. A new MAGIC-UNet crop set needs about 6.5 GB. Candidates to delete
(your choice): `runs/magic_unet/data` (run 2's crops, regenerable from seed 0, 6.1 GB), `runs/combined_full_200`
(old ConvLISTA, 3.0 GB), the prediction `.npz` files in `runs/m1_m2` (up to 3.1 GB). Keep `data/`.

---

## 5. The high-standoff plan (your main work for the next days)

> **Update, 17 November.** The 10 µm results are in
> `final_evaluation/02_source_docs/p9_high_standoff/README_HIGH_STANDOFF.md`. When presenting them, the comparison with
> CSC-LISTA and FISTA-Net is **not like-for-like**: only MAGIC-UNet was adapted to the 10 µm scale (0.7 µm pixels, a
> 44.8 µm view); the other two kept their 3 µm settings (view, depth, penalty, data), so their 10 µm failure says their
> 3 µm configurations do not transfer, not that the architectures cannot work. Without scale adaptation no learned
> method beats FISTA on average (§7.3 of that README).

### 5.1 What the evidence already says about 10 µm

- **Most of the 0.5 µm field is not in a 10 µm measurement.** Information-budget floor: 56% (L1_C1) and 73.5% (IC_4um)
  of the target norm lies above what a 10 µm measurement can see (`RESEARCH_BEYOND_ISTA.md`). Raw scores will be high
  for every method; report low-passed and in-band scores, and the height read-out, alongside raw.
- **COMSOL disagrees with free space at 10 µm by 64–140%** (`README_SIMULATED_FIELDS.md`). Until Hari's answers
  explain this, a 10 µm COMSOL score partly measures agreement with an unexplained artefact.
- **Existing models at 10 → 0.5 µm:** run 2 raw 92–95%; run 3 corrected at h* 77–78%, at the true height 68–72%.
- **A full 15-pair 10 → 0.5 µm set already exists and has never been scored:** data3 has a 10 µm plane for all three
  configurations, plus L1_C1 and IC_4um. Scoring run 2, run 3, the seeds and FISTA on it is inference only.
- **The tile is too small for 10 µm.** Run 2 works with the standoff at 15 pixels (3 µm / 0.2 µm) and a 64-pixel tile
  about four standoffs wide. At 10 µm with 0.2 µm pixels the tile is barely one standoff wide, which is part of why run 3
  weakens at large standoff. A 10 µm model should keep the run 2 ratio: about 0.6–0.7 µm pixels with 64 × 64 tiles
  (about 40 µm), or 0.2 µm pixels with 192–256 pixel tiles. This changes the published recipe, so label it ours.

### 5.2 The GPU calendar does not fit "all architectures again"

One RTX A6000. Measured costs from the logs:

| Run | GPU time |
|---|---|
| MAGIC-UNet, 50 epochs | about 4 h alone (286 s per epoch); 8 h if two share the GPU |
| MAGIC-UNet crop generation, 2,400 layouts | not logged exactly; budget about 1 h |
| CSC-LISTA, 64 layers, 1,000 steps | about 3 h |
| FISTA-1000 with λ selection | about 5 h |
| FISTA-Net run 2 | about 27 h |

Everything serially is about 40 hours before data generation and evaluation. If Hari answers on the 16th or 17th,
about 72 hours remain, and writing has to happen in them too.

**Recommendation:** one learned model (MAGIC-UNet, a narrow standoff band centred on the high standoff, with data
consistency), plus FISTA-1000 with λ re-picked on simulation, as the baseline. CSC-LISTA-64 only if GPU time is left.
Drop FISTA-Net: the review already says to stop field-domain variants, and it alone would take 27 hours.

**Do not wait for Hari to start.** The GPU is idle today. A free-space version of the high-standoff run can start now
with the current simulator. When the COMSOL details arrive, the refined-simulator run becomes the comparison
("does matching COMSOL's model help?"), not the only result. Set a hard launch cutoff: anything not running by the
night of 17 November does not go into the evaluation.

### 5.3 Hari's answers arrived (13 Nov): what they settled

Full analysis: [README_COMSOL_MODEL.md](README_COMSOL_MODEL.md). In short:

- **Not the boundaries.** The model is a 140 × 140 × 90 µm box with B·n = 0 on every face, and the walls are visible
  in the data, but exact box propagation does not reduce the mismatch.
- **Not the conductor.** Zero-thickness sheet at z = 0, 2.00 µm wires, 600 µA; amplitude 1.00, best height exactly 3.00 µm.
- **The cause is a non-physical decay in COMSOL's solution:** rate √(k² + k₀²), k₀ ≈ 0.10 rad/µm, the same at every
  height, larger on the finer mesh. With it (plus exact geometry and walls) the simulator matches COMSOL to 10–12% at
  3–12.5 µm, including a held-out configuration; without it, 65 / 61 / 134% at 10 µm.
- **The 90° rotation** is COMSOL's cut-plane frame (cpl1x = −Y, cpl1y = X; B columns global).

Decision (14 November): keep free space. **Confirmed on 17 November** by Ritaban's re-solve of L1_C1 at solver
tolerance 1e-8 (`COMSOL_params_v2/`; the direct solver fails, the system is singular): k₀ is gone, free space matches
COMSOL to 3.8 / 4.5 / 3.1% from 0.5 to 10 µm, our exact-coil simulator matches it to 3.2 / 3.2 / 9.3% at 10 µm, and
Ampère's law holds in the air (single-plane ratio 1.00 in every band; the original export followed the screened law).
The extra decay and the air current were the default solver tolerance. Details: `README_COMSOL_MODEL.md` §9–10.

### 5.4 Real QDM data (16–17 Nov)

Six measured B_NV maps of the same chip (`Experimental data/`), analysed in
`final_evaluation/02_source_docs/p10_experimental/README_EXPERIMENTAL.md`. No ground truth; the reference is the
designed coil fitted to each map. MAGIC-UNet run 10b recovers the coil layout in all six (including two coils at once
and 9–12 µm); on the near field, Tikhonov (3–4 µm) and FISTA (9 µm) are closer than every learned method.

---

## 6. What is left, in order

### Needs your decision first

- [ ] **Report and deck format.** IITB template? Page limit? Talk length? Is the booklet (`booklet/main.tex`) the
      report's base, or a separate thesis-style report?
- [ ] **Server runs and deletions.** Approve: the free-space high-standoff MAGIC-UNet run now; FISTA-1000 at the same
      standoff; which folders to delete for disk space.

### Cheap, no training (can be done now)

- [ ] Score existing models (run 2, run 3, three seeds, FISTA-1000) on the unscored 10 → 0.5 µm data3 pairs, and at
      5 µm. Inference only, on the server. Needs a small change to the pair list in `m1_rescore.py`.
- [ ] **Timing table.** Inference time per method on the same GPU (FISTA-1000 vs one MAGIC-UNet pass + correction).
      Evaluators will ask about speed. Only the September ConvLISTA runs have timing (0.24–0.46 s vs 5–13 s for
      long ISTA/FISTA); the final methods have none.
- [ ] Wiener oracle rerun with data3 on the server: marked "pending" in `README_WIENER_ORACLE.md`, never run (no
      Wiener output on the server). Optional; the local two-configuration result is enough for the talk.

### Main experiment (needs the server)

- [ ] 12–13 Nov: free-space high-standoff MAGIC-UNet run (data ~1 h, training ~4 h, evaluation with correction).
- [ ] 13–14 Nov: FISTA-1000 at the same standoff, λ chosen on simulation.
- [ ] 16–17 Nov: if Hari's answers arrive, refined simulator → one more MAGIC-UNet run. Hard cutoff: launched by the
      night of the 17th.
- [ ] 18 Nov: score everything four ways (`scripts/m2_fourway.py`), with seeds where they exist.

### Writing (start now with current results; do not wait for the new runs)

- [ ] 13–16 Nov: report draft. Sections follow §2 above; results from §3; limits (noise edge, information floor,
      two shapes) stated plainly.
- [ ] 14–17 Nov: deck draft. The last deck is from 4 November, before MAGIC-UNet and M1/M2.
- [ ] 18 Nov: insert high-standoff results. 19 Nov: rehearse.

### Stale statements to fix in the booklet and docs

- [ ] `booklet/chapters/19_magic_results.tex` says run 3 is "still running"; the booklet has nothing on M1/M2 (data
      consistency, FISTA-1000, noise edge, seeds). `booklet/main.pdf` (8 Nov) predates the latest chapter edits.
- [ ] `README_SIMULATED_FIELDS.md` opens with "No model has been trained on it".
- [ ] `PROJECT_CONTEXT.md` (29 September section) still says IC_2um and IC_4um are different geometries; they are the
      same coil.
- [ ] `RESEARCH_BEYOND_ISTA.md` §4 says MAGIC-UNet used a fixed ~50 µm standoff; `README_MAGIC_UNET.md` says
      50 ± 10 µm, unknown to the network. Re-read Supplement D of Reed et al. before the talk: it decides how much of
      "works without the standoff" is new.
- [ ] `FABLE_REVIEW.md` lists the geometry-capacity experiment as "no result on record"; results exist on the server
      (`runs/geometry_capacity_v1`, `logs/geometry_capacity_v1.launch.log`).

### Questions to prepare for

- Why not just use FISTA? (Answer: corrected MAGIC-UNet beats FISTA-1000 by 2.2–2.4 points over three seeds, needs no
  standoff, and gives a height estimate; speed from the timing table.)
- Why does it fail on IC_4um? (Closely spaced antiparallel wires look like one deeper source; seed range 4.4 points
  on that shape.)
- Is the simulator tuned to the test coils? (Yes, its layout families and widths came from them; say so. That is why
  new COMSOL geometries are the first item of future work.)
- What is the floor? (15.7% of the target norm is mesh noise above 2.9 rad/µm; low-passed scores remove it.)
- What next? (Data-consistent current-domain solver with explicit height search, `FABLE_REVIEW.md` §4.1–4.2;
  real QDM data with a calibrated standoff.)
