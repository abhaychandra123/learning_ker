# Critical review: standoff reconstruction of QDM magnetic fields

10 November 2026. Reviewer: Claude (Fable 5). Scope: every document, script, test and saved run named in
`FABLE_PROMPT.md`, plus read-only NumPy checks on saved predictions (`scripts/review_checks.py`, about 2 minutes,
no training). Nothing was trained and nothing was committed.

How to read the evidence tags:

- **[file]** a number I read in a repository file or saved run.
- **[check]** a number I computed myself with `scripts/review_checks.py` from saved arrays. These use the two COMSOL
  configurations that exist on the Mac (L1_C1 and IC_4um at 3 → 0.5 µm). They are diagnostics on test data, not results.
- **[web]** verified by a web search in this session (title, authors or venue seen in the results).
- **[memory]** a standard reference I know but did not re-fetch.
- **[not verified]** stated in your documents or a third-party page, and I could not confirm it.
- **[M1/M2]** a number from the M1 and M2 runs on the server, under the fixed protocol in `M1_M2_PROTOCOL.md`. Full
  tables and steps in `M1_M2_RESULTS.md`.

---

## Update, 17 November 2026

The "unexplained physics anomaly" (finding of §2.2(b), risk 1) is explained: COMSOL's default solver tolerance. A
re-solve of L1_C1 at tolerance 1e-8 obeys free-space decay and Ampère's law (`README_COMSOL_MODEL.md` §9–10). The
real-QDM milestone (M8) has a first analysis on six measured maps
(`final_evaluation/02_source_docs/p10_experimental/README_EXPERIMENTAL.md`); no calibrated standoff yet.

## Update after M1 and M2 (10 November 2026)

Milestones M1 and M2 (§4.2) were run on the server under a protocol written and hashed before any new COMSOL score.
The review text below is kept as written; where M1/M2 changed a claim, a marked note says so next to it. In short:

1. **Finding 1, corrected.** The noise edge from a fixed rule on the targets alone is **2.9 rad/µm**, not the 2.2 I
   used. Above it every method's error equals the target's own content there (15.6–16.2% of the target norm, against
   a share of 15.7%) [M1/M2]. My low-passed scores at 2.2 rad/µm are superseded by the M2b table. My framing "18–20%
   of every score is mesh noise" was also wrong in kind: band errors add in quadrature, so the noise band lifts a raw
   score by about 4–6 points (24.3% → about 18.4% without it, 30.5% → about 26.2%). And the "70–74% of the target's
   energy" in finding 1 is a norm ratio (`review_checks.py` takes a square root); as an energy share it is 49–55% (0.70² to 0.74²).
2. **Finding 3, confirmed on 15 pairs, slightly smaller.** With μ chosen on simulation only, and the height read from
   the network's own current (standoff not given), run 2 goes from 27.0% to **24.3%** (C1 family 23.8%, IC_4um 26.4%).
   On the same two configurations I used, the honest figure is 25.1%, so my 24.2% (three μ values tried on test data)
   was about 0.9 points optimistic [M1/M2].
3. **The cheap rescue of run 3 failed.** One global height does not bring run 3 near run 2 on IC_4um: 52.0% →
   46.9% with the estimated height (3.86 µm at a true 3 µm), 37.3% with the true height [M1/M2]. A height read off a
   biased current inherits the bias. M5's height search must re-run the reconstruction per candidate height.
4. **FISTA-300 was early-stopped, and the real classical baseline is FISTA-1000.** Chosen by 64-crop validation
   before its COMSOL score was read, FISTA-1000 scores **26.7%** on COMSOL (C1 26.6%, IC 27.2%), against 30.5% at 300
   and 27.4% at 3,000 [M1/M2]. Raw MAGIC-UNet run 2 (27.0%) does not beat it; only run 2 with the correction does (24.3%;
   on IC_4um by 0.8 points).
5. **The "learned field-domain methods sit next to the Wiener oracle" explanation holds on average.** In the prior
   band (k_c to 2.9 rad/µm), the error norm of strict CSC-LISTA-64 and FISTA-Net run 2 is on average 82% and 79% of the
   target's norm in that band (lowest single component 65% and 71%), above the two-thirds threshold I set. (The
   threshold and these ratios are norm ratios, as in my original check.) FISTA-1000 is at 59%: run to its best depth,
   it recovers more of the prior band than the learned field-domain networks. MAGIC-UNet run 2 is at 45% [M1/M2].
6. **Every saved Group B model reproduced its COMSOL scores exactly** when re-run with its own frozen code [M1/M2].
7. **Seeds (M2e), finished 11 November.** Three seeds of run 2 with the correction score 24.3 / 24.5 / 24.5% on the 15
   pairs (range 0.2 points) and 23.8 / 23.7 / 22.9% on the C1 family: every seed is below FISTA-1000 (26.7%, 26.6%). On
   IC_4um they score 26.4 / 27.5 / 30.9% against 27.2%, so by the rule fixed beforehand (amendment A4) the IC result is
   "not distinguishable". Raw run 2 (27.0 / 27.2 / 26.7%) ties FISTA-1000 [M1/M2].

---

## Executive summary

**Verdict.** The project is asking the right question (unknown standoff, with an honest treatment of ambiguity) and
its last move (a current-domain network, MAGIC-UNet) was the right one. But most of the effort so far went into
ranking field-domain unrolled networks by 1–3 points on a benchmark that cannot support such rankings. The benchmark
has two wire shapes, one seed per method, a noisy target, an unexplained physics anomaly, and it has now been used to
compare more than twenty method variants. No paper can rest on it as it stands.

**The five findings that matter most.**

1. **About 18–20% of every score is mesh noise in the COMSOL target.** Above 2.2 rad/µm, 70–74% of the target's
   energy violates Maxwell's equations between components, so it is noise, not field [check]. That band holds
   15–23% of the target norm. Against a target low-passed at 2.2 rad/µm, MAGIC-UNet run 2 scores 20.2% on L1_C1 and
   22.2% on IC_4um, and FISTA-300 scores 26.6% and 20.2% [check]. The "lost-band floor" of 34.5% mixes this noise with
   real lost signal, so it overstates what the data cannot see.
   *Update [M1/M2]: by a fixed rule on the targets alone the edge is 2.9 rad/µm, and the noise band is 15.7% of the
   target norm on the 15 pairs. It lifts raw scores by about 4–6 points in quadrature, not 18–20%. The 70–74% above
   is a norm ratio (49–55% of the energy). Use the M2b low-passed scores, not the 2.2 rad/µm ones above.*
2. **MAGIC-UNet wins outside the measured band and loses inside it.** In the band just above the SNR = 1 crossing it
   halves FISTA's error (0.12–0.13 against 0.19–0.24 of the target norm on L1_C1). Inside the measured band it is
   worse than FISTA on five of six components (0.14–0.21 against 0.08–0.12; the exception is L1_C1 Bz, whose crossing
   is higher), because nothing forces its output to agree with the data: its
   re-blurred output misses the 3 µm input by 17–33%, where the COMSOL truth itself misses by 10–17% [check].
3. **One data-consistency step fixes part of that with no training.** Adding a single Tikhonov correction to the saved
   MAGIC-UNet output, with the μ = 0.0316 that was chosen on simulation, gives 22.8% on L1_C1 and 25.6% on IC_4um
   (mean 24.2%), against 27.9% for MAGIC-UNet alone and 29.4% for FISTA-300 on the same two configurations [check].
   This is a diagnostic on two test configurations, but it points clearly at the method family: a learned
   current-domain prior inside a data-consistent solver.
   *Update [M1/M2]: confirmed on 15 pairs with μ chosen on simulation and the height estimated: 27.0 → 24.3%. Like for
   like on these two configurations it is 25.1%, not 24.2%.*
4. **The "CSC-LISTA got worse with 8,000 layouts" result is most likely noise.** Both runs took their best checkpoint
   at step 100–200, before the extra layouts could matter, and 2-point COMSOL swings appear between runs with identical
   validation scores [file]. The claims "data volume is not the limit" and "the limit is the simulator gap" are not
   supported by this experiment.
5. **Height is identifiable only through the prior, and only as an upper bound without one.** Run 2's predicted
   current, pushed forward, explains the 3 µm input best at exactly 3.0 µm on both coils [check], so a current-domain
   network already carries a usable height estimate. Run 3 shows the failure when that estimate is made tile by tile.
   *Update [M1/M2]: on a 0.01 µm grid run 2 reads 2.93 and 2.94 µm at 3 µm, but it is pulled toward 3 µm elsewhere
   (2.32 at 2 µm, 3.34–3.61 at 4 µm). Run 3 reads the C1 shapes well and IC_4um too high (3.86 at 3 µm), and one global
   height does not rescue it (46.9%).*

**What to stop.** More CSC-LISTA, LFISTA and FISTA-Net variants; ranking methods by a few points; scoring 5–10 µm COMSOL
pairs before the decay anomaly is explained; processing Bx, By and Bz as three unrelated images.

**What to do first (under one hour on the server, no training).** Re-score run 2 and run 3 on all 17 COMSOL pairs and
on simulated validation with (a) the data-consistency step and (b) a single global height read out by re-projection.
Section 4.2, milestone M1.

**Publication.** CVPR 2027 is not realistic: paper registration closes today and submission is 16 November 2026 [web].
The realistic targets are Physical Review Applied or IEEE Transactions on Computational Imaging in spring 2027, with
ICCP 2027 as a conference option. All of them need new COMSOL geometries with a corrected model and, for the physics
journals, a real QDM measurement with an independently calibrated standoff.

---

## Phase 1. What the project is

### 1.1 Problem statement

**Narrow benchmark (everything scored so far).**

| Item | What the code does |
|---|---|
| Input | Bx, By, Bz maps on a plane at height z_in above a planar micro-coil; 1400 × 1400 pixels at 0.1 µm, 140 µm field of view. z_in = 3 µm for the main benchmark. |
| Output | The same three components at z = 0.5 µm on the same grid. |
| Forward operator | Upward continuation, `H(k) = exp(−k·Δz)` with Δz = 2.5 µm, applied by FFT on a zero-padded canvas (128 px padding and a window mask in `csc_lista.py`; 700 px in the simulator). |
| Known at test time | Pixel pitch and target height for every method. Δz is given to Wiener, FISTA, CSC-LISTA and FISTA-Net. MAGIC-UNet is not told z_in, but its training range is a prior on it. |
| Unknown | Source geometry, noise level, the source offset below the reference plane (0–0.2 µm in the simulator). |
| Noise model | COMSOL has no sensor noise. Its "noise" is mesh speckle (about 10% relative L2 at 3 µm, 15–19% at 0.5 µm, 0.38 µm correlation) plus mismatch with free-space physics. Training noise is simulated: correlated speckle at 1–2.5% of each component's range for CSC-LISTA and FISTA-Net, white noise at 1–2.5% of each crop's range for MAGIC-UNet. |
| Metric | Relative L2, `‖output − target‖ / ‖target‖`, per component over the full image, averaged over 3 components × 5 configurations ("15 pairs"). Range NRMSE is also stored. |
| How methods are chosen | Settings and checkpoints on simulated validation only (64 signal-rich 512 px crops for CSC-LISTA and FISTA-Net; 10–50 full images for MAGIC-UNet's tile stride). COMSOL is read after training. |

**Research question (stated 7 November, `booklet/chapters/17_question.tex`).** Estimate an unknown standoff and
reconstruct the near-plane field across unseen geometries, fields of view and pixel pitches, and say how sure the
method is. Nothing scored so far tests geometry, field of view, pitch or noise generalisation, and nothing outputs a
height or an uncertainty.

**One disagreement with the framing.** `PROJECT_CONTEXT.md` makes the field at a reference plane the primary output and
the current an auxiliary quantity. I would reverse that. The field at 0.5 µm is a linear, exactly known function of the
equivalent sheet current, so the two carry the same information. But the current is what a real experiment can
check (against a lithography mask and the driven current), while a 0.5 µm field at 0.1 µm pitch cannot be measured on a
real QDM, whose optical resolution is of the order of the wavelength [memory; confirm with your instrument]. A paper
whose ground truth can never exist outside simulation is hard to defend at a physics venue.

### 1.2 Data

| Set | Content | Where | Independent geometries |
|---|---|---|---|
| L1_C1 | One spiral coil, 11 heights 0.5–10 µm: 10 pairs to 0.5 µm | Mac `cache/`, `runs/combined_full_200/comparison` | 1 |
| IC_4um | One IC coil, heights 0.5, 3, 4, 5, 6, 7, 10 µm: 6 pairs | Mac | 1 |
| IC_2um | Heights 1 and 2 µm: 1 pair (2 → 1 µm) | Mac | 0: same coil as IC_4um (confirmed 31 October) |
| data3 | C1+C2, C1+C2+C3, C1–C4, 20 heights 0.5–12.5 µm; only 3 → 0.5 µm has been used | Server only | 0: C2–C4 mirror C1 (correlation 0.97–0.98) |
| Simulated v3 | 200 training and 50 validation layouts (spirals, meanders, mirrored arrays), 3 → 0.5 µm | Recipes in `runs/simulated_3_to_05_v3` | Training only |
| Streamed layouts | 8,000 fresh layouts for two CSC-LISTA runs | Generated on the fly | Training only |
| MAGIC crops | 153,600 + 4,096 crops of 64 × 64 at 0.2 µm, from 2,400 + 64 layouts | Server only | Training only |

**Real independent test geometries: two.** The "15 pairs" are 2 wire shapes × 3 components, with one shape repeated
four times. The repeats are not even independent noise draws: `README_SIMULATED_FIELDS.md` reports that
`C1_C2 − L1_C1` leaves about 0.05 µT where the mesh matches, so the speckle is shared.

**The raw data are not on the Mac.** `data/` does not exist locally (it is in `.gitignore`), so I could not check
the raw CSVs, the data3 files or the unit conversion. Everything I verified uses `cache/` and the 17 saved pairs.

**Leaks and overlaps.**

- No COMSOL array enters training or normalisation in the single-standoff and MAGIC-UNet phases. I checked
  `simulate_fields.py`, `train_csc_lista.py` and `train_magic_unet.py`; this holds.
- The simulator was designed from the test coils: the layout families ("as in data3"), the width range (1.6–3.2 µm
  around the 2.08 µm fitted on L1_C1), the source offset, the 600 µA current, and the speckle level and correlation
  length all come from the COMSOL test configurations. Your documents say this openly. It means the test set measures
  in-distribution performance on the two shapes the simulator was built around.
- The test set has been reused heavily. I count at least 22 scored variants on the same 15 pairs (table below). FISTA's
  λ was retuned after the network scores were known, FISTA-Net got a second run because of its first COMSOL score, and
  an interim COMSOL score was read during that run. Each step is defensible; together they make the best score optimistic.
- The older ConvLISTA experiments (Kaggle, combined, synthetic augmentation) trained and scored on the same pairs.

### 1.3 Every method tried

"Comparable" means: same 15 COMSOL pairs at 3 → 0.5 µm, full-image relative L2, COMSOL unseen in training.

**Group A. Early work. Not comparable with anything below.**

| Method | Data and setting | Headline number | Why not comparable |
|---|---|---|---|
| Fourier kernel learning (`res.md`, `kernel*.py`) | L1_C1, forward direction | Joint kernel 0.8–1.0% range NRMSE; analytic 0.7% | Forward problem; range NRMSE |
| IC_4um crop kernel (`kernel_crops.py`) | IC_4um, forward height holdout at 5 µm | 4.4/5.9/5.6% against analytic 6.3/9.2/10.1% | Forward problem |
| Field-domain ISTA/FISTA, λ = 1e-4 µT (`iterative.py`, `fista.py`) | L1_C1 3 → 0.5, operator fitted on the same pair | ISTA-500: 33.8/36.2/36.3% relative L2 (`notebooks/runs_results/comparison_table-2.csv`) | Operator fitted on the test pair |
| In-house ConvLISTA, 163 parameters (`lista.py`) | Same pair, trained and scored on it | 16 layers: 42.1/43.9/48.4% | Train = test; no published source |
| Shared ConvLISTA, 17 pairs (`combined_lista.py`) | All local pairs, trained and scored on them | Mean 65.5% at 16 layers; diverges beyond (3,250% at 64) | Train = test; all standoffs mixed |
| Synthetic augmentation, 51 vs 1,080 components | Rotations, heights, noise of the same three COMSOL fields | 65.06% → 64.66% | Same geometries in train and test |
| Geometry capacity (tied/untied, 16/32 layers) | L1_C1 only | **No result on record**: only a one-update local smoke run exists | Never run, or never copied back |
| Dense ALISTA (`alista*.py`) | Synthetic sparse codes, 250 × 500 | −28.0 dB plain, −54.7 dB with support selection | Different problem entirely |
| Depth extension of ConvLISTA to 32/64 layers | Inference only | Diverges; per-layer gain 1.02–1.11 | Train = test |

**Group B. Single-standoff restart, 3 → 0.5 µm, 15 COMSOL pairs. Comparable with each other.** All from
`README_CSC_LISTA.md`; I re-read each number from the saved `results.json` and they match.

| Method | Training data | Simulated validation | COMSOL | Notes |
|---|---|---:|---:|---|
| ISTA-16, learned dictionary | Dictionary from 200 layouts | 64.3% | 64.4% | |
| Tikhonov ("Wiener"), μ = 0.0316 | μ from simulated crops | 60.3% | 49.3% | μ about 18× too large for COMSOL |
| CSC-LISTA, plain, MSE, 16 layers | 200 layouts, 2,000 steps | 53.4% | 39.1% | Best checkpoint at step 250 |
| CSC-LISTA, sparse, MSE, 16 | 200 layouts | 48.6% | 40.2% | Threshold floor (yours) |
| CSC-LISTA, sparse, relative loss, 16 | 200 layouts | 45.6% | 42.0% | Floor and loss (yours) |
| CSC-LISTA, sparse, relative loss, 32 | 200 layouts | 42.5% | 38.9% | |
| CSC-LISTA, strict, MSE, 32 | 200 layouts | 51.9% | 35.4% | Best at step 300 of 1,000 |
| CSC-LISTA, strict, MSE, 64 | 200 layouts | 48.8% | 33.8% | Best at step 200 of 1,000 |
| CSC-LISTA, strict, MSE, 64, streamed | 8,000 layouts | 49.3% | 36.0% | Best at step 100 |
| CSC-LISTA, relative loss, 64, stored | 200 layouts | 41.0% | 35.6% | |
| CSC-LISTA, relative loss, 64, streamed | 8,000 layouts | 40.9% | 36.2% | |
| Convolutional LFISTA-style, 32 | 200 layouts, Lasso loss | 42.8% | 34.2% | Your own combination |
| FISTA-Net run 1 | 200 layouts, 750 updates | 65.2% | 39.4% | Unconverged |
| FISTA-Net run 2 | 200 layouts, 9,000 updates | 47.9% | 33.3% | Best of 91 noisy checks; interim 34.6% at step 4,800 |
| FISTA-32, λ = 0.03 | Dictionary only | 50.9% | 41.8% | |
| FISTA-64, λ = 0.009 | Dictionary only | 45.3% | 37.9% | |
| FISTA-300, λ = 0.03 | Dictionary only | 36.6% | 32.1% | |
| FISTA-300, λ = 0.009 | Dictionary only | 35.5% | 30.5% | Best field-domain result; not converged (see 2.2) |
| FISTA-1000, λ = 0.009 [M1/M2] | Dictionary only | 34.0% | 26.7% | Best depth by validation; 3,000 gives 36.3% / 27.4% |
| MAGIC-UNet run 1 | 2,400 layouts, standoff N(3, 0.6) µm | 74% | 82% | Invalid: BatchNorm statistics bug |
| MAGIC-UNet run 2 | Same crops, per-crop noise | 15.2% (full images) | **27.0%** | Standoff not given |
| MAGIC-UNet run 3 | Standoff uniform 1–10 µm | 22.7% | 34.8% | IC_4um 52% |
| MAGIC-UNet run 2 + one Tikhonov step at the estimated height [M1/M2] | μ(h) from simulation | – | **24.3%** | Standoff not given |

Comparable in protocol, but not like-for-like in ingredients: MAGIC-UNet sees all three components together, trained on
12× more layouts with white instead of correlated noise, works at 0.2 µm, and passes its output through exact
Biot–Savart. The simulated-validation columns are also not comparable across families (64 signal-rich crops against 50
full images).

**Group C. Limits and diagnostics. Two local configurations only; not comparable with the 15-pair column.**

| Quantity | Value | Source |
|---|---:|---|
| Wiener oracle (in-sample ceiling for linear filters) | 34.4% | `runs/wiener_standoff_local/table.md` |
| Wiener fitted on the other configuration | 37.8% | same |
| Scalar μ fitted on the other configuration | 37.3% | same |
| Lost-band floor (truth with k above the SNR = 1 crossing removed) | 34.5% | `information_budget.py` |
| MAGIC-UNet run 2, two-configuration mean | 27.9% | [check] |
| FISTA-300, two-configuration mean | 29.4% | [check] |
| MAGIC-UNet run 2 + one Tikhonov correction, μ = 0.0316 | 24.2% | [check], diagnostic only; superseded by M1 (25.1% like for like with μ from simulation) |

### 1.4 Contradictions and numbers I could not reproduce

1. **Stale status lines.** `PROJECT_CONTEXT.md` (top entry) says the MAGIC-UNet run has "no results yet";
   `README_MAGIC_UNET.md` holds runs 2 and 3. `booklet/chapters/19_magic_results.tex` says run 3 is "still running"; the
   README has its results. `README_SIMULATED_FIELDS.md` opens with "No model has been trained on it".
2. **IC_2um and IC_4um.** `PROJECT_CONTEXT.md` line 167 still says they are different geometries; the 31 October entry
   says they are the same coil. The newer statement is the right one.
3. **What MAGIC-UNet's authors did about standoff.** `RESEARCH_BEYOND_ISTA.md` §4 says "**fixed** ~50 µm standoff";
   `README_MAGIC_UNET.md` says the standoff was drawn per image from 50 ± 10 µm and never given to the network. The second
   is the basis of your whole port and decides how much novelty is left, so re-read the paper's Supplement D and correct
   whichever file is wrong. [not verified by me: a search summary described the training standoff as fixed]
4. **Mixed sample sizes in one table.** `15_floor.tex` puts two-configuration Wiener numbers above "FISTA-300 (15 pairs)
   30.5%", and the run 3 table in `README_MAGIC_UNET.md` compares a five-configuration mean (34.8%) with a
   two-configuration floor (34.5%). On the same two configurations FISTA-300 is 29.4%, not 30.5%.
5. **FISTA λ table.** The three largest λ values in `09_results.tex` (29.4%, 41.3%, 55.3% at 0.09, 0.3, 0.9) are not in
   any local file. `runs/fista_depths_lambda/results.json` holds five values up to 0.03, and
   `runs/csc_lista_v2/full_plain/results.json` stores only the winner. Probably only in a server log.
6. **Not on the Mac, so not checked:** MAGIC-UNet runs 1 and 3 (all outputs), the IC diagnosis scripts
   (`logs/ic_diag.py`, `logs/implied_height.py`), every data3 number, and the exploratory analyses behind the "walls",
   "faster decay" and "speckle" findings, which `README_SIMULATED_FIELDS.md` says are not in the repository.
7. **Geometry-capacity experiment:** documented protocol, no results anywhere.
8. Everything else I spot-checked reproduced: all Group B means from `results.json`, MAGIC-UNet run 2's per-configuration
   scores from the saved predictions (26.3% and 29.6%), FISTA-300's (31.2% and 27.6%), and the 286 s epochs.

---

## Phase 2. Diagnosis

### 2.1 What went right

- **Test hygiene in the restart.** COMSOL is kept out of training, normalisation and selection, the code enforces it,
  and the documents list every deviation from the published methods. This is better than most student projects.
- **Faithful baselines.** The MAGIC-UNet port is checked layer by layer against the released model, and the run 1
  BatchNorm failure was diagnosed properly instead of being tuned away.
- **Asking for the limit.** The Wiener oracle and the information budget are the right kind of question, even though I
  disagree with how the floor is read (2.3).
- **The degeneracy analysis** in `RESEARCH_BEYOND_ISTA.md` §1 is correct and is the intellectual core of a possible paper.
- **The run 3 diagnosis** (different tiles implying different standoffs) is a real finding, not a bug report.

### 2.2 What went wrong, by cause

**(a) Information limits.**

- The SNR = 1 crossing at 3 µm sits at 1.05–1.6 rad/µm on L1_C1 [check, matches your table]. Beyond it a linear filter
  recovers nothing. That part is sound.
- The floor is not all lost signal. Split by band on L1_C1 [check]: the band from the crossing to twice the crossing
  holds 20–34% of the target norm, and this is real wire-edge content that a prior can recover. Above that the target
  holds 15–18%, and both MAGIC-UNet and FISTA have an error there equal to the target's own energy, because they output
  nothing and the target is noise. So the recoverable gap is the middle band, and the top band is a fixed cost.
- These numbers are specific to COMSOL's mesh "noise". A real QDM has photon shot noise, a different spectrum and a
  different level, so the crossing and the floor will move. Do not carry 34.5% into a paper as a property of the problem.

**(b) Modelling errors.**

- At 3 µm the free-space operator is adequate but not exact. Blurring the COMSOL 0.5 µm field to 3 µm misses the COMSOL
  3 µm field by 10–12% on L1_C1 and 14–17% on IC_4um. Restricted to k < 1 rad/µm, where speckle is small, the mismatch is
  5–8% on L1_C1 and 8–15% on IC_4um; the empirical k0 = 0.095 term reduces it to 4–6% and 7–11% [check].
- The amplitude shortfall of every method (slope 0.85–0.94 for both MAGIC-UNet and FISTA [check]) is **not** mainly this
  mismatch. The operator's own amplitude error is 2–6% (slope 0.94–0.98). The rest is the missing upper band and the
  noise in the target's norm.
- At 10 µm the mismatch is 64–140% [file]. COMSOL decays faster than any harmonic field can. A static field in a
  source-free region must be harmonic, so this is a property of the COMSOL model (boundary conditions, domain size,
  study type, or the export), not of magnetostatics. Until someone opens the model file and explains it, scores above
  about 4 µm measure agreement with an artefact.
- COMSOL's components are inconsistent with each other. Projecting the 0.5 µm target onto fields that one current sheet
  can produce leaves 15% (L1_C1) and 23% (IC_4um) of its norm outside [check]. Part of that is truncation at the image
  edge at low k and part is speckle at high k, so read it as an upper bound, in line with your 10.9% figure.
- MAGIC-UNet's own output is not physical either: its Jx and Jy come from separate decoders with no continuity
  constraint, and its field has an inconsistent part of 8% (L1_C1) and 15% (IC_4um) [check].

**(c) Data problems.**

- Two test shapes, shared speckle, simulator built around them, heavy reuse (1.2).
- No sensor noise anywhere. The only noise the methods have faced is a mesh artefact.
- No real measurement, although `PROJECT_CONTEXT.md` says the group has the QDM and the Bathla et al. dataset.
- Simulated validation does not rank methods as COMSOL does (`09_results.tex`, Figure "rank"). Part of that is
  the simulator; part is that validation uses signal-rich crops and COMSOL uses full images; part is one seed.

**(d) Optimisation and architecture.**

- **CSC-LISTA is under-trained, not data-limited.** It has 49 million parameters at 64 layers, runs 1,000 steps and keeps
  the checkpoint from step 100–300. It starts as ISTA with λ = 0.3, far from the λ = 0.009–0.03 that FISTA prefers. On
  its own training distribution it loses to FISTA at equal depth with the same dictionary (51.9 against 50.9% at 32,
  48.8 against 45.3% at 64) [file]. A learned unrolled network that cannot beat its own initialisation family on
  in-distribution data has an optimisation problem.
- **It is not doing sparse coding.** Code density is 99% in the strict runs and 9–16 of 64 thresholds are zero [file].
  With dense codes the network is close to a learned linear filter, which is why it sits near the linear ceiling.
- **FISTA-Net's selection is noisy.** Constant learning rate, validation moving between 48% and 68% in the last 1,000
  steps, best of 91 checks.
- **FISTA-300 is not converged.** Error falls monotonically from 32 to 64 to 300 iterations and nobody ran 1,000. The
  "classical baseline" is an early-stopped solver whose λ was tuned for that stopping point.
  *Update [M1/M2]: confirmed. FISTA is best at 1,000 iterations on simulation and on COMSOL (26.7%) and worse at 3,000.*
- **Everything in Group B treats Bx, By and Bz as separate scalar images.** Above the sources all three follow from
  one scalar function. The field-domain methods throw that constraint away; MAGIC-UNet uses it only at the output.
- **MAGIC-UNet has no data-consistency step and a small view.** A 12.8 µm tile is about four standoffs wide, so most of
  the field in a tile comes from wires outside it, and each tile decides its own standoff.

**(e) Evaluation.**

- The metric is taken against a noisy target, over the full image, with no error bars.
- One seed per run; differences of 0.5–3 points are read as findings.
- Selection on 64 crops (16 for FISTA's λ).
- No timing, so no speed claim. No uncertainty, no height output.

### 2.3 The surprising results, one by one

**CSC-LISTA dropped from 33.8% to 36.0% with 8,000 layouts.**

- *Best-supported explanation:* checkpoint-selection and seed noise. The stored run kept step 200 and the streamed run
  kept step 100 [file]. By then the streamed run had seen 800 crops and the stored run 1,600, so the streamed run had
  not yet seen more data. Their validation scores are 48.8% and 49.3%, indistinguishable on 64 crops. Elsewhere, runs
  with the same validation score differ by similar amounts on COMSOL (the two relative-loss runs: 41.0 and 40.9%
  validation, 35.6 and 36.2% COMSOL).
- *What it does not show:* that more data cannot help, or that the simulator gap is the limit. The experiment never
  delivered more data to the checkpoint that was scored.
- *Cheap test:* three seeds of the stored strict run and three of the streamed run, 300 steps each (every best
  checkpoint so far came by step 300; about 50 minutes per run at the logged 1,025 s per 100 steps), logging the COMSOL
  score at every check as a diagnostic. If the seed spread is 1.5 points or more, the "drop" is noise. If both groups
  are tight and 2 points apart, it is real and I am wrong.

**Learned field-domain methods sit within 4 points of the Wiener oracle.**

- *Best-supported explanation:* with dense codes they act almost linearly and can only restore the measured band. The
  band split shows it for FISTA-300: its in-band error is small (0.08–0.12 on five of six components) and its error
  above the crossing (0.22–0.31) is 66–98% of the truth's energy there (0.27–0.40) [check]. The oracle is the best
  possible linear filter, so near-linear methods end up next to it.
- *Cheap test:* save the full-image COMSOL predictions of strict CSC-LISTA-64 and FISTA-Net run 2 (inference only,
  minutes) and run `scripts/review_checks.py` on them. The explanation holds if their error above the crossing is at
  least two thirds of the truth's energy there, as it is for FISTA-300, and fails if they recover a much larger share
  of that band.
- *Result [M1/M2]:* holds on average. In the prior band their error norm is 82% (CSC-LISTA-64) and 79% (FISTA-Net)
  of the target's norm there; the lowest single component is 65%.

**FISTA-300 beats every learned field-domain model.**

- *Best-supported explanation:* the two are not at the same point. FISTA is still improving at 300 iterations; the
  networks are capped at 64 layers and stopped after 100–300 steps. The dictionary carries the prior in both, and FISTA
  simply applies it for longer.
- *Cheap test:* FISTA at 1,000 and 3,000 iterations, λ re-picked on the 16 validation crops, on simulation and COMSOL
  (under an hour). If the error keeps falling, short unrolled networks are the wrong tool here. If it rises, "300" was
  acting as a hidden regulariser and must be reported as a tuned setting.
- *Result [M1/M2]:* both, in turn. From 300 to 1,000 iterations the error falls (COMSOL 30.5 → 26.7%), from 1,000 to
  3,000 it rises (27.4%). The iteration count is a second regulariser, and the fair baseline is FISTA-1000. (It took
  about 5 hours on the GPU, not under an hour.)

**MAGIC-UNet run 2 beats methods that were given the standoff.**

- *Best-supported explanation:* a current-domain prior fills the band just above the crossing (error 0.12–0.13 there
  against FISTA's 0.19–0.24 on L1_C1), and it wins despite being worse inside the measured band [check].
- *Caveat:* the win is on one of two shapes. On IC_4um it loses (29.6 against 27.6%), and against the low-passed target
  it loses there too (22.2 against 20.2%) [check].
- *Cheap test:* milestone M1.
- *Result [M1/M2]:* against FISTA-1000, raw run 2 no longer wins (27.0 against 26.7%). With the correction at its own
  estimated height it wins on both shapes: C1 family 23.8 against 26.6%, IC_4um 26.4 against 27.2%.

**MAGIC-UNet run 3 fails on the IC coil (52%).**

- *Best-supported explanation:* yours. Closely spaced antiparallel wires look like one deeper source, each tile decides
  alone, and a 1–10 µm training range lets it choose "far". I could not re-check it because run 3's outputs are on the
  server.
- *Cheap test:* take run 3's saved IC_4um current, find the single height that best re-projects to the input over the
  whole image, and apply one data-consistency step at that height. If the score drops from 52% to near run 2's 30%,
  the failure is the per-tile height decision and not the network's current prior.
- *Result [M1/M2]:* it does not. The read-out gives 3.86 µm at a true 3 µm, and the score drops only to 46.9% (37.3%
  even at the true height). The run 3 current itself is wrong on IC_4um: its prior-band error for By stays at 31% after
  the correction. My framing (height decision versus current prior) was too clean; the per-tile height decision is
  baked into the predicted current, so a global height applied afterwards cannot undo it.

**Simulated validation does not predict COMSOL.**

- *Best-supported explanation:* a mix of three things I cannot separate from the files: selection noise on 64 crops,
  crops against full images, and real simulator differences (no fast decay, no walls, clean targets).
- *Cheap test:* score every saved model on 50 full simulated validation images with the COMSOL protocol, once on v3 and
  once on v2 (which has k0 = 0.095). If the rank correlation with COMSOL rises sharply on v2, the decay law is the
  main difference.

### 2.4 Direction

**Right:** leaving field-domain sparse coding for a current-domain prior; treating height and reconstruction as
one problem; refusing to tune on COMSOL; wanting uncertainty.

**Wrong, or not supported by the evidence:**

- "FISTA-Net run 2 is the selected network" and similar rankings inside a 1-point band on one seed.
- "Data volume is not what limits CSC-LISTA" and "the limit is the gap between simulator and COMSOL" (`16_followups.tex`).
- "Any method below 34.5% must be filling in lost frequencies" (`15_floor.tex`). It is true of MAGIC-UNet in the middle
  band, but the floor is inflated by target noise, and MAGIC-UNet gets there while being worse in-band.
- "Height cannot come from the field alone", stated as symmetric. Without a prior the data still give a hard upper
  bound on depth (3.1). The ambiguity is one-sided.
- "h stays fixed because it is physics and we know it exactly" (`13_defence.tex`). True of physics, not of this COMSOL
  export, which does not obey it.
- The working rule "only published, unmodified methods". It is a good rule for baselines. It cannot produce a paper,
  and the project already breaks it wherever it gets results (the blurred dictionary, J_eff, the Biot–Savart output).

**Stop:**

1. New CSC-LISTA, LFISTA or FISTA-Net variants (losses, floors, depths, start values).
2. Reading any difference under about 3 points on the current test set.
3. Reporting 5–10 µm COMSOL scores as physics before the decay anomaly is explained.
4. Per-component scalar processing.
5. The old 163-parameter ConvLISTA line, including the unrun geometry-capacity experiment.
6. Quoting the lost-band floor and the Wiener oracle as limits of the problem.

---

## Phase 3. Nuances and gaps

### 3.1 Identifiability

- **The degeneracy is exact but one-sided.** A source at depth h + δ must be sharper by `e^{kδ}`. A real conductor's
  spectrum decays only algebraically (a box of width w gives a sinc), so multiplying it by a growing exponential gives a
  source with unbounded energy. A source at depth h − δ, smoother by `e^{−kδ}`, is always a perfectly good current. So
  with no prior at all the feasible depths are everything *up to* the true one, and the true depth is the deepest
  hypothesis whose source stays finite. This is sharper than "impossible" and it gives the estimator its form: push the
  depth down until the source starts to ring.
- **With noise the bound blurs by about one inverse cut-off.** Only k below the SNR = 1 crossing k_c is seen, so
  hypotheses differing by δ ≲ 1/k_c are hard to separate. At 3 µm, k_c ≈ 1.1 rad/µm gives δ of the order of 1 µm. Run 2's
  tile-wise implied heights of 2.2–4.5 µm (`README_MAGIC_UNET.md`) are consistent with that. This is a prediction you
  can test at other standoffs: the height uncertainty should grow roughly in proportion to the standoff.
- **What the prior really fixes is edge sharpness, so height trades against width and thickness.** A wire of width w
  multiplies the spectrum by `1 − (kw)²/24 + …`, a conductor of thickness t by roughly `e^{−kt/2}`. Thickness is almost
  exactly a height shift. So the estimable quantity is an *effective* standoff, and a network trained on widths of
  1.6–3.2 µm has learned those widths. A reviewer will test it on a 0.5 µm or a 6 µm wire.
- **Known drive current does not break the degeneracy by itself** (the k → 0 limit does not depend on depth), but
  together with a wire model it does. That is exactly how NV groups calibrate standoff on a known wire, and it is your
  route to ground truth on the real instrument.
- **One height per image is free information that nothing uses yet.** Globally, run 2's current explains the 3 µm input
  best at 3.0 µm on both coils, with a shallow valley: ±0.25 µm changes the residual by 1–3 points [check]. Because run 2
  only ever saw 1.2–4.8 µm, this shows consistency, not blind estimation. The real test is the same read-out at 2 and
  4 µm for run 2, and at every standoff for run 3 (M1).

### 3.2 What the learned prior is learning

- **Field-domain dictionary (CSC-LISTA, FISTA):** 32 atoms of 7.5 µm in the field at 0.5 µm. Fields of wires are smooth
  and overlapping, so codes come out dense and the prior is weak.
- **Current-domain network (MAGIC-UNet):** currents are compact, piecewise constant, of one amplitude and continuous.
  That is a far stronger prior and it supports extrapolation past the cut-off. The band split confirms where it acts:
  from the crossing to about twice the crossing. Above 2.2 rad/µm the COMSOL target is noise, so nothing can be shown there.
- **Open question:** is the network recovering structure or recalling the simulator's shapes? Both test coils are
  Manhattan spirals of the simulated width. Until an out-of-family geometry is tested (curves, diagonals, pads, two
  current levels, two layers) there is no evidence either way.
- **A product no other method offers:** for each reconstruction, report which part of the output spectrum is supported
  by data (below k_c for the estimated height and noise) and which part came from the prior. This is cheap, physical
  and directly answers "how do you handle ambiguity".

### 3.3 Generalisation axes: status

| Axis | Tested? | Evidence |
|---|---|---|
| Standoff, known | Only 3 µm for field-domain methods | No FISTA or network scores at other heights |
| Standoff, unknown | Run 2 inside 2–4 µm; run 3 across 1–10 µm | Works in a ±20% band; fails tile-wise in a wide one |
| Geometry | No | Two shapes, both in the simulator's families |
| Wire width, thickness | No | |
| Field of view | No | Fixed 140 µm; MAGIC-UNet tiles are a fixed 12.8 µm |
| Pixel pitch | No | 0.1 µm, or 0.2 µm binned |
| Noise level and type | No | One synthetic family; no sensor noise |
| Current amplitude | Implicitly | Per-tile scaling gives amplitude invariance; one current in all data |
| Real measurement | No | |

A useful simplification for standoff and pitch: magnetostatics has no intrinsic length. If the input is resampled so
that the hypothesised standoff is a fixed number of pixels, every standoff becomes the same problem with a different
wire width in pixels. Run 2 (a narrow standoff band) beats run 3 (a wide one) by 8 points at 3 µm [file]; resampling
lets every case be the narrow one. This also gives pitch generalisation for free. It needs a height hypothesis, which
the height search in 4.1 supplies.

### 3.4 Calibration and uncertainty

Nothing exists yet. In order of cost:

1. Three seeds of the main model: spread of the score and a per-pixel ensemble spread (two more 4-hour runs).
2. The height profile: data misfit against hypothesised height, reported as an interval, checked for coverage on
   simulated layouts where the height is known.
3. The data-supported and prior-filled bands of 3.2.
4. Conformal intervals fitted on simulated validation, with coverage then measured on COMSOL and on real data. Expect
   coverage to break under the simulator gap; reporting that honestly is itself useful.
5. Posterior sampling with a diffusion prior (4.1, candidate B), if the venue is an ML one.

### 3.5 Physics errors a learned model could absorb or hide

- The non-harmonic COMSOL decay. A field-to-field network trained on COMSOL would learn it silently. Your choice to keep
  it out of the simulator is right. MAGIC-UNet's Biot–Savart output cannot represent it at all, which is safer.
- The 90° component-label rotation. It is handled consistently in the code, but it should be traced to the COMSOL export.
- Shared mesh speckle between configurations. Harmless while COMSOL stays out of training; a leak the moment it enters.
- Sheet currents against conductors of finite thickness, and the NV layer's own thickness and the optical blur on a real
  QDM. All three act like extra standoff (3.1).
- Currents that do not conserve charge in MAGIC-UNet's output.
- Per-tile amplitude scaling, which hides the absolute scale from the network and makes empty tiles pure noise (the
  run 1 bug was a symptom).
- Zero padding of a field that COMSOL has forced to zero normal component at the walls.

### 3.6 What is already published

| Topic | Work | Status | What it closes off |
|---|---|---|---|
| Fourier inversion, field → current | Roth, Sepulveda, Wikswo, J. Appl. Phys. 1989 | [memory] | The classical baseline |
| Vector data improves current reconstruction | Broadway et al., Phys. Rev. Applied 14, 024076 (2020) | [web] | "Use all three components" is not new |
| Fourier against Bayesian reconstruction on QDM data | Midha, Parashar, Bathla, Broadway, Tetienne, Saha, arXiv 2402.17781 | [web]; your group | Regularised current reconstruction on QDM |
| L1-curl prior, divergence-free wavelets, 1–10 µm | arXiv 2408.16550 | [web] title and abstract; standoff range from your notes | A training-free current prior; required baseline |
| Dictionary + LASSO, 3D and time | Bathla et al., arXiv 2506.05491 | [web] title; your group | Sparse dictionary reconstruction |
| U-Net, field → current | Reed et al. (MAGIC-UNet), arXiv 2407.14553; Phys. Rev. Applied 23, 034035 (2025) | [web]; journal reference from the DOI in search results | Learned current reconstruction; unknown standoff inside a narrow band if Supplement D says so |
| Untrained network with physics loss | Dubois et al., Phys. Rev. Applied 18, 064076 (2022) | [web] | Training-free neural reconstruction (magnetisation) |
| Physics-informed deep image prior, scanning NV | arXiv 2604.23431 | [web] title only | Same idea, in-plane magnetisation |
| Joint sensor distance and source, physics prior | Setescak et al., arXiv 2602.17180; Phys. Rev. Letters (2026) | [web] | Joint distance + reconstruction exists, for magnetisation, with a micromagnetic prior, not learned |
| Neural networks for magnetisation textures | arXiv 2412.19381 | [web] title only | |
| Depth-resolved currents in chips | Garsi et al., Phys. Rev. Applied 21, 014055 (2024); Oliver et al., arXiv 2202.08135 | [web] | Layer localisation from vector QDM data |
| Unrolled solver conditioned on kernel, scale, noise | USRNet (Zhang, Van Gool, Timofte, CVPR 2020); DPIR (TPAMI 2021) | [memory] | "One unrolled model for all kernels" |
| Alternating kernel estimation and restoration | IKC (CVPR 2019); DAN (NeurIPS 2020) | [memory] | Blind unrolled restoration |
| Diffusion priors for inverse problems, and blind variants | DPS (Chung et al., ICLR 2023); DDRM (Kawar et al., 2022); BlindDPS (CVPR 2023) | [memory] | Posterior sampling with operator estimation |
| Plug-and-play, deep image prior, neural operators | Venkatakrishnan et al. 2013; Ulyanov et al. 2018; Li et al. (FNO), ICLR 2021 | [memory] | The generic toolbox |
| Geophysical downward continuation with networks | Entries in `RESEARCH_BEYOND_ISTA.md` | [not verified] | Same mathematics in another field |

**What my searches did not find** (two targeted searches, not an exhaustive survey): a learned method that estimates
standoff jointly with a current or field reconstruction in NV or QDM imaging, or any diffusion or score-based prior
applied to this problem.

**So do not claim** as new: learning the field-to-current map; using vector data; a sparse or total-variation current
prior; joint distance fitting as an idea; unrolled networks conditioned on the operator; "a network that works without
being told the standoff" if Reed et al. already trained over 50 ± 10 µm.

**What can still be claimed, if the experiments deliver:** explicit, image-level standoff inference with a learned
current prior and exact physics; a one-sided identifiability analysis with a measured height resolution; a
reconstruction that reports which frequencies came from data and which from the prior; validation across standoff on a
corrected simulation and on a real QDM with calibrated standoff.

---

## Phase 4. Recommendations

### 4.1 Method family

**Recommended: a data-consistent current-domain solver with explicit height search.** In full:

1. **Unknown:** the stream function g of the equivalent sheet current, one scalar image, `J = curl(g ẑ)`. Current
   continuity then holds by construction, and Bx, By, Bz at any height follow from g by the exact Fourier relations
   already in `simulate_fields.py`.
2. **Scale normalisation:** for a hypothesised standoff h, resample the input so h is a fixed number of pixels (run 2
   used 15). One network trained at that normalised standoff then serves every physical standoff and pitch.
3. **Network:** a MAGIC-UNet-style encoder–decoder with one output (g), used as the learned prior inside a few unrolled
   stages. Each stage alternates a closed-form Fourier data-consistency step on all three components with a learned
   correction to g, in the manner of USRNet. Tiles of at least 128–256 normalised pixels, so a tile spans 8 or more standoffs.
4. **Height:** an outer one-dimensional search. For each h on a logarithmic grid, reconstruct, then score by whitened
   data misfit plus the size of the network's correction. Report the minimiser, the interval where the score stays
   within a calibrated tolerance, and one reconstruction per supported height when the interval is wide.
5. **Training:** your simulator with wider randomisation (curved and diagonal wires, widths over a decade in units of
   h, two current levels, finite thickness, white and correlated noise, optical blur). COMSOL and real data stay test-only.

Why this, for this project:

- The only training-free evidence in hand points to it: one data-consistency step takes MAGIC-UNet from 27.9% to 24.2%
  on the two local configurations [check]. *M1 confirms it on 15 pairs, with μ from simulation and the height
  estimated: 27.0 → 24.3% [M1/M2].*
- *Caveat added after M1:* the run 3 result shows that "one height per image, chosen by the physics" must be chosen by
  reconstructing at each candidate height, not by re-projecting one feed-forward prediction.
- It turns the run 3 failure into a design choice: one height per image, chosen by the physics.
- Every block has a published ancestor (MAGIC-UNet, USRNet, the L1-curl stream-function prior, Setescak's joint distance
  fit), which respects your professor's preference, while the combination and the height inference are yours.
- It reuses your simulator, your Biot–Savart code and the MAGIC-UNet port. Training cost stays at hours per run.

**Candidates compared.**

| Rank | Candidate | Expected gain | Risk | Compute on one A6000 |
|---|---|---|---|---|
| 1 | **A. Data-consistent current-domain solver + height search** (above) | Highest that the evidence supports: in-band error down to FISTA's level while keeping MAGIC-UNet's band extension; standoff and pitch handled by resampling | Medium. The height score may be flat where the prior is weak; that is then reported as an interval | 4–6 h per single-stage run; 12–24 h unrolled |
| 2 | **B. Diffusion prior on g with measurement-consistent sampling and a height variable** (DPS or BlindDPS style) | Posterior samples, so uncertainty and competing hypotheses come naturally; prior is trained once, independent of standoff | High. Guidance through an operator spanning four decades in gain is unstable; same simulator gap; slow tiled sampling | 1–3 days to train; minutes per image |
| 3 | **D. Untrained network or implicit representation of g with exact physics loss**, height as a free parameter (Dubois style) | Unknown and probably modest; no simulator gap at all | Low. Likely weak on height | Minutes per image, no training |
| 4 | **E. Classical stream-function total variation or L1-curl, on a grid of heights** | A baseline and the identifiability test in one | Low | Minutes per image |
| 5 | **C. Fourier neural operator, field → field or field → current** | No clear advantage over a CNN after resampling; mode truncation hurts sharp wires | Medium | Similar to A |
| — | More unrolled field-domain sparse coding | None supported by the evidence | — | — |

Build A. Implement D and E in any case, because reviewers will ask for both. Treat B as the second paper or the
uncertainty arm if A works. I cannot predict A's score at 5–10 µm, and I would not trust anyone who did before the
COMSOL model is fixed.

### 4.2 Roadmap

Times assume one RTX A6000 and no training on the Mac.

**M1. Re-score what you already have. CHEAPEST DECISIVE EXPERIMENT. Under 1 hour, no training.**

- *Experiment:* on the server, for run 2 and run 3 `final.pt`: (a) apply one Tikhonov data-consistency step to the
  output field; choose μ on the 50 simulated validation images, then score all 17 COMSOL pairs once; (b) for every
  pair and for simulated images at several known standoffs, find the single height at which the predicted current best
  re-projects to the input. `scripts/review_checks.py` has both computations in NumPy.
- *Expected:* (a) improves simulated validation and, if the two local configurations are representative, reaches about
  24% on the 15 pairs. (b) Height error within 15% on
  simulated layouts for run 3, and the IC_4um score of run 3 falls sharply once one global height is imposed.
- *Go:* both hold. Method family A is confirmed and joint height estimation is feasible with a learned prior.
- *Failure signature:* (a) helps on COMSOL but hurts on simulation (then it is a COMSOL-specific accident); (b) run 2's
  height sticks at 3 µm for 2 and 4 µm inputs, or run 3's sticks at the top of its range on the IC coil.
- *Outcome [M1/M2]:* (a) passes: simulation at 3 µm 15.1 → 12.7%, COMSOL 27.0 → 24.3%, and it helps at every simulated
  standoff and on all 40 COMSOL rows. (b) partly: run 3's median height error is within 15% at every simulated
  standoff except 1 and 10 µm, but on IC_4um the read-out is biased high (0.7–1.8 µm at 3–6 µm) and the score falls only
  to 46.9%. Neither failure signature appears in full: run 2 is pulled toward 3 µm on COMSOL (2.32, 3.34–3.61) but not
  on simulation (2.05, 4.01), and run 3 does not hit the top of its range. It took 2.9 hours on a shared GPU (12
  standoffs × 50 layouts × two runs, with a 17-value μ sweep), not under one.

**M2. Repair the measuring stick. 1–2 days of work, plus 8 GPU-hours for seeds.**

- *Experiment:* report every score four ways: raw, against the target low-passed at a cut-off fixed in advance,
  in-band, and prior-band. Add two more seeds of MAGIC-UNet run 2. Run FISTA at 1,000 and 3,000 iterations. Save the
  full-image predictions of every Group B model.
- *Expected:* a seed spread of 1–2 points, which sets the smallest difference worth discussing.
- *Go:* always go; this is hygiene. If the seed spread exceeds 3 points, stop comparing methods on this test set at all.
- *Outcome [M1/M2]:* noise edge 2.9 rad/µm; four-way table for every method; all Group B predictions saved and
  reproduced exactly; FISTA best at 1,000 iterations (26.7%). Seeds: see `M1_M2_RESULTS.md` §M2e.

**M3. Fix and widen the test data. Days of your time, mostly outside the GPU.**

- *Experiment:* open the COMSOL model. Record the boundary conditions, domain size, study type and mesh. Reproduce the
  k0 decay or remove it (larger domain, open boundaries, mesh refinement). Then export at least ten new geometries the
  simulator has never drawn, several widths, at five or more heights. Inventory the "extensive COMSOL database".
- *Expected:* the fast decay is a boundary or domain effect and disappears with a larger open domain.
- *Go:* a corrected set where free-space propagation between planes agrees to a few per cent in-band at every height.
- *Failure signature:* the anomaly survives a larger domain. Then it is in the export or the study type; do not publish
  large-standoff COMSOL scores until it is found.

**M4. Classical identifiability test. Half a day to one day, no training.**

- *Experiment:* stream-function total variation (or L1-curl) at a grid of heights, on simulated fields with known
  noise and on COMSOL. Plot objective against height.
- *Expected:* a one-sided valley, as 3.1 predicts: flat or slowly rising for too-shallow hypotheses, steep for too-deep
  ones, with a resolution near 1/k_c.
- *Go:* the edge of the valley tracks the true height within about 20% on simulation.
- *Failure signature:* no edge at any regularisation weight. Then height must be calibrated externally and the paper
  becomes "reconstruction given a calibrated standoff, with the sensitivity to its error". That is still publishable.

**M5. First version of method A. 3–5 days.**

- *Experiment:* retrain the MAGIC-UNet-style network with a stream-function output at normalised standoff, tiles of 128
  pixels or more, wider randomisation. Inference with the height grid and the data-consistency step. Three seeds.
- *Expected:* at 3 µm, at or below run 2 with the M1 correction; at other standoffs, clearly below run 3.
- *Go:* better than run 3 at every standoff on the corrected COMSOL set, and height error under 15% on held-out
  simulated families.
- *Failure signature:* good on simulation, poor on the new COMSOL shapes: the prior memorised the simulator. Widen the
  simulator before touching the architecture.

**M6. Unrolled version and ablations. 1–2 weeks.**

- *Experiment:* 4–8 stages with shared weights. Ablate: no data consistency; field-domain against current-domain
  output; scalar against vector input; known against estimated height; tile size; noise type.
- *Go:* each claimed ingredient is worth more than the seed spread.

**M7. Generalisation suite. 2–3 weeks, partly parallel with M6.**

- *Experiment:* held-out simulated families; the new COMSOL shapes; pitch 0.1/0.2/0.4 µm; cropped fields of view; noise
  levels and white against correlated; widths outside the training range; finite thickness.
- *Failure signature:* height biased by width or thickness. Expected to some degree (3.1); quantify it and report
  effective standoff.

**M8. Real QDM data. 2–4 weeks, critical for physics venues.**

- *Experiment:* image a wire of known width and current; fit its standoff by the standard line-profile method; then
  image a second structure in the same session. Compare estimated with calibrated standoff, the current map with the
  mask (edge position, width), and the integrated current with the driven current.
- *Go:* height within the calibration uncertainty and current edges within the optical resolution.

**M9. Uncertainty. 2–3 weeks, optional for a journal, required for an ML venue.**

- *Experiment:* ensemble spread, height intervals and conformal intervals, with coverage on held-out simulation, COMSOL
  and real data; optionally candidate B.

Realistic span: three to four months to a submission that includes M8.

### 4.3 Publication strategy

**The one claim.** *The standoff of a magnetic image can be inferred, with an honest interval, from the image itself
once a learned current prior and exact field physics are combined; doing so gives better current and near-field
reconstructions than methods that are told the standoff, across a tenfold range of standoffs, and the reconstruction
states which part of it the data support.*

**Title-level story.** "Standoff-aware current reconstruction for quantum diamond microscopy with a learned source
prior." Abstract in four moves: the standoff blur is one-parameter and exactly known, but the parameter is uncertain
in practice and trades against source sharpness; we show the ambiguity is one-sided and measure the height resolution
it allows; we give a solver that searches over height with a learned current prior and exact data consistency; on
simulation, corrected finite-element data and a real QDM it recovers the standoff to X% and lowers reconstruction
error by Y against Fourier, L1-curl, MAGIC-UNet and FISTA.

**What a reviewer will demand, and what is missing.**

| Item | Needed | Have |
|---|---|---|
| Baselines | Fourier inversion with a Hann or Tikhonov filter; L1-curl or TV on the stream function; untrained network (Dubois); MAGIC-UNet as published; FISTA with the dictionary; a known-height version of your own model | MAGIC-UNet port and FISTA only |
| Datasets | Held-out simulated families; at least ten unseen finite-element geometries with correct physics; one real QDM dataset with calibrated standoff | Two shapes, anomalous decay, no real data |
| Ablations | Data consistency; current against field domain; vector against scalar; known against estimated height; tile size; noise model | One informal diagnostic |
| Metrics | Current-domain error and edge localisation; near-field error against a noise-free or low-passed target; height error and interval coverage; in-band and prior-band error; run time | Raw relative L2 on a noisy target |
| Statistics | Three or more seeds; bootstrap intervals over geometries; a frozen test protocol written before the final runs | One seed |
| Reproducibility | Code, simulator, trained weights, the COMSOL model file | Code and simulator |

**Venues.**

| Venue | Deadline | Fit | Minimum to be competitive | Verdict |
|---|---|---|---|---|
| CVPR 2027 | Registration 10 Nov 2026, papers 16 Nov 2026 [web: official dates page, as reported by search] | Weak: niche modality, no real data | A finished method today | **Not realistic** |
| ICLR 2027 | 25 Sep 2026 [web: official] | — | — | Passed |
| ICML 2027 | Late Jan 2027 [not verified: tracker projections] | Weak | A general method plus theory | Too soon |
| ICCV 2027 (Hong Kong, 2–8 Oct 2027) | About 5–8 Mar 2027 [not verified: projected, no official call yet] | Moderate, as computational imaging | M1–M8 done by February, strong visuals on real data, generic baselines | Possible only if M1 and M5 succeed quickly; high risk |
| ICCP 2027 (College Park, 2–4 Aug 2027) | 16 Mar 2027 [not verified: aggregator, not the official call] | Good: physics-based computational imaging | M1–M7 and at least a first real measurement | **Best conference option** |
| NeurIPS 2027 | Mid to late May 2027 [not verified: projected] | Only if framed as blind inverse problems with a one-parameter operator family and ambiguity-aware posteriors | Candidate B working, a second problem domain, identifiability analysis | Stretch; otherwise the ML-for-physical-sciences workshop |
| IEEE Trans. Computational Imaging | Rolling | Good for the algorithm | M1–M7, full ablations, one real dataset strongly advised | **Realistic** |
| Physical Review Applied | Rolling | Best: MAGIC-UNet, Dubois, Broadway and Garsi all appeared there | M3 and M8 are mandatory; height validated against an independent calibration | **Realistic, and my first choice** |
| Nature Communications | Rolling | Needs a new capability on real devices | Several real samples and a result the field could not get before (for example failure analysis at large standoff) | Not on current evidence |
| Optica | Rolling | Weak unless the optics or the instrument is new | — | Not a fit |
| Review of Scientific Instruments | Rolling | Fallback for a methods paper | M1–M5 and real data | Safe fallback |

Check every unverified date on the official site before planning around it.

### 4.4 Top five risks

1. **The benchmark is not valid evidence.** Two shapes, an unexplained non-physical decay, inconsistent components, a
   noisy target. *Action:* M2 and M3 before any further method comparison; real data (M8) for the paper.
2. **The learned prior has memorised the simulator's shapes and widths.** Every gain so far is on coils the simulator was
   built around. *Action:* held-out families and new COMSOL shapes (M7); keep the untrained-network and L1-curl baselines
   beside every result, since they have no simulator to memorise.
3. **Height is not identifiable tightly enough to be useful.** The valley is shallow at 3 µm and will be shallower
   higher up; width and thickness bias it. *Action:* M4 early; define and report effective standoff with an interval;
   keep the fallback paper (calibrated standoff plus sensitivity analysis) in view.
4. **Novelty is thinner than it looks.** MAGIC-UNet already maps field to current, possibly over an unknown ±20%
   standoff; Setescak et al. already fit the sensor distance jointly. *Action:* re-read both in full; centre the claim on
   explicit image-level height inference, data consistency, the one-sided analysis and real validation; port MAGIC-UNet
   exactly as a baseline (done).
5. **Statistical fragility and test-set reuse.** One seed, 22 variants on 15 pairs, selections informed by earlier
   scores. *Action:* write the final protocol down first; freeze a new test set from M3 that no method has seen; three
   seeds; bootstrap intervals; report every run.

A sixth, practical one: a single GPU and a single person. Method A fits that budget; candidate B and a wide
architecture search do not.

---

## Appendix A. Numbers behind the [check] tags

`python3 scripts/review_checks.py`, from the repository root. Inputs: `runs/combined_full_200/comparison/*_z03.0_to_z00.5.npz`
and `runs/magic_unet/run2_server_outputs/{comsol_3_to_05_predictions,fista300_comsol_predictions}.npz`.
All values are fractions of the target norm for Bx / By / Bz unless marked.

| Quantity | L1_C1 | IC_4um |
|---|---|---|
| Relative L2, MAGIC-UNet run 2 | 0.239 / 0.264 / 0.285 (mean 0.263) | 0.263 / 0.299 / 0.325 (mean 0.296) |
| Relative L2, FISTA-300 | 0.295 / 0.287 / 0.353 (mean 0.312) | 0.264 / 0.246 / 0.318 (mean 0.276) |
| Amplitude slope, MAGIC-UNet | 0.885 / 0.886 / 0.860 | 0.890 / 0.907 / 0.861 |
| Amplitude slope, FISTA-300 | 0.886 / 0.888 / 0.846 | 0.923 / 0.936 / 0.865 |
| Cosine between the two methods' residuals | 0.63 / 0.64 / 0.74 | 0.64 / 0.46 / 0.62 |
| SNR = 1 crossing k_c (rad/µm) | 1.05 / 1.15 / 1.60 | 1.15 / 1.15 / 1.15 |
| Error below k_c, MAGIC-UNet | 0.145 / 0.162 / 0.181 | 0.142 / 0.207 / 0.196 |
| Error below k_c, FISTA-300 | 0.075 / 0.097 / 0.238 | 0.116 / 0.118 / 0.080 |
| Target energy, k_c to 2k_c | 0.337 / 0.331 / 0.200 | 0.289 / 0.284 / 0.332 |
| Error, k_c to 2k_c, MAGIC-UNet | 0.119 / 0.133 / 0.129 | 0.119 / 0.134 / 0.123 |
| Error, k_c to 2k_c, FISTA-300 | 0.244 / 0.218 / 0.194 | 0.152 / 0.144 / 0.212 |
| Target energy above 2k_c | 0.150 / 0.158 / 0.175 | 0.182 / 0.159 / 0.225 |
| Error above 2k_c, MAGIC-UNet | 0.148 / 0.161 / 0.179 | 0.187 / 0.170 / 0.228 |
| Error above 2k_c, FISTA-300 | 0.147 / 0.158 / 0.174 | 0.182 / 0.161 / 0.223 |
| Target energy above 2.2 rad/µm | 0.145 / 0.161 / 0.199 | 0.184 / 0.162 / 0.228 |
| Share of that band that is Maxwell-inconsistent (all components) | 0.70–0.73 | 0.73–0.74 |
| Maxwell-inconsistent share, whole target | 0.149 | 0.226 |
| Maxwell-inconsistent share, MAGIC-UNet output | 0.077 | 0.154 |
| Relative L2 against the target low-passed at 2.2 rad/µm, MAGIC-UNet | 0.191 / 0.210 / 0.205 | 0.185 / 0.247 / 0.235 |
| Same, FISTA-300 | 0.260 / 0.241 / 0.297 | 0.192 / 0.184 / 0.229 |
| Re-blurred prediction against the 3 µm input, MAGIC-UNet | 0.185 / 0.264 / 0.275 | 0.168 / 0.203 / 0.333 |
| Same, FISTA-300 | 0.095 / 0.099 / 0.095 | 0.114 / 0.103 / 0.156 |
| Same, COMSOL truth at 0.5 µm | 0.111 / 0.124 / 0.100 | 0.137 / 0.151 / 0.166 |
| Forward mismatch for k < 1 rad/µm, free space | 0.076 / 0.077 / 0.048 | 0.079 / 0.146 / 0.116 |
| Same, with k0 = 0.095 | 0.058 / 0.052 / 0.038 | 0.067 / 0.103 / 0.109 |
| MAGIC-UNet + one Tikhonov correction, μ = 0.0316 | 0.205 / 0.226 / 0.254 (mean 0.228) | 0.243 / 0.248 / 0.278 (mean 0.256) |
| Same, μ = 0.01 / μ = 0.003 (means) | 0.225 / 0.236 | 0.262 / 0.301 |
| FISTA below 1.0 rad/µm blended with MAGIC-UNet above (mean) | 0.226 | 0.252 |
| Height at which the MAGIC-UNet current best explains the input (0.25 µm grid) | 3.0 µm, residual 0.240 | 3.0 µm, residual 0.228 |

Caveats on these checks:

- Two configurations, one seed, saved predictions only. The correction and blend rows look at test data: I tried three
  μ values and four blend cut-offs. Only μ = 0.0316 was fixed beforehand (it is the value your Tikhonov baseline chose
  on simulation). M1 repeats this properly. *[M1/M2]: with μ chosen on simulation, the two-configuration mean is
  25.1%; on 15 pairs 24.3%.*
- *[M1/M2]: the 2.2 rad/µm noise edge in the rows above was my eyeball choice. The fixed rule of M2a gives 2.9 rad/µm;
  the low-passed rows here are superseded by `M1_M2_RESULTS.md` §M2b.*
- The Maxwell test assumes a single current sheet, free space and no truncation. Low-k inconsistency partly reflects
  the finite image, especially on IC_4um, whose field reaches the edges.
- The band edges use the same first-crossing rule as `information_budget.py`, so they inherit its dependence on
  COMSOL's mesh noise.

## Appendix B. Sources checked on the web

- CVPR 2027 dates: https://cvpr.thecvf.com/Conferences/2027/Dates
- ICLR 2027 call for papers: https://www.iclr.cc/Conferences/2027/CallForPapers
- ICCV 2027 (projected): https://www.opencurious.com/ai-conference-deadlines/iccv-2027
- ICML and NeurIPS 2027 (projected): https://www.opencurious.com/ai-conference-deadlines/icml-2027 , https://www.opencurious.com/ai-conference-deadlines/neurips-2027
- ICCP 2027: https://www.computer.org/conferences/cfp-ieee-iccp-2027 , https://www.myhuiban.com/conference/5608
- Reed et al., MAGIC-UNet: https://arxiv.org/abs/2407.14553 , https://www.doi.org/10.1103/PhysRevApplied.23.034035
- Dubois et al.: https://arxiv.org/abs/2207.13720
- Broadway et al.: https://www.doi.org/10.1103/PhysRevApplied.14.024076
- Midha et al.: https://arxiv.org/abs/2402.17781
- L1-curl: https://arxiv.org/html/2408.16550v1
- Setescak et al.: https://arxiv.org/abs/2602.17180 , https://journals.aps.org/prl/pdf/10.1103/nc5g-3yw9
- Physics-informed deep image prior: https://arxiv.org/abs/2604.23431
- Garsi et al.: https://arxiv.org/abs/2112.12242
- Oliver et al.: https://arxiv.org/abs/2202.08135
- Bathla et al.: https://arxiv.org/abs/2506.05491
- Neural networks for magnetisation textures: https://arxiv.org/abs/2412.19381
