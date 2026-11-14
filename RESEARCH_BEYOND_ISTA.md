# Research landscape: unknown standoff, all standoffs, unseen geometries

7 November 2026. Literature survey plus three cheap checks on local COMSOL data.
Nothing new is implemented. Replaces the earlier single-standoff version of this file.

**Research question.** Can a physics-based learned inverse estimate unknown standoff and
reconstruct near-plane QDM magnetic fields consistently across unseen geometries, FOVs and
sampling scales? It must also explain how it handles source–height ambiguity, with
uncertainty or competing hypotheses.

## Short answer

- **ISTA is not the problem; the soft-threshold prior is.** Unrolled methods are the right
  *skeleton* for this question: the physics step `exp(−k·Δz)` is written in physical units,
  so one network can take a different height, pitch and FOV at every input. The strongest
  published "one model for all kernels, scales and noise levels" method, USRNet, is itself
  an unrolled solver. What has to change is the prior block (soft-threshold on the field →
  a learned or current-domain prior) and an added height-estimation loop.
- **Height and near-plane field are one problem.** A source at depth `h` and a sharper
  source at depth `h + δ` give exactly the same measurement (§1). Only a source prior can
  separate them. The same prior also fills in the frequencies the data cannot see. So
  the main research effort goes into a strong, physically correct source prior.
- **Gap.** In NV/QDM work, standoff is fitted with parametric models on a known feature
  (a wire, an edge, a dipole), or jointly with a micromagnetic-energy prior for
  magnetisation (Setescak et al. 2026). We found no learned method that estimates standoff
  jointly with reconstruction, reports competing height hypotheses, or is tested across
  pitch and FOV. Blind-deblurring literature (DAN, IKC, BlindDPS, Fast Diffusion EM) has
  the machinery; nobody has applied it to magnetic imaging. Our blur family has only one
  parameter, so the method can be more rigorous than generic blind deblurring.

## 1. The physics behind everything: an exact degeneracy

For a thin current sheet (stream function `g`) at depth `h` below the sensor, every
component is `B(k) = (μ0/2)·e^{−kh}·K(k)·g(k)`, with `K` a fixed angular factor. Then

```text
source g at depth h  ≡  source g·e^{kδ} at depth h + δ      (identical B at the sensor)
```

The Bx/By/Bz components share `e^{−kh}`, so vector data does not break this. The two
hypotheses predict *different* fields at the reference plane (a fixed height above the
source), so a wrong height directly corrupts the reconstruction. Consequences:

1. **Field-only height estimation is impossible.** A prior must say what sources look like
   (sharp wire edges, piecewise-constant current, divergence-free, known current scale).
   The data then give roughly an *upper bound* on depth: deeper means a sharper source,
   which eventually needs ringing or negative current.
2. **"Standoff" must be defined as the effective sheet depth** (NV layer thickness, conductor
   thickness, optical blur all add to it). Report physical gap and effective depth separately.
3. **Where the prior is weak, the honest output is a set of height hypotheses, each with its
   own field.** This is the guide's A5 candidate bank, and the direct answer to the
   "how do you handle ambiguity" question.

## 2. Information budget at every standoff (new check)

All 17 local pairs, 128 px padding, free-space operator. "Effective noise" = COMSOL residual
`B_in − H·X_target` (mesh noise plus free-space mismatch, **not** sensor noise; real QDM
noise moves these numbers). Floor = error of the truth with everything above the
SNR = 1 crossing removed, mean of Bx/By/Bz. This is roughly what the best linear filter
reaches; anything better must come from the prior.

| Pair | SNR = 1 at k (rad/µm, Bx/By/Bz) | Lost-band floor |
|---|---|---:|
| L1_C1 1 → 0.5 | 2.05 / 1.90 / 2.10 | 17.5% |
| L1_C1 2 → 0.5 | 1.60 / 1.70 / 1.75 | 21.0% |
| L1_C1 3 → 0.5 | 1.05 / 1.15 / 1.60 | 33.4% |
| L1_C1 4 → 0.5 | 0.90 / 1.00 / 1.00 | 39.8% |
| L1_C1 5 → 0.5 | 0.85 / 0.85 / 0.90 | 41.3% |
| L1_C1 6 → 0.5 | 0.80 / 0.85 / 0.80 | 42.6% |
| L1_C1 7 → 0.5 | 0.75 / 0.75 / 0.75 | 44.8% |
| L1_C1 8 → 0.5 | 0.70 / 0.70 / 0.70 | 47.4% |
| L1_C1 9 → 0.5 | 0.60 / 0.65 / 0.65 | 53.6% |
| L1_C1 10 → 0.5 | 0.60 / 0.65 / 0.60 | 56.0% |
| IC_4um 3 → 0.5 | 1.15 / 1.15 / 1.15 | 35.6% |
| IC_4um 4 → 0.5 | 1.00 / 1.00 / 0.90 | 42.6% |
| IC_4um 5 → 0.5 | 0.85 / 0.80 / 0.65 | 52.4% |
| IC_4um 6 → 0.5 | 0.75 / 0.70 / 0.60 | 58.2% |
| IC_4um 7 → 0.5 | 0.55 / 0.60 / 0.50 | 65.4% |
| IC_4um 10 → 0.5 | 0.45 / 0.45 / 0.35 | 73.5% |
| IC_2um 2 → 1 | 1.40 / 1.30 / 1.50 | 21.2% |

At 3 → 0.5 µm the floor (34.5% over both coils) matches the Wiener oracle (34.4%), and
FISTA-300 (30.5%), CSC-LISTA-64 (33.8%) and FISTA-Net (33.3%) all sit within 4 points of
it. **From 4 µm up, over 40–70% of the target is prior-filled.** At large standoff any
method is mostly drawing wires from its prior, so uncertainty reporting is required, not
optional. Script: `python3 scripts/information_budget.py` (also prints the blind estimates of §3; the oracle-band estimates were an inline check, band `k ≤ 3.5/z_true`).

## 3. The ambiguity on our own data (demonstration, not an estimator)

Fitting the decay slope of the Bz radial power spectrum gives a naive height estimate.

- **With a fitting band chosen using the true height** (oracle): IC_4um 3–7 µm comes out
  0.2–0.6 µm too high, and both coils at 0.5 µm about 1.1 µm too high (the source's own
  thickness/width). L1_C1 above 4 µm comes out **1–5 µm too low** (10 µm reads as 5.2). The
  likely cause is the coil's own spectrum at low k: a closed loop's Bz power rises with k
  near zero, which fights the exponential once only k < 0.5 is usable. That is the
  degeneracy of §1 in practice: the estimate depends on what the source looks like.
- **With the band chosen from the data alone** (power above 100× the high-k floor): every
  height above 3 µm reads as 0.9–2 µm. COMSOL mesh noise is not white, so the band includes
  noise. Blind spectral estimation fails outright.

So height must come from a source prior plus the reconstruction, not from the spectrum.

## 4. Literature mapped to each part of the question

Provenance: **[read]** full text read; **[summary]** model summary of truncated text,
check before quoting numbers; **[abstract]** abstract or search summary only; **[std]** standard
reference from memory, not re-fetched this session.

### Magnetic microscopy reconstruction

| Work | What it does | What it gives us |
|---|---|---|
| Bathla et al. 2025, arXiv 2506.05491 (our group) [read, partly] | Measured coil-map dictionary + LASSO, 3D/temporal, ~7 µm standoff | The group's lineage; our CSC work extends it |
| MAGIC-UNet, arXiv 2407.14553 [read + summary] | U-Net, (Bx, By, Bz) → (Jx, Jy); 153,600 Biot–Savart wire layouts; **fixed** ~50 µm standoff, 64×64, white noise | Strongest learned current-latent method. Standoff not an input; stated weaknesses: correlated noise, multiple planes, large standoff |
| Dubois et al. 2022, PRApplied 18 064076 [abstract] | Untrained network outputs the source; exact forward model is the loss | No training data, so no simulator gap; geometry-agnostic by construction |
| PI-DIP, arXiv 2604.23431 [abstract] | Physics-informed deep image prior, scanning NV; calls standoff "a critical source of uncertainty" | Supports making height a first-class unknown |
| Setescak et al. 2026, PRL, arXiv 2602.17180 [abstract] | Micromagnetic energy as prior, FFT upward continuation, **jointly fits sensor distance** | Closest prior art for joint height + reconstruction. Magnetisation, not current; not learned; no uncertainty |
| L1-curl, div-free wavelets, arXiv 2408.16550 [read + summary] | `min ‖BJ − B̂‖² + τ‖curl J‖₁`, ADMM; tested at 1–10 µm standoff | Published, training-free current prior |

### How NV practice estimates standoff today

Parametric fits on a known feature [abstract]: stripe line profiles (arXiv 2503.04244),
two perpendicular film edges (arXiv 2109.14388), dipole fits to 58 micromagnets giving 3.6 µm
mean QDM standoff (arXiv 2002.07952), line-profile fits because full 2D fits were too costly
(arXiv 2407.15319). **This is the experimental calibration route:** image a wire of known
width on the real QDM, fit its standoff, and use it as ground truth for the learned estimator.

### Unified and blind inverse methods (the machinery)

| Part of the question | Published method | Mapping |
|---|---|---|
| One model across height, pitch, noise (known height) | USRNet (Zhang, Van Gool, Timofte, CVPR 2020) [std]; DPIR (TPAMI 2021) [std] | Unrolled half-quadratic splitting, conditioned on kernel, scale factor and noise. **Same shape as guide A2**; required baseline |
| Joint height + reconstruction | IKC (Gu et al., CVPR 2019) [std]; DAN (Huang et al., NeurIPS 2020) [std] | Alternate kernel estimation and restoration: guide A4 |
| Joint estimation with a generative prior | BlindDPS (Chung et al., CVPR 2023) [abstract; code github.com/BlindDPS/blind-dps]; Fast Diffusion EM (Laroche et al., 2024) [std]; GibbsDDRM (Murata et al., ICML 2023) [std] | Diffusion prior on the image, parametric operator estimated alongside. With one height parameter, EM or a profile over `h` is cheap |
| Operator shift at test time | Model adaptation (Gilton, Ongie, Willett 2021, arXiv 2012.00139) [abstract] | Adapt a trained network to a changed forward model without new labels; relevant to COMSOL vs free-space mismatch and to real QDM |
| Ambiguity vs estimation uncertainty | "Separating Intrinsic Ambiguity from Estimation Uncertainty in Deep Generative Models for Linear Inverse Problems" (ICML 2026) [abstract] | Exactly the reviewer's question; demonstrated on MRI/EEG, not magnetic data |
| Calibrated uncertainty | Conformal image-to-image regression (Angelopoulos et al., ICML 2022) [std] | Distribution-free intervals on the reconstruction and on `h` |
| Self-supervised from real data only | Scale-equivariant imaging (Scanvic et al., arXiv 2312.11232) [abstract] | Learns lost high frequencies from blurred data alone; route for the real QDM archive |

### Pitch and FOV

The Fourier physics step in rad/µm is exact on any grid. **What breaks across pitch is the
learned part:** convolution kernels are sized in pixels. Options, in order of simplicity:
resample inputs to a fixed physical pitch; condition layers on pitch (FiLM, as in guide A2);
or use a discretisation-invariant prior such as a Fourier neural operator. The earlier
"FNO is not worth our time" line was written for one fixed pitch and is withdrawn. FOV
changes mainly the padding and boundary handling and the lowest resolved k.

### Geophysics (same mathematics)

D-Unet supervised downward continuation (Front. Earth Sci. 2022) [abstract]; physics-trained
network with upward continuation in the loss (arXiv 2502.05190) [abstract]; depth-to-source
estimation (Euler deconvolution, CNN depth estimators) [abstract] treats non-uniqueness as
sensitivity, not as multiple hypotheses.

## 5. A candidate "distinct inference method"

Profile inference over height with a current-domain prior:

1. For each candidate effective depth `h_j` on a grid, run one shared reconstructor
   conditioned on `h_j`, pitch and noise (USRNet-style unrolled, prior on current or stream
   function, exact physics step).
2. Score each `(h_j, x_j)` by data fit plus prior. A normalised posterior is available only
   with a generative prior (diffusion/BlindDPS-style); otherwise call it a predictive weight
   and calibrate it on held-out data (guide A5's warning).
3. Output the height posterior or set, and one field per supported hypothesis. Collapse to
   one answer only when the posterior is unimodal and narrow.
4. Refine continuously around the best `h` (DAN-style alternation or a gradient on `h`).

The professor wants published, unmodified methods; the question asks for a distinct
inference method. **The contribution can be new, but every component and every baseline
should be published.** This design does that: USRNet/DAN/BlindDPS components, published
current priors, new combination plus the ambiguity treatment.

## 6. Test plan

1. **Information budget** (done above for local data; rerun on the server with data3 and the
   larger COMSOL database).
2. **Identifiability, go/no-go for the whole project (½–1 day, no training).** On COMSOL at
   known heights, fit a regularised sheet current (TV on `g` or L1-curl) at a grid of
   candidate `h`. Does the best total objective land on the true effective depth, and how wide
   is the valley? If no prior gives a minimum near the truth, joint estimation cannot work and
   the paper becomes "height must be calibrated externally", which is still a result.
   Apply the repo's quarter-turn rotation before fitting Bx/By with Bz.
3. **Known-height conditioned single model** across all standoffs (USRNet-style). Required
   baseline: shows what is lost when height is unknown.
4. **Joint estimation:** DAN-style alternation, or the profile in §5. Compare with a
   two-stage baseline (height regressor, then step 3) as in guide A3.
5. **Generalisation splits:** leave-geometry-out, FOV by cropping, coarser pitch by area-averaged
   downsampling of the 0.1 µm grid, noise levels. Finer pitch needs new COMSOL exports.
6. **Uncertainty:** candidate bank versus BlindDPS-style sampling; check calibration (coverage
   of the true `h` and of the field) on held-out data.
7. **Experimental QDM:** calibrate standoff on a wire of known width (NV practice above), then
   test the estimator on other structures in the same session.

## 7. Data limits and ceilings

- Locally there are only **two wire shapes** (IC and C1). Unseen-geometry claims need the
  server data (data3 and the larger COMSOL database). Training geometries come from the
  simulator; COMSOL stays held-out test data.
- All local data are 0.1 µm pitch and one FOV. Crops and downsampling test smaller FOV and
  coarser pitch honestly; nothing local tests finer pitch.
- COMSOL is not fully self-consistent: Bx/By disagree with Bz by 10.9% after rotation, and a
  free-space wire model scored 19–25% against COMSOL at 0.5 µm. Any physics-consistent method
  is bounded by this on COMSOL, so also report held-out simulated data.

## Sources

- Bathla et al., arXiv 2506.05491 — https://arxiv.org/abs/2506.05491
- MAGIC-UNet, arXiv 2407.14553 — https://arxiv.org/abs/2407.14553
- Dubois et al., arXiv 2207.13720 — https://arxiv.org/abs/2207.13720
- PI-DIP, arXiv 2604.23431 — https://arxiv.org/html/2604.23431v1
- Setescak et al., arXiv 2602.17180 — https://arxiv.org/abs/2602.17180
- L1-curl div-free wavelets, arXiv 2408.16550 — https://arxiv.org/abs/2408.16550
- Fourier vs Bayesian QDM current reconstruction, arXiv 2402.17781 — https://arxiv.org/abs/2402.17781
- Standoff in scanning NV, arXiv 2503.04244 — https://arxiv.org/abs/2503.04244
- Integrated widefield probe, arXiv 2109.14388 — https://arxiv.org/pdf/2109.14388
- Micromagnet PUF with QDM, arXiv 2002.07952 — https://arxiv.org/pdf/2002.07952
- Compact QDM, arXiv 2407.15319 — https://arxiv.org/pdf/2407.15319
- BlindDPS, arXiv 2211.10656 — https://arxiv.org/abs/2211.10656
- Model adaptation, arXiv 2012.00139 — https://arxiv.org/abs/2012.00139
- Intrinsic ambiguity vs estimation uncertainty, ICML 2026 — https://icml.cc/virtual/2026/70403
- Scale-equivariant imaging, arXiv 2312.11232 — https://arxiv.org/abs/2312.11232
- D-Unet downward continuation — https://www.frontiersin.org/articles/10.3389/feart.2022.897055
- Physics-trained downward continuation, arXiv 2502.05190 — https://arxiv.org/abs/2502.05190
- Standard references, not re-fetched: USRNet (CVPR 2020), DPIR (TPAMI 2021), IKC (CVPR 2019),
  DAN (NeurIPS 2020), Fast Diffusion EM (2024), GibbsDDRM (ICML 2023), conformal
  image-to-image regression (ICML 2022), FNO (ICLR 2021).
