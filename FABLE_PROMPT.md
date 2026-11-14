# Prompt for a deep project review (paste into a Fable session opened in this repo)

You are a senior researcher in computational imaging / inverse problems with publication experience at CVPR, ICCV, NeurIPS and ICLR (also Nature Communications / Physical Review Applied for the physics side). I need a deep, critical, evidence-based review of my project, not a summary and not encouragement. Disagree with me where the evidence says so.

## Phase 1: Understand before you opine (do this first, read the actual files)

1. Read `PROJECT_CONTEXT.md` (newest entries first), `RESEARCH_BEYOND_ISTA.md`, `README_MAGIC_UNET.md`, `README_CSC_LISTA.md`, `README_SIMULATED_FIELDS.md`, `README_WIENER_ORACLE.md`, `README_GEOMETRY_CAPACITY.md`, `README_SYNTHETIC_*.md`, `LISTA_FISTA.md`, `ALISTA.md`, `info.md`, `res.md`, and the booklet in `booklet/chapters/*.tex` (especially 09 results, 13 defence, 15 floor, 16 followups, 17 question, 18-19 magic).
2. Read the code that defines the physics and the models: `scripts/fields.py`, `propagator.py`, `kernel*.py`, `fista.py`, `fista_net.py`, `csc_lista.py`, `lista.py`, `alista*.py`, `simulate_fields.py`, `magic_data.py`, `magic_unet.py`, `information_budget.py`, `wiener_oracle.py`, and the tests. Inspect the data folders (`data/`, `runs/`, `output/`, `cache/`) and the saved metrics/logs.
3. Before giving any opinion, write back to me, in your own words:
   - the exact problem statement (inputs, outputs, forward operator, what is known/unknown at test time, noise model, metric and how it is computed),
   - exactly what data we have (real COMSOL pairs, simulated data, how many independent geometries really exist, what leaks or overlaps between train and test),
   - a table of EVERY method tried (Wiener, ISTA/FISTA, LISTA, ALISTA, ConvLISTA, CSC-LISTA, FISTA-Net, MAGIC-UNet, anything else you find) with data used, standoff setting, test set and the headline number, flagged by whether the comparison is apples-to-apples.
   If you find contradictions between documents, or numbers that cannot be reproduced from the logs, list them.

## Phase 2: Diagnose

- What has gone right, what has gone wrong, and why. Separate (a) fundamental information limits (the `information_budget.py` floor, source/height ambiguity), (b) modelling errors (forward operator exp(-kz) vs the empirical decay we saw, mismatch between our simulator and COMSOL), (c) data problems (only two wire shapes, mirrored configs counted as separate, simulated-to-real gap), (d) optimisation or architecture problems, (e) evaluation problems (metric choice, held-out design, noise-floor accounting).
- For each result that looked bad or surprising (for example the CSC-LISTA drop on 8000 layouts, learned methods only 4 points from the Wiener oracle at 3 µm, FISTA-300 still beating learned models), say which explanation is best supported by the evidence and what single cheap experiment would confirm or kill it.
- Say plainly whether I am moving in the right direction, and what I should STOP doing.

## Phase 3: Find the nuances and the gaps

Look for the small, specific things a reviewer would find or a novel contribution could hide in:
- identifiability (is standoff + near-plane field recoverable at all, under what prior),
- what the learned prior is actually learning (current-domain vs field-domain prior, spectral content outside the measurable band),
- generalisation axes: standoff, geometry, FOV, pixel pitch, noise, unseen wire shapes,
- calibration/uncertainty (reviewers of this problem will ask for it),
- physics errors that a learned model could absorb or hide.
Use web search to check the literature (QDM / NV magnetometry reconstruction, current-density inversion from magnetic images, magnetic field downward continuation, deep unfolding and USRNet-style generalisable unrolled solvers, MAGIC-UNet of Reed et al. 2024, neural operators, diffusion / score-based priors for inverse problems, plug-and-play, implicit neural representations, equivariant or Fourier-domain networks). Only cite papers you actually verified; mark anything unverified as such. Tell me what is already published, so I do not claim novelty that does not exist.

## Phase 4: Recommend

1. The best-suited architecture or method family for THIS problem (not generically), with the reasoning tied to our data limits and physics. Compare at least 3 candidates (for example unrolled solver with learned current-domain prior, physics-informed Fourier-domain network, diffusion/score prior with measurement-consistent sampling, neural operator, hybrid with joint standoff estimation) and rank them by expected gain, risk, and compute (we have one RTX A6000 server; training never runs on the Mac).
2. A concrete roadmap in ordered milestones with go/no-go checks, each with the one experiment, the expected outcome, the failure signature, and the time cost. Mark the cheapest decisive experiment to run first.
3. Publication strategy: what is the single clear contribution and claim, what the title/abstract-level story is, what baselines, datasets, ablations and metrics a reviewer will demand, and what is currently missing. Give me an honest verdict on which venues fit (CVPR 2027, ICCV/ECCV, NeurIPS, ICLR, IEEE TCI, Physical Review Applied, Nature Communications, Optica, Review of Scientific Instruments, a workshop) with deadlines you can verify, and the minimum result needed to be competitive at each. If CVPR 2027 is not realistic, say so and tell me what is.
4. A list of the top 5 risks that could sink the paper, and what to do about each.

## Rules

- Do not run training on the Mac. Tiny smoke checks and reading are fine. Do not commit.
- Ground every claim in a file, number or citation. Say "I did not verify this" when true.
- Do not invent results. If the answer needs an experiment, name it rather than guessing.
- Write the final report to `FABLE_REVIEW.md` in the repo root, with a one-page executive summary at the top, then the sections above. Keep the chat reply short and point to the file.
