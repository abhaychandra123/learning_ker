# Simulated fields from random current layouts

Training and COMSOL results on this data: [README_CSC_LISTA.md](README_CSC_LISTA.md).

31 October 2026. Restart at one standoff pair, 3 → 0.5 µm. This generates
training data only. No model has been trained on it, and no reconstruction
score exists yet.

## Method

`scripts/simulate_fields.py` draws closed Manhattan current paths for each sample:
a two-arm spiral, an uneven meander, or an array of 2–4 mirrored spirals in the image
quadrants, as in `data3`. Each sample gets a random width (1.6–3.2 µm), source offset
(0–0.2 µm), noise level and placement, with a fixed 600 µA current. Bx, By and Bz at
both heights come from that single current map:

```text
E(k, z) = (mu0 / 2) * exp(-k * (z + d))
Bx(k) = E * Kx(k)
By(k) = E * Ky(k)
Bz(k) = E * i(ky * Kx(k) - kx * Ky(k)) / k
```

`d` is the source offset. This is free-space decay, matching the solver's `exp(-kz)`
operator. `--decay-k0` can add an empirical COMSOL-like term, but it is off by default
and off in the current dataset. Each corner shares its current equally
between the two wires, and the divergent part of the drawn current is removed in
Fourier space. Speckle-like noise goes on the input only, at 1, 1.5, 2 or 2.5% of each
component's range, with a 0.38 µm exponential correlation and independently per
component (`--noise-corr-um 0` gives white noise). Optional Gaussian blur
(`--psf-sigma-um`) also applies to the input only and stays off for COMSOL work.

The normalisation scale comes from simulated training targets only. COMSOL never
enters generation, normalisation or training.

## Run

```bash
python3 scripts/simulate_fields.py --output runs/<fresh_name> --train 200 --validation 50
python3 scripts/plot_simulated.py --dataset runs/<fresh_name> --output runs/<fresh_report> --data3 <folder>
python3 -m unittest tests.test_simulate_fields
```

`--z-input` and `--z-target` select another pair. Output folders must be new.
The dataset stores recipes only (536 KB). `SimulatedDataset.iter_components`
regenerates arrays on demand, with the same `sharp`, `clean` and `blurred` keys as
`SyntheticDataset`. No trainer is connected yet.

## Current dataset

`runs/simulated_3_to_05_v3`, seed 0: 200 training and 50 validation layouts (750
component pairs), 80 spirals, 97 meanders and 73 arrays, scale 145.487 µT, free-space
decay. The split is by layout. Report: `runs/simulated_3_to_05_v3_report/index.html`.
It shares its layouts with `runs/simulated_3_to_05_v2`, which used `k0 = 0.095` and is
kept only for reference. `runs/simulated_3_to_05_v1` is the earlier white-noise version.

**The forward operator stays `exp(-kz)`.** A static field above its sources must be
harmonic, so `k0` cannot be real physics. The user chose to leave it out of the data.

## Evidence

Fifteen CPU tests pass, covering closed-form square-loop and strip fields, Maxwell
consistency in the export frame, free-space and `k0` decay, speckle correlation,
closed and clear paths, mirrored arrays, repeatability, input-only noise and version
1 recipes.

Real coils rebuilt from the traced L1_C1 centreline, compared with COMSOL. Each cell
gives relative L2 for Bx / By / Bz:

| Coils, height | Free space | With `k0` | Slope with `k0` |
|---|---|---|---|
| L1_C1, 0.5 µm | 16.8 / 18.8 / 23.0% | unchanged | 1.007 / 1.008 / 1.010 |
| L1_C1, 3 µm | 12.7 / 13.8 / 25.8% | 11.2 / 12.1 / 21.5% | 1.014 / 1.020 / 0.946 |
| L1_C1, 10 µm | 63.7 / 64.1 / 139.7% | 15.1 / 12.6 / 64.9% | 0.936 / 0.942 / 0.639 |
| C1–C4, 3 µm | 13.2 / 14.0 / 27.3% | 11.5 / 12.4 / 22.5% | 1.012 / 1.024 / 0.940 |
| C1–C4, 10 µm | 61.6 / 63.1 / 146.0% | 13.5 / 11.4 / 67.2% | 0.952 / 0.956 / 0.624 |

The width (2.08 µm), offset (0.08 µm) and corner handling were set on L1_C1, and `k0`
was fitted on C1–C4 Bz, so this is a regression check and not independent validation.
The IC geometry has not been rebuilt this way.

## Known mismatches

- **Large standoff.** COMSOL decays faster than free-space physics, and the simulated
  data deliberately does not copy this. The cause is unknown. At 3 µm the mismatch is
  small; by 10 µm it is large, so do not expect transfer at large standoffs.
- **Mesh speckle.** Simulated inputs now match COMSOL's spectrum, including the
  high-frequency floor, for all five configurations. Targets stay clean on purpose,
  so they sit far below COMSOL beyond about 0.3 cycles/µm.
- **Peaks and RMS.** Every COMSOL peak lies above the simulated 5–95% range, and
  COMSOL 3 µm Bz reaches about 105 µT against an RMS of about 13 µT in the denser
  configurations. These spikes are not modelled. Dense multi-coil RMS sits slightly
  above the simulated range.
- **Simulated scores.** Inputs come from a simplified model of COMSOL, so simulated
  scores will be optimistic. Only held-out COMSOL scores count.
- **Prior knowledge.** The width range, speckle level and layout families all
  came from the COMSOL configurations used for testing, though no reconstruction
  score informed them. State this in any write-up.

## What `data3` taught us (1 November 2026)

`data3` holds three COMSOL configurations (C1+C2, C1+C2+C3, C1+C2+C3+C4), each at
20 heights from 0.5 to 12.5 µm. With L1_C1 and IC_4um this gives five configurations
and 15 component pairs at 3 → 0.5 µm. They contain only two wire shapes: C2, C3 and
C4 are mirror copies of the C1 spiral (correlation 0.97–0.98), all in one plane, and
subtracting configurations isolates each coil. The speckle and array layouts below
are in the generator; the decay law is optional and off. The analyses themselves were exploratory
scripts and are not in the repository.

- **Side walls.** The field component pointing into the image edge is about 100×
  smaller at the top and bottom edges and 10–30× smaller at the sides than 10 µm
  inside, so the 140 µm window appears bounded by walls. Modelling them with mirror
  currents barely helped (10 µm Bz error 140 → 128%). A top wall did not help, and
  each cosine mode decays as a straight exponential from 0.5 to 12.5 µm.
- **Faster decay.** Modes decay faster than a harmonic field in a walled box can
  (mode (1,3): 0.122 against 0.071 per µm). The cause is unknown. It needs the COMSOL
  mesh, domain and material settings.
- **Empirical decay.** `exp(-sqrt(k^2 + k0^2) * dz)` with k in rad/µm, fitted on the
  C1–C4 Bz only (`k0 = 0.095`), carries over to the other exports:

| Case | Bx error | By error | Bz error |
|---|---|---|---|
| Simulated L1_C1 vs COMSOL, 10 µm | 64 → 15% | 64 → 13% | 140 → 65% |
| Simulated L1_C1 vs COMSOL, 3 µm | 13 → 11% | 14 → 12% | 26 → 22% |
| COMSOL L1_C1 forward, 0.5 → 10 µm | 56 → 12% | 57 → 12% | 40 → 14% |
| COMSOL IC_4um forward, 0.5 → 10 µm | 49 → 24% | 73 → 48% | 36 → 22% |
| COMSOL IC_4um forward, 0.5 → 3 µm | 14 → 13% | 15 → 13% | 17 → 17% |

- **Speckle.** It repeats between exports where the mesh matches (`C1_C2 - L1_C1`
  leaves about 0.05 µT), so it is not random per run. Mirror pairs estimate it at
  1.4–1.9% of range for Bx/By and 2–3.4% for Bz, with 15–19% relative L2 at 0.5 µm and
  about 10% at 3 µm. It fades with height at about the same rate as the field, and it
  is no stronger over the wires. Its correlation falls to 0.27 by 0.5 µm and about 0
  by 1 µm, and it is uncorrelated between 0.5 and 3 µm. These figures assume the
  mirror copies differ only by speckle, so treat them as upper estimates.
- **Score floor.** Clean-model error against COMSOL at 0.5 µm cannot be expected to
  fall below about 15–19% relative L2. The simulator's 17–23% is already near it.

## Data questions

- COMSOL Bx/By only satisfy Maxwell against Bz when the image is rotated 90° from
  the component labels (10.9% residual, about 100% as labelled). `convert.py` is
  not the likely cause: a reshape error gives a transpose, and that scored worst.
  The repo's quarter-turn rotation preserves this residual.
- The user confirmed on 31 October 2026 that IC_2um and IC_4um are the same coil.
  Both fit about 2.0–2.4 µm wire width at matching positions, and IC_4um propagated
  from 0.5 to 1 µm matches IC_2um with slope 1.02–1.04 and 15–20% relative L2. Count
  them as one test coil.
