# Is Wiener filtering a good method for COMSOL standoff reconstruction?

5 November 2026. Script: `scripts/wiener_oracle.py`. Tests: `python3 -m unittest tests.test_wiener_oracle`
(the Tikhonov-equivalence test needs torch, so it runs only on the server).

The existing "Wiener" baseline (`csc_lista.Dataless`) is a Tikhonov filter `H / (H² + μ)`
with one scalar μ = 0.0316, chosen on simulated validation (49.3% COMSOL at 3 → 0.5 µm).
This experiment estimates Wiener filters from COMSOL pairs themselves, one group per
(input height, target height). **In-sample scores are an oracle upper bound** for linear
shift-invariant filters, not a deployable result. Held-out scores fit on other
configurations and/or other heights.

## Bottom line (local data, 3 → 0.5 µm)

| Filter | Relative L2 | vs existing |
|---|---:|---:|
| Existing Tikhonov, μ = 0.0316 (simulated noise) | 49.7% | — |
| Real Wiener H S_x/(H² S_x + S_n), fitted on the other configuration | 37.8% | −12 points |
| Real Wiener, other configuration and other heights | 36.1% | −14 points |
| Real Wiener, oracle (upper bound) | 34.4% | −15 points |
| Scalar μ, fitted on the other configuration | 37.3% | −12 points |
| FISTA-300 / FISTA-Net (simulated-only tuning, for reference) | 30.5% / 33.3% | |

- **The real Wiener filter works: about 12–14 points better held out.** Almost all of that
  comes from using COMSOL's real noise level (the old μ came from simulated noise about 18×
  too strong). A scalar μ fitted on the same COMSOL data gets the same score.
- **The frequency shape of SNR(k) adds about 0–0.5 points from 3 µm up,** because SNR drops
  from high to below 1 within a narrow band, so only the cut-off position matters. It adds
  2–4 points at 1–2 µm (in-sample, one configuration only).
- **It does not change the ranking:** held-out Wiener (36–38%) and even the oracle (34.4%)
  stay behind FISTA-300 and FISTA-Net.
- **Fairness caveat:** the old μ never saw COMSOL; the held-out Wiener filter saw COMSOL data
  from another configuration. Part of the gain is access to real COMSOL statistics, which a
  simulated-only method does not have.
- Only two configurations exist at 3 µm locally. Rerun on the server with data3 and the
  extra COMSOL data before quoting these numbers.

## Method

Model `B_z = H_z X + n_z`, `H_z = exp(−2π|f|(z − z_target))`, same 128 px zero padding as
the baseline. Spectra are radial averages, and each image is weighted by 1/‖X‖², so fits
target the mean squared relative L2.

| Filter | Definition |
|---|---|
| `tikhonov` | existing μ = 0.0316 |
| `tikhonov_best_mu` | scalar μ chosen on the fitting pairs |
| `cross` | E[X B*] / E\|B\|² (Wiener–Hopf, needs no noise model) |
| `model` | H S_x / (H² S_x + S_n), S_x = E\|X\|², S_n = E\|B − H X\|², this group only |
| `shared` | model form; S_x from every truth image at this target height (all input heights share one truth) |
| `shared_trend` | shared form; S_n at this height interpolated (log-linear) from the other input heights |
| `cross_2d` | per-frequency fit; memorises few pairs, ignore |

Protocols: `oracle` (in-sample), `loco` (leave one configuration out), `loso` (leave one
wire shape out, only when it differs from loco), `loho` (leave this input height out,
configurations in-sample), `loho_loco` (leave this height **and** this configuration out:
the most honest test).

## Run

```bash
# discovers every <name>_z<in>_to_z<target>.npz in runs/combined_full_200/comparison,
# plus data/data3 at 3 -> 0.5 if that folder exists
python3 scripts/wiener_oracle.py --output runs/wiener_standoff_all
# or any set of pairs
python3 scripts/wiener_oracle.py --output runs/wiener_standoff_manifest --manifest pairs.csv
# only score some groups (spectra for sharing still use every pair)
python3 scripts/wiener_oracle.py --output runs/wiener_3um --groups 3:0.5
```

Manifest columns: `name, shape, z_input, z_target, sharp, blurred`. Paths are relative to
the manifest. `.csv` files are COMSOL exports in Tesla (columns 3–5); `.npz` files use the
`sharp`/`blurred` arrays in µT. `name` must be the same for one configuration at every
height (this is how S_n trends and leave-height-out find it). Give mirrored or duplicate
coils the same `shape`. Outputs: `table.md`, `scores.csv`, `results.json`, `spectra.npz`,
`standoff_target_<z>um.png`.

## Local result: 17 pairs (L1_C1 at 1–10 µm, IC_4um at 3–7 and 10 µm, IC_2um 2 → 1 µm)

`runs/wiener_standoff_local` (~90 s). Mean relative L2 (%):

| z_in → target | configs | best μ | SNR = 1 at (cyc/µm) | existing μ | best μ | Wiener oracle | Wiener, other config | Wiener, other config + height |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 → 0.5 | 1 | 0.056 | 0.33 | 24.7 | 21.7 | 18.0 | — | — |
| 2 → 0.5 | 1 | 0.01 | 0.27 | 33.1 | 26.3 | 23.8 | — | — |
| 3 → 0.5 | 2 | 0.0018 | 0.19 | 49.7 | 34.8 | 34.4 | 37.8 | 36.1 |
| 4 → 0.5 | 2 | 0.001 | 0.16 | 63.3 | 41.2 | 41.2 | 42.6 | 42.5 |
| 5 → 0.5 | 2 | 0.00056 | 0.13 | 74.1 | 45.9 | 45.9 | 47.3 | 47.3 |
| 6 → 0.5 | 2 | 0.00018 | 0.12 | 81.4 | 50.4 | 51.0 | 54.3 | 54.2 |
| 7 → 0.5 | 2 | 0.0001 | 0.11 | 86.2 | 55.8 | 56.7 | 63.3 | 63.3 |
| 8 → 0.5 | 1 | 3.2e-05 | 0.12 | 94.7 | 50.6 | 50.3 | — | — |
| 9 → 0.5 | 1 | 1.8e-05 | 0.10 | 96.5 | 55.2 | 55.3 | — | — |
| 10 → 0.5 | 2 | 5.6e-05 | 0.08 | 93.2 | 71.8 | 72.8 | 88.0 | 87.7 |
| 2 → 1 | 1 | 0.032 | 0.24 | 23.7 | 23.7 | 21.5 | — | — |

Rows with one configuration (L1_C1 at 1, 2, 8, 9; IC_2um) are not comparable with
two-configuration rows; IC_4um is harder than L1_C1 at every height (see the plot).

Findings:

- **The existing μ is badly tuned at every height** and gets worse with distance (it was
  fixed on simulated 3 µm noise). The best μ falls about 1000× from 1 to 10 µm.
- **The frequency-dependent Wiener filter only helps close to the target**: 18.0 vs 21.7%
  at 1 µm, 23.8 vs 26.3% at 2 µm, 21.5 vs 23.7% at 2 → 1 µm. From 3 µm on, a well chosen
  scalar μ matches it. Where the input SNR is high over a wide band, the filter's shape
  matters; where SNR collapses quickly, only the cut-off position matters.
- **The linear ceiling rises steeply with standoff**: the SNR = 1 frequency falls from
  0.33 to 0.08 cycles/µm, and the oracle error from 18% to 73%.
- **S_n changes smoothly with height**: interpolating it from other heights (`loho`) is
  within ~0.5 points of the oracle. A shared truth spectrum S_x changes nothing.
- **Transfer to an unseen coil gets worse with height**: the gap between held-out and oracle
  is ~2–3 points at 3–5 µm and ~15 points at 10 µm. Unseen coil plus unseen height
  (`loho_loco`) is no worse than unseen coil alone.
- **S_n falls with height at most frequencies** (smoother COMSOL solution higher up), but
  **rises with height at the lowest frequencies** (below ~0.05 cycles/µm). That low-k
  residual growth is what an unmodelled extra decay such as the empirical k0 would produce.
