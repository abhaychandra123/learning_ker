# What the COMSOL model files show

14 November 2026. Analysis of `COMSOL_params/` (sent by Hari on 13 November): full and intermediate reports for the
L1_C1 model at two mesh densities, new L1_C1 exports at 0.5 and 3 µm for each mesh, mesh screenshots, and the coil
geometry `Gemetry_C1.dxf`. All numbers below were computed by the scripts in `scripts/comsol_check/` (commands at the
end). Nothing was trained. The simulator and the operators were not changed.

## Update, 17 November 2026: cause confirmed (section 9)

Ritaban re-solved L1_C1 (Extra Fine mesh) with the iterative solver at **relative tolerance 1e-8**
(`COMSOL_params_v2/Extra_Fine_Mesh/FGMRES_RT1e-8/`, 0.5, 3 and 10 µm). The direct solver could not be used: it fails
on this model with "relative error 1.1e+04, returned solution is not converged" (screenshot in the same folder), the
signature of a singular system (vector-potential formulation without gauge fixing). In the re-solve:

- **k₀ is gone.** Every band decays at rate k, as free space requires (k = 0.048 → 0.048, k = 0.109 → 0.103 rad/µm,
  3 → 10 µm), where the original solve gave k₀ ≈ 0.10.
- **Free-space propagation now agrees with COMSOL:** COMSOL's 0.5 µm field propagated to 10 µm misses its 10 µm plane
  by **3.8 / 4.5 / 3.1%** (original: 57 / 52 / 41%).
- **Our exact-coil simulator now matches COMSOL in pure free space** (walls, no k₀): **3.2 / 3.2 / 9.3% at 10 µm**
  (original: 65 / 58 / 129%), 9.5 / 8.7 / 8.5% at 3 µm.
- The original exports differ from the re-solve by **3 / 3 / 8% at 0.5 µm, 10 / 10 / 23% at 3 µm and 60 / 67 / 120%
  at 10 µm**. So the main 3 → 0.5 µm benchmark inputs were also off by 10–23%.

- **Ampère's law now holds in the air.** On a single plane, B_z / [i(k·B_t)/k] must be 1 in every band in
  source-free air (section 10). The original export gives 0.32–0.97 for k = 0.03–0.3 rad/µm, tracking the screened law
  k/√(k² + 0.10²); the re-solve gives **1.00 in every band at 3 and 10 µm**.

So the extra decay, and the Ampère violation that came with it, was the default solver tolerance, and the free-space
generator was right. Figure:
`final_evaluation/04_figures/p8_comsol_check/tolerance_resolve.png`. The only re-solved data are L1_C1 at 0.5, 3 and
10 µm; every other COMSOL file (other heights, data3, IC_4um from `interconnects_F.mph`) still carries the artefact
until it is re-exported with the tight tolerance.

## Short answer (14 November, before the re-solve)

**Why our simulated fields failed at high standoff:** COMSOL's field in the air above the coil does not obey
magnetostatics. Between every pair of exported planes, its low spatial frequencies decay at the rate √(k² + k₀²), not
k. Here k₀ ≈ 0.10 rad/µm (decay length about 10 µm). At 10 µm, 70% of the field lies below k = 0.1 rad/µm, exactly
where this extra decay acts; at 0.5 µm only 11% does. That is why the free-space simulator was fine near the coil and
off by 65 / 61 / 134% (Bx/By/Bz) at 10 µm.

With the exact coil from the DXF, a zero-thickness sheet at z = 0, the box walls and k₀ = 0.10, the simulator
reproduces COMSOL to **10–12% at 3, 5 and 10 µm, with amplitude 1.00 and the best-fitting height exactly 3.00 µm**.
The remaining 10% is about the COMSOL mesh noise. A held-out configuration (C1+C2+C3+C4, not used to fit k₀) gives the
same: 83 / 75 / 66% → 10 / 11 / 10% at 12.5 µm.

**k₀ is not physics.** A static field in empty, non-magnetic air must satisfy ∇²B = 0; COMSOL's satisfies ∇²B ≈ k₀²B.
It also gets **larger on the finer mesh** (Extremely Fine: k₀ ≈ 0.12–0.17), so it is a numerical property of how this
model was solved, not something a real coil or a real QDM produces. The exact mechanism is not confirmed (the solver
log reports a small linear-error estimate, 8e-6, so "unconverged solve" is not established). Refining the mesh further
will not remove it.

## What the model is

From `C1_EF_FR.html` / `C1_EEF_FR.html` (COMSOL 5.5, `Multilayer square coil 2um 4um 3L 4 COILS.mph`).

| Item | Value |
|---|---|
| Physics | Magnetic Fields (AC/DC), **Stationary** study, magnetic vector potential A, quadratic curl elements |
| Domain | One air block, **140 × 140 × 90 µm**, centred on the four-coil array; mesh volume 1,764,000 µm³ |
| Outer boundaries | **Magnetic Insulation on all six faces** (n × A = 0, so B·n = 0) |
| Materials | Air: μr = 1, σ = 0. Coil: aluminium (σ = 3.774e7 S/m) |
| Coil | Boundary (surface) coil, single conductor, **600 µA**, conduction thickness **200 nm** (sets resistance only; the current is a zero-thickness sheet in the plane z = 0) |
| Geometry | 15 rectangles, all **2.00 µm** wide, 4 µm gaps, 3 turns ("2um 4um 3L"); leads run to the box wall at Y = 42 µm; mirrored to four coils |
| Terminals | Input and output edges on the box wall; the current returns through the wall |
| Excitation sweep | One model, one mesh: (I1..I4) = each coil alone and the cumulative sets. L1_C1 and data3 (C1+C2, C1+C2+C3, C1–C4) all come from this one solve, which is why their mesh speckle is shared |
| Mesh, Extra Fine | 284,696 tetrahedra, element size 0.212–4.94 µm, growth 1.35. **The original L1_C1 exports are this mesh** (they match the new EF export to 0.03–0.13%) |
| Mesh, Extremely Fine | 1,372,173 tetrahedra, element size 0.0282–2.82 µm, growth 1.3 |
| Solver | FGMRES with multigrid, 1 linear iteration, reported LinErr 8e-6 (EF) / 7.8e-6 (EEF) |
| Export | Cut plane, point (0, 0, z), normal +z, 1400 × 1400 points at 0.1 µm, tesla |

IC_4um and IC_2um come from a **different** model, `interconnects_F.mph`, which we have no report for. Its exports show
the same two signatures (walls at the window edges; extra low-k decay, k₀ ≈ 0.07–0.095 from a tapered estimate).

## Findings, one by one

### 1. The 90° rotation is COMSOL's cut-plane frame

The CSV columns `cpl1x, cpl1y` are the cut plane's **local** axes, which COMSOL rotated from the global ones:
cpl1x = −Y, cpl1y = X (the export window [−42, 98] × [−39, 101] µm is exactly the block's [−98, 42] in Y and
[−39, 101] in X). The B columns are **global** components. So along the array axes, B_col = −By_csv and
B_row = Bx_csv. Check: Bz = i(k·B_t)/k holds to 6% with this mapping and fails (100%) with the naive labels
(0.3–1.5 rad/µm, L1_C1 at 3 µm). The repo's existing quarter-turn handled this correctly.

### 2. The window is the whole box, and the walls are visible in the data

The normal field component at each window edge is 0.2–0.6% of the rms field, and the mean of Bz is zero to four
decimals at every height, as a closed box with B·n = 0 requires. Our operators and simulator zero-pad instead, as if
the field continued in free space.

**But the walls are not why the decay is wrong.** Exact box propagation (mirror images at all walls, top wall at
45 µm) gives the same mismatch as free space: 0.5 → 10 µm, 50.1% vs 50.8% (mean of components). Fitting the top-wall
height gives no improvement at any value.

### 3. The decay anomaly, measured in the box's own modes

Decay rate per band between COMSOL planes (L1_C1, Extra Fine; `box_modes.py`, mirror-extended, no taper). Exact box
physics predicts rate = k here.

| Band k (rad/µm) | 0.027 | 0.075 | 0.109 | 0.155 | 0.216 | 0.303 | 0.432 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Measured 0.5 → 3 µm (Bz) | 0.103 | 0.128 | 0.144 | 0.189 | 0.240 | 0.329 | 0.445 |
| Measured 3 → 6 µm (Bz) | 0.104 | 0.126 | 0.141 | 0.187 | 0.238 | 0.327 | 0.435 |
| Measured 6 → 10 µm (Bz) | 0.106 | 0.127 | 0.141 | 0.188 | 0.237 | 0.329 | 0.420 |
| √(k² + 0.10²) | 0.104 | 0.125 | 0.148 | 0.184 | 0.238 | 0.319 | 0.443 |

The excess is the same at every height and fits one constant, k₀ ≈ 0.10 rad/µm, across bands (0.093–0.106). Bx and By
show the same. On the Extremely Fine mesh the excess is larger (0.5 → 3 µm Bz: 0.170 at k = 0.027, k₀ ≈ 0.12–0.17).

### 4. One constant closes the gap

COMSOL L1_C1 at 0.5 µm propagated upward, compared with COMSOL at z (mean relative L2 over components; full band /
low-passed below 1 rad/µm). `k0_test.py`.

| z (µm) | Free space | Box walls, k₀ = 0 | Box walls, k₀ = 0.10 |
|---:|---:|---:|---:|
| 2 | 14.5 / 4.4 | 14.5 / 4.5 | 14.2 / 3.6 |
| 3 | 11.2 / 6.1 | 11.4 / 6.5 | 10.4 / 4.6 |
| 5 | 17.1 / 14.6 | 17.5 / 15.1 | 11.0 / 6.7 |
| 7 | 27.8 / 26.9 | 28.1 / 27.2 | 10.6 / 7.9 |
| 10 | 50.8 / 50.6 | 50.1 / 49.8 | **10.2 / 8.8** |

Held out (C1+C2+C3+C4, k₀ fixed at 0.10; `heldout_c1234.py`), Bx/By/Bz: 0.5 → 10 µm 57.8 / 51.2 / 44.3% →
9.8 / 11.1 / 9.6%; 0.5 → 12.5 µm 82.6 / 74.9 / 66.3% → 10.3 / 11.4 / 10.2%.

### 5. The simulator with the exact coil reproduces COMSOL

`dxf_forward.py`: current from a Laplace conduction solve in the DXF conductor (terminals at the wall, 600 µA,
current conserved to 4e-13), field of a zero-thickness sheet at z = 0. Relative L2 Bu/Bv/Bz against COMSOL, unscaled,
with the best amplitude scale a in brackets.

| Target | Free space | Box walls | Box walls + k₀ = 0.10 |
|---|---|---|---|
| 0.5 µm, Extra Fine | 19.5 / 18.2 / 22.3 [0.95] | 19.5 / 18.0 / 22.4 [0.95] | 19.4 / 17.9 / 21.2 [0.96] |
| 0.5 µm, Extremely Fine | 16.3 / 14.7 / 20.6 [0.96] | 16.3 / 14.6 / 20.8 [0.95] | 16.2 / 14.5 / 18.5 [0.97] |
| 3 µm, Extra Fine | 14.0 / 12.0 / 24.7 [0.92] | 14.0 / 13.0 / 25.1 [0.91] | **11.0 / 10.5 / 10.9 [1.00]** |
| 3 µm, Extremely Fine | 11.8 / 11.3 / 33.5 [0.89] | 11.8 / 12.3 / 34.5 [0.88] | 7.5 / 8.2 / 13.2 [0.98] |
| 5 µm, Extra Fine | 23.2 / 19.9 / 45.9 [0.81] | 23.1 / 21.4 / 45.7 [0.80] | **11.3 / 11.6 / 12.3 [1.00]** |
| 10 µm, Extra Fine | 64.9 / 60.5 / 134.2 [0.52] | 64.5 / 57.7 / 128.9 [0.53] | **10.1 / 11.8 / 12.4 [0.99]** |

Against the target low-passed below 1 rad/µm, box + k₀ agrees to 2.9–4.4% at 0.5 µm and 8.9–10.9% at 10 µm. The best
sheet height at the 3 µm plane is exactly 3.00 µm (2.9 and 3.1 µm are both worse), so no source offset is needed.
The free-space column at 10 µm reproduces the 1 November spike (68 / 64 / 142% with a hand-traced coil).

### 6. Mesh noise, measured directly

The two meshes of the same model differ by 19.4 / 21.8 / 25.6% at 0.5 µm and 11.7 / 12.7 / 15.8% at 3 µm. Per band
at 0.5 µm, |EF − EEF| / |EEF| is 0.02–0.05 below 1 rad/µm, 0.1–0.2 at 1–2, about 1.0 at 2–2.9 and about 1.4 above
2.9 rad/µm. Independent noise on both would give √2 ≈ 1.41, so **above 2.9 rad/µm both targets are noise**, 2–2.9 is
mixed, and below 2 is signal. This confirms the noise edge fixed in M2a (2.9 rad/µm) by a route that does not use
Maxwell consistency. Each mesh carries roughly 19–26% / √2 ≈ 14–18% noise at 0.5 µm, in line with M2b's 15.7% noise
band. At 3 µm the two meshes already disagree by 73–88% (Bx, By) in the 1–1.5 rad/µm band, so the 3 µm input is
noise-dominated above about 1 rad/µm, consistent with the SNR = 1 crossings (1.05–1.6 rad/µm). The Extremely Fine
export is less noisy (closer to the noise-free DXF field: 16 / 15 / 21% against 19 / 18 / 22%) but not clean.

### 7. COMSOL's air carries current: Ampère's law fails

`ampere_test.py`, figure `figures/comsol_check/ampere_air_current.png`. In empty air curl B = 0. The horizontal
components of curl B, estimated half-way between planes 1 µm apart (fields low-passed below about 0.35 rad/µm), as a
fraction of the size of the two terms that should cancel:

| Planes | COMSOL | Exact magnetostatics (control, DXF coil) | Screened model, k₀ = 0.10 | COMSOL vs screened pattern |
|---|---:|---:|---:|---|
| 2 → 3 µm | 0.27 | 0.011 | 0.25 | correlation 0.83, amplitude 0.87 |
| 3 → 4 µm | 0.33 | 0.010 | 0.30 | 0.90, 1.02 |
| 6 → 7 µm | 0.41 | 0.007 | 0.44 | 0.95, 0.87 |
| 9 → 10 µm | 0.52 | 0.004 | 0.57 | 0.97, 0.87 |

The control shows the estimate itself is accurate to about 1%. COMSOL's implied air current at 3.5 µm reaches
8e5 A/m² and runs opposite to the coil current beneath it, the pattern of curl B = −k₀²A (a screening current). This is the
same anomaly as the decay law, seen through Maxwell's equations without any propagation or wall model: COMSOL's
solution does not satisfy Ampère's law in the air. The air in the model is μr = 1, σ = 0, with a Stationary study,
so no physical current can flow there.

**Possible causes, not yet distinguished:** (a) the iterative linear solve (FGMRES + multigrid, one iteration)
stopped with an error that is smooth in space, which a residual-based error estimate can miss and which tends to grow
on finer meshes; (b) a regularisation or gauge term in the A-formulation that acts like a weak screening medium in
the air; (c) some other setting in the .mph file not visible in the report. Tests that separate them are listed under
Options, item 3.

### 8. Our generator checked against exact Biot–Savart (no COMSOL)

`simulator_vs_biot_savart.py` runs `simulate_fields.fields_from_current` (the code behind every training set) on a
2 µm wide square loop (centre-line half-side 10 µm, 600 µA) and compares it with direct Biot–Savart line integrals
over 81 nested filaments, a completely independent method. Relative L2 error, Bx/By/Bz:

| z | 0.5 µm | 1 µm | 3 µm | 10 µm | 20 µm |
|---|---|---|---|---|---|
| Error | 0.62 / 0.61 / 0.54% | 0.42 / 0.41 / 0.44% | 0.27 / 0.26 / 0.36% | 0.18 / 0.19 / 0.74% | 0.13 / 0.21 / 1.98% |

The rise in Bz at 10–20 µm comes from the periodic FFT canvas (205 µm here): the loop's far field wraps around. Keep
the padding generous when generating high-standoff data. The generator returns fields in COMSOL's CSV label
convention (physical B_col = By_out, B_row = −Bx_out, Bz = −Bz_out, overall sign set by current direction).

### 9. The tight-tolerance re-solve (17 November)

Side-by-side figures of the two solves: `final_evaluation/04_figures/p8_comsol_check/resolve_vs_old_maps.png` (Bz at 0.5, 3 and 10 µm: original export, re-solve and our simulator for the exact coil, with differences), `resolve_vs_old_linecut.png` (line cuts at 3 and 10 µm, all three versions) and `resolve_vs_old_why.png` (amplitude lost per frequency, and how much of the field is broad structure at each height); script `scripts/comsol_check/plot_resolve_vs_old.py`.

`tolerance_check.py` (and `plot_tolerance.py` for the figure) on `COMSOL_params_v2/Extra_Fine_Mesh/FGMRES_RT1e-8/`.
Same model, same Extra Fine mesh, iterative solver (FGMRES) at relative tolerance 1e-8 instead of the default. Other
files in `COMSOL_params_v2` are byte-identical to `COMSOL_params` (checked by MD5).

Decay rate per band (box modes, all components pooled), 3 → 10 µm; free space predicts rate = k:

| Band k (rad/µm) | 0.027 | 0.048 | 0.075 | 0.109 | 0.155 | 0.216 | 0.303 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original export | 0.103 | 0.110 | 0.125 | 0.141 | 0.187 | 0.237 | 0.327 |
| Re-solve, tolerance 1e-8 | 0.025 | 0.048 | 0.076 | 0.103 | 0.158 | 0.219 | 0.314 |

| Check (Bx/By/Bz, relative L2) | Original | Re-solve |
|---|---|---|
| Re-solve vs original, same plane | — | 0.5 µm: 2.9 / 3.3 / 7.9%; 3 µm: 9.8 / 10.3 / 22.8%; 10 µm: 59.5 / 66.7 / 119.9% |
| COMSOL 0.5 µm propagated to 10 µm (box walls, free space) vs COMSOL 10 µm | 56.7 / 52.0 / 41.4% | **3.8 / 4.5 / 3.1%** |
| Same with k₀ = 0.10 | 9.7 / 11.9 / 9.1% | 37.9 / 35.7 / 34.2% |
| Exact DXF coil (walls, free space) vs COMSOL, 3 µm | 14.0 / 13.0 / 25.1% | **9.5 / 8.7 / 8.5%** |
| Same, 10 µm | 64.5 / 57.7 / 128.9% | **3.2 / 3.2 / 9.3%** |

Read: the original export carried a solver error that is smooth in space and large where the field is weak; a
residual-based convergence test judged it converged. With the tight tolerance COMSOL agrees with exact free-space
magnetostatics, and with our simulator, to a few per cent at 10 µm. Ritaban's point that "the curl of the field is
zero as there is no source there" is true of the physics; section 7 measured that the *original export* did not
satisfy it, and the re-solve now does (it obeys the free-space decay law, which is equivalent in source-free air).

**Consequences for earlier results.** Every COMSOL score in this project used the original exports. At 3 → 0.5 µm
the inputs were off by 10–23% and the targets by 3–8%; all methods saw the same data, so rankings at 3 µm are
unlikely to flip, but absolute numbers will move. At 10 µm the COMSOL test (phase 9) was not a valid test of
physics; it becomes one once the coils are re-exported. What is needed from COMSOL: every configuration (L1_C1 all
heights, data3, the IC model) re-exported with the tight tolerance.

### 10. Did the re-solve fix the Ampère violation? Yes (17 November)

`ampere_single_plane.py`, figure `final_evaluation/04_figures/p8_comsol_check/ampere_single_plane.png`.

Section 7 tested Ampère's law with planes 1 µm apart. The re-solve only has 0.5, 3 and 10 µm, so that test cannot be
repeated. An exact single-plane form exists: because B = ∇ × A, div B = 0 always, so each Fourier mode
B ∝ exp(i k·r − γz) has B_z = i(k·B_t)/γ. Curl B = 0 then holds **if and only if γ = k**. So in source-free air

    R(k) = B_z / [ i (k·B_t) / k ]   (box modes, × tanh(k(H − z)) for the top wall)

must be 1 in every band. A screened field (γ = √(k² + k₀²)) gives k/√(k² + k₀²).

| Band k (rad/µm) | 0.027 | 0.048 | 0.075 | 0.109 | 0.155 | 0.216 | 0.303 | 0.432 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original export, 3 µm | 0.32 | 0.42 | 0.60 | 0.72 | 0.85 | 0.92 | 0.97 | 0.99 |
| Original export, 10 µm | 0.35 | 0.42 | 0.60 | 0.72 | 0.84 | 0.90 | 0.94 | 1.00 |
| Screened law, k₀ = 0.10 | 0.27 | 0.44 | 0.60 | 0.74 | 0.84 | 0.91 | 0.95 | 0.97 |
| Re-solve, 3 µm | 1.01 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Re-solve, 10 µm | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

The original export obeys the screened law within a few hundredths in every band: its "air current" (section 7) is
exactly what a k₀ = 0.10 screening term produces. The re-solve satisfies Ampère's law to 1–2% in the lowest bands and
to better than 0.5% above. **Answer to the open question of section 7:** the air current was numerical; with a tight
solver tolerance it disappears. Ritaban's remark that "the curl of the field is zero as there is no source there" is
true of the physics and of the re-solve; it was not true of the original export.

What remains unexplained: why COMSOL's default convergence test accepted the loose solution (it reported a linear
error of 8e-6). The likely reason is that a smooth, low-frequency error in the weak far field barely changes the
residual it checks. That is a statement about COMSOL's settings, not about this project.

## How our simulator differs from the COMSOL model

Ranked by how much each matters at high standoff.

| # | Difference | Our simulator (`simulate_fields.py`) | COMSOL | Effect |
|---|---|---|---|---|
| 1 | Decay law | exp(−kz), free space | √(k² + k₀²) with k₀ ≈ 0.10 (EF), larger on EEF; numerical | **Dominant.** Bz at 3 µm 24.7 → 10.9%; all components at 10 µm 65 / 61 / 134 → 10 / 12 / 12% |
| 2 | Wire width, source height | 1.6–3.2 µm, offset 0–0.2 µm (fitted 2.08 µm, 0.08 µm) | 2.00 µm, zero-thickness sheet exactly at z = 0 | A few points at 0.5 µm: hand-traced 22.9 / 19.3 / 24.5% → exact 19.5 / 18.2 / 22.3% |
| 3 | Boundaries | Free space, zero padding | 140 × 140 × 90 µm box, B·n = 0, window = whole box | At most 1–2 points on propagation; visible near the edges |
| 4 | Leads | Closed loops inside the image | Leads run to the wall; current returns through the wall | Part of the low-k content; captured by the mirror-image model |
| 5 | Noise | Correlated speckle, 1–2.5% of range, 0.38 µm | Mesh noise of 14–18% of the 0.5 µm target norm, above about 2 rad/µm | Sets the floor of every score |

The current (600 µA) and the vector frame were already right: amplitude scale 1.00 at 3–10 µm.

## Options (the user's decision)

The k₀ question also touches the existing 3 → 0.5 µm benchmark: the free-space forward model misses COMSOL's 3 µm
Bz by 25% with the exact coil, 11% with k₀; the in-band forward mismatch of the benchmark pair falls from 7.6 / 6.5 /
4.1% to 5.2 / 5.6 / 3.0% (below 1 rad/µm).

1. **Add k₀ (and optionally the box) to the simulator and to the solvers' operator** (`--decay-k0` already exists in
   `simulate_fields.py`; the operator would become exp(−√(k² + k₀²)Δz)). COMSOL scores at 5–12.5 µm become
   meaningful. The cost: the models learn a numerical property of this COMSOL dataset that a real QDM will not have.
   Use k₀ per model and mesh if so (EF 0.10; IC model 0.07–0.095, uncertain).
2. **Keep free space** and declare COMSOL above about 4 µm invalid as a test, stating this analysis as the reason.
3. **Ask Hari for one re-solve** of L1_C1 with a direct solver (PARDISO or MUMPS) or a much tighter linear tolerance,
   exported at 0.5, 3 and 10 µm; or for curl B evaluated in the air domain (it should be zero). If k₀ disappears, the
   data are fixed at the source. More mesh refinement will not help: the finer mesh made k₀ larger. Diagnosis in order:
   - **No re-solve:** in the existing solution, integrate B·dl around a closed 20 × 20 µm square in a vertical plane
     in the air (for example from z = 2 to 22 µm, beside the coil, not enclosing any wire). Physics gives 0; a
     non-zero result confirms the air current inside COMSOL itself.
   - **Direct solver:** same model and mesh, Stationary solver switched from Iterative to Direct. If k₀ goes to ~0,
     the cause is the iterative solve (fix: direct solver, or a much tighter tolerance).
   - **Tighter tolerance** with the iterative solver (relative tolerance 1e-6 instead of the default): k₀ should shrink if (a).
   - **If the direct solver still shows it:** a calibration case with an exact answer (one straight wire or one
     circular loop, same box and mesh) and a check of the Ampère's Law / gauge / discretisation settings in the .mph.
   - The scripts here re-measure k₀ and curl B on any new export in minutes.

## Commands

```bash
# CSV -> npz (arrays [comp, row, col] in tesla, raw CSV labels); run where the CSVs are
python3 scripts/comsol_check/csv2npz.py <out.npz> <csv files...>
# reports -> text
python3 scripts/comsol_check/html2txt.py COMSOL_params/Extra_Fine_Mesh/C1_EF_FR.html
# analyses: <dir> holds L1_C1_all.npz (all L1_C1 heights), new_EF.npz, new_EEF.npz, C1234_sel.npz
python3 scripts/comsol_check/box_analysis.py <dir>      # frame, mesh noise, free vs box propagation
python3 scripts/comsol_check/box_modes.py <dir>         # wall checks, decay per band in box modes
python3 scripts/comsol_check/decay_rate.py <dir>        # same with a Hann taper, plane by plane
python3 scripts/comsol_check/k0_test.py <dir>           # propagation with k0
python3 scripts/comsol_check/dxf_forward.py <dir> COMSOL_params/Gemetry_C1.dxf   # simulate from the DXF
python3 scripts/comsol_check/heldout_c1234.py <dir>     # held-out check on C1+C2+C3+C4
python3 scripts/comsol_check/ic_rate.py                 # on the server: IC_4um decay and walls
python3 scripts/comsol_check/tolerance_check.py <dir> <rt8.npz>   # tight-tolerance re-solve vs original (section 9)
python3 scripts/comsol_check/plot_tolerance.py <dir> <rt8.npz> <out.png>
python3 scripts/comsol_check/ampere_single_plane.py <dir> <rt8.npz> <out.png>   # single-plane Ampère test (section 10)
```

The input arrays were made from `data/L1_C1_Z*.csv` and `data/data3/Coil_C1_C2_C3_C4/*Z{0p5,3p0,10p0,12p5}*.csv` on
the server and from the four new CSVs in `COMSOL_params/`. Local runs take 1–4 minutes each on the Mac (NumPy and
SciPy only).
