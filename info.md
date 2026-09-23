This is a massive result. The joint fitting completely changed the game. Here's the full comparison:

┌──────────────────────┬───────┬───────┬───────┐
│        Method        │  Bx   │  By   │  Bz   │
├──────────────────────┼───────┼───────┼───────┤
│ Per-component unreg. │ 581%  │ 1109% │ 575%  │
├──────────────────────┼───────┼───────┼───────┤
│ Joint unreg.         │ 0.87% │ 0.93% │ 0.98% │
├──────────────────────┼───────┼───────┼───────┤
│ Per-component reg.   │ 5.10% │ 5.01% │ 4.46% │
├──────────────────────┼───────┼───────┼───────┤
│ Joint reg.           │ 0.81% │ 0.86% │ 0.91% │
├──────────────────────┼───────┼───────┼───────┤
│ Analytic (exact)     │ 0.73% │ 0.74% │ 0.72% │
└──────────────────────┴───────┴───────┴───────┘

Two things to notice:

1. The joint unregularised kernel went from 575-1109% to 0.87-0.98%. It's nearly as good as the analytic propagator. The joint fitting itself acts as regularisation: the mesh noise in Bx, By, and Bz is independent (each component cuts through the mesh differently), so when you average across 3 components in the denominator (|Sx|^2 + |Sy|^2 + |Sz|^2), the noise partially cancels while the signal adds coherently.
2. The regularised version barely improves over the unregularised (0.81-0.91% vs 0.87-0.98%). The joint fitting already killed most of the noise that lambda was compensating for. Adding lambda on top gives a small extra improvement, but the heavy lifting was done by the joint constraint.

The learned kernel is now within ~0.15% of the analytic propagator -- essentially at the mesh noise floor. Your instinct about using all 3 components jointly was the key insight.

Look at that. The joint unregularised kernel's training NRMSE is no longer zero -- it's ~0.5-1.0%, because the shared kernel can't perfectly fit each component's individual noise. And the holdout went from 575-1109% down to 0.87-0.98%. That's within striking distance of the analytic propagator's 0.72%.

Now the regularised version.

/btw so we have learned each ki with just 3 pairs ? bx, by, bz, of sharp and blur image?

Yes, exactly. For each dz value, we learn the kernel from 3 pairs: one Bx pair, one By pair, one Bz pair. But actually, each pair gives us an **independent** kernel estimate. We're not combining them -- we learn a separate K_hat for each (dz, component) combination.

So for example, for dz = 0.5:
- K_hat(0.5, Bx) = B_x(z=1.0) / S_x(z=0.5)
- K_hat(0.5, By) = B_y(z=1.0) / S_y(z=0.5)
- K_hat(0.5, Bz) = B_z(z=1.0) / S_z(z=0.5)

The physics says the propagator exp(-2pi*k*dz) is the **same** for all three components (Bx, By, Bz all obey the same Laplace equation in source-free space). So in principle, all three should give identical kernels. The fact that we learn them separately and they all give similar NRMSE is actually a consistency check -- it confirms the physics is working.

We could in theory **pool** all three components to get a single more robust kernel per dz (averaging over 3 independent estimates), which would reduce the noise. But right now the code keeps them separate, which is the simpler and more transparent approach.

And to be precise about what "one pair" gives us: each pair is a 1400x1400 sharp image and a 1400x1400 blurred image. In Fourier space that's 1400x1400 = 1,960,000 frequency bins, each giving one value of K_hat. So from one image pair, we get the full 2D kernel. The issue is that this is also 1,960,000 free parameters -- which is why the unregularised version overfits so badly.



Regularised kernel results:

Figure 1 (Fourier space): The right panel tells the story clearly. The blue curve (unregularised B/S) follows the analytic exp(-2pikdz) at low k, then plateaus at high k because it's fitting noise. The red curve (Wiener-regularised) tracks the analytic curve much better and gets suppressed at high k instead of plateauing. The regularisation is doing exactly what it should: trusting the data where signal dominates, killing it where noise dominates.

Figure 2 (Real space): Still noisy-looking, which is expected. The Poisson kernel has very long tails (20% of its mass outside r = 5*dz), so it doesn't look like a clean bump in a small crop. But the dz=0.5 panel now shows a visible central peak, unlike the pure speckle of the unregularised version.

Figure 3 (Holdout test): This is the payoff. The regularised kernel applied 3 times gives NRMSE of 4.5-5.1% across Bx, By, Bz. Compare that to:
- Unregularised: 500-1100% (catastrophic, noise compounds each application)
- Analytic propagator: 0.7% (best possible, exact physics)
- Regularised: ~5% (captures the physics, residual is mesh noise)

The residual maps (rightmost column) show the error is spatially uniform noise, not a systematic pattern, which means the kernel has learned the correct physics and the remaining error is just COMSOL mesh interpolation artifacts.

Let me also show the radial profile figure.

  Read 1 file

Figure 4 (Radial profile): For dz=0.5 (leftmost), the learned kernel (blue) matches the Poisson kernel (red dashed) near the center peak beautifully. For larger dz values (2.5, 4.5, 9.5), the learned curve is flat near zero while the analytic Poisson is a broad smooth bump. This is because the Wiener regularisation kills the high-k content that encodes the tails. The kernel is effectively being truncated in Fourier space, which wipes out the slow real-space decay. This is fine -- the Fourier-space representation is what matters for prediction, and that's where it matches well.

Summary of where we stand:

┌─────────────────────┬───────────────┬──────────────────────────────────────────┐
│       Method        │ Holdout NRMSE │                  Notes                   │
├─────────────────────┼───────────────┼──────────────────────────────────────────┤
│ Unregularised (B/S) │ 500-1100%     │ Memorises noise, catastrophic on holdout │
├─────────────────────┼───────────────┼──────────────────────────────────────────┤
│ Wiener regularised  │ 4.5-5.1%      │ Learns real physics, noise suppressed    │
├─────────────────────┼───────────────┼──────────────────────────────────────────┤
│ Analytic propagator │ 0.7%          │ Exact theory, best possible              │
└─────────────────────┴───────────────┴──────────────────────────────────────────┘

The gap between 5% (learned) and 0.7% (analytic) is the cost of mesh interpolation noise in the COMSOL data. The regularisation brought us from useless to within an order of magnitude of the theoretical limit.




kernel_vs_analytic.png — the kernel check

Column 1 (actual): the real z=2.0 data. Field peaks at ~50 uT.

Column 2 (analytic propagator): near-perfect prediction. NRMSE = 0.7%. This uses the known exp(-2pikdz) formula, no learning involved.

Column 3 (learned kernel applied 3 times): total disaster. NRMSE = 500-1100%. The image is pure noise with residuals of thousands of microtesla when the actual field is only ~50. The coil structure is completely gone.

What went wrong: The learned kernel K_hat = B/S is correct at low k (where there is real signal) but is noise/noise at high k. Each time you apply the kernel, those high-k noise ratios get multiplied together. After three applications, the noise has been cubed:

- 1 application: noise level ~0.1 at high k (from the kernel_vs_analytic plot)
- 3 applications: (0.1)^3 is not the issue. The problem iach application scrambles the high-k content into randomvalues, and after 3 applications those random values have been amplified and mixed across the image.

Meanwhile the analytic propagator gives 0.7% error because it has no noise at any frequency. It just multiplies by the clean exponential.

This is the answer to "why lambda?" Without regularisation, the learned kernel memorises noise at high k. On training data it looks perfect  (zero error), but the moment you try to use it on anythine explodes. Lambda suppresses the kernel at high k (wherewe know it's noise anyway), which would prevent this blowup. The kernel would then match the analytic propagator at low k and smoothly go to zero at high k instead of holding random noise values.




11 COMSOL 2D cut-plane exports, one per standoff height, all from the same model:

- Model: Multilayer_square_coil_2um_4um_3L_4_COILS.mph (COMSOL 5.5) — 2 µm tracks, 4 µm pitch, 3 layers, 4 coils
- Drive: 600 µA, 0p1PR = 0.1 µm point resolution (the grid pitch)
- Standoffs: z = 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 µm (11 planes)


What you're calling a blur kernel is physically a known propagator. In source-free space above the coil, Bz at height z₂ from Bz at z₁ is exactly a Fourier-space multiplication by exp(−k·Δz) — analytic, isotropic, zero free parameters. So "learning" it isn't discovery; it's validation, and that's still worth doing: the gap between the fitted kernel and exp(−kΔz) is exactly how much extra smoothing COMSOL's mesh is injecting. That's a real result about your simulation's trustworthiness.

CORRECTED (an earlier note here said the coil was truncated at the FOV boundary — it is not). The spiral sits entirely inside the field of view, at x ∈ [−20, 22], y ∈ [−22, 22] µm. What reaches the x_min boundary is two feed traces still carrying current at ~90 µT. The top-right half of the FOV is empty. So the field does not decay to zero at the x_min edge, but the cause is the feed routing, not a cut-off coil.

That matters because the propagator is a Fourier multiply, and a naive FFT will wrap that hot left edge into the empty right side. Any design here has to deal with it explicitly — I'll fold that in rather than ask you about it.


Phase 1 design — standoff propagator validation

1. What we're measuring

In source-free space above the coil, for any component:

B̃(k, z₂) = B̃(k, z₁) · exp(−2π·k·Δz)     k = √(fx²+fy²) in cycles/µm, Δz = z₂−z₁

Zero free parameters. Three things get measured against it:

┌─────┬──────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────┐
│  #  │                  Check                   │                                       What it answers                                        │
├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
│ P   │ phase of H(k) = B̃₂/B̃₁ must be            │ Are the 11 planes laterally registered? A sub-pixel shift masquerades as "extra smoothing"   │
│     │ identically 0                            │ and would poison every amplitude number. Runs first, before any amplitude work.              │
├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
│ B   │ NRMSE between propagate(B_z₁, Δz) and    │ The headline result. No division, no conditioning issues. This number decides Phase 2.       │
│     │ actual B_z₂, interior only               │                                                                                              │
├─────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
│ A   │ radially-averaged |H(k)| vs exp(−2πkΔz), │ Explains B: at what k does it break. This is the usable-bandwidth curve.                     │
│     │  SNR-masked                              │                                                                                              │
└─────┴──────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────┘


(a) Your K can't be shared across the sum. You wrote Σᵢ ‖bᵢ − sᵢ*K‖² with one K. But the kernel is the standoff — the operator for z=2 is not the operator for z=10. It's 10 independent least-squares problems, one per Δz. (The only way to keep one sum is to parameterize K = K(Δz) and fit the parameter jointly, which is a different, much smaller problem.) WHYYY? 
cant we just learn one K and apply it to all?

**Because the propagator is not linear in Δz. The Fourier-space multiplication is exp(−kΔz), which is not a linear function of Δz. So the kernel for z=2 is not the same as the kernel for z=10. Each standoff height has its own unique kernel that must be learned independently. SO for the data i have will will have one 3 pairs for each standoff height, and we will learn a separate kernel for each of those 11 heights? is that so?**


(c) Don't fit a compact real-space kernel. This is the one that would silently ruin the result. The exact kernel is the 2-D Poisson kernel:

$$K(r) = \frac{\Delta z}{2\pi,(r^2+\Delta z^2)^{3/2}}, \qquad \text{tail mass beyond } R = \frac{\Delta z}{\sqrt{R^2+\Delta z^2}}$$

At R = 5Δz that's 20% of the kernel mass still outside the window. At R = 20Δz, 5%. For Δz = 9.5 µm (95 px) you'd need a ~3800 px kernel to capture 95% — wider than the image. A 31×31 or 64×64 conv kernel doesn't just truncate, it forces least squares to distort the center to compensate.

Do it in Fourier instead, where the same objective has a closed form:

$$\hat K(f) = \frac{\overline{S(f)},B(f)}{|S(f)|^2 + \lambda} \quad\text{(minimizes } |b-s*K|^2 + \lambda|K|^2\text{)}$$

That's Wiener-regularized spectral division — it is exactly the H(k) estimator from the plan. Your least-squares and my validation are the same equation, so they're the same code.

**can you explain in simple , why are we not fitting a compact real-space kernel and instead doing it in Fourier space**
We are not fitting a compact real-space kernel because the exact kernel that describes how the magnetic field propagates with height (the 2-D Poisson kernel) has a long tail. This means that a significant portion of the kernel's mass is located far from the center, and if we try to fit a small, compact kernel (like 31x31 or 64x64), we would be truncating this important information. This truncation would force the fitting process to distort the center of the kernel to compensate for the missing tail, leading to inaccurate results.



**Why those modules — the blunt version**

K̂ = S̄B/(|S|²+λ) returns a kernel for any input, including garbage. Nothing in the output tells you it's wrong. We happen to know the true answer here, so we can grade it — and the grade produces three numbers you need before you can even set λ:

- Phase check → if the planes are laterally misregistered by even a fraction of a pixel, K̂ absorbs the shift and comes out asymmetric. You'd publish a registration bug as physics.
- Interior mask → which pixels are usable at all, given the coil is truncated at 45% of peak on the x_min edge. Without it you fit K partly to FFT wraparound.
- Usable k-range → above some k the COMSOL data is mesh noise, not field. That crossover is how you choose λ.

And the modules map onto your pipeline anyway: convert (get data fast) → fields (make the pairs) → propagator (FFT machinery + K̂ + masks) → validate (grade it).

Plan for choosing 0.5 vs 1.0 — the ladder test

Falls out of the pipeline for free:

1. Anchor at 0.5, fit z_eff for targets 1…10 (Δz = 0.5…9.5). Anchor at 1.0, fit z_eff for targets 2…10 (Δz = 1…9).
2. Plot z_eff vs Δz for both. A clean anchor gives slope 1, intercept 0.
3. A contaminated anchor shows as a nonzero intercept, and the sign diagnoses it: if the anchor plane is itself over-smoothed by δ, the target looks less blurred relative to it, so z_eff = Δz − δ → negative intercept.
4. Fit-free cross-check: propagate 0.5→1.0 and compare to the actual z=1.0 plane; then 1.0→2.0 vs actual z=2.0. If the first residual is much worse, z=0.5 is the culprit.
5. Cheapest of all, and it's in the visualization below: the measured ratio B̃(k,1.0)/B̃(k,0.5) must equal exp(−πk) exactly. Where it peels off is where the mesh fails at the lowest plane.

Decision rule: use 0.5 if |intercept| < 0.1 µm and the spectral ratio tracks exp(−πk) out to k ≥ 1 cyc/µm; otherwise anchor at 1.0.


What's built

learning_ker/
  scripts/convert.py      CSV → .npy cache (run once, ~3 min)
  scripts/fields.py       loader, Grid, pair enumeration   ← your pairs come from here
  scripts/propagator.py   k-grid, exp(-2πkΔz), radial spectra
  scripts/visualize.py    all 10 figures, 15 s end to end
  cache/                  44 arrays + grid.json (260 MB, replaces 2.5 GB of CSV)
  figures/                overview, edges, zoom_Bz, linecuts, decay, spectra, montage_{Bx,By,Bz,Bnorm}


fields.pairs(reference_z) already yields {z_sharp, z_blur, dz, comp, sharp, blur} — that's the seam your least-squares plugs into. I spot-checked the cache against a raw CSV line (field[189,0] = −86.9339 µT vs -8.693396477214041E-5 T): exact.



The COMSOL export has a numerical noise floor from mesh interpolation, and it's not small. You can see triangular facets directly in zoom_Bz.png and edges.png, and stair-steps in linecuts.png.


Where the measured transfer |B̃(k,z₂)|/|B̃(k,z₁)| peels away from exp(−2πkΔz):

┌─────────┬──────────────────┬────────────────┐
│ Δz (µm) │ k_break (cyc/µm) │ finest feature │
├─────────┼──────────────────┼────────────────┤
│ 0.5     │ 0.375            │ 2.7 µm         │
├─────────┼──────────────────┼────────────────┤
│ 1.0     │ 0.30–0.31        │ 3.2 µm         │
├─────────┼──────────────────┼────────────────┤
│ 1.5     │ 0.283            │ 3.5 µm         │
├─────────┼──────────────────┼────────────────┤
│ 2.0     │ 0.18             │ 5.6 µm         │
├─────────┼──────────────────┼────────────────┤
│ 3.0     │ 0.15             │ 6.6 µm         │
├─────────┼──────────────────┼────────────────┤
│ 5.0     │ 0.11             │ 9.0 µm         │
└─────────┴──────────────────┴────────────────┘

Four independent lines say this is real data noise, not my analysis: the breakpoint is invariant to windowing and cropping (full / Tukey-tapered / interior-crop agree within a few percent, so it isn't edge leakage); the facets are directly visible; analytic propagation leaves 13–17% NRMSE at small Δz where an exact relation should give <1%; and k_break falls with Δz exactly as a fixed additive floor predicts.
**The number that decides your project: your tracks are 2 µm at 4 µm pitch, so the winding fundamental is at 0.25 cyc/µm. Compare that to the table — for Δz ≥ 2 µm, the coil's own fundamental is already below the noise floor. Those pairs carry almost no independent information about K. Least squares will still return a kernel; it'll be fitting mesh artifacts.**


**wait what, so you are saying that there is no point in making pairs out of these given standoff heights because the noise floor is already too high for the coil's fundamental frequency? So, for standoff heights of 2 µm and above, the data is dominated by noise rather than the actual signal from the coil, making it unreliable for learning the kernel. Is that correct?**
Yes, that's correct. For standoff heights of 2 µm and above, the noise floor is already above the coil's fundamental frequency (0.25 cyc/µm), meaning that the measured data is dominated by noise rather than the actual signal from the coil. As a result, any pairs created from these standoff heights would not provide reliable information for learning the kernel, as the least squares fitting would be influenced more by the noise artifacts than by the true signal. Therefore, it is not useful to make pairs from these standoff heights for the purpose of learning the kernel.
How to prove it though? 


MAIN POINT  **Usable pairs are Δz ≲ 1.5 µm,**

Your formulation is sound — this is a data-quality ceiling, not a method problem.

- Usable pairs are Δz ≲ 1.5 µm, so anchored at z=0.5 that's targets z=1 and z=2. Call it 2–3 genuinely informative pairs, not 10.
- λ isn't a free knob anymore — it has to roll off at k_break, and k_break depends on Δz. One λ per pair.
- Your instinct to stack Bx/By/Bz gets more valuable, not less. Independent constraints are now scarce, so 3× the data per Δz matters.
- The anchor question partly dissolves. At equal Δz=1.0, anchoring at 1.0 (k_break 0.309) beats anchoring at 2.0 (0.283) — no sign z=1.0 is degraded. There's no equal-Δz comparison available for z=0.5, so the ladder fit is still the way to settle it, but it'd be fitting z_eff inside a band we've now shown is noise-dominated for most pairs. Lower priority than it was an hour ago.



Now the four figures.

---

decay.png — amplitude vs standoff

Left: peak |field|. Right: RMS. Both log-y, all four components.

The headline is simple: over 0.5 → 10 µm the field drops ~43× at peak (215 → 4.9 µT) and ~27× in RMS (22.9 → 0.86 µT).

Two things worth reading past that:

The curves bend. On log-y against linear z, a pure exponential would be a straight line. These are steep early and flatten out. That's the propagator showing up in a scalar: the field is a sum of spatial frequencies, each dying as exp(−2πkz), so the sharp high-k content is gone by z ≈ 2 µm and only the broad low-k content survives higher up. The curve flattening is the blur.

Left panel wobbles, right panel doesn't. Peak |Bz| goes 8.19 at z=8 and 8.11 at z=9 — essentially flat where it should drop ~20%. Bz at z=2 (110 µT) sits above the trend too. But every RMS curve is perfectly smooth and monotone. That's diagnostic, not cosmetic: peak is one pixel, so it latches onto whichever mesh facet happens to spike; RMS averages 1.96 M pixels, so the noise cancels. The disagreement between the two panels is the mesh noise, measured.

Also note Bz > Bx ≈ By in RMS at every height — expected for a planar spiral viewed from above.

---

edges.png — where the field touches the boundary

Left panel is Bz at z=0.5 hard-clipped to ±20 µT when the peak is 215. Deliberately blown out: saturating the coil to solid red/blue is what makes the weak structure visible. You can now see the square spiral (~4.5 turns) and the two feed traces running left out of the domain — invisible in the percentile-scaled montages, which is exactly how I got this wrong the first time.

Middle panel zooms on those traces. The red/blue/red banding is the textbook Bz signature of a straight current-carrying wire — positive on one side, negative on the other. This panel is also the clearest look at the triangular mesh facets; they're everywhere.

Right panel is Bz sampled along the single leftmost column (x = −41.95 µm), every standoff overlaid. Two sharp bipolar spikes at y ≈ −21 and −15 reaching ±90 µT at z = 0.5 (dark), collapsing toward flat by z = 10 (yellow). Everything else along that edge is ~0.

Why it matters: an FFT treats the plane as periodic. At x = +98 µm the field is 0.001 µT; at x = −42 µm it's 90 µT. Wrapping those together manufactures a step edge that isn't in the physics, and steps leak across all spatial frequencies. Only this one edge has the problem — and it fades fast with z, so it's mostly a low-standoff concern.

---

linecuts.png — one row through the coil, every standoff

A horizontal slice at y = 20.05 µm (through the spiral's top arm), plotted for all 11 heights.

Top panel, absolute. Sharp positive spikes at x ≈ −19 and +20 (the outer conductor edges) with a deep negative plateau between. At z = 0.5 the plateau runs −60 to −160 µT; by z = 10 the whole feature is nearly flat. This is the decay.

Bottom panel, each curve divided by its own peak. Amplitude removed, so only shape remains — this is the blur, isolated. Watch the edges at x ≈ ±20: dark (low z) curves turn over in about a micron; yellow (z = 10) curves take ~10 µm. And the tails beyond |x| > 25 rise with height, because flux spreads laterally as you move away.

Now look at the jaggedness. At z = 0.5–1 the plateau oscillates by ±30 µT on ~1 µm scales, in a pattern that isn't periodic. The real field 0.5 µm above 2 µm traces on a 4 µm pitch would be a smooth, regular ripple at 0.25 cyc/µm — not random steps. Those steps are the piecewise-linear mesh interpolant. And in the normalized panel the green/yellow curves are visibly noisier than the dark ones in the middle, which is the additive-noise-floor signature: the true signal shrinks with height while the numerical noise doesn't.

This is the single most convincing figure to put in front of the COMSOL person, because "the field has stair-steps" needs no analysis to accept.

---

spectra.png — the actual finding

Left: radially averaged spectrum of Bz at each standoff. Big peak near k ≈ 0.05 cyc/µm (the coil's overall ~20–40 µm size), a knee around k ≈ 0.1–0.15, then a long shallow tail out to k = 2.

That tail is the giveaway. Physical content at k = 1 cyc/µm and z = 10 µm would be suppressed by exp(−2π·1·10) ≈ 10⁻²⁷ relative to the surface — nothing survives. Yet the z = 10 curve sits at ~10⁻⁵ µT out there. The entire high-k tail is numerical, by about twenty orders of magnitude.

Right: the measurement. Solid = measured ratio |B̃(k,z)| / |B̃(k, 0.5 µm)|. Dashed = the exact analytic exp(−2πkΔz), color-matched to its pair.

Read it left to right: solid sits on dashed at low k — the physics is right and the data is clean there. Then solid peels off and flattens into a plateau while dashed plunges. The k where they part company is the usable bandwidth for that pair (0.375 cyc/µm at Δz = 0.5, down to ~0.11 at Δz = 9.5). Past that point you're measuring the ratio of two noise floors, which is why it's flat rather than decaying.

The small bumps in the solid curves around k ≈ 0.25–0.4 are plausibly the 4 µm winding pitch briefly poking above the floor — which is precisely the frequency your whole study cares about, and precisely where it's being lost.

---

One caveat I'd keep attached to this figure: it's Hann-windowed to suppress that x_min edge, so it's a diagnostic, not the final unbiased estimate. The breakpoints themselves are robust — I re-measured them with no window, a Tukey taper, and an interior crop, and they agreed within a few percent, which is what ruled out edge leakage as the cause.


Gist: past Δz ≈ 2 µm the mesh noise buries the winding-scale content (0.25 cyc/µm), so those pairs can't constrain the kernel at the length scale you actually care about.

Two corrections to the phrasing:

1. It's Δz — the gap within the pair — not absolute height. z = 5 → 6 µm is a better-conditioned pair than z = 0.5 → 5 µm, even though it sits higher up. Your anchoring-everything-to-0.5 scheme is what pushes most pairs past the limit.
2. The images aren't useless, the fine structure is. Low-k content still propagates correctly at every Δz — analytic propagation matches to ~13% even at Δz = 4.5. What dies is the high-k band, and the 4 µm winding pitch happens to sit right at the edge of it. So a wide-Δz pair will still yield a kernel; it'll just be well-determined where you already knew the answer and noise-fitted where you don't.

Practical upshot: anchored at z = 0.5, you have two genuinely informative pairs (→1, →2). Anchored at z = 1, you have one more usable one. That's the constraint — and it's why the convergence pair from the COMSOL side is worth more than any algorithm work right now.



Here's what to send. The key move is to give them a number they can verify themselves, not "the data looks noisy."

The ask in one line

▎ The exported cut-planes are limited by mesh interpolation, not by physics — I need the air region above the coil meshed finer than the 0.1 µm export grid, plus one convergence pair so we can confirm the fix.

Paste-ready message

▎ Hi — I've been working with the 11 cut-plane exports from Multilayer_square_coil_2um_4um_3L_4_COILS.mph (z = 0.5…10 µm, 600 µA, 0.1 µm export grid). Found something I think is a meshing artifact rather than a physics result, and I'd like your read.
▎
▎ The check. Above the conductors the field is source-free, so any two cut-planes are related exactly, with no free parameters: B(k, z₂) = B(k, z₁)·exp(−2π·k·Δz), the same for Bx, By and Bz. That means I can predict any plane from any lower plane and measure the error.
▎
▎ What I get. The measured transfer follows the analytic curve at low spatial frequency, then peels away and flattens into a floor:
▎
▎ | Δz (µm) | departs above k = | i.e. features finer than |
▎ |---------|-------------------|--------------------------|
▎ | 0.5     | 0.375 cyc/µm      | 2.7 µm                   |
▎ | 1.0     | 0.31              | 3.2 µm                   |
▎ | 2.0     | 0.18              | 5.6 µm                   |
▎ | 5.0     | 0.11              | 9.0 µm                   |
▎
▎ Propagating one plane to another leaves 13–17% RMS error even at Δz = 1 µm, where an exact analytic relation on converged data should give well under 1%.
▎
▎ Why I don't think it's my analysis. The breakpoint is unchanged whether I use the full plane, a tapered window, or an interior crop — so it isn't FFT edge leakage. And the artifact is directly visible: the field maps show triangular facets across the windings, and line cuts through the coil have stair-steps of tens of µT. That's the shape of a piecewise-linear interpolant of a mesh coarser than my 0.1 µm sample grid.
▎
▎ Why it matters here. The tracks are 2 µm wide on a 4 µm pitch, so the winding pattern's own fundamental is at 0.25 cyc/µm. For Δz ≥ 2 µm that fundamental is already at or below the floor — the exports stop carrying usable information about the coil structure at exactly the length scale the study is about.
▎
▎ What would help, roughly in order of leverage:
▎
▎ 1. Refine the mesh in the air slab above the coil — z ≈ 0 to 12 µm over the coil footprint — to a maximum element size at or below ~0.5 µm. A swept/mapped mesh in z over that slab would be cheaper than refining everything. This is the root fix.
▎ 2. Raise the discretization order for A (quadratic → cubic). Since B = ∇×A, quadratic A gives only piecewise-linear B, which is exactly the faceting I'm seeing. This may buy a lot without much mesh cost.
▎ 3. One convergence pair, so we can prove it's fixed: the same z = 1 µm and z = 2 µm planes exported at two mesh refinements (current, and refined). If the breakpoint moves up with refinement, that confirms the diagnosis and tells us where the new ceiling is. This is the single most useful thing you could send — more than a full re-run.
▎ 4. Mesh statistics in the sampled region — element count across a 2 µm trace and across the 2 µm gap, and ideally a mesh plot on the z = 1 µm cut plane. Lets me sanity-check the export grid against the mesh instead of guessing.
▎
▎ Two smaller things while the model is open:
▎
▎ 5. The FOV is off-centre: the spiral sits at x ∈ [−20, 22], y ∈ [−22, 22] inside a 140 × 140 µm window, so more than half the export is empty while two feed traces leave through the x_min boundary still carrying ~90 µT. Re-centring the cut-plane window (or extending it left) would cost nothing and would remove a real edge artifact from the frequency analysis.
▎ 6. Finer z spacing near the surface — z = 0.5, 0.75, 1.0, 1.25, 1.5 — would let me compare reference planes at matched Δz, which I currently can't do below 1 µm.
▎
▎ Happy to send the figures and the spectral plots, or to walk through the check on a call.

What to send with it

Attach three, they carry the whole argument:

- figures/planes/Bz_z02.0.png — the facets, visible by eye
- figures/linecuts.png — the stair-steps, hard to explain as physics
- figures/spectra.png — measured transfer vs analytic; the right panel is the finding

If you only get 30 seconds with them

▎ "The exports are mesh-limited above ~0.2 cycles/µm, and our track pitch is 0.25 — so we're right at the edge. Can you refine the air mesh above the coil below 0.5 µm and send me z = 1 and z = 2 at the old and new mesh so I can confirm it moved?"

Item 3 is the one to push for. A convergence pair is cheap for them, and it converts "I think this is a mesh artifact" into a measurement — if refining doesn't move the breakpoint, my diagnosis is wrong and we've learned that for the price of two exports.


past Δz ≈ 2 µm the mesh noise buries the winding-scale content (0.25 cyc/µm), so those pairs can't constrain the kernel at the length scale you actually care about.

Two corrections to the phrasing:

1. It's Δz — the gap within the pair — not absolute height. z = 5 → 6 µm is a better-conditioned pair than z = 0.5 → 5 µm, even though it sits higher up. Your anchoring-everything-to-0.5 scheme is what pushes most pairs past the limit.
2. The images aren't useless, the fine structure is. Low-k content still propagates correctly at every Δz — analytic propagation matches to ~13% even at Δz = 4.5. What dies is the high-k band, and the 4 µm winding pitch happens to sit right at the edge of it. So a wide-Δz pair will still yield a kernel; it'll just be well-determined where you already knew the answer and noise-fitted where you don't.

Practical upshot: anchored at z = 0.5, you have two genuinely informative pairs (→1, →2). Anchored at z = 1, you have one more usable one. That's the constraint — and it's why the convergence pair from the COMSOL side is worth more than any algorithm work right now.