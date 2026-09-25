1. Recovering near-field magnetic structure

Present the goal as field-plane reconstruction, not current-density inversion. All reported observations and references are actual COMSOL exports. This deck reports two controlled, same-sample experiments. Main slides 1–20; technical backup follows.

2. Actual COMSOL fields define the input and target

Each component is a signed 1400×1400 field image with 0.1 µm pixel spacing. The observation is the z=3 cached COMSOL array, not an analytically propagated z=0.5 image. The target is the actual z=0.5 reference. Cropped display below shows the Bz row from the original first-run figure; the complete images appear in backup.

3. Upward continuation suppresses fine spatial detail

In a source-free homogeneous upper half-space with the decaying solution, each lateral Fourier mode decays exponentially with height. f is in cycles per micrometre. Reversing this attenuation amplifies discrepancies. This motivates regularization, but the experiment does not substitute the ideal exponential for its fitted transfer. See scripts/propagator.py for the analytical utility.

4. One shared transfer is fitted from all three component pairs

learn_kernel_joint in deblur.py minimizes the sum of squared Fourier residuals across Bx, By and Bz on the zero-padded grid. It is unregularized least squares with H set to zero where the absolute denominator is <=1e-12. H is fixed during LISTA training. The fit is not forced to be radial, positive or bounded by one. The reference is already used at this calibration stage, including for the baselines. Full padded fitting and cropped reconstruction losses have different domains.

5. Padding and cropping are part of the operator

P zero-embeds the original image and C crops the same region, so C=P*. A=C T_H P, and the adjoint uses the conjugate transfer. Intermediate crop and re-padding in A* A are essential: replacing the composition by a single |H|² multiplier would generally change the operator. 32 pixels equals 3.2 µm per side. The padded FFT still implements circular convolution, with a zero-extension assumption; it does not establish an artifact-free physical boundary.

6. ISTA and FISTA solve the same pixel-domain L1 problem

x is a signed field image. The first term is a sum of squared residuals, not a mean; the second is the sum of absolute field values. λ=10^-4 in microtesla units is inherited from the existing ISTA experiment, not selected by validation. L1 on pixels encourages small backgrounds but can suppress legitimate weak fields. There is no TV, wavelet, positivity, current prior or explicit Maxwell constraint. Treat this as a chosen convex objective, not a proof of physical truth.

7. ISTA corrects the residual, then soft-thresholds

For f(x)=0.5||Ax-b||², the gradient is A*(Ax-b). The proximal map of αλ||x||1 is soft thresholding. ISTA majorizes the smooth objective with a quadratic upper bound and minimizes that bound plus the L1 penalty. x0=b matches the repository initialization. The name is Iterative Shrinkage-Thresholding Algorithm. The norm bound and unit conventions are covered in backup.

8. FISTA adds momentum to the same proximal update

Standard Beck–Teboulle FISTA uses y1=x0 and t1=1. The first step matches ISTA because momentum is initially zero. Return xk, not the extrapolated y. This implementation has no restart, backtracking or monotonicity modification. Under convex assumptions the objective-gap rate is O(1/k²), versus O(1/k) for ISTA. These are not guarantees on reference-image error or monotonic objective decrease at every step. Source: https://www.ceremade.dauphine.fr/~carlier/FISTA

9. ConvLISTA learns corrections to a finite ISTA sequence

This is a structured supervised LISTA variant, not a pure finite-filter replacement of the physics operator. Wb=αA*+Cb and Wx=I−αA*A+Cx. Both correction filters are bias-free, one input/output channel, stride one, 9×9, and tied across depth. θ is one scalar shared over space, components and layers. Input shape [3,1,1400,1400] treats components as batch samples, not feature channels. The whole layer retains global FFT interactions. No claim of convergence beyond trained depth. Source: scripts/lista.py.

10. The supervised target changes what LISTA is trying to do

Original LISTA learns to approximate sparse codes obtained by optimization. Here the user-selected target is the actual sharp COMSOL field. Only the final layer is supervised with full-image MSE. No additional data-consistency or L1 penalty is in the training loss. Therefore learned intermediate iterates need not reduce J, and learned maps need not be proximal gradient maps of any convex objective. Original paper: https://icml.cc/Conferences/2010/papers/449.pdf

11. Only depth and baseline iteration budget changed

The two JSON files have identical input and source hashes. Both CSV tables agree numerically with their JSON endpoint metrics. Run 1 uses 8 layers and 200 baseline iterations; run 2 uses 16 and 500. Both include matched-depth baselines. Training uses 200 full-batch Adam updates with lr=1e-4 and best training-loss selection. Same λ, pad, normalization, seed, hardware and code. Equal epochs do not imply equal training compute. Deterministic float32 GPU execution without AMP or TF32.

12. At matched depth, LISTA has lower reference error

Every bar is taken from the saved JSON endpoint metrics, checked against CSVs. At both matched depths all three components have lower range NRMSE under trained LISTA. This comparison does not imply equal operation counts: LISTA adds convolutional corrections. Short-run batch inference is 0.216/0.211/0.240 s for ISTA8/FISTA8/LISTA8 and 0.418/0.431/0.460 s for ISTA16/FISTA16/LISTA16. These are same-sample supervised results, not held-out accuracy.

13. More computation affects the three methods differently

Cross-run reference-error reductions: LISTA 3.46%, 2.66%, 3.44%; ISTA 6.38%, 5.43%, 9.28%. FISTA reference-error increases: 32.85%, 34.92%, 11.57%. The settings other than depth and iteration budget are controlled. This is one deterministic experiment per depth; no variability estimates or generalization claims. Baseline iteration count does not influence LISTA training.

14. FISTA reference error reaches a minimum before 500 steps

The full run-2 history is available. FISTA's lowest recorded range NRMSE occurs at k=60 for Bx (1.616636%) and By (1.460708%), and k=250 for Bz (1.795615%). Metrics are sampled every ten iterations, so these are best recorded points, not exact minima. Selecting an iteration using the reference is an oracle selection; deployment requires separate validation or a justified observation-only rule. ISTA's best recorded points are at the 500-step endpoint in all components. The LISTA path ends at trained depth 16.

15. Optimization improves even when reference agreement worsens

For Bx, FISTA200→500 reduces J from 16576.105 to 5691.415 and PG RMS from 0.00327225 to 0.000637219, while range NRMSE increases from 1.86991% to 2.48416%. Analogous endpoint worsening occurs in By and Bz. This is evidence of objective/reference mismatch, not algorithm divergence. Semi-convergence is a useful description of the observed reference-error path; the physical cause of discrepancies is not isolated. LISTA is trained against a different loss.

16. Sixteen layers improve training MSE at twice the training cost

Run 1 best training epoch=200, loss=0.001674862229; run 2 best epoch=200, loss=0.001565922867. Training durations include post-update evaluation: 150.1225 and 309.3839 seconds. Final physical thresholds θq are 0.746789 and approximately 0.31694 µT. Their difference is not an interpretable lambda change because the learned linear maps also change. Both use one shared nonnegative threshold. Each curve is final-output loss across epochs, not convergence across layers.

17. The useful comparison is reconstruction error versus cost

All endpoint latencies are warmed synchronized medians of three repeats, on the same single GPU and batch of three, excluding metrics, file I/O, kernel fitting and training. Color indicates algorithm, marker indicates run, labels give iterations/layers. LISTA8 0.239503 s; LISTA16 0.459688 s. ISTA200/500 5.180782/13.073968 s; FISTA200/500 5.303321/13.485866 s. Long baselines have different accuracy, so no equal-accuracy 20x/30x speedup is claimed. Three timing repeats do not provide a broad benchmark distribution.

18. Run 2: sharper structure and background artifacts must both be assessed

This is the Bz row of reconstructions-2.png, cropped by PowerPoint without modifying the source image. Columns are observed, reference, ISTA500, LISTA16, FISTA500. Shared row display limits are ±99.5th percentile of |reference|, so extreme values saturate; there is no independent autoscaling per reconstruction. Smoothness is not sufficient evidence of fidelity. FISTA500 has lower Bz endpoint error than ISTA500, and both improve on LISTA16. FISTA200 has lower Bz endpoint error than FISTA500. The speckle's cause has not been isolated.

19. The evidence supports a tradeoff, not a universal winner

Among the endpoint rows in the two CSVs, ISTA500 is best for Bx and By; FISTA200 is best for Bz. Across saved histories, FISTA250 slightly improves Bz further, but that is reference-based oracle selection, not a validated configuration. LISTA wins reference error at matched depth in both runs. FISTA500 gives the smallest objective and stationarity residual among endpoints. Range NRMSE divides by full reference range and is not percent accuracy. LISTA16 relative L2 errors are 42.15/43.95/48.44%.

20. Next: separate the objective, calibration and evaluation questions

Proposed follow-up work, not completed experiments. First clarify whether the research goal is solver acceleration or physical reference reconstruction. Use separate validation data for λ and stopping/depth selection, and independent test observations for final comparisons. Assess forward residuals and boundaries separately; consider analytical or independently calibrated transfers if justified. Compare at matched accuracy or latency and count training costs. Repeat over seeds/data to assess robustness. Main presentation ends here; backup slides follow.

21. Backup: derive shrinkage from the proximal subproblem

Start with f(x)=0.5||Ax-b||². Gradient is A*(Ax-b). The quadratic upper bound with step α leads to minimizing 0.5||u-v||²+αλ||u||1, where v=x−α∇f(x). The problem separates over pixels. For positive u, derivative gives u=v−θ; for negative u, u=v+θ; if |v|≤θ the solution is zero. This explains soft thresholding and why signed field values are preserved. L1 penalizes amplitudes; it is not an edge-preserving TV penalty.

22. Backup: norm bound, units and initial threshold

C and P have operator norm one; padded circular convolution has norm max|H|. Thus Ltrue=||A||²≤Lbound=max|H|²=4.478589576. α=0.99/Lbound=0.221051736. With x=q x_tilde and q=215.1441345 µT, J(q x_tilde)=q²[0.5||A x_tilde−b_tilde||²+(λ/q)||x_tilde||1]. Normalized λ=4.64804677e-7 and initial θ=1.02745879e-7. Physical fixed threshold=2.21051736e-5 µT. The shared scale uses both reference and observed images; saved inference must reuse the checkpoint scale.

23. Backup: training and tensor implementation

DataParallel splits batch size three into 2+1. MSE is computed after gathering predictions, preventing unequal chunk weighting. Model parameters are only the two filters and threshold; H real and imaginary arrays are nontrainable buffers. Direct θ parameter is projected nonnegative after Adam; clamp_min also protects forward evaluation. Activation checkpointing recomputes layers during backpropagation to reduce memory. Best weights are selected after post-update reevaluation. Kernel fitted in float64, GPU inference float32; no AMP or TF32. These details do not establish generalization.

24. Backup: the metrics answer different questions

Reference RMSE is in microtesla. Range NRMSE normalizes by max(s)−min(s); relative L2 normalizes by ||s||. The denominator difference explains seemingly small range NRMSE despite sizeable relative L2. PG is the proximal-gradient mapping of the original objective, not the training gradient of the learned network. PG=0 is optimality for the convex baseline objective, but small PG alone does not bound reference error or distance to a unique optimizer in an ill-conditioned system. All field metrics use the full field of view.

25. Backup: map the mathematics to source files

Notebook extracts and verifies the bundle, runs tests, launches scripts/lista.py and displays/export results. lista.py CLI delegates to compare_lista.py, while its ConvLISTA class is independently importable. compare_lista orchestrates fitting, scaling, baseline solving, training, metrics and benchmarks. GPU baselines use torch_inverse.py; standalone CPU FISTA uses iterative.py. Original deblur.py remains the fit/operator reference. Loading the checkpoint includes saved H, scale and original norm bound to preserve inference behavior.

26. Backup: checks support implementation, not physical optimality

Tests in test_iterative.py and test_lista.py cover adjoint identities, dense references, known L1 solution, ISTA initialization parity, finite-difference gradients, normalization, checkpointing and uneven two-GPU gradients. The first local run previously passed 12 with the two-GPU test skipped; notebook includes that check, but these result files do not store its test log. Therefore this slide describes coverage and run-time initialization result, without inventing a GPU test count. Both run JSONs report initialization max error 0. No independent samples or uncertainty estimates.

27. Backup: run 1 endpoint metrics

Numbers are read directly from the corresponding metrics JSON; CSVs were checked for agreement. Latency is a median for the batch of all three components, not per component. Range NRMSE values are percentages; JSON stores fractions. All comparisons are same-sample. The listed best endpoints must not be confused with minima selected over histories.

28. Backup: run 2 endpoint metrics

Numbers are read directly from the corresponding metrics JSON; CSVs were checked for agreement. Latency is a median for the batch of all three components, not per component. Range NRMSE values are percentages; JSON stores fractions. All comparisons are same-sample. The listed best endpoints must not be confused with minima selected over histories.

29. Backup: complete reconstruction grid — run 1

Original supplied figure, preserved without pixel editing. Columns compare observed, reference, long ISTA, trained LISTA, long FISTA. Shared per-row display range is ±99.5th percentile of absolute reference; extremes can saturate. The same-sample label and unequal iteration budgets are explicit. Numerical arrays were not provided in this results folder, so no new physical ROI metrics or residual maps were synthesized.

30. Backup: complete reconstruction grid — run 2

Original supplied figure, preserved without pixel editing. Columns compare observed, reference, long ISTA, trained LISTA, long FISTA. Shared per-row display range is ±99.5th percentile of absolute reference; extremes can saturate. The same-sample label and unequal iteration budgets are explicit. Numerical arrays were not provided in this results folder, so no new physical ROI metrics or residual maps were synthesized.

31. Backup: provenance and primary references

Results are supplied in notebooks/runs_results/metrics.json and metrics-2.json, with matching source/input SHA256 dictionaries. CSVs numerically agree with the JSONs. Source references are Gregor & LeCun (2010), Learning Fast Approximations of Sparse Coding, ICML; Beck & Teboulle (2009), A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems, SIAM J Imaging Sciences; PyTorch 2.10 Conv2d and Adam documentation. No new reconstruction or training was run to create the slides; plots are regenerated from saved metrics. Build script: presentations/build_professor_deck.py.