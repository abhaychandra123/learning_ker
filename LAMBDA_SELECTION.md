# Patch-based Morozov lambda selection

Run one component at a time on the primary coil cache:

```bash
python scripts/select_lambda.py --z 3 --comp Bz --select-roi --output runs/lambda_bz_z3
```

1. Drag a rectangle on the observed blurred image over an approximately constant field region. Avoid edges, field gradients and numerical boundary artifacts. The region need not have zero mean. Prefer enough pixels to estimate variance reliably, without extending into spatial variation.
2. Press Enter to preview the patch with a local color scale, row/column mean profiles and a mean-subtracted histogram. Press R to reselect, or Enter to accept and start the search. Closing the preview cancels. The histogram is a diagnostic, not proof of Gaussian independent noise.
3. The script estimates `variance = np.var(patch, ddof=1)`. Only the patch mean is subtracted; no filtering or detrending is applied. Variance is in µT², standard deviation in µT.
4. Set the discrepancy target to `b.size * variance`, counting the full observed image, excluding FFT padding. The estimated noise variance must be representative of that image. Patch size controls variance estimation, not the full-image noise-energy multiplier.
5. Learn the same joint kernel as `deblur.py` from all three .5/target component pairs, and freeze it throughout the search. This remains a same-sample, reference-assisted experiment. Kernel/model error is not automatically explained by the noise estimate; COMSOL artifacts may be structured. A patch noise estimate does not remove those limitations.
6. Starting from λ=1e-4 (clipped at the zero-solution threshold `max(abs(A.T b))`), bracket the discrepancy target by factors of ten and refine by logarithmic bisection. Each trial solves `0.5*sum((Ax-b)**2) + lambda*sum(abs(x))` to the requested PG epsilon. FISTA is the default; `--solver ista` uses ISTA for the same objective. There is no user-facing iteration budget. Live progress and actual update counts are recorded.
7. Accept a candidate when `abs(sum((b-Ax)**2)/(n*variance)-1) <= 0.05`. Change the 5% tolerance with `--discrepancy-tolerance`. This is separate from solver `--epsilon` (default .001).
8. Verify that candidate with a ten-times tighter epsilon. Accept only if both solves converged, the tighter solution remains within the discrepancy tolerance, and the change in squared residual is at most one-fifth of that tolerance times the target. Otherwise save diagnostics, report that verification failed, and require a tighter epsilon in a new run. Unconverged solves and unmatched targets are never reported as successful selections. An internal 35-trial guard stops an unbracketed/unresolved search.
9. Read `index.html`: selected patch, λ/discrepancy curve, reconstruction error for evaluation only, actual update counts, convergence, reconstruction, baseline λ=1e-4 when evaluated, and reblurred residuals. The sharp truth never determines λ. `noise.json`, `lambda_trials.csv`, `histories.json`, `metrics.json`, kernel and reconstruction arrays preserve the numerical results and provenance. Trial CSV/history files are saved after every solve.

To inspect only the noise patch before spending time on reconstructions:

```bash
python scripts/select_lambda.py --z 3 --comp Bz --select-roi --estimate-only --output runs/noise_bz_z3
python scripts/select_lambda.py --z 3 --comp Bz --roi-from runs/noise_bz_z3/noise.json --output runs/lambda_bz_z3
```

If an interactive matplotlib window is unavailable, pass `--roi X0 X1 Y0 Y1` instead of `--select-roi`. These are zero-based pixel coordinates; upper bounds are exclusive. The image origin is at the lower left in the displayed plot. A reused ROI must have the same component/height; variance is recomputed from the current data. Each output directory must be new.

Do not use a tiny patch or choose the flattest numerical region merely to force a small noise estimate. A region with genuine signal gradients overestimates noise; a quiet patch with unusually low numerical error may underestimate error elsewhere. Ideally inspect multiple physically flat regions and compare estimates before interpreting selected λ scientifically.
