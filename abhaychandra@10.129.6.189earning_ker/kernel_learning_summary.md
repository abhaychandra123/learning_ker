# Kernel Learning: What We Did and What We Found

## Step 1: Unregularised Kernel (per-component)

**Script:** `scripts/kernel.py` (original version)

**Approach:** For each pair (reference z=0.5, target z), learn a separate kernel per component (Bx, By, Bz) by pointwise division in Fourier space:

```
K_hat = B / S
```

This is the closed-form least-squares solution to `minimize ||b - K*s||^2`. One pair of images per kernel, holdout pair z=0.5 -> z=2.0 (dz=1.5). Holdout test: apply the dz=0.5 kernel 3 times (0.5 -> 1.0 -> 1.5 -> 2.0).

**Training NRMSE:** 0.000000 for every pair (exact interpolation).

**Holdout NRMSE (applied 3x):**

| Component | Analytic | Learned x3 |
|-----------|----------|------------|
| Bx        | 0.73%    | 581%       |
| By        | 0.74%    | 1109%      |
| Bz        | 0.72%    | 575%       |

**What happened:** Zero training error is a red flag. The kernel has 1.96M parameters for 1.96M data points, so it memorises the noise. At high k, K_hat = noise/noise ~ 1.0 instead of the correct exp(-2pi*k*dz) ~ 0.04. When applied 3 times, the noise amplifies by ~25^3 = 15,000x.

**Figures:** `kernel_vs_analytic.png`, `kernel_real_space.png`, `kernel_radial_profile.png`, `kernel_holdout_test.png`

---

## Step 2: Wiener Regularisation (per-component)

**Script:** `scripts/kernel_regularised.py` (original version)

**Approach:** Add an L2 penalty on the kernel (Wiener deconvolution):

```
minimize ||b - K*s||^2 + lambda * ||K||^2

Solution: K_hat = conj(S) * B / (|S|^2 + lambda)
```

Lambda estimated automatically from the high-k noise floor of the reference plane: median of |S|^2 at k > 1.0 cycles/um.

**Lambda:** 2.62e+06

**Training NRMSE:** 0.09 - 0.25% (nonzero because regularisation deliberately doesn't fit the noise).

**Holdout NRMSE (applied 3x):**

| Component | Analytic | Regularised x3 | Unregularised x3 |
|-----------|----------|----------------|-------------------|
| Bx        | 0.73%    | 5.10%          | 581%              |
| By        | 0.74%    | 5.01%          | 1109%             |
| Bz        | 0.72%    | 4.46%          | 575%              |

**What happened:** Regularisation reduced holdout error by ~100x. Lambda acts as a threshold: frequencies with |S|^2 >> lambda pass through (data dominates), frequencies with |S|^2 << lambda get suppressed (penalty dominates). The remaining 5% error comes from regularisation bias (kernel under-propagates high-k) compounding over 3 steps.

**Figures:** `kernel_reg_vs_analytic.png`, `kernel_reg_real_space.png`, `kernel_reg_radial_profile.png`, `kernel_reg_holdout_test.png`

---

## Step 3: Joint Kernel Across Components

**Script:** `scripts/kernel.py` and `scripts/kernel_regularised.py` (current versions)

**Approach:** The physics says Bx, By, Bz all share the same propagator. Instead of learning 3 separate kernels, fit one joint kernel using all 3 pairs simultaneously:

```
Unregularised:
  K_hat = (Sx*Bx + Sy*By + Sz*Bz) / (|Sx|^2 + |Sy|^2 + |Sz|^2)

Regularised:
  K_hat = (Sx*Bx + Sy*By + Sz*Bz) / (|Sx|^2 + |Sy|^2 + |Sz|^2 + lambda)
```

Same holdout setup: z=2.0 held out, dz=0.5 kernel applied 3 times.

**Training NRMSE (joint unreg):** 0.49 - 1.01%
**Training NRMSE (joint reg):** 0.50 - 1.01%

**Holdout NRMSE (applied 3x):**

| Component | Analytic | Joint Unreg x3 | Joint Reg x3 | Per-comp Unreg x3 | Per-comp Reg x3 |
|-----------|----------|----------------|--------------|---------------------|-----------------|
| Bx        | 0.73%    | 0.87%          | 0.81%        | 581%                | 5.10%           |
| By        | 0.74%    | 0.93%          | 0.86%        | 1109%               | 5.01%           |
| Bz        | 0.72%    | 0.98%          | 0.91%        | 575%                | 4.46%           |

**What happened:** The joint fitting itself acts as regularisation. The mesh noise in Bx, By, Bz is independent (each component cuts through the mesh differently), so in the denominator |Sx|^2 + |Sy|^2 + |Sz|^2, noise partially cancels while signal adds coherently. The joint unregularised kernel is nearly as good as the analytic propagator. Adding lambda on top gives a small further improvement (0.81-0.91% vs 0.87-0.98%), but the joint constraint did the heavy lifting.

**Figures:** `kernel_vs_analytic_joint.png`, `kernel_real_space_joint.png`, `kernel_radial_profile_joint.png`, `kernel_holdout_test_joint.png`, `kernel_reg_vs_analytic_joint.png`, `kernel_reg_real_space_joint.png`, `kernel_reg_radial_profile_joint.png`, `kernel_reg_holdout_test_joint.png`

---

## Step 4: Mesh Noise Measurement

**Script:** `scripts/mesh_noise.py`

**Approach:** The analytic propagator B_hat(k, z2) = B_hat(k, z1) * exp(-2pi*k*dz) is exact in source-free space. So propagate z1 analytically to z2 and subtract the actual z2 data. Any nonzero residual is mesh interpolation artifacts from COMSOL, not physics.

Tested all 55 plane pairs (every combination of source and target).

**Results (selected pairs for Bz):**

| z1   | z2   | dz   | NRMSE  |
|------|------|------|--------|
| 0.5  | 1.0  | 0.5  | 1.12%  |
| 0.5  | 2.0  | 1.5  | 0.72%  |
| 0.5  | 5.0  | 4.5  | 1.37%  |
| 0.5  | 10.0 | 9.5  | 5.20%  |
| 1.0  | 2.0  | 1.0  | 0.70%  |
| 1.0  | 3.0  | 2.0  | 0.80%  |
| 5.0  | 10.0 | 5.0  | 2.82%  |

**Key findings:**
- Mesh noise is never zero for any pair, proving the COMSOL data contains artifacts that violate the exact physics.
- Error grows with dz (0.7% at dz=1 to 5.2% at dz=9.5).
- Error grows when the target z2 is large (weaker signal, so noise is a larger fraction).
- The signal-vs-noise spectrum shows that at large dz, the mesh noise dominates above k ~ 0.3 cycles/um.
- The holdout pair (z=0.5 -> z=2.0) has a mesh noise floor of 0.72%, which matches the analytic propagator NRMSE exactly. The joint learned kernel at 0.81-0.91% is within ~0.15% of this floor.

**Figures:** `mesh_noise_heatmap.png`, `mesh_noise_residuals.png`, `mesh_noise_spectrum.png`, `mesh_noise_vs_dz.png`

---

## Final Summary Table

| Method                      | Holdout NRMSE (Bz) | Notes                                    |
|-----------------------------|---------------------|------------------------------------------|
| Per-component unreg (B/S)   | 575%                | Catastrophic overfitting                 |
| Per-component Wiener reg    | 4.46%               | Lambda kills noise, but bias compounds   |
| Joint unreg                 | 0.98%               | Joint fitting regularises naturally      |
| Joint Wiener reg            | 0.91%               | Small improvement over joint unreg       |
| Analytic propagator         | 0.72%               | Exact theory, this is the noise floor    |
| Mesh noise floor            | 0.72%               | Hard limit from COMSOL data quality      |

The learned kernel is within 0.19% of the theoretical best achievable on this dataset.
