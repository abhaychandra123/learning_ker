# CSC-LISTA at one standoff, 3 → 0.5 µm

1 November 2026. CSC-LISTA (convolutional sparse coding LISTA) is Eq. 20, the
convolutional LISTA, of the ALISTA paper: Liu, Chen, Wang and Yin, "ALISTA: Analytic
Weights Are As Good As Learned Weights in LISTA" (ICLR 2019). It is trained on
simulated dataset v3 and scored on 15 COMSOL component pairs. One seed. Code:
`scripts/csc_lista.py`, `scripts/train_csc_lista.py`, `tests/test_csc_lista.py`.

It is not the old in-house ConvLISTA (`scripts/lista.py`, `combined_lista.py`,
`geometry_lista.py`), which learns two tied corrections on field pixels and has no
published source. CSC-LISTA works on codes in a learned dictionary. Until 5 November
2026 it was also called Conv LISTA, with files and runs named `conv_lista`. See
`runs/RENAMED.md` for the mapping.

## Model

```text
x_m <- shrink(x_m - w_m' * (sum_m d_m * x_m - b), theta)
d_m = h * g_m
field = sum_m g_m * x_m
```

`h` is the fixed `exp(-kz)` kernel and `g_m` is a learned convolutional dictionary. Eq.
20 and the dictionary-first procedure come from the paper. Placing `h` inside `d` for
deblurring is our addition, because the paper's experiment is denoising.

## Differences from the published methods

Compared with Eq. 20 and Appendix H of Liu et al.:

| Item | Paper | Here | Status |
|---|---|---|---|
| Layer update | Eq. 20 | Same | Matches |
| Output field | Codes convolved with the original dictionary | Same | Matches |
| Thresholds | One per layer | Same | Matches |
| Dictionary kernels | Small learned filters | `d_m = h * g_m`, long blurred kernels | Our extension, needed for deblurring |
| Problem tested | Denoising | Deblurring | No published result for this use |
| Filter `w` size | Same size as `d` | 155 px, shorter than the long `d` | Our change |
| Dictionary learning | Algorithm of Liu et al. 2018 | Alternating FISTA and Adam | Our substitute |
| Sparse threshold floor | None | `theta` held at or above its start | Our addition, sparse runs only |
| Relative loss | MSE | Per-crop relative L2 | Our addition, relative runs only |

The plain MSE run has no floor and no relative loss. It is the run closest to the paper.

Momentum is not in Liu et al. The layer has the form of FISTA-Net Eq. 8a-c (Xiang et al.),
but FISTA-Net fixes `W` to an analytic matrix scaled by a learned step, uses a CNN proximal
step, and gives its weight a softplus schedule. Here `w_m` and each `beta` are free. The
layer also equals LFISTA (Moreau and Bruna) with `W_g = (1 + beta)(I - WD)`, `W_m = -beta(I - WD)`
and `W_e = W`, but LFISTA uses dense unconstrained matrices. This combination is our own.

## Setup

- 32 atoms of 75 px and `w` of 155 px per layer. 16 layers give 12.3 M parameters.
- Dictionary lambda 0.3: 6.4% relative L2 and 2.4% density on simulated validation.
- Trained on 512 px crops of the 0.1 µm fields, batch 8, Adam with a rate of 3% of
  each parameter's starting scale and cosine decay. MSE runs took 2,000 steps and
  relative-loss runs 1,000.
- Settings and checkpoints were chosen on simulated validation crops. COMSOL did not
  steer any choice. Scores use the full 1400 × 1400 images.
- Filter size did not matter in 800-step trials: 155/255/355 px gave 52.35/52.02/51.99%.

## Results

Mean relative L2. Validation uses 64 signal-rich crops, so it is not comparable with
the full-image COMSOL scores.

| Method | Simulated validation | COMSOL (15 pairs) | COMSOL NRMSE |
|---|---:|---:|---:|
| ISTA-16, same dictionary | 64.3% | 64.4% | 4.60% |
| Wiener | 60.3% | 49.3% | 3.53% |
| CSC-LISTA, plain, MSE, 16 layers | 53.4% | 39.1% | 2.80% |
| CSC-LISTA, sparse, MSE, 16 layers | 48.6% | 40.2% | 2.88% |
| CSC-LISTA, sparse, relative loss, 16 layers | 45.6% | 42.0% | 3.01% |
| CSC-LISTA, sparse, relative loss, 32 layers | 42.5% | 38.9% | 2.80% |
| CSC-LISTA, strict, MSE, 32 layers | 51.9% | 35.4% | 2.54% |
| CSC-LISTA, strict, MSE, 64 layers | 48.8% | 33.8% | 2.43% |
| Convolutional LFISTA-style, Lasso loss, 32 layers | 42.8% | 34.2% | 2.47% |
| FISTA-Net run 1, paper start values, 750 updates | 65.2% | 39.4% | 2.85% |
| **FISTA-Net run 2, authors' start values, 9,000 updates** | **47.9%** | **33.3%** | **2.41%** |
| FISTA-32, same dictionary, lambda 0.03 | 50.9% | 41.8% | 3.00% |
| FISTA-64, same dictionary, lambda 0.009 | 45.3% | 37.9% | 2.72% |
| FISTA-300, same dictionary, lambda 0.03 | 36.6% | 32.1% | 2.33% |
| FISTA-300, same dictionary, lambda 0.009 | 35.5% | 30.5% | 2.20% |

Strict means no threshold floor and no relative loss. Its 32-layer and 64-layer best
checkpoints came at steps 300 and 200 of 1,000.

FISTA lambda was picked per depth on the first 16 validation crops, from 7e-7, 0.0003,
0.003, 0.009, 0.03, 0.09, 0.3 and 0.9. The value 7e-7 is the old 1e-4 µT default in these
normalised units, and it is not a direct equivalent, because lambda here acts on
dictionary codes. On those crops FISTA-300 scored 40.0% at 7e-7 and 22.4% at 0.009.

COMSOL relative L2 by configuration:

| Method | L1_C1 | IC_4um | C1+C2 | C1+C2+C3 | C1–C4 |
|---|---:|---:|---:|---:|---:|
| Wiener | 49.3% | 50.1% | 49.1% | 49.0% | 49.0% |
| CSC-LISTA, plain, MSE, 16 layers | 39.7% | 37.4% | 39.5% | 39.4% | 39.4% |
| CSC-LISTA, sparse, MSE, 16 layers | 40.6% | 39.1% | 40.4% | 40.3% | 40.3% |
| CSC-LISTA, sparse, relative loss, 16 layers | 42.3% | 41.6% | 42.1% | 42.0% | 42.0% |
| CSC-LISTA, sparse, relative loss, 32 layers | 39.6% | 36.8% | 39.4% | 39.3% | 39.3% |
| CSC-LISTA, strict, MSE, 32 layers | 36.3% | 32.3% | 36.3% | 36.1% | 36.1% |
| CSC-LISTA, strict, MSE, 64 layers | 34.6% | 30.7% | 34.6% | 34.5% | 34.5% |
| Convolutional LFISTA-style, 32 layers | 35.3% | 30.1% | 35.3% | 35.2% | 35.2% |
| FISTA-Net run 1 | 39.0% | 42.1% | 38.9% | 38.7% | 38.5% |
| FISTA-Net run 2 | 33.1% | 34.6% | 33.2% | 33.1% | 32.7% |
| FISTA-32 | 41.2% | 45.0% | 41.0% | 40.9% | 41.0% |
| FISTA-64, lambda 0.009 | 38.3% | 36.9% | 38.2% | 38.1% | 38.1% |
| FISTA-300, lambda 0.03 | 33.0% | 28.7% | 33.1% | 33.0% | 32.9% |
| FISTA-300, lambda 0.009 | 31.2% | 27.6% | 31.4% | 31.2% | 31.1% |

COMSOL relative L2 by component:

| Method | Bx | By | Bz |
|---|---:|---:|---:|
| CSC-LISTA, strict, MSE, 32 layers | 33.6% | 33.6% | 39.1% |
| CSC-LISTA, strict, MSE, 64 layers | 31.8% | 32.0% | 37.5% |
| Convolutional LFISTA-style, 32 layers | 32.8% | 31.4% | 38.5% |
| FISTA-Net run 1 | 36.2% | 36.2% | 46.0% |
| FISTA-Net run 2 | 30.2% | 30.7% | 39.1% |
| FISTA-32 | 40.0% | 40.2% | 45.2% |
| FISTA-64, lambda 0.009 | 36.2% | 36.4% | 41.2% |
| FISTA-300, lambda 0.03 | 30.2% | 28.5% | 37.8% |
| FISTA-300, lambda 0.009 | 29.0% | 27.8% | 34.7% |

The selected model is FISTA-Net run 2. Among the runs free of our floor and loss changes
it has the lowest simulated validation error, and that rule picks it without using
COMSOL. It has 74,599 parameters. Strict CSC-LISTA at 64 layers, with 49.2 M parameters, is
effectively tied. The lower validation error of the sparse relative-loss run is set
aside because that run uses our additions, not because of its COMSOL score.

## Convolutional LFISTA-style run

Job 3 adds learned momentum to CSC-LISTA and trains on the Lasso objective with lambda
0.03. It is not the published LFISTA, whose dense matrices would need about 16 PB here.
Treat it as exploratory. Its codes are 16.6% dense, and validation improved up to the
last step.

## FISTA-Net

`scripts/fista_net.py`, `scripts/train_fista_net.py` and `tests/test_fista_net.py` follow
Xiang, Dong and Yang (arXiv 2008.02683) and the authors' `M5FISTANet.py`. Choices made
with the user: the code's gradient step and 10-layer CNN, the paper's Eq. 14 loss, Adam
at 1e-3 with the code's decay skipped, the paper's start values, 512 px crops, batch 24
and 30 epochs. It has 74,599 parameters.

Besides crop size and batch, the run differs from the authors' code in its start values
(paper values give a summed gradient step 17 times weaker), its loss, its skipped decay,
and its removed final ReLU, because fields are signed. The score is one unconverged run: validation fell from 106% at step 575 to 65.2% at step 725, moved
between checks by up to 30 points, and training stopped at step 750. The reported
`mu`, `theta` and `rho` hardly moved. 39.4% was not FISTA-Net's limit.

Run 2 used the same code and settings with `--start code` and 360 epochs, or 9,000
updates, about what 30 epochs give on the paper's dataset. It took about 27 hours and
scored 33.3% on COMSOL at step 8,700 of 9,000, against an interim 34.6% at step 4,800.
Validation still moved between 47.9 and 68.5% in the last 1,000 steps, and the learning
rate never decays, so the best step is partly a lucky draw. Its code is frozen in
`experiments/fista_net_code_start`.

## Limits

- With MSE, every best checkpoint came between steps 200 and 300, and validation then
  got worse. With relative loss, validation improved until about step 700–1,000. This
  suggests a loss mismatch, though only one run of each was made.
- Simulated validation does not rank methods the way COMSOL does. The sparse
  relative-loss run beats strict 64 layers on validation (42.5 against 48.8%) and loses
  on COMSOL (38.9 against 33.8%). Simulation and COMSOL differ.
- Strict CSC-LISTA codes are about 99% dense, so these networks are not sparse coders in
  practice. The sparse relative-loss run reached 22% and the dictionary 2.4%.
- At equal depth, strict CSC-LISTA beats FISTA on COMSOL: 35.4 against 41.8% at 32 and 33.8
  against 37.9% at 64. It does not beat FISTA-300 (30.5%). On simulated validation the
  32-layer network loses to FISTA-32 (51.9 against 50.9%), so the gain shows on COMSOL only.
- The FISTA-64 lambda is uncertain: the 16 selection crops chose 0.009, but on all 64
  crops 0.03 was slightly better (44.9 against 45.3%). Both give 38% on COMSOL.
- FISTA-Net run 2 was started because of what run 1 showed, and no other network got a
  second run, so its lead of 0.5 points over strict 64 layers means little.
- Run time was not measured, so no speed advantage for the networks can be claimed yet.
- COMSOL speckle probably puts a floor of about 15–19% under these scores.
- The 15 pairs hold only two wire shapes, so the configuration scores are not
  independent. The dictionary, widths and speckle level were also designed with these
  coils in view.

## Run

The server holds frozen code in `~/learning_ker/experiments/csc_lista_v2` and `_v3`, and
full outputs in `~/learning_ker/runs/csc_lista_v2` and `_v3`. Metrics are copied to the
same local `runs` folders. Code inside the frozen folders keeps the file names it ran
with (`conv_lista.py`, `train_conv_lista.py`). Run from `~/learning_ker`:

```bash
~/qdm-env/bin/python -u experiments/csc_lista_v2/scripts/train_conv_lista.py \
  --dataset runs/simulated_3_to_05_v3 --output runs/<fresh_name> \
  --dictionary runs/csc_lista_v2/dictionary_strong/dictionary_0.3.pt \
  --filter-width 155 --layers 32 --steps 1000 --eval-every 100 \
  --theta-floor 1 --loss relative --skip-fista
```

The strict, LFISTA-style and FISTA-Net code is frozen in `experiments/csc_lista_v4`,
`csc_lista_lfista` and `fista_net`, with outputs in `runs/csc_lista_strict`,
`csc_lista_lfista` and `fista_net`.

`scripts/score_fista_depths.py` gives the FISTA depth scores, frozen in
`experiments/fista_depths`, with outputs in `runs/fista_depths` and
`runs/fista_depths_lambda`.

All 23 tests pass on the server's PyTorch: 15 for CSC-LISTA and 8 for FISTA-Net. There is
no PyTorch on the Mac.
