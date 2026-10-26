# Synthetic training optimization and ROI assessment

Prepared 25 October 2026. Status: user-provided terminal output confirms remote
deployment, all seven optimization tests passing in 2.871 seconds, and the
switch preserving **epoch 7**. Optimized launcher PID: **67736**. Subsequent
user-provided logs confirm cache preparation, all eight CUDA benchmark trials,
and completion of **both 200-update arms and their final comparison**.
Log: `/home/abhaychandra/learning_ker/runs/synthetic_data_hypothesis_fast_v2/launch.log`.
Performance results below are from these logs. Final comparison shows a small
aggregate error reduction with mixed geometry effects and worse mean amplitude
recovery; see the experiment README for the complete interpretation.

This continues the hypothesis and evaluation plan in
[README_SYNTHETIC_EXPERIMENT.md](README_SYNTHETIC_EXPERIMENT.md).

## Measured remote speedup and presentation evidence

Recorded 25 October 2026 from the user's RTX A6000 run. All **1,080 component
pairs at 1,400 × 1,400 resolution** remain in the experiment. Cache preparation
completed once in **95.9 s**, before the optimized epochs. Epoch 7's model and
Adam state were retained; training continued at epoch 8.

| Work | Legacy time | Optimized time | Speedup and evidence |
|---|---:|---:|---|
| Full gradient pass | ~747.7 s | 203.5 s mean | ~3.67×; old time extrapolated from 780/1,080 components in ~540 s; new mean of three completed passes |
| Full training-loss evaluation | 619.4 s | 34.3 s | ~18.06×; both are logged complete evaluation passes |
| Gradient + evaluation every epoch | ~1,367.1 s | ~237.8 s | ~5.75×; summed stage times at the same evaluation cadence |
| Average epoch with evaluation every five updates | ~1,367.1 s (22.8 min) | ~210.4 s (3.51 min) | ~6.50×, or ~84.6% less time; amortized estimate using measured optimized stages and the projected legacy baseline |

The ~6.5× figure includes both faster execution and less frequent loss
measurement. It is **not** a measured ratio of two completed 200-epoch runs,
nor a claim that GPU utilization increased by that factor. Initial cache/tuning,
the control arm and final actual-COMSOL evaluation are outside this comparison.
Even when evaluating every epoch, the stage timings imply about 5.75× speedup.
These are combined implementation gains; individual contributions have not
been isolated with timing ablations.

### Logged epoch evidence

| Epoch | Gradient pass (s) | Post-update evaluation (s) | Total epoch (s) | Pre-update MSE | Post-update MSE |
|---|---:|---:|---:|---:|---:|
| 8 | 203.2 | Skipped | 203.3 | 0.00342925638 | Not measured |
| 9 | 203.7 | Skipped | 203.7 | 0.00345084234 | Not measured |
| 10 | 203.6 | 34.3 | 237.9 | 0.00345068960 | 0.00343422336 |

`post-update MSE=None` on epochs 8 and 9 is intentional sparse measurement,
not a failed update. Epoch 10's loss decreased during that update, but these
early training losses do not establish improved actual-data reconstruction or
model confidence. As an earlier projection, at that rate the 190 updates remaining
after epoch 10 would take roughly **11.1 hours**, including 38 scheduled loss
passes, assuming unchanged throughput. This excludes control training and final
evaluation and is a projection, not a completion promise.

### CUDA tuning evidence

All eight trials passed the tuner's memory budget. The highest measured rate
was **batch size 8 with activation checkpointing disabled**, which the tuner
selects. Here, checkpointing means activation recomputation during backward;
model/optimizer checkpoints are still saved every epoch.

| Batch size | Activation checkpointing | Seconds / benchmark batch | Components / s | Peak allocated GPU memory (GB, decimal) |
|---|---|---:|---:|---:|
| 1 | Off | 0.1947 | 5.136 | 10.177 |
| 2 | Off | 0.3771 | 5.304 | 11.184 |
| 4 | Off | 0.7480 | 5.348 | 13.200 |
| 8 | Off | 1.4882 | 5.376 | 17.232 |
| 1 | On | 0.2262 | 4.421 | 9.429 |
| 2 | On | 0.4405 | 4.541 | 9.688 |
| 4 | On | 0.8686 | 4.605 | 10.208 |
| 8 | On | 1.7376 | 4.604 | 11.246 |

Batch 8 without checkpointing used about **16.05 GiB** of peak allocated GPU
memory in the probe. This includes resident tensors; it is not total framebuffer
usage, reserved allocator memory, or a measured full-epoch peak. Larger batches
show diminishing throughput gains: batch 8 is only about 4.7% faster per
component than batch 1 without checkpointing. Filling all 48 GB is not itself
a performance goal. The three full gradient passes sustain about **5.31
components/s**, close to the selected probe's 5.38 components/s. New sustained
GPU-utilization samples and explicit cache-placement metadata were not included
in this log excerpt; the CPU fallback/prefetch path is not verified by it.

### Contribution statement for a presentation

> Integrated compact synthetic recipes into a physics-consistent training and
> matched-control evaluation pipeline, then optimized full-resolution training
> using reusable propagation/input caches, shared targets, operator-compatible
> batches, real FFTs, measured GPU batch/checkpoint selection, and sparse loss
> evaluation. Preserved model and Adam state when migrating at epoch 7. On an
> RTX A6000, gradient passes fell from an estimated 748 s to a measured 203.5 s,
> and measured loss evaluation fell from 619.4 s to 34.3 s. The resulting
> estimated average epoch cost is 3.51 min versus the projected 22.8 min legacy
> baseline, about 6.5× faster, with all 1,080 full-resolution pairs and one Adam
> update per epoch retained. The completed matched-control comparison found
> only a 0.78% relative reduction in mean actual-COMSOL NRMSE, with mixed
> geometry effects and worse mean amplitude recovery; computational speedup
> and reconstruction-quality outcomes are separate contributions.

## Diagnosis

The legacy run processes 1,080 full-resolution component pairs sequentially.
Clean synthetic propagation is recomputed on the CPU repeatedly because the
one-entry cache is displaced by the next component. Every epoch also includes
a second full pass to measure training loss. The reported loss pass alone took
619.4 seconds. A partial gradient-pass measurement implied about 748 seconds,
or approximately 22.8 minutes per epoch and 76 hours for 200 epochs at that rate.

The user's 15-sample NVIDIA monitor showed 12 samples at 0% GPU utilization and
an arithmetic mean of about 9%. This is a short monitoring window, not a full
profile, but is consistent with the CPU preparation bottleneck found in code.
It does not imply a broken or inadequate RTX A6000, and its 793 MB sampled
framebuffer reading is not peak training memory.

## Implemented changes

| Change | Behavior and reason |
|---|---|
| Three-component CPU propagation cache | Reuses clean propagation across noise and quarter-turn variants during preparation. For this full dataset, at most 90 distinct reference-component/height propagations are needed instead of repeating them for 1,080 inputs per pass. |
| Fixed input cache | Generates each normalized noisy input once using the original NumPy loader and seed. The inputs consume 8,467,200,000 bytes, about 7.89 GiB. |
| Shared target cache | Stores each geometry/rotation/component target once, approximately 282 MB, rather than repeating sharp targets for every noise/height recipe. |
| Device storage | `--cache auto` selects GPU storage if inputs plus targets occupy less than 45% of currently free GPU memory. Epochs then need no CPU generation or input transfers. Operator storage and training activations are additional. |
| CPU fallback | Pinned host cache plus a CUDA transfer stream overlaps next-batch copies with current computation if device caching cannot fit. Host RAM must accommodate the input/target cache. |
| Same-operator batches | Groups adjacent examples with identical grid, shape, height difference and padding. Supports larger batches without mixing operators. Scalar sample order is preserved; tail batches receive their correct sample weight. |
| No per-batch scalar synchronization | Accumulates detached loss on-device and reads it at pass end. Nonfinite loss/gradients still fail before the optimizer step. |
| Real FFTs | Stores the real analytical half-spectrum and uses `rfft2/irfft2`. The same full field, physical spacing, transfer function and zero-padding are used. No padding/FFT-size change. |
| Measured batch/checkpoint settings | Times batch sizes 1/2/4/8 and checkpointing on/off on the GPU with the data cache resident. Chooses measured throughput among candidates within a memory budget, retaining 20% of initially free memory as headroom. |
| Sparse loss measurement | Full post-update loss is measured every 5 epochs, at epoch 0, at the final epoch, and on an epoch-boundary stop request. Every gradient pass still logs its pre-update MSE. |
| Checkpoint every epoch | Saves model, Adam moments and completed epoch even when no post-update loss was measured. Final evaluation always uses `final.pt`. |

The cache and benchmark timings are recorded separately from epoch training
time. The benchmark uses one representative cached operator group and no
optimizer steps. It does not guarantee every batch's future peak memory or a
particular speedup under competing GPU jobs. Actual epoch timings are the
authoritative performance result. Remote CUDA tuning and epochs 8–10 are now
confirmed above; the CPU fallback's CUDA prefetch branch remains unverified.

The scientific training settings remain: seed 0, full 1,400-square images,
pad 256, the same 1,080/51 examples and noise draws, 16 layers, tied 9×9
convolutions, global scale, Adam lr=1e-4, physical lambda=1e-4, clipped full-data
gradient, and **one optimizer update per epoch**. There is no stochastic
minibatch Adam, AMP, TF32, image resize, new split, or changed noise definition.
Batching and FFT arithmetic can introduce floating-point rounding differences;
this is not a promise of bitwise agreement with the old full-complex path.

`best.pt` now means best among **measured** training-loss checkpoints, including
the inherited legacy best. It is a secondary artifact. Skipped evaluations are
stored as null, never represented as measured post-update losses.

## Checkpoint migration and paired experimental control

All legacy Python source files are left unchanged. Optimized code is in new
files, so the old checkpoint's recorded source hashes remain verifiable.
`train_synthetic_fast.py --migrate-from ...` accepts only the audited legacy
synthetic trainer, checks every old source hash, data/actual-archive hashes,
normalization, model settings and scientific configuration, and then restores
the model **and Adam optimizer state**. There is no generic hash-check bypass.

New output: `runs/synthetic_data_hypothesis_fast_v2/`. The old experiment is
preserved. `legacy_last.pt` is an additional snapshot; `migration.json` records
its checksum, epoch and original contract. Migration from epoch K means the
next update is K+1, not a fresh 200 epochs. It continues to epoch 200.

For a fair comparison, the control arm uses the original scalar/full-complex/
checkpointed training for its first K updates, then the same new batch size,
checkpointing mode, real FFT and evaluation cadence as the augmented arm.
Thus both arms follow the same computational transition schedule. The original
51-pair control's data remain the matched synthetic subset, not actual COMSOL
training. Final actual-data evaluation still uses the existing full-complex
analytical evaluator for both arms. The existing comparison script verifies
matching contracts and update budgets.

## Switch the running remote experiment

The switch below has already completed for this run. **Do not rerun it.**
For reference, the one-time command from the Mac project directory was:

```bash
bash hpc/optimize_synthetic_remote.sh abhaychandra@10.129.6.189
```

This copies only new source/tests and this README. It does not overwrite old
training sources. It runs the new CPU tests in the remote Python environment,
validates the dataset, all actual archives and the legacy checkpoint, then:

1. Identifies the old launcher through its PID file and validates the trainer's
   parent PID, script name and exact output directory. It refuses ambiguity.
2. Stops that launcher from starting the control arm, terminates only its
   identified legacy trainer, and stops the old wrapper. There is no blanket
   process-name kill. **Only the current incomplete epoch may be lost.**
3. Copies the latest atomic completed-epoch checkpoint into the new directory.
4. Starts a detached optimized launcher. It caches inputs, benchmarks the GPU,
   resumes the augmented arm, then runs the matched control and comparison.

The legacy trainer has no graceful epoch-stop handler. The new trainer does:
SIGTERM requests a checkpoint and exit at the next epoch boundary. Its wrapper
will not start the control if the augmented status is paused/incomplete.

Monitor on the server:

```bash
tail -n 60 ~/learning_ker/runs/synthetic_data_hypothesis_fast_v2/launch.log
cat ~/learning_ker/runs/synthetic_data_hypothesis_fast_v2/augmented_1080/status.json
nvidia-smi dmon -s pucm -c 15
```

Initial cache-building and benchmarking take time before the next epoch starts.
The selected batch/checkpoint settings, individual benchmark trials and cache
memory/preparation statistics are in each arm's `setup.json`. The new PID is in
`runs/synthetic_data_hypothesis_fast_v2/launcher.pid`. Original PID 66593 is the
legacy launcher, not the new one.

If the script refuses a mismatched PID, changed source or existing optimized
directory, inspect the message and preserved files. Do not delete checkpoints
or bypass the checks. If it stops after preserving the checkpoint, the original
run and `legacy_last.pt` remain available. Repeated execution is intentionally
refused to prevent duplicate GPU jobs.

For a manual restart of an already-created optimized augmented arm:

```bash
cd ~/learning_ker
"$HOME/qdm-env/bin/python" -u scripts/train_synthetic_fast.py \
  --dataset runs/synthetic_v1 --actual-run runs/combined_full_200 \
  --output runs/synthetic_data_hypothesis_fast_v2/augmented_1080 \
  --subset full --device cuda --epochs 200 --layers 16 --kernel-size 9 \
  --lr 1e-4 --lambda-ista 1e-4 --seed 0 --save-predictions \
  --cache auto --fft real --eval-every 5 --resume
```

Batch size, checkpoint mode and prefix count are restored from the optimized
checkpoint when omitted. Do not pass `--autotune` or `--migrate-from` on resume.
If caching or tuning failed before the new arm created `last.pt`, use the
preserved `legacy_last.pt` with `--migrate-from` and a fresh arm output instead.
The cache is rebuilt on each process restart; it does not need to be exported
to disk. Keep new source files unchanged while either arm is running.

## Should the image be cropped?

Potentially for **L1_C1**, but not as a blanket optimization for this dataset.
The actual array audit found these rectangles, retaining at least 99% of
squared-field energy **in every component** (union of per-component marginal
energy bounds, not necessarily the smallest possible rectangle):

| Geometry | Reference field box | Highest propagated field box |
|---|---:|---:|
| IC_2um | 1,253 × 1,315 | 1,285 × 1,316 |
| IC_4um | 1,236 × 1,313 | 1,365 × 1,374 |
| L1_C1 | 514 × 641 | 844 × 860 |

The IC geometry includes a spatially extended circuit, so reducing every input
to a small central crop would remove relevant signal. L1_C1 is concentrated
near one corner, not the image center. A centered crop would be inappropriate.
Magnetic propagation is nonlocal: fields spread beyond the visible conductor,
and the solver assumes zero field beyond the computational boundary.

For L1_C1 only, a reference-based box plus a 128-pixel halo yields a 768 × 771
crop. In a forward check at the highest distance (9.5 µm), propagating that crop
instead of the full field changed the prediction **inside the reference ROI**
by relative L2 errors of about 0.277% (Bx), 0.674% (By), 0.941% (Bz). These are
additional boundary/truncation discrepancies, not inverse-reconstruction error
estimates. With the same halo the IC proposals expand back to the whole image.

Therefore the optimized resumed run keeps full images. A future L1-specific
crop study should compare crop+halo operators against full-field propagation,
retain identical physical pixel spacing, apply rotations to crop coordinates
consistently, and assess ROI reconstruction and full-field results separately.
Even cropping noise changes what should be considered the original
full-field noise range unless generation and cropping are carefully ordered.
ROI-weighted loss can focus learning without discarding exterior physics, but
does not reduce the FFT workload and changes the objective. Downsampling also
changes resolution and can remove the fine details we want to reconstruct.

Audit outputs: `runs/synthetic_roi_audit_v1/roi_audit.json` and
`field_coverage.png`. Reproduce into a fresh directory with:

```bash
python scripts/audit_synthetic_roi.py --dataset runs/synthetic_v1 \
  --output runs/synthetic_roi_audit_new
```

## Validation and results tracking

- The complete synthetic dataset/training test selection passed: **21 tests**
  on CPU with PyTorch 2.14.1. Python compilation and both new shell scripts'
  syntax checks passed.
- Seven new CPU tests passed: cached inputs/targets exactly equal the original
  generator; 12 unique propagations for 96 fixture components; no generation
  during repeated cached passes; weighted batch gradients match scalar
  accumulation; even/odd rectangular real-FFT outputs, adjoints and gradients
  match full FFT; migration retains model and Adam state; replayed prefix and
  resumed updates match; sparse evaluation leaves final weights unchanged;
  optimized resume is exact in fixtures and rejects changed source contracts;
  process identity and energy-box checks.
- Full-resolution CPU check on all three geometries/components at their highest
  distance with 90° rotation: real/full FFT maximum normalized difference below
  1.94e-7; physical RMSE differences at most 4.26e-6 µT. Results in
  `runs/synthetic_roi_audit_v1/full_resolution_fft_check.json`.
- User-provided remote logs confirm cache preparation in 95.9 s, all eight
  accepted CUDA tuning trials, and completed epochs 8–10. Mean gradient-pass
  time is 203.5 s; the epoch-10 loss pass took 34.3 s. Benchmark peak allocation
  for the fastest candidate was 17,231,754,752 bytes. Full-epoch peak memory,
  sustained GPU utilization and the CPU fallback/prefetch path remain unmeasured.
- Final user-supplied `comparison/results.json` and `comparison/summary.csv`
  confirm both 200-update arms completed. Mean actual-COMSOL NRMSE improves
  0.778% relatively and relative L2 improves 0.619%; only 16/51 components
  improve in NRMSE, and mean amplitude slope falls from 0.6189 to 0.5663.
  See the experiment README for per-geometry results and qualified conclusions.

Next optimization decisions should follow measured epoch timings. Multi-GPU
training would need explicit distributed gradient accumulation; AMP/TF32 would
need numerical validation; changing padding to an FFT-friendly size would
change the current physical operator. None of these are silently enabled.
