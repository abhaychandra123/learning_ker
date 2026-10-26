# Does more synthetic training data improve QDM reconstruction?

Experiment: `synthetic_data_hypothesis_v1`  
Protocol recorded: 25 October 2026 (India time)  
Current status: **both arms completed 200 updates and final actual-COMSOL
evaluation** in `synthetic_data_hypothesis_fast_v2`. The user supplied the final
`comparison/results.json` and `comparison/summary.csv` on 25 October 2026.
The directional criterion passed, but the aggregate gains are below 1%, vary
by geometry, and accompany worse mean amplitude recovery. Original v1 outputs
are preserved; the optimized run continued from its completed epoch 7.

**Optimization follow-up:** [README_SYNTHETIC_OPTIMIZATION.md](README_SYNTHETIC_OPTIMIZATION.md)
documents the new cached/batched trainer, checked migration to a separate v2
run, GPU autotuning and the measured ROI audit. Gradient passes now average
**203.5 s** and full loss evaluation took **34.3 s**. Including evaluation every
five epochs gives an estimated **3.51 min/epoch**, about **6.5× faster** than the
projected 22.8 min legacy baseline. The linked README records raw timing tables,
benchmark memory, calculation assumptions and presentation-ready contribution
wording. Final reconstruction results and their limitations are below. Neither
launcher needs to be rerun.

## Final results: small aggregate gain, mixed reconstruction behavior

Follow-up: [README_SYNTHETIC_DIAGNOSIS.md](README_SYNTHETIC_DIAGNOSIS.md) records
new local full-resolution probes of clean-versus-actual inputs and matched
analytical ISTA/FISTA. The model also struggles on clean high-standoff inputs;
longer FISTA runs improve substantially. These selected Bz probes guide the
next experiments and do not replace the full comparison reported here.

Evidence: user-pasted remote JSON/CSV, not an independent download or local
verification of the remote checkpoint hashes. Both arms used 200 Adam updates;
the evaluation contains the same 17 vector / 51 component actual-COMSOL pairs.

| Metric, mean over 51 components | Control: 51 training pairs | Augmented: 1,080 training pairs | Interpretation |
|---|---:|---:|---|
| Range NRMSE (%) | 3.740098 | 3.711002 | 0.029096 percentage points lower; **0.778% relative reduction** |
| Relative L2 error (%) | 65.064585 | 64.661981 | 0.402604 percentage points lower; **0.619% relative reduction** |
| RMSE (µT) | 13.601660 | 13.531605 | 0.070055 µT lower; **0.515% relative reduction** |
| Signed amplitude slope (ideal 1) | 0.618911 | 0.566331 | Mean moves farther from 1; not improved amplitude recovery |

The relatively small range NRMSE must not be described as roughly 96% accuracy.
It divides RMSE by the target's range, whereas relative L2 uses the target norm.
The latter remains about 0.647, so appreciable reconstruction error remains.
The mean amplitude slope is an average of per-component projections onto the
truth; it is not a claim that every pixel has the same multiplicative bias.

| Geometry | Evaluation components | Control NRMSE (%) | Augmented NRMSE (%) | Relative NRMSE change | Components improved |
|---|---:|---:|---:|---:|---:|
| IC_2um | 3 | 3.496537 | 2.888403 | **17.39% lower** | 3/3 |
| IC_4um | 18 | 4.545969 | 4.576551 | **0.67% higher** | 4/18 |
| L1_C1 | 30 | 3.280932 | 3.273933 | **0.21% lower** | 9/30 |
| All | 51 | 3.740098 | 3.711002 | **0.78% lower** | **16/51** |

The aggregate improvement is largely driven by IC_2um, which has only one
evaluated vector pair. IC_4um worsens on both mean error measures; L1_C1 changes
very little. Only 16/51 component cases (31.4%) have lower NRMSE; the summary
does not distinguish ties from regressions among the other 35. Mean amplitude
slope improves toward 1 for IC_2um (1.2371 → 1.1482), but moves farther below 1
for IC_4um (0.6117 → 0.5598) and L1_C1 (0.5614 → 0.5120).

The automatic verdict correctly implements the predeclared rule that both
mean errors must decrease. It has no minimum effect-size or statistical
significance requirement. **Passing it does not mean a substantial or consistent
improvement.** Expanding the recipe count by 21.18× yielded a small, mixed gain
under this model and update budget. This does not support the strong claim
that insufficient data was the only cause of poor reconstruction. The weaker
claim that this augmentation can help some same-geometry cases has limited
support, strongest for IC_2um. There is no calibrated confidence result.

Limits remain: one seed, three shared source geometries/targets, changed
height/rotation/noise coverage and geometry weights, and no unseen-geometry
test. More recipes do not supply more independent geometries. Equal update
budgets also do not establish that both arms reached their best achievable fit.

Before another long run, inspect the existing paired results by height and
component, visualize sharp/input/control/augmented fields on shared color scales,
and compare amplitude slopes with residuals. Then test specific explanations
such as analytical-versus-COMSOL forward mismatch, optimization/underfitting,
or augmentation weighting. These are proposed diagnostics, not demonstrated
causes or experiments already performed.

Presentation wording:

> Built and evaluated a 1,080-pair synthetic augmentation pipeline against a
> matched 51-pair control at 200 updates each. Actual-COMSOL mean NRMSE fell by
> 0.78% relatively, but only 16/51 components improved and mean amplitude
> recovery worsened. The result shows limited, geometry-dependent benefit,
> rather than resolving reconstruction quality through more data alone.
> Separately, computational optimization reduced estimated average epoch cost
> by about 6.5× while retaining full-resolution inputs and the update policy.

## Hypothesis and what this experiment can establish

Hypothesis: the previous shared ConvLISTA model's reconstruction quality was
limited partly by the small training set of 17 vector pairs / 51 component pairs.
Increasing height, rotation and noise coverage to 360 vector pairs / 1,080
component pairs may improve reconstruction of actual COMSOL fields.

We measure reconstruction error and amplitude recovery, not a model confidence
probability. The existing model does not produce calibrated uncertainty.
Improvement supports the usefulness of this augmentation. It cannot establish
that too little data was the **only** problem. Operator mismatch, model capacity,
optimization and information lost at large standoffs remain possible causes.

All 1,080 pairs descend from the same three reference geometries. The actual
COMSOL evaluation shares geometries and sharp targets with training. This is a
same-geometry reconstruction/forward-model-transfer experiment, not an unseen
geometry generalization test. Do not randomly split augmented siblings and
describe them as independent validation data.

## Why a direct old-versus-new comparison is insufficient

The archived `runs/combined_full_200` model used jointly fitted operators with
32 pixels of padding. The synthetic dataset was generated using analytical
propagation with 256 pixels of padding. Feeding synthetic pairs through the old
fitted operators would violate the generation physics.

The primary comparison therefore uses two fresh models:

| Arm | Training data | Scalar pairs | Operator | Purpose |
|---|---|---:|---|---|
| Historical | Actual COMSOL, archived 200-epoch run | 51 | Fitted, pad 32 | Contextual reference only |
| Control | Synthetic, original COMSOL height pairs, rotation 0°, noise 0% | 51 | Analytical, pad 256 | Matched small-data control |
| Augmented | Existing `runs/synthetic_v1`, every recipe | 1,080 | Analytical, pad 256 | Test expanded coverage |

All 17 original height pairs have exactly one matching noiseless, unrotated
synthetic recipe. The control is a subset of the augmented dataset. Its count is
computed and verified from the manifests, not hard-coded by the loader.

This comparison changes coverage and distribution, not merely the number of
duplicates: heights, rotations, noise levels and geometry weighting change.
For example, the control's geometries have unequal numbers of original heights,
while the augmented arm gives each geometry 360 scalar pairs. It measures the
effect of this augmentation package. A later ablation can isolate these factors.

## Integration: what changed and why

| File | Change | Why it is needed |
|---|---|---|
| `scripts/train_synthetic.py` | New training entry point using the existing `SharedConvLISTA`, `PaddedFFT`, checkpoint utilities and full-dataset gradient accumulation | Existing `compare_combined.py` expects paths to stored COMSOL planes, not synthetic recipes |
| `scripts/train_synthetic.py`: `RecipeBatches` | Calls `SyntheticDataset.component()` and streams one normalized scalar input/target at a time | Uses the compact 70.7 MB dataset without materializing roughly 25.4 GB |
| `RecipeBatches.get_operator()` | Builds the analytical transfer from each sample's distance, rotated grid and manifest padding; retains one operator at a time | Keeps the training physics consistent with generation, including rectangular grids and quarter-turns |
| `scripts/synthetic_experiment.py` | Selects the 51-pair control, validates actual archives/reference identity, and computes physical metrics | Prevents mismatched references, units, heights or accidental use of a different evaluation set |
| `scripts/compare_synthetic_experiment.py` | Joins paired results and checks matching experiment contracts and update counts | Makes the small-versus-large comparison auditable |
| `hpc/run_synthetic_experiment.sh` | Checks CUDA, runs tests, then runs the augmented arm and control sequentially and compares them | Provides one launch command and avoids two competing jobs on one GPU |
| `hpc/launch_synthetic_remote.sh` | Copies required files, copies the dataset only if absent, and starts a logged background launcher | Lets the user's terminal perform the SSH operation blocked in this agent session |
| `tests/test_synthetic_training.py` | Checks selection/provenance, rotated physics, resume equivalence and evaluation isolation | Verifies the new connection to the trainer |

The existing model architecture, original trainer, original COMSOL files,
synthetic dataset and archived results are not modified by these new commands.
The original trainer remains available for original-format experiments.

## Fixed training protocol

The list below records the original v1 execution protocol. Starting after
completed update 7, both arms use the v2 computational schedule: cached inputs,
real FFTs, operator-compatible batches (tuner selects size 8), activation
checkpointing off, and post-update loss measurement every five epochs and at
the endpoint. The control replays seven legacy updates first. Full-dataset
gradient weighting, one Adam update per epoch, data, precision, physical
operator and the 200-update budget remain fixed. See the optimization README
for migration validation and floating-point equivalence limits.

- Fresh deterministic initialization in both arms, seed 0. No warm start from the historical model.
- Existing shared ConvLISTA: 16 layers, tied 9×9 corrections, 163 parameters,
  blurred initialization, gradient checkpointing, float32. No AMP.
- Adam, learning rate `1e-4`, gradient clipping norm 1, physical lambda `1e-4` µT.
- One global normalization scale from the synthetic training references:
  **215.14413452148438 µT**. It also matches the historical run's recorded scale.
- Fixed generated noise (`epoch=0`), preserving exactly the declared dataset.
- One component per forward/backward pass; each loss is weighted by
  `1 / number_of_training_components`. Gradients accumulate across the entire
  dataset before a single Adam step, matching the old trainer's update policy.
- **200 epochs = 200 Adam updates**, in each arm. Epoch 0 is an initial loss
  evaluation, not an optimizer update. Equal epochs do not mean equal GPU time:
  the augmented arm processes about 21.2 times as many component pairs per pass.
- Re-evaluate synthetic training loss after each update. Actual COMSOL error is
  never used for early stopping or checkpoint selection.
- Primary endpoint: `final.pt` at the predeclared update budget. `best.pt` is
  saved separately using synthetic training loss only, and is not substituted
  for the primary checkpoint after looking at actual-data scores.
- No changes to the data split during this run. A future unseen-geometry study
  requires fresh geometry-level splits and fresh training.

## Evaluation and decision rule

Evaluate both final models on the **same 17 actual COMSOL vector pairs / 51
component pairs** stored in `runs/combined_full_200/comparison/*.npz`.
Both new models use analytical pad-256 operators here. Archive checksums,
grids, physical units, reference heights and exact target-array identity are
verified before training; archives are checked again before evaluation.

Primary metric: unweighted mean of per-pair range NRMSE. Lower is better.
Secondary checks: relative L2 error, RMSE in µT, per-geometry results and the
number of individual pairs that improved. `amplitude_slope = <prediction,truth>
/ <truth,truth>` measures signed amplitude alignment: 1 is ideal; values near
0 indicate substantial amplitude collapse. It is not a confidence score.
CSV NRMSE and relative L2 values are **fractions**, not percentages.

Predeclared directional criterion: the augmented model must reduce **both**
mean range NRMSE and mean relative L2 against the matched control. Report all
per-geometry results even when the overall mean improves. No statistical
significance claim is made from one seed and three shared geometries.

Also compare with the archived historical LISTA predictions for context. The
historical mean range NRMSE is **3.789666%** and mean RMSE **13.823991 µT** across
51 components. Its different operator and training distribution mean beating
it does not isolate the effect of dataset size.

If synthetic training loss improves but actual COMSOL errors do not, the result
does not support this augmentation as a remedy for actual-input reconstruction.
If both new arms remain poor, dataset size alone is an inadequate explanation
under this model and training budget. Inspect large-standoff amplitude recovery
and training curves before deciding which factor to test next.

## Launch on the GPU machine

This agent session cannot establish SSH to `10.129.6.189` (`Operation not permitted`).
The user successfully ran the launcher from their Mac and supplied terminal
output confirming remote dataset inspection and launcher PID 66593. The
commands below document that procedure; do not relaunch the same experiment.
Subsequent logs confirm migration after epoch 7 and optimized training through
epoch 10; see the evidence ledger and optimization README. The v1 commands in
this section and the legacy resume section are historical procedures, not
instructions to start a second job while v2 is running.

**One-command launch from your Mac's project directory**, targeting the
dedicated GPU server with the existing `~/learning_ker` and `~/qdm-env`:

```bash
bash hpc/launch_synthetic_remote.sh abhaychandra@10.129.6.189
```

This copies the listed files, uses the existing remote synthetic dataset (or
copies the local one if absent), verifies it, and launches the experiment under
`nohup`. It refuses existing experiment output or a previous launch PID file.
A launcher PID is not proof that CUDA tests passed: inspect the log and status.
Do not use this direct-server helper on a scheduled cluster's login node.
The equivalent manual steps follow for transparency and custom paths.

From the local project root, copy only the relevant source and documentation:

```bash
rsync -av --relative \
  scripts/train_synthetic.py scripts/synthetic_experiment.py \
  scripts/compare_synthetic_experiment.py scripts/synthetic_data.py \
  scripts/propagator.py tests/test_synthetic_training.py tests/test_synthetic_data.py \
  hpc/run_synthetic_experiment.sh README_SYNTHETIC_EXPERIMENT.md \
  abhaychandra@10.129.6.189:~/learning_ker/
```

This assumes the existing project/model dependencies and completed actual
COMSOL run are already on that machine. On the remote machine:

```bash
cd ~/learning_ker
export QDM_PYTHON="$HOME/qdm-env/bin/python"
"$QDM_PYTHON" scripts/synthetic_data.py inspect --dataset runs/synthetic_v1
```

If the synthetic directory is absent remotely, copy the verified local dataset
from your Mac; no generation is required:

```bash
rsync -av runs/synthetic_v1/ \
  abhaychandra@10.129.6.189:~/learning_ker/runs/synthetic_v1/
```

Then launch on the dedicated GPU server (or execute the wrapper inside an
allocated GPU job, not a cluster login node):

```bash
cd ~/learning_ker
mkdir -p logs
nohup env QDM_PYTHON="$HOME/qdm-env/bin/python" \
  bash hpc/run_synthetic_experiment.sh \
  > logs/synthetic_data_hypothesis_v1.launch.log 2>&1 &
echo $! > logs/synthetic_data_hypothesis_v1.pid
```

Monitor:

```bash
tail -n 40 logs/synthetic_data_hypothesis_v1.launch.log
cat runs/synthetic_data_hypothesis_v1/augmented_1080/status.json
nvidia-smi
```

The wrapper refuses an existing experiment directory. It runs the requested
1,080-pair arm first, then the 51-pair control, then writes the paired report.
Set `CUDA_VISIBLE_DEVICES` before launch if a specific GPU has been assigned.
`QDM_DATASET`, `QDM_ACTUAL_RUN`, `QDM_EXPERIMENT`, `QDM_EPOCHS` and `QDM_THREADS`
can override paths or budgets. Keep the same budget in both arms. Measured v2
runtime and the estimated legacy comparison are in the optimization README.

## Interrupted runs and tracked outputs

Resume an interrupted augmented arm from its latest completed epoch:

```bash
"$HOME/qdm-env/bin/python" -u scripts/train_synthetic.py \
  --dataset runs/synthetic_v1 --actual-run runs/combined_full_200 \
  --output runs/synthetic_data_hypothesis_v1/augmented_1080 \
  --subset full --device cuda --epochs 200 --layers 16 --kernel-size 9 \
  --lr 1e-4 --lambda-ista 1e-4 --seed 0 --checkpoint-layers \
  --save-predictions --resume
```

For the control, substitute `control_51` and `--subset matched-original`.
Omit `--resume` if that arm has not started. Resume verifies data/code/settings
hashes. A partial epoch is recomputed. If interrupted before the epoch-0
checkpoint exists, use a fresh output directory; there is nothing to resume.
Do not edit experiment code while either arm is running.

After both arms finish (including after manual resumes):

```bash
"$HOME/qdm-env/bin/python" scripts/compare_synthetic_experiment.py \
  --control runs/synthetic_data_hypothesis_v1/control_51 \
  --augmented runs/synthetic_data_hypothesis_v1/augmented_1080 \
  --output runs/synthetic_data_hypothesis_v1/comparison
```

| Artifact | What it tracks |
|---|---|
| Each arm's `setup.json` | Configuration, environment, exact recipe IDs, code/data/archive hashes |
| `status.json` | Training/evaluation/completed/interrupted/failed and last completed epoch |
| `training.json` | Per-epoch normalized training MSE and cumulative training time |
| `last.pt` | Model, optimizer, training history and resume contract |
| `final.pt`, `best.pt` | Primary final checkpoint and secondary best-training-loss checkpoint |
| `actual_comsol.csv`, `actual_summary.json` | Actual-data metrics plus historical model context |
| `actual_predictions/*.npz` | New physical-unit LISTA outputs; originals remain in the source archive |
| `comparison/paired_comparison.csv` | Control, augmented and delta for every actual pair/component |
| `comparison/summary.csv`, `results.json` | Overall/per-geometry results, verdict, limitations and provenance |

## Evidence ledger

| Item | Current evidence |
|---|---|
| Local dataset | Complete; 360 vector / 1,080 scalar recipes; checksums verified |
| Original-height control | All 17 original pairs match uniquely; 51 scalar pairs |
| Actual evaluation archives | All 17 archives verified; references exactly match synthetic references |
| Integration checks | All 14 synthetic dataset/training tests passed on CPU with PyTorch 2.14.1, including training, exact epoch-resume equivalence, rotated operators and actual-input isolation; Python compilation and shell syntax passed |
| Remote dataset | User-provided remote inspection passed: 3 geometries, 360 vector / 1,080 scalar pairs, all training |
| Remote launcher | Started by user on `10.129.6.189`; PID 66593; log `~/learning_ker/logs/synthetic_data_hypothesis_v1.launch.log` |
| Remote training | User log: epoch 6 complete, normalized MSE 0.0033977439, training elapsed 8721.7 s; epoch 7 gradient pass at 780/1080 by about 540 s |
| GPU observation | User snapshot: RTX A6000, Python PID 66663, 776 MiB process VRAM, 0% sampled GPU utilization; not a sustained-utilization or peak-memory measurement |
| Optimized remote progress | User log: epoch 7 preserved; launcher PID 67736; cache prepared in 95.9 s; epochs 8–10 complete; epoch 11 started |
| CUDA tuning | Eight accepted trials; fastest candidate batch 8, activation checkpointing off, 5.376 components/s; benchmark peak allocated memory 17.232 GB (16.05 GiB) |
| Optimized runtime | Gradient passes 203.2 / 203.7 / 203.6 s (mean 203.5 s); epoch-10 loss pass 34.3 s; total epochs 203.3 / 203.7 / 237.9 s |
| Speedup estimate | ~3.67× gradient vs extrapolated legacy pass; ~18.06× measured loss evaluation; ~6.50× amortized epoch vs projected legacy baseline, including the evaluation-cadence change |
| Latest measured training loss | Epoch 10 pre-update MSE 0.00345068960, post-update MSE 0.00343422336; actual-data quality remains unevaluated |
| Control results | Completed 200 updates; mean actual NRMSE 3.740098%, relative L2 0.650646; final training MSE 0.004416467622 |
| Augmented results | Completed 200 updates; mean actual NRMSE 3.711002%, relative L2 0.646620; 16/51 components improved |
| Hypothesis outcome | Directional rule passes; aggregate relative gains <1%, mixed geometry effects, worse mean amplitude slope; no support for data scarcity as the sole explanation |

Update this ledger with the machine/job or PID, start time and final report
paths after a verified launch. The run directories update their own status and
metrics during execution; this README does not invent remote progress.

CPU validation used an isolated temporary Python 3.13 environment assembled
from locally cached wheels. It exercised tiny training fixtures, not a
full-resolution CUDA run. The remote wrapper reruns the tests using the remote
environment before starting the full experiment. These local checks do not
constitute a full-resolution CUDA benchmark or a completed 200-epoch experiment.

### Runtime investigation after epoch 6

The user reported 619.4 seconds for epoch 6's full training-loss evaluation.
Epoch 7 processed 780 of 1,080 gradient components by roughly 540 seconds.
Linear extrapolation gives about 748 seconds for gradients plus 619 seconds
for evaluation: approximately 22.8 minutes/epoch, or 76 hours for 200 epochs
at unchanged speed, excluding the control arm and final actual-data evaluation.
This is a projection from partial progress, not a measured full-run duration.

Code inspection identifies avoidable work: every epoch performs a second
complete dataset pass solely to measure training loss; synthetic clean fields
are regenerated on the CPU with a one-entry propagation cache; loading and
GPU execution are sequential; checkpointing trades activation memory for
recomputation. Each epoch produces only one Adam update after all 1,080 pairs.
The model still performs substantial full-resolution physics work: 16 layers
on 1,400-square images, with 1,912-square padded FFTs.

A bounded local trace of three recipes at the same height/rotation (0%, 1%,
1.5% noise) confirmed nine propagation calls for just three unique inputs.
Nine component loads took 2.742 seconds on the Mac; this is not a remote CPU
or GPU timing measurement. The supplied A6000 snapshot's 0% utilization is
consistent with waiting on the loader but cannot establish sustained utilization.
Repeated GPU samples/profiling are needed to quantify the runtime breakdown.

No training code or checkpoint contract was changed during this diagnosis.
Potential improvements: reuse clean propagations across noise/rotation variants,
prefetch data, evaluate full loss less often, and benchmark batching/checkpoint
settings using measured peak VRAM. Minibatch Adam would change optimization and
must be treated as a revised protocol for both arms, not silently applied to
this running experiment. Hypothesis outcome remains pending.
