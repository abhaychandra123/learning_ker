# L1_C1 depth and parameter-sharing experiment

Protocol: 26 October 2026. User requested more parameters/layers and one geometry.
This is a fresh fitting diagnostic following README_SYNTHETIC_DIAGNOSIS.md, not
an extension of the completed 51-versus-1,080-pair hypothesis experiment.

## Question and controlled arms

Can increased depth or layer-specific correction weights fit one geometry
better, before introducing more geometries, rotations or noise?

| Arm | Layers | Correction-weight sharing across layers | Parameters |
|---|---:|---|---:|
| tied_16 | 16 | Shared | 163 |
| tied_32 | 32 | Shared | 163 |
| untied_16 | 16 | Separate per layer | 2,608 |
| untied_32 | 32 | Separate per layer | 5,216 |

Each layer uses the physical gradient plus two single-channel 9 × 9 convolution
corrections and a nonnegative shrinkage threshold. Untying gives each layer
its own kernels and threshold; weights remain shared over examples/components.
At zero-kernel initialization all arms reproduce their corresponding number
of ISTA iterations. More tied layers alone do not increase parameter count.
This is a controlled capacity change, not a switch to a large generic CNN.

## Fixed first-stage protocol

- L1_C1 only, reference height 0.5 µm. Clean input heights 1,2,...,10 µm.
- Exactly **10 vector / 30 component pairs**, Bx/By/Bz, rotation 0°, noise 0%,
  realization 0. No geometric crops or image resizing. Full 1,400-square fields.
- Same existing NumPy synthetic loader and analytical pad-256 forward model.
- Original global normalization 215.14413452148438 µT is retained for all arms.
- Fresh deterministic ISTA initialization in every arm, seed 0. No warm start
  from the old model, and no checkpoint migration across architectures.
- Adam lr 1e-4, initial physical lambda 1e-4 µT, gradient clipping norm 1,
  float32, AMP/TF32 off, original full-dataset mean MSE.
- **200 updates per arm** initially, one accumulated Adam update over all 30
  components each epoch. Equal updates do not imply equal compute or convergence.
- Cache inputs once; real FFTs; per-arm CUDA tuning of batch 1/2/3 and activation
  checkpointing. These change computation/memory, not the update policy.
- Post-update training MSE and per-height/component reconstruction scores at
  update 0, every 20 updates, and final update. Gradient norm before clipping,
  clipping flag, update L2 magnitude, and threshold ranges/zero counts every update.
- Every-update resumable `last.pt`; primary `final.pt` at update 200. No
  selection using actual COMSOL scores. Final actual evaluation covers all
  ten L1_C1 COMSOL heights, saved as arrays and physical metrics.

This deliberately removes augmentation while testing fitting. It does not
replace the previous augmented dataset, or establish unseen-geometry accuracy.
If the model fits clean inputs but not actual COMSOL, investigate the forward
gap. If fitting remains poor, inspect the logged gradients/update curves and
test longer optimization or single-pair fitting before declaring a capacity limit.

## Start on the dedicated GPU server

From the Mac project directory:

```bash
bash hpc/launch_geometry_capacity_remote.sh abhaychandra@10.129.6.189
```

The helper checks paths and refuses a duplicate experiment, copies the current
Python sources into a **new snapshot** at
`~/learning_ker/experiments/geometry_capacity_v1`, runs five CPU tests in the
server environment, then starts a logged background wrapper. All four arms run
sequentially on the GPU, followed by a verified comparison. It does not edit
old remote training sources or outputs. Do not use this dedicated-server helper
on a scheduled cluster login node. Existing synthetic data and actual archives
must already be present, as in the completed experiment.

Monitor on the server:

```bash
tail -n 60 ~/learning_ker/logs/geometry_capacity_v1.launch.log
```

Run outputs: `~/learning_ker/runs/geometry_capacity_v1/`.
The launcher PID is not proof of successful CUDA training; confirm benchmarks
and actual update messages. No measured full-suite runtime is available yet.
This session cannot authenticate through the user's terminal; remote launch
status must be confirmed from a successful launcher response/log.

## Outputs and interpretation

Each arm stores `setup.json`, `training.json`, `last.pt`, `final.pt`, `status.json`,
`synthetic_epoch_XXXX.csv`, `synthetic_final.csv`, `actual_final.csv`, and final
`actual_predictions/*.npz`. CSV NRMSE/relative L2 values are fractions, not percent.
The new checkpoints require `geometry_lista.load_model`; the old shared-only
loader cannot load layer-specific parameter names.

The final summary verifies completed statuses, checkpoint contracts, identical
data/source hashes/update budgets and matching evaluation cases. Allowed arm
differences are depth, sharing and measured batch/checkpoint runtime settings.
Results are `comparison/summary.csv`, `comparison/per_case.csv`, and
`comparison/results.json` with source hashes and limitations.

Assess clean fitting first. Compare tied_16 vs tied_32 for depth with the same
parameter count, and tied vs untied at a fixed depth for parameter sharing.
Then inspect actual-data errors and amplitude slopes by height/component.
An aggregate improvement does not imply all cases improved. No significance
claim is justified by one geometry and one seed.

## Pause, resume or extend

SIGTERM to an identified trainer requests saving and stopping at its next
update boundary. The wrapper will not start the next arm after a paused run.
Do not use a broad process-name kill.

Example: resume/extend an existing untied_32 arm to 1,000 updates on the server:

```bash
cd ~/learning_ker
"$HOME/qdm-env/bin/python" -u experiments/geometry_capacity_v1/scripts/train_geometry_capacity.py \
  --dataset runs/synthetic_v1 --actual-run runs/combined_full_200 \
  --output runs/geometry_capacity_v1/untied_32 --geometry L1_C1 \
  --sharing untied --layers 32 --kernel-size 9 --lr 1e-4 --lambda-ista 1e-4 \
  --seed 0 --device cuda --cache auto --eval-every 20 --epochs 1000 --resume
```

Resume restores Adam and the measured batch/checkpoint settings. It refuses
changed code/data/model/settings; use the frozen source copy. Do not pass
`--autotune` on resume. Retain all arms at the same final budget before rerunning
the formal four-arm summarizer. An extension of just one arm is a separate
optimization diagnostic, not the original equal-update comparison.

## Validation and launch status

Five CPU tests passed: expected parameter counts; tied-output equivalence and
summed untied gradients; correct per-layer checkpoint recomputation; geometry/
clean-height selection; exact model/Adam resume with configuration rejection;
actual-input isolation; four-arm summary and incomplete-arm rejection.
Shell syntax, Python compilation and whitespace checks also passed locally.

A separate local full-resolution smoke run uses only L1_C1 at 1 → 0.5 µm,
all three components, untied 32 layers, one update, CPU, and no actual evaluation.
Its source snapshot and artifacts are in `runs/geometry_capacity_smoke_v1/`.
It completed successfully: 5,216 parameters; gradient pass 92.5 s on this CPU;
finite pre-clipping gradient norm 0.00109538; parameter update L2 0.00715859;
initial/final measured normalized MSE 0.0001164050 / 0.0001180913. That one update
increased loss slightly (~1.45%), so it is not evidence that more parameters
improve reconstruction. All 32 learned thresholds projected to zero on this
update, which is visible in the new logs and should be monitored during training.
Training-counter time including initial/post-update diagnostics was 197.9 s;
this is a CPU smoke timing, not the remote experiment ETA.

It is an execution check, not the four-arm scientific experiment. Direct SSH
from this agent session was attempted and blocked with `Operation not permitted`.
The remote four-arm experiment **has not been launched by the agent**. Run the
Mac command above to authenticate and start it. No existing run was stopped.
