# Using IIT Bombay HPC for the QDM reconstruction project

**22 October extension:** For the user's new shared-model experiment across
`data/` and `data/data2/`, use [README_COMBINED_EXPERIMENT.md](README_COMBINED_EXPERIMENT.md)
and `hpc/combined.sbatch`. That new runner supports epoch-boundary optimizer
resume and geometry-separated data preparation. The single-pair commands and
their checkpoint limitations described below still refer to the original CLI.
No new combined-data GPU training has been run on the cluster yet.

Reviewed: 16 October 2026. This guide is based on all three supplied manuals, the current project context, and the relevant implementation. It is an operational plan, not a record of jobs already run on these clusters.

## 1. What is confirmed, and what is still unknown

The user confirms access to all three facilities and has logged in and changed the initial passwords. The Prajna access email confirms username `23b0313`, group `ajitvr`, and hostname `prajna.iitb.ac.in`. No passwords belong in scripts, experiment metadata, or this guide. The email does not specify quotas or permitted queues.

### Live storage observations supplied by the user, 16 October 2026

These are user-supplied command outputs, not commands executed remotely by this assistant. They take precedence over generic manual descriptions for the observed mounts. Shell prompts confirm username `23b0313` on all three clusters.

| Cluster and mount | Reported capacity | Used | Available | Scope |
|---|---|---|---|---|
| Prajna `/home`, Lustre | 1.3P | 782T (64%) | 444T | Shared filesystem, not a personal allocation |
| Praganak `/home`, Lustre | 5.2P | 690T (14%) | 4.5P | Shared filesystem, not a personal allocation |
| Rudra `/home`, Lustre | 1.1P | 244T (24%) | 774T | Shared filesystem, not a personal allocation |
| Prajna `/scratch`, source `slurm:/scratch` | 250G | 39G (16%) | 212G | Capacity of this mount as seen from `login2`; not a personal quota |
| Prajna `/lustre-scratch`, Lustre | 820T | 6.2M (rounded 1%) | 812T | Shared filesystem; user/group quota reports both show zero limits |
| Prajna `/lustre-flash`, Lustre | 399T | 2.0T (1%) | 393T | Shared filesystem; user/group quotas have not been checked |

Prajna `lfs quota -u 23b0313 /home` reports 112 KiB and 27 files, with both soft/hard block limits and soft/hard file-count limits set to zero. `id -nG` reports only group `ajitvr`. The group's `/home` usage is 2,943,252,872 KiB (about 2.74 TiB) and 4,716,059 files, also with zero block/file limits. Thus no user/group limits are reported by those queries; zero is not a zero-byte allowance. This does not establish a guaranteed unlimited entitlement or exclude project/pool/site policy. Lustre supports project quotas separately from user/group quotas; see the [official project quota documentation](https://wiki.lustre.org/Lustre_Project_Quotas) and [quota operations manual](https://doc.lustre.org/lustre_manual.pdf).

On Prajna, `lfs quota -u 23b0313 /scratch` fails because this mount is not Lustre. A subsequent `findmnt` confirms filesystem type `nfs4`, source `slurm:/scratch`. This is network-mounted storage, not node-local disk. The 250G figure describes this observed filesystem, not a 250G allocation to this account. Its intended role, compute-node visibility, quota and retention/backup policy need confirmation.

The same session identifies three Lustre mounts: `/home` from server filesystem `/home`, `/lustre-flash` from `/flash`, and `/lustre-scratch` from `/scratch`. Thus `/lustre-scratch` is a distinct Lustre working-storage candidate, rather than the small NFS mount `/scratch`. Subsequent user/group quota queries on `/lustre-scratch` both report zero usage/files and zero soft/hard byte/file limits. This confirms no user/group limits are reported there; it does not establish an assigned writable directory, project quota, retention policy or compute-node visibility. The very low total usage is not proof of an initialization problem, nor proof that unrestricted use is permitted. A name containing `flash` alone does not establish a user entitlement or suitable destination for this dataset.

### Live Prajna scheduler observations supplied by the user

All displayed partitions report availability `up`. The `sinfo` node counts are partition membership, not idle-node counts or personal allocations. Partitions can overlap, so these counts must not be added as distinct hardware inventory.

| Partition | Reported maximum time | Member nodes | GPU resources per node |
|---|---|---|---|
| `dgx` | 6 days | 9 | 8 |
| `dgx-mpi` | 6 days | 3 | 8 |
| `a40` | 4 days | 19 | 4 |
| `l40` (marked as the default partition) | 2 days | 6 | 8 |
| `interactive` | 4 hours | 2 plus 1 | Two nodes report 8 GPUs each; one reports 4 |
| `debug` | 30 minutes | 1 | 4 |

These are partition-level time limits; account/QoS limits can be stricter. The GRES output is untyped (`gpu:8`/`gpu:4`). Subsequent one-GPU batch diagnostics on `dgx` succeeded and reported an A100-SXM4-80GB; this verifies the sampled device, rather than every node's configuration. There is no `l4` partition in this output, so do not assume an accessible L4 queue.

Slurm associations exist for both accounts `23b0313` and `ajitvr`, with a blank partition restriction field. A wider query has now resolved the previously truncated QoS lists:

| Account | Reported allowed QoS | Default QoS field |
|---|---|---|
| `23b0313` | `a40,debug,dgx,dgx-mpi,gh200,interactive,l4,l40,normal` | Blank |
| `ajitvr` | `a40,debug,dgx,dgx-mpi,gh200,interactive,l4,l40` | Blank |

QoS names `gh200` and `l4` do not establish a currently visible matching partition or available hardware. QoS membership must be combined with partition rules.

`scontrol show partition dgx` reports `AllowGroups=ALL`, `AllowAccounts=ALL`, `AllowQos=dgx`, maximum time six days, no partition minimum node count beyond zero, and 72 GPU resources across nine nodes. Both account associations include QoS `dgx`, so `--account=ajitvr --partition=dgx --qos=dgx` is a supported combination to try for a small diagnostic allocation. The explicit account is a practical choice for this group project; the default or preferred charging account has not been checked. QoS-specific resource minima/limits, fair share and scheduling can still affect acceptance/start.

The user attempted `srun --pty` with `ajitvr`/`dgx`/`dgx`, one GPU, four CPUs, 16G memory and one-minute walltime. Slurm rejected it with `Interactive jobs are only allowed on partition 'interactive'`. This is an observed site submission rule that was not visible in the partition output. The request did not allocate resources. Do not repeat an interactive shell request on `dgx`; reserve interactive sessions for the `interactive` partition after checking its applicable rules.

Use a short **batch** diagnostic on `dgx`, issued on the login node:

```bash
sbatch --account=ajitvr --partition=dgx --qos=dgx \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 \
  --mem=16G --time=00:02:00 --job-name=gpu-check \
  --output=gpu-check-%j.out --error=gpu-check-%j.err \
  --wrap='nvidia-smi'
```

Slurm returns a job ID if the batch request is accepted. Use `squeue --me` to inspect its state. Once complete, inspect `gpu-check-JOBID.out` and `gpu-check-JOBID.err`, replacing JOBID with the returned number. This job runs GPU inventory and then exits, releasing its allocation automatically. The two-minute walltime is a ceiling, not a request to keep running for two minutes. A pending request is normal; do not repeatedly submit identical jobs.

**Observed result, 16 October 2026:** the user submitted this batch command twice, producing job IDs `353387` and `353388`. Both output files show a working `nvidia-smi` query and one visible `NVIDIA A100-SXM4-80GB`, 81920 MiB memory, driver `550.144.03`, header CUDA version `12.4`, and MIG disabled. Both jobs were absent from `squeue --me` when inspected. For such a short command this is consistent with having already finished; `squeue` is not completed-job history. No training workload was run and the displayed zero utilization is an idle inventory snapshot, not evidence of poor training performance. Exact accounting exit states can be checked with `sacct -j 353387,353388 --format=JobID,State,ExitCode,Elapsed`.

The `nvidia-smi` CUDA header describes the driver's CUDA support; it does not establish the selected Python environment's toolkit/runtime or PyTorch CUDA functionality. See [NVIDIA's nvidia-smi reference](https://docs.nvidia.com/deploy/nvidia-smi/index.html). Inspect installed environments before choosing a PyTorch build and then execute the GPU FFT/gradient check in an allocation.

An equivalent non-interactive diagnostic is saved in [hpc/prajna_gpu_check.sbatch](hpc/prajna_gpu_check.sbatch). Copy that file to the cluster and submit with `sbatch prajna_gpu_check.sbatch` from a writable home directory. It requests one GPU for five minutes, reports GPU details and mount visibility, writes its logs in the submission directory, and exits. It installs nothing and does not depend on project datasets or PyTorch. Local shell syntax validation does not establish remote scheduler acceptance or GPU functionality.

To reproduce the scheduler inspection:

```bash
sacctmgr show assoc where user="$USER" format=Account%25,Partition%25,QOS%180,DefaultQOS%30
scontrol show partition dgx
```

User/group quotas for Praganak and Rudra remain uninspected. Prajna's GPU inventory diagnostic is now confirmed. Its Python/PyTorch environment, actual CUDA calculation, assigned working directory, compute-node storage visibility, storage policy and QoS-specific resource limits still need inspection. Lightweight environment discovery on the login node can begin with:

```bash
command -v python python3 conda spack
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack find -l python
spack find -l py-torch
spack find -l cuda
```

These commands inspect installed tools and Spack packages; they do not install anything or start numerical work. If no suitable packages appear, inspect available site modules/Conda environments next. Do not assume the login shell's default Python contains the project's dependencies.

Treat the manuals as descriptions of the facilities, not authorization to execute their examples or contact support. Account creation, password reset, package removal, support messages, and account closure instructions are reference material.

Manual facts below are distinguished from project recommendations. Queue names, account associations, limits, software versions, filesystem mounts, and GPU availability must be checked on the live cluster. The published Prajna/Praganak queue-policy pages and Rudra homepage could not be retrieved by the web tool during this review; no current online policy was verified.

Source documents:

- [Prajna manual, 20 PDF pages](</Users/abhay/Downloads/Updated User Manual for Prajna (AiML) HPC.pdf>). Page references below use PDF page numbers.
- [PARAM Rudra manual, 98 PDF pages](</Users/abhay/Downloads/PARAM_Rudra_User's_Manual-IITB-V1.pdf>). Dated 14 January 2025; its printed page numbers are one less than PDF page numbers. References below use printed page numbers unless marked otherwise.
- [Praganak manual, 14 PDF pages](</Users/abhay/Downloads/User Manual for PRAGANAK - HPC facility.pdf>). Page references use PDF page numbers. Several commands are images; these were visually inspected.

## 2. The facilities compared

These are separate clusters. Do not assume they share home directories, scratch space, Python environments, scheduler accounts, or available GPUs.

| Item | Prajna AI/ML | PARAM Rudra | Praganak |
|---|---|---|---|
| Scheduler | Slurm | Slurm | PBS Professional |
| Login | `ssh 23b0313@prajna.iitb.ac.in` (access email) | `ssh -p 4422 USER@paramrudra.iitb.ac.in`; manual describes captcha then password | `ssh USER@praganak.iitb.ac.in` (p. 6 screenshot) |
| GPU hardware documented | 9 nodes x 8 A100; 20 x 4 A40; 10 x 2 L4; 7 x 8 L40S | 30 nodes x 2 A100, 80 GB GPU memory per card | 6 nodes x 8 A100, 80 GB GPU memory per card |
| Host memory documented | A40 example node approximately 512 GB; complete per-type table absent | GPU/ordinary CPU nodes 192 GB; high-memory nodes 768 GB | CPU and GPU nodes 512 GB |
| CPU nodes | GPU-focused manual; do not infer CPU partition entitlement | 482 ordinary CPU nodes, 90 high-memory nodes; 48 cores per node | 240 CPU nodes, 112 cores each |
| Total storage advertised | About 2 PB, Lustre; flash/disk tiers described | 4.4 PiB, Lustre | About 5 PB/PiB; terminology varies in manual |
| Personal quota | Manual gives no numbers; live `/home` user/group reports show zero limits (see observations above) | 50 GB home and 200 GB scratch **soft limits in the 2025 manual**; current quota unverified | Group quotas mentioned, numbers absent |
| Home layout | `/home/<group>/<user>/`; likely `/home/ajitvr/23b0313/`, verify using `$HOME` | `/home/<user>/` | `/home/<group>/<user>/` |
| Scratch | Live `/scratch` is a separate 250G NFS4 mount; `/lustre-scratch` is a distinct Lustre mount to inspect for working data | `/scratch/<user>/` documented | Shared across login/compute nodes within this cluster according to manual; user path must be checked |
| Scratch retention | Files not accessed in three months may be deleted | Same; explicitly not backed up | Same |
| Package setup | Spack | Modules, Spack, Conda | Spack |

Sources: Prajna pp. 4-10, 17, 19; Rudra pp. 6-12, 17-24, 38-40, 75-83; Praganak pp. 2-12. Total machine capacity is not a user entitlement. RAID and a parallel filesystem do not establish a backup service.

Prajna's A100 memory capacity is not specified in the supplied manual, but the user's live `dgx` diagnostic now confirms an A100-SXM4-80GB. This does not measure full-resolution training memory requirements or every other node. Praganak calls its A100 GPUs “Ada”; this is a terminology error: A100 belongs to NVIDIA's Ampere family, also stated in Rudra's introduction. Do not infer an Ada GPU from that sentence.

## 3. Recommended division of work

**Start by establishing one reliable, single-GPU workflow on Prajna.** This is a practical starting choice because its AI/ML focus fits the code and the login is confirmed, not a measured claim that it will be fastest.

| Project activity | Initial placement | Reason and qualification |
|---|---|---|
| ConvLISTA / future unrolled QDM model training | One Prajna GPU, preferably an A100 if available and permitted | Establish memory use, correctness, and runtime before requesting more resources. Benchmark another accessible GPU if queue delay dominates. |
| Independent GPU seeds, noise conditions, ablations | Prajna or Rudra jobs | Independent runs suit arrays or separate jobs. Keep data, code, dependencies, and scientific protocol consistent. |
| Full-resolution GPU solver comparisons | One GPU on either Slurm cluster | Current comparison code runs inference comparisons on one GPU. Compare methods on the same device. |
| CPU kernel fitting, conversion, classical solver sweeps | Rudra CPU initially | Current NumPy programs are easy to place in small Slurm allocations; request cores according to actual parallelism. |
| Large parallel CPU preparation or future simulation campaigns | Praganak, when the program and policy fit | Its manual prohibits serial jobs and says CPU requests must be multiples of 56. Reserving 56 cores alone does not parallelize a Python program. |
| High-memory processing | Rudra high-memory partition, when measured need justifies it | Host RAM solves CPU-memory pressure; it does not enlarge GPU VRAM. |
| Praganak GPU experiments | After confirming a site-supported GPU PBS example | GPU hardware is documented, but the manual does not specify a complete GPU resource request or whether CPU-granularity rules differ for GPU jobs. |
| COMSOL generation | Only after checking installed software, licence availability, and permitted execution | Access to an HPC account does not establish a COMSOL licence or that the intended study benefits from GPUs. |

The present ConvLISTA training batch has three images: Bx, By, Bz. `--gpus 2` uses single-node `torch.nn.DataParallel`; it is not multi-node distributed training. Three images provide little data-parallel work for an eight-GPU node. Start with one GPU, then measure one versus two using identical settings. The synthetic ALISTA training CLI uses one selected device and has no multi-GPU option.

More compute should first buy independent geometries, physical standoffs, noise tests, seeds, and trustworthy validation. More epochs on the existing three images do not establish generalization.

## 4. First session: inspect access without running research on the login node

On the Mac, log into Prajna:

```bash
ssh 23b0313@prajna.iitb.ac.in
```

On the cluster, collect these lightweight facts:

```bash
hostname
whoami
id
printf '%s\n' "$HOME"
pwd
df -h "$HOME" /scratch
quota -s
```

`df` shows filesystem capacity, not your quota. `quota` may be unavailable or not report Lustre/group limits. If the paths are Lustre and `lfs` is installed, inspect both user and group quotas on the actual mounts:

```bash
lfs quota -u "$USER" "$HOME"
lfs quota -g ajitvr "$HOME"
lfs quota -u "$USER" /scratch
lfs quota -g ajitvr /scratch
```

These group commands are specifically for Prajna's confirmed group. Check both byte and inode/file-count quotas. A command failure or empty report does not mean unlimited space. Verify the assigned scratch directory before creating it; do not guess ownership from a generic `/scratch` path.

Update: Prajna's user/group `/home` quota queries above have now been supplied by the user. Its observed `/scratch` is not Lustre, so do not repeat `lfs quota` against that mount. Use filesystem-appropriate quota tooling after identifying its type and the site's designated work area.

On Prajna/Rudra, inspect Slurm:

```bash
sinfo -o '%P %a %l %D %G'
scontrol show partition
squeue --me
sacctmgr show assoc where user="$USER" format=Cluster,Account,Partition,QOS,DefaultQOS
```

Account-association details may be restricted. Record the permitted account, partition, QoS, GPU count, CPU/memory limits, and walltime from the live system or support. A Unix group name does not prove an identical Slurm account name. GPU inventory in `sinfo` is not an entitlement or a promise of immediate availability.

On Praganak:

```bash
qstat -q
qstat -Qf
qstat -u "$USER"
```

If permitted, `pbsnodes -av` can expose node resource definitions; use it to inspect the site's GPU resource names. Ask for the official PyTorch GPU PBS example if policy/resource granularity remains unclear.

Use login nodes for editing, file transfer, environment inspection, compilation permitted by the site, and job submission. Run training, large conversions, numerical tests, and substantial plotting on allocated compute nodes.

## 5. Cluster-specific scheduler facts

### Prajna

The manual shows `a40` and `l40` partitions. The hardware called L40S is represented by `l40` in those examples. It does not establish the current names for the A100 or L4 partitions.

The pp. 16-17 example explicitly shows a job pending because QoS `normal` is not permitted in partition `l40`; that partition allows QoS `l40`. Thus partition and QoS must both be checked. Do not assume every current partition requires an identically named QoS.

The `infinite`/`UNLIMITED` values in example partition output do not override account/QoS limits. Current policy is delegated to [Prajna queue policy](https://hpcverse.iitb.ac.in/queue-policy/prajna), which was inaccessible during this review.

Spack initialization documented on p. 19:

```bash
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack find
spack find --loaded
```

Load exact installed package versions after inspection. `spack list` lists available recipes, whereas `spack find` lists installed software.

### PARAM Rudra

SSH and SCP/SFTP use port **4422**. For SCP, the port flag is uppercase `-P`:

```bash
scp -P 4422 local-file USER@paramrudra.iitb.ac.in:/confirmed/remote/directory/
```

The manual's Slurm examples use `standard`, `cpu`, `gpu`, `hm`. Prose capitalizes CPU/GPU, so use the live partition names exactly. Selecting `gpu` is insufficient by itself: explicitly request the GPU resource, e.g. `--gres=gpu:1` when confirmed appropriate.

Historical policy on p. 24: ten simultaneous jobs; eight-node jobs up to four days, sixteen-node jobs up to two days, thirty-two-node jobs up to one day; default walltime two hours. These are 2025 documented limits, not verified present entitlements. Explicitly request an appropriate runtime.

The manual discourages long interactive work; use short allocated sessions for diagnosis and batch jobs for experiments. The documented password validity is 90 days.

Software entry points documented on pp. 40 and 75-78:

```bash
module avail
module load spack
spack find
```

An alternative documented Spack setup is `source /home/apps/spack/share/spack/setup-env.sh`. Conda is documented through `module load miniconda`. The manual lists PyTorch 2.2.0 and GPU PyTorch 2.2.1; the project currently declares `torch>=2.6`. Inspect current modules and create a compatible user environment rather than silently assuming the historical module meets the requirement.

### Praganak

Use PBS commands and `#PBS` directives:

| Action | Slurm (Prajna/Rudra) | PBS (Praganak) |
|---|---|---|
| Submit | `sbatch job.sh` | `qsub job.pbs` |
| Your jobs | `squeue --me` | `qstat -u "$USER"` |
| Job details | `scontrol show job JOBID` | `qstat -f JOBID` |
| Cancel | `scancel JOBID` | `qdel JOBID` |
| Submission directory in job | `$SLURM_SUBMIT_DIR` | `$PBS_O_WORKDIR` |

Manual snapshot queues: `debug`, `small`, `medium`, `large`, `gpu`, `gpu-debug`, `himem`. Snapshot walltimes: debug/GPU debug 30 minutes; small/GPU/high-memory 96 hours; medium 72 hours; large 48 hours. The live [Praganak policy](https://hpcverse.iitb.ac.in/queue-policy/praganak) takes precedence; it was inaccessible during review.

The CPU example in the p. 10 image requests `select=1:ncpus=56:mpiprocs=56`, uses `place=pack`, and launches MPI. Page 11 says CPU count must be a multiple of 56. Do not launch 56 copies of `scripts/lista.py` with MPI: the code is not an MPI program and copies could contend for GPUs/output files.

Documented Spack setup:

```bash
source /apps/spack/v0.23/share/spack/setup-env.sh
spack find
```

The manual does not establish the exact GPU request fields, permitted fractional GPU allocation, or CPU allocation rule for a single-GPU Python job. Confirm these before preparing a production GPU PBS script.

## 6. Storage and transferring this project

Recommended layout within each cluster, using verified assigned paths:

```text
HOME/project/                 source, small configuration, dataset manifests
HOME/environments/            user Python environment if quota permits
ASSIGNED_SCRATCH/project/     active numerical arrays and run directories
INDEPENDENT_BACKUP/project/   irreplaceable raw data, selected models and results
```

Keep a single authoritative raw-data archive plus a backup independent of the scratch filesystem. Maintain active copies only on the clusters actually running those experiments. Do not treat `/home` as automatically backed up; confirm its backup policy too. Copy selected results out soon after each successful run, rather than waiting for scratch retention to expire.

For this repository, Git alone does not deliver the experiment data: `data/` and `runs/` are ignored. The QDM comparison requires `cache/grid.json` and six component arrays for the reference/target heights. Source files and data must belong to the same experiment definition.

Recommended transfer process: create the destination directories after checking paths and quotas; transfer source/configuration separately from bulky raw data and results; use SCP as documented or `rsync` when installed on both ends and supported by the site's authentication. For Rudra, an rsync SSH transport must specify port 4422. Do not upload Mac virtual environments; build a Linux environment on each cluster. Do not use transfer options that delete destination files unless deliberately synchronizing a disposable directory.

The primary cache contains 1400 x 1400 float32 fields. One array is about 7.84 MB (7.48 MiB); six input/reference fields are about 47.0 MB; all 44 height/component arrays about 345 MB. This excludes raw CSVs, temporary arrays, FFT buffers, gradients, activations, and output artifacts. Dataset size is not an estimate of training VRAM.

Keep geometry identity explicit. The existing cache names encode height and component, not geometry. Use separate dataset roots/manifests for different geometries; do not overwrite the primary cache with another geometry. `data/data2` is a separate source family.

Prefer binary arrays for repeated numerical reads over reparsing large CSVs every epoch. If a future dataset has millions of small crops, consider measured, moderately sized shards rather than creating millions of files on shared storage. Leave Lustre striping at site defaults initially; the Rudra large-file examples concern much larger files and are not a prescription for ordinary NumPy arrays.

## 7. Python environment and a first GPU check

Project dependencies: NumPy >=1.24, pandas >=2, Matplotlib >=3.7, PyTorch >=2.6 (`requirements-lista.txt`). An installed compatible environment can be reused; otherwise create a user-owned environment with Python and CUDA-enabled PyTorch compatible with the cluster driver. Choose versions after inspecting the actual software stack; do not install a CPU-only PyTorch build by accident or modify shared environments.

Load/activate the same environment explicitly inside every batch script. Installing a CUDA toolkit module alone does not establish that PyTorch can use the GPU. Check the driver and run a real calculation inside the GPU allocation.

Example diagnostic Python to run **on an allocated GPU compute node**, after environment activation:

```python
import platform
import torch

print('Python:', platform.python_version())
print('PyTorch:', torch.__version__, 'build CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA is unavailable to PyTorch'
print('Visible GPUs:', torch.cuda.device_count())
p = torch.cuda.get_device_properties(0)
print('GPU:', p.name, 'VRAM GiB:', p.total_memory / 2**30)
x = torch.randn(1, 1, 256, 256, device='cuda', requires_grad=True)
y = torch.fft.ifft2(torch.fft.fft2(x)).real
assert torch.allclose(x, y, atol=2e-6, rtol=2e-5)
y.square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all()
print('CUDA FFT and gradient check passed')
```

This checks functionality, not the full training memory budget or scientific results. Then run the repository's numerical suite on an allocated compute node. The existing two-GPU test requires an allocation with two GPUs; one-GPU or CPU sessions can legitimately skip it.

## 8. A Slurm starting template for the current QDM experiment

This is a **template pending filesystem paths and environment values**. For Prajna, the observed account/partition/QoS combination `ajitvr`/`dgx`/`dgx` can now be supplied explicitly; remote resource acceptance still needs a pilot. The resource amounts below are proposed pilot values, not manual requirements or verified sufficient memory/runtime. Adjust after measuring. Replace the marked paths with existing verified paths. The output run directory must not already exist.

```bash
#!/bin/bash
#SBATCH --job-name=qdm-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=qdm-%j.out
#SBATCH --error=qdm-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Replace with this cluster's verified module loading/environment activation.
source /ABSOLUTE/PATH/TO/PROJECT_ENV/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg

hostname
nvidia-smi
python -m pip freeze > "environment-${SLURM_JOB_ID}.txt"

# Run from a source checkout with a verified primary cache under the data root.
python -u scripts/lista.py \
  --data-root /ABSOLUTE/PATH/TO/DATA_ROOT \
  --device cuda --gpus 1 \
  --z 3 --layers 8 --kernel-size 9 --epochs 10 --lr 1e-4 \
  --epsilon 1e-3 --pad 32 --lambda-ista 1e-4 \
  --checkpoint-layers \
  --output "/ABSOLUTE/ASSIGNED/SCRATCH/qdm-pilot-${SLURM_JOB_ID}"
```

Supply the verified partition/account/QoS at submission, or insert literal values in directives. Shell variables inside `#SBATCH` lines are not expanded like ordinary shell commands. A confirmed default account/QoS can be omitted. Use `sbatch --test-only` with the same resource/account options before real submission; this checks scheduler acceptance/estimates, not Python dependencies or data correctness.

The ten-epoch run exercises the existing full-resolution pipeline, including converged ISTA/FISTA baselines; it is not necessarily a short smoke test. Start with the small CUDA check first. After a successful pilot, measure runtime and memory, and then choose the production epoch count and walltime.

For two-GPU ConvLISTA, change the allocation to two GPUs and the Python argument to `--gpus 2`; retain one task and one node. Requesting two nodes does not distribute this implementation.

**Checkpoint limitation:** `--checkpoint-layers` saves activation memory by recomputing intermediate values; it is not training-restart support. The current ConvLISTA CLI writes its selected model after training and exposes no resume option. The ALISTA CLI saves logs/validation data during training but saves its model at the end and also has no optimizer-resume option. A timeout can therefore lose training progress. Use bounded pilots; add periodic resumable checkpoints before launching long campaigns that need recovery. This guide does not implement that feature.

## 9. Monitor, diagnose, and copy results

Slurm commands after submitting a real job:

```bash
squeue --me
scontrol show job JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS
```

Inspect job-step rows as well as the top-level job; memory accounting availability varies. `MaxRSS` measures host memory, not GPU VRAM. Sample GPU use inside the allocation, or use the site's supported GPU monitoring. A diagnostic `nvidia-smi` before training only confirms visibility; it does not show peak training memory.

| Symptom | Likely next check |
|---|---|
| Pending, QoS/account reason | Permitted association, partition and QoS; Prajna includes an explicit mismatch example |
| Pending, resources/priority | Queue load, fair share, requested size, walltime; do not repeatedly resubmit |
| CUDA unavailable | Was a GPU actually allocated? Is the activated PyTorch CUDA-enabled? Is the driver compatible? |
| CUDA out of memory | GPU VRAM and activation memory; more Slurm `--mem` alone will not fix GPU memory exhaustion |
| Host out of memory | Accounted RAM, CPU-side kernel/data arrays, actual requested host memory |
| TIMEOUT | Pilot-based runtime estimate and absence/presence of resumable checkpoints |
| Output already exists | Use a fresh run directory; current QDM/ALISTA CLIs protect existing outputs |
| Dataset missing after Git transfer | Transfer ignored external data separately and verify cache paths |

For PBS, use `qstat -f JOBID`; completed-job history depends on site configuration. The manual's shorthand `qstat -x/H` is not a literal command to paste.

Preserve source revision plus working-tree changes/source hashes, CLI settings, dataset hashes/manifests, seeds, environment versions, GPU model, scheduler job ID, logs, convergence status, numerical metrics, and selected model weights. The current repository has uncommitted changes, so a commit ID alone cannot identify the complete code used. Keep local snapshots of the actual run source.

Copy important outputs to durable storage and an independent backup promptly. Validate transferred artifacts with a checksum manifest when practical. A successful scheduler exit is not sufficient: check expected files, finite metrics, solver convergence, and meaningful scientific results.

## 10. Scaling the research sensibly

1. Establish access, storage paths, quotas, environment, and one GPU diagnostic.
2. Run allocated numerical checks and one bounded QDM pilot with the existing scientific definitions.
3. Reproduce a reference configuration, measure training/inference time and peak memory, and archive its inputs and outputs.
4. For a future generalization study, split source geometries/acquisitions before cropping or augmentation. Keep train, validation, and test data independent and prevent reference-assisted operator fitting from leaking held-out targets.
5. Parallelize independent experimental configurations with limited-concurrency arrays. Give every task a unique output directory and explicit dataset/configuration entry. A shared input archive is fine; a shared mutable output directory is not.
6. Use a fixed device and precision for method timing comparisons. Report results from other GPUs separately. Preserve the project's physical units, padding/boundary definitions, solver stopping, and normalization contracts.
7. Only then investigate multiple GPUs, larger models, or broader parameter sweeps using measured benefit per GPU-hour and actual queue delay.

The current QDM ConvLISTA comparison fits/evaluates the same three component images; its results are same-sample reconstruction scores. The implemented ALISTA benchmark is dense synthetic sparse recovery, not an already implemented full-resolution QDM ALISTA model. Larger HPC allocations do not change those scientific scopes.

A future learned QDM model may condition on standoff and sampling/FOV, but this still needs an explicit experiment and dataset interface. Physical super-resolution must be validated against held-out truth or defensible experimental evidence; changing tensor size is insufficient.

## 11. Manual corrections and cautions

| Manual item | Correct interpretation for this project |
|---|---|
| `//` annotations in sample shell/Slurm scripts | These are explanatory markup, not portable shell comments. Use separate `#` comments and clean directives. |
| Migration table says `#BATCH` | Slurm directives are `#SBATCH`. |
| Migration table maps PBS submit directory to `$PBS_JOBID` | Use `$PBS_O_WORKDIR`; Slurm uses `$SLURM_SUBMIT_DIR`. |
| Rudra DL example uses `$SLURM_WORKDIR` and CPU partition | Use the actual submission/working directory and explicitly request a GPU for GPU work. |
| `--ntasks-per-node` described as a core count | Tasks are processes/ranks. For the current single-process Python GPU job use one task plus appropriate `--cpus-per-task`. |
| Manuals suggest `singleton` for prerequisite work | It serializes same-user/same-name jobs; it does not require predecessor success. Use an explicit `afterok:JOBID` dependency for success-dependent processing. |
| Praganak describes `select=...:mem=256gb` as memory per core | PBS `mem` inside a select chunk is the memory for that chunk, not 256 GB for every CPU. |
| Rudra generic Jupyter example uses `--allow-root`, `0.0.0.0`, and an HTTPS URL | Ordinary users do not need `--allow-root`. Preserve token authentication, run on an allocated node, and use a site-supported SSH tunnel. Use the URL/protocol Jupyter actually prints. The manual's command does not itself configure TLS. |
| Rudra example node `gpu007` | It is an illustration. Use only a node assigned to your own active allocation, and the site's supported launch method. |
| Prajna partition output says unlimited | Account/QoS limits may still constrain the job. |
| Cluster has petabytes of storage | Personal/group quota and retention policy still apply. |

The task/process distinction and success dependency are checked against the [official Slurm sbatch reference](https://slurm.schedmd.com/sbatch.html). Chunk memory is checked against the [PBS Professional User Guide, section 4.3.3](https://2021.help.altair.com/2021.1.2/PBS%20Professional/PBSUserGuide2021.1.2.pdf). These general scheduler semantics do not verify local site policy.

## 12. Administrative details worth retaining

- Prajna/Praganak manual support: [IIT Bombay helpdesk](https://help.cc.iitb.ac.in/) and `hpc@iitb.ac.in`.
- Rudra manual support: [Rudra support](https://paramrudra.iitb.ac.in/support) and `rudrasupport@iitb.ac.in`.
- For support, provide the cluster, username, job ID, clean submission script, error excerpt, requested resources, environment versions, and data-path/quota symptom. Do not include passwords.
- Rudra p. 89 requests acknowledgement of NSM/PARAM Rudra in research outputs and reporting of resulting publications. Retain that section when preparing a thesis/paper; confirm any updated wording before submission. No equivalent specific wording is supplied by the other two manuals.
- Rudra p. 96 describes account closure through support and supervisor/project clearance. Export needed data and results before ending access.

## 13. Information needed to make the first production job concrete

For each account, collect: current permitted queue/partition and QoS/account, runtime/GPU/CPU limits, assigned scratch path, home/scratch byte and inode quotas, backup policy, and installed compatible Python/PyTorch environments. For Praganak also obtain its supported GPU PBS resource request and CPU allocation rules. For Prajna, the login, Unix group, home/Lustre-scratch user/group quota reports, complete account QoS lists, `dgx` partition permissions and distinct NFS/Lustre scratch mounts are confirmed. Site policy allows interactive jobs only in `interactive`; the `ajitvr`/`dgx`/`dgx` batch diagnostic has run twice and reports an A100-SXM4-80GB with driver 550.144.03. The Python/PyTorch environment, actual CUDA computation, preferred/default charging account, broader QoS resource limits, assigned writable working directory, compute-node storage visibility and storage policy remain to be checked. Other quota classes and policies are not established by zero user/group limits.

No remote login, file transfer, environment installation, job submission, or support communication was performed during this manual review.

## 14. Additional SSH access reported on 22 October 2026

The user confirms access to two additional SSH endpoints:

| Host | Username | Login command | Hardware and usage policy |
|---|---|---|---|
| `10.129.7.82` | `abhaychandra` | `ssh abhaychandra@10.129.7.82` | Confirmed hostname `user-ESC4000A-E11`; two Quadro RTX 5000 GPUs, 16 GB each; usage policy unverified |
| `10.129.6.189` | `abhaychandra` | `ssh abhaychandra@10.129.6.189` | Presumed to be the supplied `user-ESC4000A-E10` output with RTX A6000 48 GB; IP-to-host mapping still needs confirmation |

The user supplied hardware discovery output for both hostnames on 22 October. The second output explicitly follows an SSH login to `10.129.7.82`, establishing its E11 mapping. The first output does not show its SSH destination; assigning E10 to the remaining address `10.129.6.189` is an inference, not yet independently verified. `hostname -I` can help confirm the mapping. No connection has been attempted by this assistant.

| Hostname | Observed GPUs | Host RAM | Home filesystem | Driver / nvidia-smi CUDA header |
|---|---|---|---|---|
| `user-ESC4000A-E10` | One RTX A6000, 49140 MiB reported (48 GB class) | 62 GiB total, 58 GiB available | `/dev/nvme0n1p5`, 449G total, 89G available, 80% used | 590.48.01 / 13.1 |
| `user-ESC4000A-E11` | Two Quadro RTX 5000, 16384 MiB each (16 GB per GPU) | 125 GiB total, 122 GiB available | `/dev/nvme0n1p5`, 706G total, 49G available, 93% used | 580.178.04 / 13.0 |

These are snapshots of shared resources, not guaranteed personal allocations. Both GPU listings show only Xorg graphics processes and no listed numerical workload at that moment; this does not establish an exclusive reservation. E11's login banner identifies Ubuntu 22.04.5 LTS, x86_64, kernel 6.8.0-138-generic. One login-banner connection attempt to Ubuntu's release service failed; this does not by itself establish that package repositories or model downloads are inaccessible.

No paths were printed by `command -v sbatch srun qsub` on either host. Thus those scheduler commands were not found in the reported shells' PATH. Do not infer there is no scheduler anywhere or that direct GPU execution is automatically permitted; establish the lab's sharing/booking rules. Prajna's account names, paths, Spack setup and Slurm restrictions must not be copied to these hosts without evidence.

### Practical role for the project

- The A6000 48 GB server is a strong candidate for day-to-day development, numerical debugging and bounded single-GPU QDM pilots once direct execution/sharing policy and the Python environment are established. Actual full-resolution memory/runtime must be measured.
- The two 16 GB Quadro GPUs suit smaller experiments, tests or independent runs, with one process assigned to each permitted GPU. Their VRAM is not automatically pooled into a 32 GB device. The current combined-data runner is single-GPU; the original ConvLISTA CLI's DataParallel support splits a batch and does not pool all GPU memory for an arbitrary oversized sample.
- Prajna's confirmed A100 80 GB remains a useful candidate for the largest single-GPU runs, subject to measured performance and queue time. Do not present this suggested division of work as a measured speed comparison.
- Home filesystem headroom is limited on both new hosts, especially E11 (49G available). Inspect other data mounts and assigned storage before copying the complete COMSOL archive or downloading large model collections. Do not delete other users' data or change system packages to create space.

The Quadro RTX 5000 is compute capability 7.5, while the RTX A6000 is 8.6, per [NVIDIA's GPU table](https://developer.nvidia.com/cuda/gpus). Verify that the selected PyTorch binary contains support for the actual GPU architecture, then run an actual CUDA FFT/gradient check. The driver header's CUDA 13.x value is not proof of the Python package's CUDA build or a requirement to install CUDA 13.x. [Official PyTorch installation archives](https://pytorch.org/get-started/previous-versions/) list CUDA-specific builds; choose an environment based on the project's dependencies and tested hardware support rather than that header alone.

Next lightweight discovery on each host:

```bash
hostname -I
df -hT
command -v python3 conda
python3 --version
python3 -m pip show torch
```

These commands do not install software or start GPU computation. Missing Python/Pip/Torch or an inactive environment should be reported as discovery results; they do not mean the server cannot run PyTorch. The environment used for the actual experiment must be activated and recorded explicitly.
