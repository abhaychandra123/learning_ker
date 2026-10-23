#!/bin/bash
#SBATCH --job-name=prajna-gpu-check
#SBATCH --account=ajitvr
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=prajna-gpu-check-%j.out
#SBATCH --error=prajna-gpu-check-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

printf 'Job: %s; account: %s; partition: %s\n' \
  "$SLURM_JOB_ID" "${SLURM_JOB_ACCOUNT:-unset}" "${SLURM_JOB_PARTITION:-unset}"
hostname
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

for storage_path in /home /scratch /lustre-scratch /lustre-flash; do
  if mountpoint -q "$storage_path"; then
    findmnt -M "$storage_path" -o TARGET,SOURCE,FSTYPE
    df -h "$storage_path"
  else
    printf 'Not mounted here: %s\n' "$storage_path"
  fi
done

printf 'GPU inventory and mount checks finished. PyTorch has not been tested.\n'
