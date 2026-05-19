#!/usr/bin/env bash
# SLURM job script for DiffMamba on Northeastern Explorer (gpu partition, 8h limit).
# Do NOT run directly. Submit via scripts/submit_hpc.sh, which passes env vars.
#
# Auto-rechains: submits the next job segment before training starts so the
# chain survives even if this job is hard-killed at wall time.
# Each new segment resumes from last.ckpt. Once global_step >= MAX_STEPS the
# next submitted segment exits immediately.
#
# Env vars injected by submit_hpc.sh / sbatch --export:
#   RUN_NAME    e.g. runD1
#   EXPERIMENT  e.g. runD_130m
#   MAX_STEPS   e.g. 76000
#   EXTRA_ARGS  e.g. "seed=1"  (optional, space-separated hydra overrides)

#SBATCH --partition=gpu
#SBATCH --time=7:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

SCRATCH="/scratch/${USER}"
REPO="${HOME}/DiffMamba"
RUN_DIR="${SCRATCH}/DiffMamba/runs/${RUN_NAME}"
CKPT="${RUN_DIR}/checkpoints/last.ckpt"
LOG_DIR="${SCRATCH}/DiffMamba/logs"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

echo "=== ${RUN_NAME} | job ${SLURM_JOB_ID} | $(date) ==="
echo "  experiment : ${EXPERIMENT}"
echo "  max_steps  : ${MAX_STEPS}"
echo "  extra_args : ${EXTRA_ARGS:-}"
echo "  run_dir    : ${RUN_DIR}"
echo

# ---------- environment ----------

module load anaconda3/2024.06 cuda/13.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffmamba

# Build CUDA extensions if not already installed (first job only, ~15 min).
python -c "import mamba_ssm" 2>/dev/null || {
    echo "mamba_ssm not found — building CUDA extensions (one-time, ~15 min)..."
    MAX_JOBS=4 pip install "causal-conv1d>=1.4.0" --no-build-isolation -q
    MAX_JOBS=4 pip install "mamba-ssm>=2.0.0" --no-build-isolation -q
    MAX_JOBS=4 pip install "flash-attn>=2.5.0" --no-build-isolation -q
    echo "  CUDA extensions built."
}

export WANDB_DIR="${SCRATCH}/wandb"
mkdir -p "${WANDB_DIR}" "${SCRATCH}/data"

# ---------- early exit if already done ----------

if [[ -f "${CKPT}" ]]; then
    STEP=$(python -c "
import torch
ckpt = torch.load('${CKPT}', map_location='cpu', weights_only=False)
print(ckpt.get('global_step', 0))
")
    echo "Resuming from step ${STEP} / ${MAX_STEPS}"
    if [[ "${STEP}" -ge "${MAX_STEPS}" ]]; then
        echo "Training already complete. Exiting."
        exit 0
    fi
else
    echo "No checkpoint found — starting from scratch."
fi

# ---------- pre-emptive chain submission ----------
# Submit the next segment NOW (before training), so the chain survives
# even if this job is hard-killed (SIGKILL) at wall time.

NEXT_JID=$(sbatch \
    --dependency=afterany:"${SLURM_JOB_ID}" \
    --job-name="${RUN_NAME}" \
    --output="${LOG_DIR}/${RUN_NAME}_%j.log" \
    --export=RUN_NAME="${RUN_NAME}",EXPERIMENT="${EXPERIMENT}",MAX_STEPS="${MAX_STEPS}",EXTRA_ARGS="${EXTRA_ARGS:-}" \
    "${REPO}/scripts/hpc_job.sh" \
    | awk '{print $NF}')
echo "Next segment queued as job ${NEXT_JID} (runs after this job completes)."
echo

# ---------- training ----------

cd "${REPO}"

python main.py \
    +experiment="${EXPERIMENT}" \
    ${EXTRA_ARGS:-} \
    data.cache_dir="${SCRATCH}/data" \
    hydra.run.dir="${RUN_DIR}" \
    checkpointing.resume_from_ckpt=true \
    || echo "Training exited with non-zero status (SIGTERM checkpoint save is normal)."

echo
echo "=== Job ${SLURM_JOB_ID} done at $(date) ==="
