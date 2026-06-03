#!/usr/bin/env bash
# SLURM job for the Stage-1 VLM on Northeastern Explorer (gpu partition, 8h limit).
# Do NOT run directly. Submit via scripts/submit_vlm.sh (passes env vars).
#
# Pre-emptively chains the next 8h segment (survives SIGKILL at wall time); each
# segment resumes from last.ckpt and the SigLIP feature memmap cache is reused
# (built once on the first segment). Exits immediately once step >= MAX_STEPS.
#
# Env vars injected by submit_vlm.sh:
#   RUN_NAME    vlm_align | vlm_sft   (must match the experiment's run dir)
#   EXPERIMENT  vlm_stage1_align | vlm_stage1_sft
#   MAX_STEPS   6000 | 8000           (must be a multiple of the 2000 ckpt interval)
#   EXTRA_ARGS  optional space-separated hydra overrides

#SBATCH --partition=gpu
#SBATCH --time=7:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:TERM@120

set -euo pipefail
USER="${USER:-$(whoami)}"
HOME="${HOME:-/home/${USER}}"

SCRATCH="/scratch/${USER}"
REPO="${HOME}/DiffMamba"
RUN_DIR="${SCRATCH}/DiffMamba/runs/${RUN_NAME}"
CKPT="${RUN_DIR}/checkpoints/last.ckpt"
LOG_DIR="${SCRATCH}/DiffMamba/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

echo "=== ${RUN_NAME} | job ${SLURM_JOB_ID} | $(date) ==="
echo "  experiment : ${EXPERIMENT} | max_steps : ${MAX_STEPS} | run_dir : ${RUN_DIR}"

# ---------- environment ----------
for _mod_init in \
    /usr/share/lmod/lmod/init/bash \
    /etc/profile.d/modules.sh \
    /usr/share/Modules/init/bash \
    /opt/apps/lmod/lmod/init/bash; do
    [[ -f "$_mod_init" ]] && { source "$_mod_init"; break; }
done
unset _mod_init

module load anaconda3/2024.06 cuda/12.8.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffmamba
export PATH="${HOME}/.conda/envs/diffmamba/bin:${PATH}"

python -c "import mamba_ssm" 2>/dev/null || {
    echo "mamba_ssm not importable — compiling from source (~10 min)..."
    MAX_JOBS=4 pip install "causal-conv1d>=1.4.0" --force-reinstall --no-deps --no-build-isolation --no-cache-dir --no-binary :all: -q
    MAX_JOBS=4 pip install "mamba-ssm>=2.0.0" --force-reinstall --no-deps --no-build-isolation --no-cache-dir --no-binary :all: -q
}

export WANDB_DIR="${SCRATCH}/wandb"
mkdir -p "${WANDB_DIR}" "${SCRATCH}/data"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- early exit if already done ----------
if [[ -f "${CKPT}" ]]; then
    STEP=$(python -c "import torch; print(torch.load('${CKPT}', map_location='cpu', weights_only=False).get('global_step', 0))")
    echo "Resuming from step ${STEP} / ${MAX_STEPS}"
    if [[ "${STEP}" -ge "${MAX_STEPS}" ]]; then
        echo "Training already complete. Exiting."
        exit 0
    fi
fi

# ---------- pre-emptive chain (next segment runs after this one) ----------
NEXT_JID=$(sbatch \
    --dependency=afterany:"${SLURM_JOB_ID}" \
    --job-name="${RUN_NAME}" \
    --output="${LOG_DIR}/${RUN_NAME}_%j.log" \
    --export=ALL,RUN_NAME="${RUN_NAME}",EXPERIMENT="${EXPERIMENT}",MAX_STEPS="${MAX_STEPS}",EXTRA_ARGS="${EXTRA_ARGS:-}" \
    "${REPO}/scripts/vlm_job.sh" | awk '{print $NF}')
echo "Next segment queued as job ${NEXT_JID}."

# ---------- training ----------
cd "${REPO}"
python main_vlm.py \
    +experiment="${EXPERIMENT}" \
    ${EXTRA_ARGS:-} \
    data.cache_dir="${SCRATCH}/data" \
    hydra.run.dir="${RUN_DIR}" \
    checkpointing.save_dir="${RUN_DIR}" \
    checkpointing.resume_from_ckpt=true \
    || echo "Training exited non-zero (SIGTERM checkpoint save at wall is normal)."

echo "=== Job ${SLURM_JOB_ID} done at $(date) ==="
