#!/usr/bin/env bash
# SLURM job script for DiffMamba on Northeastern Explorer (gpu partition, 8h limit).
# Do NOT run directly. Submit via scripts/submit_hpc.sh, which passes env vars.
#
# Auto-rechains: submits the next job segment before training starts so the
# chain survives even if this job is hard-killed at wall time.
# Each new segment resumes from last.ckpt. Once global_step >= MAX_STEPS the
# next submitted segment exits immediately. A segment that starts from the same
# step as its predecessor refuses to chain (see "loop breaker" below).
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
USER="${USER:-$(whoami)}"
HOME="${HOME:-/home/${USER}}"

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

# Initialize module system (not active in SLURM batch env by default)
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

# Force diffmamba's Python to the front of PATH.
# module load anaconda prepends base conda, which can shadow diffmamba
# even after conda activate. This makes the right interpreter explicit.
export PATH="${HOME}/.conda/envs/diffmamba/bin:${PATH}"

# Build CUDA extensions if not importable. Flags explained:
#   --force-reinstall : recompile a broken .so instead of skipping
#   --no-deps         : do NOT swap torch out from under the build
#   --no-cache-dir + --no-binary :all: : compile from SOURCE, never reuse a
#                       cached wheel (a cached wheel built against the wrong
#                       CUDA/torch is exactly what keeps the broken .so around)
# causal-conv1d + mamba-ssm: needed by all dimamba (BiMamba) runs.
python -c "import mamba_ssm" 2>/dev/null || {
    echo "mamba_ssm not importable — compiling from source (~10 min)..."
    MAX_JOBS=4 pip install "causal-conv1d>=1.4.0" --force-reinstall --no-deps --no-build-isolation --no-cache-dir --no-binary :all: -q
    MAX_JOBS=4 pip install "mamba-ssm>=2.0.0" --force-reinstall --no-deps --no-build-isolation --no-cache-dir --no-binary :all: -q
    echo "  mamba_ssm built."
}

# flash-attn: the Transformer backbone (runB) AND the hybrid backbone's
# attention blocks (DDiTBlock.forward -> flash_attn_varlen_qkvpacked_func) call
# it at runtime. Gate on experiment name so pure-BiMamba runs don't waste ~20
# min building it. The import check short-circuits when it is already present,
# so covering hybrid runs is free when the env already has the wheel and saves
# a guaranteed crash-at-first-attention-layer when it does not.
if [[ "${EXPERIMENT}" == *transformer* || "${EXPERIMENT}" == *hyb* ]]; then
    python -c "import flash_attn" 2>/dev/null || {
        echo "flash_attn not importable — compiling from source (~15-20 min)..."
        MAX_JOBS=4 pip install "flash-attn>=2.5.0" --force-reinstall --no-deps --no-build-isolation --no-cache-dir --no-binary :all: -q
        echo "  flash_attn built."
    }
fi

export WANDB_DIR="${SCRATCH}/wandb"
mkdir -p "${WANDB_DIR}" "${SCRATCH}/data"

# Reduce CUDA memory fragmentation (recommended in the OOM error message).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    STEP=0
    echo "No checkpoint found — starting from scratch."
fi

# ---------- loop breaker ----------
# The chain is submitted pre-emptively below, so a segment that dies without
# training still spawns a successor. If two consecutive segments start from the
# same step, no progress is being made and chaining would recur forever. Fail
# closed instead. A segment killed before its first checkpoint save trips this
# too -- that is intended: it means something is wrong and needs a human.

PROGRESS_MARKER="${RUN_DIR}/.last_segment_start_step"
PREV_STEP=$(cat "${PROGRESS_MARKER}" 2>/dev/null || echo "-1")

if [[ "${STEP}" -eq "${PREV_STEP}" ]]; then
    echo "ERROR: previous segment also started at step ${STEP} and made no progress."
    echo "Refusing to queue another segment. Inspect the log above, then resubmit."
    exit 1
fi

echo "${STEP}" > "${PROGRESS_MARKER}"

# ---------- pre-emptive chain submission ----------
# Submit the next segment NOW (before training), so the chain survives
# even if this job is hard-killed (SIGKILL) at wall time.

NEXT_JID=$(sbatch \
    --dependency=afterany:"${SLURM_JOB_ID}" \
    --job-name="${RUN_NAME}" \
    --output="${LOG_DIR}/${RUN_NAME}_%j.log" \
    --export=ALL,RUN_NAME="${RUN_NAME}",EXPERIMENT="${EXPERIMENT}",MAX_STEPS="${MAX_STEPS}",EXTRA_ARGS="${EXTRA_ARGS:-}" \
    "${REPO}/scripts/hpc_job.sh" \
    | awk '{print $NF}')
echo "Next segment queued as job ${NEXT_JID} (runs after this job completes)."
echo

# ---------- training ----------

cd "${REPO}"

# trainer.max_steps MUST be overridden here. Every experiment config bakes in
# max_steps: 76_000, so without this the trainer stops the instant it resumes at
# 76000 while the shell check above still believes MAX_STEPS (e.g. 150000) of work
# remains -- and the pre-queued successor repeats that forever (157-job crash loop,
# 2026-07-19). The shell's notion of "done" and the trainer's must come from the
# same number.
TRAIN_RC=0
python main.py \
    +experiment="${EXPERIMENT}" \
    ${EXTRA_ARGS:-} \
    data.cache_dir="${SCRATCH}/data" \
    hydra.run.dir="${RUN_DIR}" \
    trainer.max_steps="${MAX_STEPS}" \
    checkpointing.resume_from_ckpt=true \
    || TRAIN_RC=$?

if [[ "${TRAIN_RC}" -ne 0 ]]; then
    echo "Training exited ${TRAIN_RC} (SIGTERM checkpoint save at wall time is normal)."
fi

echo
echo "=== Job ${SLURM_JOB_ID} done at $(date) ==="
