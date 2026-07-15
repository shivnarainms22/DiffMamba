#!/usr/bin/env bash
# Back up DiffMamba checkpoints from /scratch to the Hugging Face Hub.
#
# /scratch purges by access time and WILL delete checkpoints that no running job
# is touching (this is how the conda env and, potentially, idle run checkpoints
# get reaped). This job mirrors every run's checkpoints to the Hub so training
# work survives the purge. Runs unattended as a batch job — a 200 GB upload must
# not depend on an interactive session staying alive.
#
# Submit:  sbatch scripts/backup_checkpoints.sh
#          REPO=Shiv-22/diffmamba-checkpoints sbatch --export=ALL,REPO scripts/backup_checkpoints.sh
#          RUNS="hyb_e3 hyb_e6" sbatch --export=ALL,RUNS scripts/backup_checkpoints.sh   # subset
# Watch :  tail -f /scratch/$USER/DiffMamba/logs/backup_ckpt_<jobid>.log
#
# Idempotent: the Hub skips a blob whose content hash already matches, so rerun it
# after every training run — only new/changed checkpoints actually transfer.

#SBATCH --job-name=backup_ckpt
#SBATCH --partition=short
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/%u/DiffMamba/logs/backup_ckpt_%j.log

set -uo pipefail   # NOT -e: one failed upload must not abort the rest of the backup
USER="${USER:-$(whoami)}"
HOME="${HOME:-/home/${USER}}"

SCRATCH="/scratch/${USER}"
RUNS_DIR="${SCRATCH}/DiffMamba/runs"
LOG_DIR="${SCRATCH}/DiffMamba/logs"
mkdir -p "${LOG_DIR}"

# Target repo: explicit REPO override, else the one the training jobs already push to.
REPO="${REPO:-${HF_HUB_REPO_ID:-}}"
if [[ -z "${REPO}" ]]; then
    echo "ERROR: no target repo. Set REPO=... or export HF_HUB_REPO_ID." >&2
    exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set — upload will fail auth. Export it and resubmit." >&2
    exit 1
fi

echo "=== backup_checkpoints | job ${SLURM_JOB_ID:-local} | $(date) ==="
echo "  repo : ${REPO}"
echo "  from : ${RUNS_DIR}"

# ---------- environment (hf CLI lives in the diffmamba env) ----------
for _mod_init in \
    /usr/share/lmod/lmod/init/bash \
    /etc/profile.d/modules.sh \
    /usr/share/Modules/init/bash \
    /opt/apps/lmod/lmod/init/bash; do
    [[ -f "$_mod_init" ]] && { source "$_mod_init"; break; }
done
unset _mod_init

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffmamba
export PATH="${HOME}/.conda/envs/diffmamba/bin:${PATH}"

# Faster, resumable transfers for multi-GB blobs. Falls back silently if the
# accelerator package is not installed.
python -c "import hf_transfer" 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1

if ! command -v hf &>/dev/null; then
    echo "ERROR: 'hf' CLI not found in the diffmamba env." >&2
    exit 1
fi

# Which runs to back up: RUNS override, else every run directory present.
if [[ -n "${RUNS:-}" ]]; then
    RUN_LIST="${RUNS}"
else
    RUN_LIST=""
    for d in "${RUNS_DIR}"/*/; do
        [[ -d "${d}checkpoints" ]] && RUN_LIST+=" $(basename "${d}")"
    done
fi
echo "  runs : ${RUN_LIST}"
echo

OK=0; FAIL=0; SKIP=0
FAILED_LIST=""
for run in ${RUN_LIST}; do
    CKPT_DIR="${RUNS_DIR}/${run}/checkpoints"
    if [[ ! -d "${CKPT_DIR}" ]]; then
        echo "SKIP ${run}: no checkpoints dir"; SKIP=$((SKIP+1)); continue
    fi
    shopt -s nullglob
    for f in "${CKPT_DIR}"/*.ckpt; do
        name="$(basename "${f}")"
        size="$(du -h "${f}" | cut -f1)"
        echo ">>> ${run}/${name} (${size})"
        if hf upload "${REPO}" "${f}" "runs/${run}/checkpoints/${name}" >/dev/null; then
            OK=$((OK+1))
        else
            echo "!! FAILED ${run}/${name}"
            FAILED_LIST+=" ${run}/${name}"
            FAIL=$((FAIL+1))
        fi
    done
    shopt -u nullglob
done

echo
echo "=================================================================="
echo "=== backup done at $(date) — uploaded ${OK}, failed ${FAIL}, skipped ${SKIP} runs ==="
if [[ ${FAIL} -gt 0 ]]; then
    echo "Failed uploads (rerun this job — idempotent, only these will retry):"
    for x in ${FAILED_LIST}; do echo "  ${x}"; done
    exit 1
fi
echo "All checkpoints backed up to https://huggingface.co/${REPO}/tree/main/runs"
