#!/usr/bin/env bash
# Submit a DiffMamba training run to Northeastern Explorer (SLURM).
# Handles job chaining automatically — each 8h segment resumes from last.ckpt.
#
# Usage:
#   bash scripts/submit_hpc.sh <run_name> <experiment> <max_steps> [hydra overrides...]
#
# Examples:
#   bash scripts/submit_hpc.sh runD1 runD_130m 76000 seed=1
#   bash scripts/submit_hpc.sh runD2 runD_130m 76000 seed=2
#   bash scripts/submit_hpc.sh runB  runB_transformer_130m 76000
#   bash scripts/submit_hpc.sh s50   scaling_50m 30000
#   bash scripts/submit_hpc.sh s100  scaling_100m 61000

set -euo pipefail

RUN_NAME="${1:?run_name required (e.g. runD1)}"
EXPERIMENT="${2:?experiment required (e.g. runD_130m)}"
MAX_STEPS="${3:?max_steps required (e.g. 76000)}"
shift 3
EXTRA_ARGS="$*"  # e.g. "seed=1"

SCRATCH="/scratch/${USER}"
REPO="${HOME}/DiffMamba"
LOG_DIR="${SCRATCH}/DiffMamba/logs"
mkdir -p "${LOG_DIR}"

# Warn if required env vars are missing.
for VAR in HF_HUB_REPO_ID HF_TOKEN WANDB_API_KEY; do
    if [[ -z "${!VAR:-}" ]]; then
        echo "WARNING: ${VAR} is not set — set it before submitting or training will fail."
    fi
done

JID=$(sbatch \
    --job-name="${RUN_NAME}" \
    --output="${LOG_DIR}/${RUN_NAME}_%j.log" \
    --export=RUN_NAME="${RUN_NAME}",EXPERIMENT="${EXPERIMENT}",MAX_STEPS="${MAX_STEPS}",EXTRA_ARGS="${EXTRA_ARGS}" \
    "${REPO}/scripts/hpc_job.sh" \
    | awk '{print $NF}')

echo "Submitted ${RUN_NAME} as job ${JID}."
echo "  Log : ${LOG_DIR}/${RUN_NAME}_${JID}.log"
echo "  Run : ${SCRATCH}/DiffMamba/runs/${RUN_NAME}/"
echo
echo "Monitor:"
echo "  squeue -u ${USER}"
echo "  tail -f ${LOG_DIR}/${RUN_NAME}_${JID}.log"
