#!/usr/bin/env bash
# Submit a Stage-1 VLM phase to Northeastern Explorer (SLURM), self-chaining
# across the 8h wall until MAX_STEPS is reached.
#
# Two-phase run (launch SFT only after ALIGN has finished — SFT warm-starts from
# the align best.ckpt, set in configs/experiment/vlm_stage1_sft.yaml):
#   bash scripts/submit_vlm.sh vlm_align vlm_stage1_align 6000
#   # ... wait for vlm_align to complete (squeue empty for it) ...
#   bash scripts/submit_vlm.sh vlm_sft   vlm_stage1_sft   8000
#
# RUN_NAME must match the run dir (vlm_align / vlm_sft).

set -euo pipefail

RUN_NAME="${1:?run_name required (vlm_align | vlm_sft)}"
EXPERIMENT="${2:?experiment required (vlm_stage1_align | vlm_stage1_sft)}"
MAX_STEPS="${3:?max_steps required (6000 | 8000)}"
shift 3
EXTRA_ARGS="$*"

SCRATCH="/scratch/${USER}"
REPO="${HOME}/DiffMamba"
LOG_DIR="${SCRATCH}/DiffMamba/logs"
mkdir -p "${LOG_DIR}"

for VAR in WANDB_API_KEY; do
    if [[ -z "${!VAR:-}" ]]; then
        echo "NOTE: ${VAR} not set — runs will use the local CSV logger (fine)."
    fi
done

JID=$(sbatch \
    --job-name="${RUN_NAME}" \
    --output="${LOG_DIR}/${RUN_NAME}_%j.log" \
    --export=ALL,RUN_NAME="${RUN_NAME}",EXPERIMENT="${EXPERIMENT}",MAX_STEPS="${MAX_STEPS}",EXTRA_ARGS="${EXTRA_ARGS}" \
    "${REPO}/scripts/vlm_job.sh" | awk '{print $NF}')

echo "Submitted ${RUN_NAME} as job ${JID}."
echo "  Log : ${LOG_DIR}/${RUN_NAME}_${JID}.log"
echo "  Run : ${SCRATCH}/DiffMamba/runs/${RUN_NAME}/"
echo "Monitor:  squeue -u ${USER}   |   tail -f ${LOG_DIR}/${RUN_NAME}_${JID}.log"
