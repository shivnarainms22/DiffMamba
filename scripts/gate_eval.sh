#!/usr/bin/env bash
# Grounding gate evaluation for uni_grounding_v2 (attempt 2), as a BATCH job so it
# queues for a GPU without holding an interactive srun session (avoids SSH timeout
# while waiting for allocation). Runs all three gate evals sequentially and writes
# their JSON outputs into the run dir.
#
# Submit:  sbatch scripts/gate_eval.sh
# Watch :  squeue -u $USER --name=gate_eval
#          tail -f /scratch/$USER/DiffMamba/logs/gate_eval_<jobid>.log
# Results: /scratch/$USER/DiffMamba/runs/uni_grounding_v2/{uni_vqa_eval,uni_grounding_diag,uni_eval}.json

#SBATCH --job-name=gate_eval
#SBATCH --partition=gpu
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/%u/DiffMamba/logs/gate_eval_%j.log

set -euo pipefail
USER="${USER:-$(whoami)}"
HOME="${HOME:-/home/${USER}}"

SCRATCH="/scratch/${USER}"
REPO="${HOME}/DiffMamba"
CKPT="${SCRATCH}/DiffMamba/runs/uni_grounding_v2/checkpoints/last.ckpt"
LOG_DIR="${SCRATCH}/DiffMamba/logs"
mkdir -p "${LOG_DIR}"

echo "=== gate_eval | job ${SLURM_JOB_ID:-local} | $(date) ==="
echo "  ckpt : ${CKPT}"

# ---------- environment (mirrors vlm_job.sh) ----------
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found: ${CKPT}" >&2
    exit 1
fi

cd "${REPO}"
COMMON="eval.checkpoint_path=${CKPT} data.cache_dir=${SCRATCH}/data sampling.steps=64 wandb=null"

# Each eval is independent: a crash in one must NOT abort the others, so a single
# (scarce) GPU allocation harvests every result it can. The DECISIVE one runs
# first, so even a wall-time kill still leaves us the gate JSON.
RC=0
echo "=== [1/3] uni_vqa_eval (DECISIVE: image_ablation_exact_delta) ==="
python main_vlm.py +experiment=uni_grounding mode=uni_vqa_eval ${COMMON} || { echo "[1/3] uni_vqa_eval FAILED (see traceback above)"; RC=1; }

echo "=== [2/3] uni_grounding_diag (freeze check: Probe A std_vs_ref ~1; Probe B sym_kl) ==="
python main_vlm.py +experiment=uni_grounding mode=uni_grounding_diag ${COMMON} || { echo "[2/3] uni_grounding_diag FAILED (see traceback above)"; RC=1; }

echo "=== [3/3] uni_eval (no-regression) ==="
python main_vlm.py +experiment=uni_grounding mode=uni_eval ${COMMON} || { echo "[3/3] uni_eval FAILED (see traceback above)"; RC=1; }

echo "=== gate_eval done at $(date) (rc=${RC}) ==="
echo "JSON outputs in: ${SCRATCH}/DiffMamba/runs/uni_grounding_v2/"
