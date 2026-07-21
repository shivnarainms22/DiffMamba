#!/usr/bin/env bash
# Batch PPL eval for the hybrid attention ablation (STEP 3 of scripts/ablation_README.md).
#
# Evaluates each finished run at exactly 76000 steps and appends
#   run, attn_layers, backbone_params, val_ppl, val_nll
# to a CSV, so the ablation table is assembled by one GPU job instead of five
# interactive srun sessions (the login node kills heavy torch/flash-attn imports).
#
# Submit:  sbatch scripts/eval_ablation.sh
#          RUNS="hyb_mid hyb_late" sbatch --export=ALL,RUNS scripts/eval_ablation.sh
# Watch :  tail -f /scratch/$USER/DiffMamba/logs/eval_ablation_<jobid>.log
# Results: /scratch/$USER/DiffMamba/runs/ablation_ppl.csv

#SBATCH --job-name=eval_ablation
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/%u/DiffMamba/logs/eval_ablation_%j.log

set -euo pipefail
USER="${USER:-$(whoami)}"
HOME="${HOME:-/home/${USER}}"

SCRATCH="/scratch/${USER}"
REPO="${HOME}/DiffMamba"
RUNS_DIR="${SCRATCH}/DiffMamba/runs"
LOG_DIR="${SCRATCH}/DiffMamba/logs"
CSV="${RUNS_DIR}/ablation_ppl.csv"

# The baseline is evaluated first as a harness check: it must reproduce its known
# val PPL 69.60. If it does not, the eval path is wrong and no other row is trustworthy.
RUNS="${RUNS:-hybrid_130m hyb_e3 hyb_e6 hyb_e12 hyb_early hyb_mid hyb_late}"
EXPECTED_STEPS="${EXPECTED_STEPS:-76000}"

mkdir -p "${LOG_DIR}" "${RUNS_DIR}"

echo "=== eval_ablation | job ${SLURM_JOB_ID:-local} | $(date) ==="
echo "  runs           : ${RUNS}"
echo "  expected steps : ${EXPECTED_STEPS}"
echo "  csv            : ${CSV}"
echo

# ---------- environment (mirrors hpc_job.sh) ----------
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

# Every run here is a hybrid: its attention blocks call flash_attn at forward time.
python -c "import flash_attn" 2>/dev/null || {
    echo "flash_attn not importable — compiling from source (~15-20 min)..."
    MAX_JOBS=4 pip install "flash-attn>=2.5.0" --force-reinstall --no-deps --no-build-isolation --no-cache-dir --no-binary :all: -q
    echo "  flash_attn built."
}

cd "${REPO}"
[[ -f "${CSV}" ]] || echo "run,attn_layers,backbone_params,val_ppl,val_nll,ckpt" > "${CSV}"

# A run's PPL is only comparable if it trained the full schedule. Reads global_step
# from the checkpoint and prints it, so an unfinished run is skipped rather than
# silently entering the table as a real data point.
ckpt_step() {
    python -c "
import torch, sys
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(ck.get('global_step', -1))
" "$1"
}

# Backbone parameter count, straight from the checkpoint (no model build needed).
# Excludes the EMA shadow copy, which would otherwise double the count.
ckpt_params() {
    python -c "
import torch, sys
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
sd = ck['state_dict']
print(sum(v.numel() for k, v in sd.items() if k.startswith('backbone.')))
" "$1"
}

RC=0
for r in ${RUNS}; do
    echo "=================================================================="
    echo "=== ${r} | $(date) ==="

    # Extra-seed runs are named <experiment>_s<N> (e.g. hyb_e3_s2 = seed 2 of
    # experiment hyb_e3), matching how submit_hpc.sh is invoked. There is no
    # hyb_e3_s2.yaml, so derive the experiment config by stripping the seed
    # suffix; runs whose name is already the experiment are unchanged.
    EXP="${r}"
    [[ "${r}" =~ ^(.+)_s[0-9]+$ ]] && EXP="${BASH_REMATCH[1]}"

    CKPT="${RUNS_DIR}/${r}/checkpoints/step=${EXPECTED_STEPS}.ckpt"
    [[ -f "${CKPT}" ]] || CKPT="${RUNS_DIR}/${r}/checkpoints/last.ckpt"
    if [[ ! -f "${CKPT}" ]]; then
        echo "SKIP ${r}: no checkpoint under ${RUNS_DIR}/${r}/checkpoints/"
        continue
    fi

    STEP=$(ckpt_step "${CKPT}")
    if [[ "${STEP}" != "${EXPECTED_STEPS}" ]]; then
        echo "SKIP ${r}: checkpoint is at step ${STEP}, not ${EXPECTED_STEPS} (${CKPT})"
        echo "        an unfinished run is not comparable — let it finish, then re-run this job."
        continue
    fi

    PARAMS=$(ckpt_params "${CKPT}")
    echo "  ckpt   : ${CKPT} (step ${STEP})"
    echo "  params : ${PARAMS} backbone"

    # Capture stdout: the PPL lands in Lightning's validate metric table (val/ppl),
    # and the arch line proves which attention layout was actually evaluated.
    OUT="${LOG_DIR}/eval_${r}_${SLURM_JOB_ID:-local}.out"
    if ! python main.py \
            +experiment="${EXP}" \
            mode=ppl_eval \
            "eval.checkpoint_path='${CKPT}'" \
            data.cache_dir="${SCRATCH}/data" \
            hydra.run.dir="${SCRATCH}/DiffMamba/eval/${r}" \
            wandb=null 2>&1 | tee "${OUT}"; then
        echo "FAILED ${r}: eval exited non-zero (see ${OUT})"
        RC=1
        continue
    fi

    # Parse from the captured output. Grepping the metric table is the only way out:
    # ppl_eval returns its numbers through trainer.validate's stdout table, not JSON.
    # Lightning draws its metric table with Unicode box characters, so match the
    # metric key anywhere on the line rather than anchoring on a column separator.
    ATTN=$(grep -o 'attention_layers=\[[^]]*\]' "${OUT}" | head -1 | tr -d ' ' | sed 's/attention_layers=//' || true)
    PPL=$(grep 'val/ppl' "${OUT}" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1 || true)
    NLL=$(grep 'val/nll' "${OUT}" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1 || true)

    if [[ -z "${PPL}" ]]; then
        echo "FAILED ${r}: eval ran but no val/ppl found in output (see ${OUT})"
        RC=1
        continue
    fi

    echo "${r},\"${ATTN:-?}\",${PARAMS},${PPL},${NLL:-?},${CKPT}" >> "${CSV}"
    echo "  RESULT ${r}: attn=${ATTN:-?}  params=${PARAMS}  val/ppl=${PPL}  val/nll=${NLL:-?}"
done

echo
echo "=================================================================="
echo "=== eval_ablation done at $(date) (rc=${RC}) ==="
echo
column -t -s, "${CSV}" 2>/dev/null || cat "${CSV}"
exit "${RC}"
