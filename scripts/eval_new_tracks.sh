#!/usr/bin/env bash
# One-shot evaluation of the hybrid-cfg-vqa tracks on a GPU node.
#
# Runs three things in sequence, each isolated from prior runs:
#   1. the new-track test suite (evidence the code works; non-fatal)
#   2. mode=uni_vqa_eval on the existing uni_stage3 ckpt (read-only)
#   3. mode=gen_eval cfg_scale sweep + a gen_sample at the chosen scale,
#      against the gen_stage2_cfg ckpt (only if it has finished training)
#
# All outputs go under ${EVAL_ROOT} (default /scratch/$USER/DiffMamba/eval) —
# NEVER into a training run dir. Checkpoints are only read, never written.
#
# Run on a GPU node (it does its own env setup so it works under srun OR sbatch):
#   srun --partition=gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G \
#        --time=2:00:00 --pty bash -lc 'bash scripts/eval_new_tracks.sh'
#
# Override any path/param via env, e.g.:
#   STEPS=128 NUM_EVAL=500 CFG_SCALES="1.0 2.0 4.0" bash scripts/eval_new_tracks.sh

set -uo pipefail   # NOT -e: optional steps are guarded and must not abort the run

# ---------- configuration (all overridable) ----------
USER="${USER:-$(whoami)}"
HOME="${HOME:-/home/${USER}}"
SCRATCH="${SCRATCH:-/scratch/${USER}}"
REPO="${REPO:-${HOME}/DiffMamba}"

RUNS="${RUNS:-${SCRATCH}/DiffMamba/runs}"
EVAL_ROOT="${EVAL_ROOT:-${SCRATCH}/DiffMamba/eval}"

UNI_CKPT="${UNI_CKPT:-${RUNS}/uni_stage3/checkpoints/best.ckpt}"
GEN_CFG_CKPT="${GEN_CFG_CKPT:-${RUNS}/gen_stage2_cfg/checkpoints/best.ckpt}"

NUM_EVAL="${NUM_EVAL:-200}"
STEPS="${STEPS:-64}"
CFG_SCALES="${CFG_SCALES:-1.0 1.5 2.0 3.0}"
SAMPLE_SCALE="${SAMPLE_SCALE:-2.0}"

mkdir -p "${EVAL_ROOT}"

echo "================================================================"
echo " eval_new_tracks | $(date)"
echo "   repo       : ${REPO}"
echo "   uni ckpt   : ${UNI_CKPT}"
echo "   gen-cfg    : ${GEN_CFG_CKPT}"
echo "   eval out   : ${EVAL_ROOT}"
echo "   steps=${STEPS}  num_eval=${NUM_EVAL}  cfg_scales='${CFG_SCALES}'"
echo "================================================================"

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO}"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "FATAL: no CUDA GPU visible. Run this on a GPU node (srun/sbatch with --gres)."
    exit 1
fi
python -c "import mamba_ssm" 2>/dev/null \
    || echo "WARN: mamba_ssm not importable — the dimamba backbone evals will fail."

# ---------- 1. test suite (non-fatal: record result, keep going) ----------
echo; echo "----- [1/3] new-track test suite -----"
python -m pytest \
    tests/test_hybrid_schedule.py tests/test_vqa_eval_utils.py \
    tests/test_cfg_utils.py tests/test_gen_template.py \
    tests/test_hybrid_dimamba.py -v
TEST_RC=$?
[[ ${TEST_RC} -eq 0 ]] && echo "tests: PASS" || echo "tests: FAIL (rc=${TEST_RC}) — continuing to evals anyway"

# ---------- 2. unified VQA eval (existing ckpt, read-only) ----------
echo; echo "----- [2/3] uni_vqa_eval -----"
if [[ -f "${UNI_CKPT}" ]]; then
    OUT="${EVAL_ROOT}/uni_vqa"
    mkdir -p "${OUT}"
    python main_vlm.py +experiment=uni_stage3 mode=uni_vqa_eval \
        eval.checkpoint_path="${UNI_CKPT}" \
        eval.num_eval="${NUM_EVAL}" sampling.steps="${STEPS}" wandb=null \
        checkpointing.save_dir="${OUT}" \
        && echo "uni_vqa_eval: done -> ${OUT}/uni_vqa_eval.json" \
        || echo "uni_vqa_eval: FAILED (rc=$?)"
else
    echo "SKIP: ${UNI_CKPT} not found."
fi

# ---------- 3. CFG Stage-2 sweep + sample (only if trained) ----------
echo; echo "----- [3/3] gen_stage2_cfg eval sweep -----"
if [[ -f "${GEN_CFG_CKPT}" ]]; then
    for s in ${CFG_SCALES}; do
        OUT="${EVAL_ROOT}/gen_cfg_s${s}"
        mkdir -p "${OUT}"
        echo ">>> cfg_scale=${s}"
        python main_vlm.py +experiment=gen_stage2_cfg mode=gen_eval \
            eval.checkpoint_path="${GEN_CFG_CKPT}" \
            sampling.cfg_scale="${s}" sampling.steps="${STEPS}" wandb=null \
            checkpointing.save_dir="${OUT}" \
            || echo "gen_eval cfg_scale=${s}: FAILED (rc=$?)"
    done

    OUT="${EVAL_ROOT}/gen_cfg_samples"
    mkdir -p "${OUT}"
    echo ">>> qualitative samples at cfg_scale=${SAMPLE_SCALE}"
    python main_vlm.py +experiment=gen_stage2_cfg mode=gen_sample \
        eval.checkpoint_path="${GEN_CFG_CKPT}" \
        sampling.cfg_scale="${SAMPLE_SCALE}" sampling.steps="${STEPS}" wandb=null \
        checkpointing.save_dir="${OUT}" \
        && echo "gen_sample: done -> ${OUT}" \
        || echo "gen_sample: FAILED (rc=$?)"
else
    echo "SKIP: ${GEN_CFG_CKPT} not found — submit training first:"
    echo "      bash scripts/submit_vlm.sh gen_stage2_cfg gen_stage2_cfg 8000"
fi

echo; echo "================================================================"
echo " eval_new_tracks done | $(date)"
echo "   results under: ${EVAL_ROOT}"
echo "================================================================"
