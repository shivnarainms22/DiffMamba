#!/usr/bin/env bash
# One-time setup for DiffMamba on Northeastern Explorer HPC.
# Run from the login node (or a gpu-interactive session for --build-cuda).
#
# Usage:
#   cd ~/DiffMamba
#   bash scripts/setup_hpc.sh              # standard deps only (login node)
#   bash scripts/setup_hpc.sh --build-cuda # also compile CUDA extensions (GPU node)
#
# For --build-cuda, first get an interactive GPU session:
#   srun --partition=gpu-interactive --gres=gpu:a100:1 --time=2:00:00 --pty bash
#   module load anaconda3/2024.06 cuda/13.2.0
#   source activate diffmamba
#   bash scripts/setup_hpc.sh --build-cuda

set -euo pipefail

BUILD_CUDA="${1:-}"
SCRATCH="/scratch/${USER}"

echo "=== DiffMamba HPC setup (Northeastern Explorer) ==="
echo

# ---------- conda env ----------

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "^diffmamba "; then
    echo "conda env 'diffmamba' already exists — skipping create."
else
    echo "Creating conda env 'diffmamba' (Python 3.11)..."
    conda create -n diffmamba python=3.11 -y
fi

conda activate diffmamba

# ---------- PyTorch (CUDA 12.8 wheels) ----------
# Explorer GPU nodes run NVIDIA driver 570.x (max CUDA 12.8). torch MUST be a
# cu12x build — a cu130 build will load but torch.cuda.is_available() is False.

echo "Installing PyTorch + standard dependencies..."
pip install -q torch --index-url https://download.pytorch.org/whl/cu128
pip install -q \
    "lightning>=2.0" \
    "hydra-core>=1.3" \
    "omegaconf>=2.3" \
    "transformers>=4.40" \
    "datasets>=2.14" \
    "wandb>=0.16" \
    "huggingface_hub>=0.21" \
    "rich>=13.0" \
    "fsspec>=2024.2" \
    "einops>=0.7" \
    "timm>=0.9" \
    "packaging" \
    "ninja" \
    "pytest>=8.0"
echo "  done."
echo

# ---------- CUDA extensions ----------
# causal-conv1d, mamba-ssm, flash-attn must be compiled with nvcc.
# Cannot run on the login node. Either:
#   (a) run this script with --build-cuda from a gpu-interactive session, or
#   (b) they get compiled automatically on the first job run (adds ~15 min).

if [[ "$BUILD_CUDA" == "--build-cuda" ]]; then
    if ! command -v nvcc &>/dev/null; then
        echo "Loading cuda/12.8.0 for nvcc (matches cu128 torch + driver)..."
        module load cuda/12.8.0
    fi
    # --no-cache-dir --no-binary :all: forces a source compile against the
    # installed torch — never reuse a cached wheel built for a different CUDA.
    echo "Building causal-conv1d (≈2 min)..."
    MAX_JOBS=4 pip install "causal-conv1d>=1.4.0" --no-deps --no-build-isolation --no-cache-dir --no-binary :all:
    echo "Building mamba-ssm (≈5 min)..."
    MAX_JOBS=4 pip install "mamba-ssm>=2.0.0" --no-deps --no-build-isolation --no-cache-dir --no-binary :all:
    echo "Building flash-attn (≈5-20 min)..."
    MAX_JOBS=4 pip install "flash-attn>=2.5.0" --no-deps --no-build-isolation --no-cache-dir --no-binary :all:
    echo "  CUDA extensions built."
else
    echo "Skipping CUDA extension builds (login node has no GPU)."
    echo "Run 'bash scripts/setup_hpc.sh --build-cuda' from a gpu-interactive session,"
    echo "or they will build automatically on first job submission (~15 min overhead)."
fi
echo

# ---------- scratch dirs ----------

mkdir -p \
    "${SCRATCH}/data" \
    "${SCRATCH}/wandb" \
    "${SCRATCH}/DiffMamba/logs" \
    "${SCRATCH}/DiffMamba/runs"
echo "Scratch dirs created at ${SCRATCH}/"
echo

# ---------- verify ----------

echo "Verifying imports..."
python -c "import torch, lightning, hydra, wandb, datasets, transformers; print('  standard imports OK')"
echo

echo "=== Setup complete ==="
echo
echo "Add to ~/.bashrc (then: source ~/.bashrc):"
echo "  export HF_HUB_REPO_ID=<user>/<repo>"
echo "  export HF_TOKEN=<token>"
echo "  export WANDB_API_KEY=<key>"
echo
echo "Submit runs from ~/DiffMamba:"
echo "  bash scripts/submit_hpc.sh runD1 runD_130m 76000 seed=1"
echo "  bash scripts/submit_hpc.sh runD2 runD_130m 76000 seed=2"
echo "  bash scripts/submit_hpc.sh runB  runB_transformer_130m 76000"
echo "  bash scripts/submit_hpc.sh s50   scaling_50m 30000"
echo "  bash scripts/submit_hpc.sh s100  scaling_100m 61000"
