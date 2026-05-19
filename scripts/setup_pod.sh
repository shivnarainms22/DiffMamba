#!/usr/bin/env bash
# One-shot pod setup for DiffMamba on RunPod A100 spot instances.
# Run once after cloning the repo, before the first training launch.
#
# Usage:
#   git clone https://github.com/shivnarainms22/DiffMamba && cd DiffMamba
#   bash scripts/setup_pod.sh

set -euo pipefail

echo "=== DiffMamba pod setup ==="
echo

# ---------- pre-flight checks ----------

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA not available. Use a RunPod PyTorch image with CUDA support."
    exit 1
fi

if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. mamba-ssm requires a *devel* CUDA image to compile."
    echo "       Use e.g. runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    exit 1
fi

TORCH_VER=$(python -c "import torch; print(torch.__version__)")
CUDA_VER=$(python -c "import torch; print(torch.version.cuda)")
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
echo "  PyTorch ${TORCH_VER}  |  CUDA ${CUDA_VER}  |  ${GPU_NAME}"
echo

# ---------- standard deps ----------

echo "--- Installing standard dependencies ---"
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
    "packaging" \
    "pytest>=8.0"
echo "  done."
echo

# ---------- CUDA extension builds ----------
# causal-conv1d must be installed before mamba-ssm.
# --no-build-isolation: build sees the already-installed torch headers instead
# of an isolated env. MAX_JOBS caps ninja workers to avoid OOM during compile.

echo "--- Building causal-conv1d (≈2 min) ---"
MAX_JOBS=4 pip install "causal-conv1d>=1.4.0" --no-build-isolation
echo "  done."
echo

echo "--- Building mamba-ssm (≈5 min) ---"
MAX_JOBS=4 pip install "mamba-ssm>=2.0.0" --no-build-isolation
echo "  done."
echo

# ---------- import verification ----------

echo "--- Verifying installs ---"
python - <<'EOF'
import importlib.metadata as meta
import torch, mamba_ssm, lightning, transformers, datasets, wandb

def v(pkg):
    try:
        return meta.version(pkg)
    except meta.PackageNotFoundError:
        return '?'

print(f"  torch          {torch.__version__}")
print(f"  mamba-ssm      {v('mamba-ssm')}")
print(f"  causal-conv1d  {v('causal-conv1d')}")
print(f"  lightning      {lightning.__version__}")
print(f"  transformers   {transformers.__version__}")
print(f"  datasets       {datasets.__version__}")
print(f"  wandb          {wandb.__version__}")
EOF

echo
echo "--- Verifying project imports ---"
python -c "import diffusion, dataloader, callbacks_py, hf_hub_callback, utils; print('  OK')"
echo

# ---------- done ----------

echo "=== Setup complete ==="
echo
echo "Set these env vars before launching (add to ~/.bashrc or RunPod secrets):"
echo
echo "  Required:"
echo "    export HF_HUB_REPO_ID=<user>/<repo>   # HF Hub repo for checkpoint backups"
echo "    export HF_TOKEN=<token>                # HF Hub write token"
echo "    export WANDB_API_KEY=<key>             # wandb — MUST be set; tmux is non-interactive"
echo
echo "  Optional (launch_tmux.sh sets defaults):"
echo "    export DATA_CACHE_DIR=/workspace/data  # dataset .dat cache (default: /workspace/data)"
echo "    export WANDB_DIR=/workspace/wandb      # wandb local cache  (default: /workspace/wandb)"
echo
echo "Pod volume: 150 GB recommended"
echo "  ~55 GB  OpenWebText download + tokenised .dat (first run only, reused across all 5 runs)"
echo "  ~7 GB   checkpoints per active run (save_top_k=3 + last.ckpt)"
echo "  ~15 GB  wandb local cache across all 5 runs"
echo "  ~13 GB  buffer"
echo
echo "First run will spend 30-60 min downloading and tokenising OpenWebText before step 0."
echo
echo "Launch order:"
echo "  bash scripts/launch_tmux.sh runD1  runD_130m seed=1"
echo "  bash scripts/launch_tmux.sh runD2  runD_130m seed=2"
echo "  bash scripts/launch_tmux.sh runB   runB_transformer_130m"
echo "  bash scripts/launch_tmux.sh s50    scaling_50m"
echo "  bash scripts/launch_tmux.sh s100   scaling_100m"
