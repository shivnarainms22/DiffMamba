#!/usr/bin/env bash
# Launch a training run inside a detached tmux session with unbuffered stdout
# logged to a file. Survives SSH disconnect; logs persist across reconnects.
#
# Usage:
#   bash scripts/launch_tmux.sh <session_name> <experiment_name> [extra hydra overrides]
#
# Examples:
#   bash scripts/launch_tmux.sh runD1 runD_130m seed=1
#   bash scripts/launch_tmux.sh runB1 runB_transformer_130m
#   bash scripts/launch_tmux.sh scale50 scaling_50m
#
# Attach later with:   tmux attach -t <session_name>
# Detach from inside:  Ctrl-b then d
# Kill if needed:      tmux kill-session -t <session_name>
# Tail logs from anywhere:  tail -f logs/<session_name>.log

set -euo pipefail

SESSION="${1:?session name required}"
EXPERIMENT="${2:?experiment name required}"
shift 2

# Refuse to clobber a live session.
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
    exit 1
fi

mkdir -p logs

LOG="logs/${SESSION}_$(date +%Y%m%d_%H%M%S).log"
echo "Launching $EXPERIMENT in tmux session '$SESSION'"
echo "Log: $LOG"

# Sanity check: env vars for HF Hub upload (warn but don't fail).
if [[ -z "${HF_HUB_REPO_ID:-}" ]]; then
    echo "WARNING: HF_HUB_REPO_ID is unset — checkpoints will not be uploaded."
    echo "         If you want spot-eviction safety: export HF_HUB_REPO_ID=<user>/<repo>"
fi
if [[ -z "${HF_TOKEN:-}" && -n "${HF_HUB_REPO_ID:-}" ]]; then
    echo "WARNING: HF_TOKEN unset but HF_HUB_REPO_ID set — uploads will fail."
fi

# Dataset cache: openwebtext.yaml hardcodes a Cornell cluster path.
# Override to pod volume (default /workspace/data; set DATA_CACHE_DIR to change).
CACHE_DIR="${DATA_CACHE_DIR:-/workspace/data}"
echo "Dataset cache: ${CACHE_DIR}"
mkdir -p "${CACHE_DIR}"

# PYTHONUNBUFFERED=1 + python -u → line-buffered output for live `tail -f`.
# `tee` keeps a copy in the file even if the session is killed.
CMD="PYTHONUNBUFFERED=1 python -u main.py +experiment=${EXPERIMENT} data.cache_dir=${CACHE_DIR} $* 2>&1 | tee ${LOG}"

tmux new-session -d -s "$SESSION" "$CMD"
echo
echo "Session started. Useful commands:"
echo "  tmux attach -t $SESSION"
echo "  tail -f $LOG"
echo "  tmux ls"
