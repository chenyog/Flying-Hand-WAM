#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <task> <task_config> <checkpoint_path> [seed] [gpu_id] [episodes]"
  echo "Example: $0 flying_hand/blocks_ranking_rgb flying_hand_clean /path/to/600.ckpt 0 0 10"
}

if [[ $# -lt 3 || $# -gt 6 ]]; then
  usage >&2
  exit 2
fi

TASK_NAME="$1"
TASK_CONFIG="$2"
CHECKPOINT_PATH="$3"
SEED="${4:-0}"
GPU_ID="${5:-0}"
EPISODES="${6:-1}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONWARNINGS="ignore::UserWarning"

"${PYTHON_BIN}" script/eval_policy.py \
  --config policy/DP/deploy_policy.yml \
  --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --ckpt_setting flying_hand_dp \
  --expert_data_num 0 \
  --seed "${SEED}" \
  --eval_num_episodes "${EPISODES}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --device cuda:0
