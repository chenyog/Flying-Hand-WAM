#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  train_flying_hand.sh [options]

Train one ACT model on every Flying-Hand task. The converter always uses all
available episodes and assigns one global episode index across the merged set.

Options:
  --all-tasks          Explicitly select the all-task mode (default behavior)
  --gpus IDS           GPU ids, e.g. 0 or 0,1,2,3 (default: 0)
  --seed N             Training seed (default: 0)
  --epochs N           Number of training epochs (default: 6000)
  --batch-size N       Batch size per GPU (default: 8)
  --num-workers N      DataLoader workers per train/val loader and GPU (default: 1)
  --convert-workers N  Parallel workers for episode conversion (default: 32)
  --force-convert      Rebuild the merged ACT dataset
  -h, --help           Show this help

Examples:
  ./train_flying_hand.sh --gpus 0
  ./train_flying_hand.sh --gpus 0,1,2,3 --epochs 6000 --batch-size 8
EOF
}

GPU_IDS="0"
SEED="0"
EPOCHS="6000"
BATCH_SIZE="8"
NUM_WORKERS="1"
CONVERT_WORKERS="32"
FORCE_CONVERT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all-tasks) shift ;;
        --gpus)
            [[ $# -ge 2 ]] || { echo "--gpus requires a value" >&2; exit 2; }
            GPU_IDS="$2"; shift 2 ;;
        --seed)
            [[ $# -ge 2 ]] || { echo "--seed requires a value" >&2; exit 2; }
            SEED="$2"; shift 2 ;;
        --epochs)
            [[ $# -ge 2 ]] || { echo "--epochs requires a value" >&2; exit 2; }
            EPOCHS="$2"; shift 2 ;;
        --batch-size)
            [[ $# -ge 2 ]] || { echo "--batch-size requires a value" >&2; exit 2; }
            BATCH_SIZE="$2"; shift 2 ;;
        --num-workers)
            [[ $# -ge 2 ]] || { echo "--num-workers requires a value" >&2; exit 2; }
            NUM_WORKERS="$2"; shift 2 ;;
        --convert-workers)
            [[ $# -ge 2 ]] || { echo "--convert-workers requires a value" >&2; exit 2; }
            CONVERT_WORKERS="$2"; shift 2 ;;
        --force-convert) FORCE_CONVERT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown argument: $1 (ACT now trains all Flying-Hand tasks)" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "--gpus must be comma-separated integers" >&2
    exit 2
fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]] || ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || ! [[ "$CONVERT_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "seed must be non-negative; epochs/batch-size/convert-workers must be positive; num-workers may be zero" >&2
    exit 2
fi

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${PROJECT_ROOT}/../.." && pwd)
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
TORCHRUN_BIN="${REPO_ROOT}/.venv/bin/torchrun"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python
[[ -x "${TORCHRUN_BIN}" ]] || TORCHRUN_BIN=torchrun
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
GPU_COUNT=${#GPU_ARRAY[@]}
cd "$PROJECT_ROOT"

SIM_TASK_NAME="sim-flying_hand-all-tasks"
PROCESSED_DIR="processed_data/sim-flying_hand/all-tasks"
CKPT_DIR="./act_ckpt/act-flying_hand/all-tasks"

has_task_config() {
    "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
try:
    with open("SIM_TASK_CONFIGS.json", encoding="utf-8") as stream:
        configs = json.load(stream)
    if sys.argv[1] not in configs:
        raise KeyError(sys.argv[1])
except (OSError, ValueError, KeyError, TypeError):
    raise SystemExit(1)
PY
}

if (( FORCE_CONVERT == 1 )) || [[ ! -f "${PROCESSED_DIR}/conversion_complete.json" ]] || ! has_task_config "$SIM_TASK_NAME"; then
        "$PYTHON_BIN" process_data.py \
            --all-tasks \
            --output "$PROCESSED_DIR" \
            --workers "$CONVERT_WORKERS"
fi

TRAIN_ARGS=(
    --task_name "$SIM_TASK_NAME"
    --ckpt_dir "$CKPT_DIR"
    --policy_class ACT
    --kl_weight 10
    --chunk_size 32
    --hidden_dim 512
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --dim_feedforward 3200
    --num_epochs "$EPOCHS"
    --lr 1e-5
    --save_freq 2000
    --state_dim 5
    --seed "$SEED"
)

if (( GPU_COUNT > 1 )); then
    "$TORCHRUN_BIN" --standalone --nproc_per_node="$GPU_COUNT" imitate_episodes.py "${TRAIN_ARGS[@]}"
else
    "$PYTHON_BIN" imitate_episodes.py "${TRAIN_ARGS[@]}"
fi
