#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_flying_hand.sh [CONFIG [RUN_PREFIX [GPU_IDS [FSDP_DEVICES]]]] [options]

Options:
  --dataset-dir PATH  Local LeRobot dataset root; repeat to concatenate datasets
  --assets-dir PATH   Directory containing dataset-specific normalization assets
  --batch-size N      Override the global training batch size
  --num-workers N     Override the training data-loader worker count
  --run-name NAME     Run-name prefix (a UTC timestamp is always appended)
  -h, --help          Show this help

Example:
  ./train_flying_hand.sh pi05_flying_hand_lora pi05_lora 4,5,6,7 4 \
    --dataset-dir /data/lerobot/task_a --dataset-dir /data/lerobot/task_b
EOF
}

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$project_root"

config=pi05_flying_hand_lora
run_prefix=""
gpu_ids=0
fsdp_devices=""
dataset_dirs=()
assets_dir=""
batch_size=""
num_workers=""

if [[ $# -gt 0 && "$1" != --* ]]; then config="$1"; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then run_prefix="$1"; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then gpu_ids="$1"; shift; fi
if [[ $# -gt 0 && "$1" != --* ]]; then fsdp_devices="$1"; shift; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-dir)
      [[ $# -ge 2 ]] || { echo "--dataset-dir requires a value" >&2; exit 2; }
      dataset_dirs+=("$2"); shift 2 ;;
    --assets-dir)
      [[ $# -ge 2 ]] || { echo "--assets-dir requires a value" >&2; exit 2; }
      assets_dir="$2"; shift 2 ;;
    --batch-size)
      [[ $# -ge 2 ]] || { echo "--batch-size requires a value" >&2; exit 2; }
      batch_size="$2"; shift 2 ;;
    --num-workers)
      [[ $# -ge 2 ]] || { echo "--num-workers requires a value" >&2; exit 2; }
      num_workers="$2"; shift 2 ;;
    --run-name)
      [[ $# -ge 2 ]] || { echo "--run-name requires a value" >&2; exit 2; }
      run_prefix="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$gpu_ids" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_IDS must be comma-separated integers" >&2
  exit 2
fi
if [[ -n "$run_prefix" && ! "$run_prefix" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Run prefix may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
IFS=',' read -r -a gpu_list <<< "$gpu_ids"
fsdp_devices=${fsdp_devices:-${#gpu_list[@]}}
if ! [[ "$fsdp_devices" =~ ^[1-9][0-9]*$ ]]; then
  echo "FSDP_DEVICES must be a positive integer" >&2
  exit 2
fi

train_args=()
for dataset_dir in "${dataset_dirs[@]}"; do
  [[ -d "$dataset_dir" ]] || { echo "LeRobot dataset directory does not exist: $dataset_dir" >&2; exit 1; }
done
if (( ${#dataset_dirs[@]} > 0 )); then
  train_args+=(--data.dataset-dirs "${dataset_dirs[@]}")
fi
if [[ -n "$assets_dir" ]]; then
  [[ -d "$assets_dir" ]] || { echo "Normalization assets directory does not exist: $assets_dir" >&2; exit 1; }
  train_args+=(--data.assets.assets-dir "$assets_dir")
fi
if [[ -n "$batch_size" ]]; then
  [[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || { echo "--batch-size must be a positive integer" >&2; exit 2; }
  train_args+=(--batch-size "$batch_size")
fi
if [[ -n "$num_workers" ]]; then
  [[ "$num_workers" =~ ^[0-9]+$ ]] || { echo "--num-workers must be a non-negative integer" >&2; exit 2; }
  train_args+=(--num-workers "$num_workers")
fi

run_prefix=${run_prefix:-$config}
run_id="${run_prefix}_$(date -u +%Y%m%dT%H%M%S_%6NZ)"
export CUDA_VISIBLE_DEVICES="$gpu_ids"
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-1.0}
echo "[launch] config=${config} run_id=${run_id} gpus=${gpu_ids} fsdp_devices=${fsdp_devices} datasets=${#dataset_dirs[@]} assets_dir=${assets_dir:-<config-default>}"
uv run scripts/train.py "$config" \
  --exp-name="$run_id" \
  --checkpoint-base-dir="$project_root/checkpoints" \
  --fsdp-devices="$fsdp_devices" \
  "${train_args[@]}"
