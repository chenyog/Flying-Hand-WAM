#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$project_root"

config=${1:-pi05_flying_hand_lora}
experiment=${2:-flying_hand}
gpu_ids=${3:-0}
global_batch_size=${4:-32}

IFS=',' read -r -a gpu_id_list <<< "$gpu_ids"
gpu_count=${#gpu_id_list[@]}
if [[ "$gpu_count" -lt 1 || -z "${gpu_id_list[0]}" ]]; then
  printf 'GPU list must contain at least one device (got: %s)\n' "$gpu_ids" >&2
  exit 2
fi
if [[ "$global_batch_size" -lt 1 ]]; then
  printf 'Global batch size must be positive (got: %s)\n' "$global_batch_size" >&2
  exit 2
fi
if (( global_batch_size % gpu_count != 0 )); then
  printf 'Global batch size (%s) must be divisible by visible GPU count (%s)\n' \
    "$global_batch_size" "$gpu_count" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$gpu_ids"
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
uv run scripts/train.py "$config" \
  --exp-name="$experiment" \
  --checkpoint-base-dir="$project_root/checkpoints" \
  --batch-size="$global_batch_size" \
  --fsdp-devices="$gpu_count" \
  --overwrite
