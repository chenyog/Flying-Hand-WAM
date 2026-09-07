#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$project_root"

config=${1:-pi0_base_flying_hand_lora}
experiment=${2:-flying_hand}
gpu_ids=${3:-0}
IFS=',' read -r -a gpu_list <<< "$gpu_ids"
fsdp_devices=${4:-${#gpu_list[@]}}

export CUDA_VISIBLE_DEVICES="$gpu_ids"
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
uv run scripts/train.py "$config" --exp-name="$experiment" --checkpoint-base-dir="$project_root/checkpoints" --overwrite --fsdp-devices="$fsdp_devices"
