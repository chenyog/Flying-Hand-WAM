#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=pi0
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
episodes=100
checkpoint_id=50000
output_dir=""
run_id="${EVAL_RUN_ID:-}"
shift 6
while [[ $# -gt 0 ]]; do
    case "$1" in
        --episodes) episodes="$2"; shift 2 ;;
        --checkpoint-id) checkpoint_id="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --run-id) run_id="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if ! [[ "$episodes" =~ ^[1-9][0-9]*$ && "$checkpoint_id" =~ ^[1-9][0-9]*$ ]]; then
    echo "episodes and checkpoint-id must be positive integers" >&2
    exit 2
fi
if [[ -n "$run_id" && ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "run-id may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"

if [[ -z "$output_dir" ]]; then
    setting_slug=${task_config#flying_hand_}
    if [[ -z "$run_id" ]]; then
        run_id="${policy_name}_${setting_slug}_step${checkpoint_id}_${episodes}eps_seeded_$(date -u +%Y%m%dT%H%M%S_%6NZ)"
    fi
    output_dir="${project_root}/eval_results/${run_id}/${task_name}"
fi

eval_args=(--config policy/$policy_name/deploy_policy.yml)
eval_args+=(--eval_output_dir "${output_dir}")
overrides=(
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --checkpoint_id ${checkpoint_id} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --eval_num_episodes ${episodes} \
    --eval_video_log False
)
PYTHONWARNINGS=ignore::UserWarning \
/data/Flying-Hand-WAM/policy/pi0/.venv/bin/python script/eval_policy.py "${eval_args[@]}" "${overrides[@]}"
