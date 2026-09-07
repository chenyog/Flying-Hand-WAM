#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
episodes=100
enable_dynamics=""
output_dir=""
video_max_seconds=30
shift 6
while [[ $# -gt 0 ]]; do
    case "$1" in
        --episodes) episodes="$2"; shift 2 ;;
        --enable-dynamics) enable_dynamics="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --video-max-seconds) video_max_seconds="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# source .venv/bin/activate
cd ../.. # move to root

eval_args=(--config policy/$policy_name/deploy_policy.yml)
if [[ -n "${output_dir}" ]]; then eval_args+=(--eval_output_dir "${output_dir}"); fi
overrides=(
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --eval_num_episodes ${episodes} \
    --eval_video_max_seconds ${video_max_seconds}
)
if [[ -n "${enable_dynamics}" ]]; then overrides+=(--enable_dynamics "${enable_dynamics}"); fi
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py "${eval_args[@]}" "${overrides[@]}"
