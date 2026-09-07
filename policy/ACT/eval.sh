#!/bin/bash

# == keep unchanged ==
policy_name=ACT
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
DEBUG=False

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

cd ../..

eval_args=(
    --config policy/$policy_name/deploy_policy.yml
    --overrides
    --task_name ${task_name}
    --task_config ${task_config}
    --ckpt_setting ${ckpt_setting}
    --ckpt_dir policy/ACT/act_ckpt/act-${task_name}/${ckpt_setting}-${expert_data_num}
    --seed ${seed}
    --temporal_agg true
    --eval_num_episodes ${episodes}
    --eval_video_max_seconds ${video_max_seconds}
)
if [[ -n "${enable_dynamics}" ]]; then
    eval_args+=(--enable_dynamics "${enable_dynamics}")
fi
if [[ -n "${output_dir}" ]]; then
    eval_args+=(--eval_output_dir "${output_dir}")
fi

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py "${eval_args[@]}"
