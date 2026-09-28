#!/bin/bash

policy_name=openvla-oft
task_name=${1}
task_config=${2}
checkpoint_path=${3}
seed=${4}
gpu_id=${5}
unnorm_key=${6}
POLICY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${POLICY_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${POLICY_DIR}/.venv/bin/python}"

export CUDA_VISIBLE_DEVICES=${gpu_id}
export OPENVLA_ROBOT_PLATFORM=FLYING_HAND
export PYTHONPATH="${POLICY_DIR}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd "${REPO_ROOT}"

PYTHONWARNINGS=ignore::UserWarning \
"${PYTHON}" script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --checkpoint_path ${checkpoint_path} \
    --ckpt_setting ${checkpoint_path} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --unnorm_key ${unnorm_key}

# example usage 
# bash eval.sh move_can_pot demo_randomized ckpt_path 0 5 aloha_move_can_pot_builder
