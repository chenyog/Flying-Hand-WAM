#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${POLICY_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-${POLICY_DIR}/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/lerobot/flying_hand_direct_all_tasks}"
RUN_ID="${RUN_ID:-flying_hand_direct_clean_$(date -u +%Y%m%dT%H%M%S)_$$}"
RUN_ROOT="${RUN_ROOT:-${POLICY_DIR}/checkpoints}"
export OPENVLA_ROBOT_PLATFORM=FLYING_HAND
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-flying-hand-openvla-oft}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${POLICY_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${1:-}" == "--help" ]]; then
    printf '%s\n' 'Usage: VLA_PATH=/absolute/local/openvla-7b bash train_flying_hand.sh [finetune overrides]' \
      '       bash train_flying_hand.sh --dry-run' \
      'Environment: PYTHON DATA_ROOT RUN_ROOT RUN_ID NPROC_PER_NODE(default 8)' \
      '             CUDA_VISIBLE_DEVICES BATCH_SIZE(default 5/device, global 40; throughput selected) MAX_STEPS(default 80000)' \
      '             SAVE_FREQ(default 1000 steps; weights + latest resume state)' \
      '             --resume_state /absolute/path/state/latest.pt restores full training state' \
      '             WANDB_MODE(default online) WANDB_PROJECT WANDB_ENTITY'
    exit 0
fi

cd "${POLICY_DIR}"
if [[ "${1:-}" == "--dry-run" ]]; then
    exec "${PYTHON}" scripts/check_flying_hand_data.py --data-root "${DATA_ROOT}"
fi
: "${VLA_PATH:?Set VLA_PATH to a local pretrained OpenVLA checkpoint directory}"
[[ -d "${VLA_PATH}" ]] || { printf 'Missing local VLA_PATH: %s\n' "${VLA_PATH}" >&2; exit 1; }
[[ -f "${VLA_PATH}/config.json" ]] || { printf '%s\n' 'VLA_PATH must contain config.json' >&2; exit 1; }
[[ ! -e "${RUN_ROOT}/${RUN_ID}" ]] || { printf 'Run already exists: %s\n' "${RUN_ROOT}/${RUN_ID}" >&2; exit 1; }
for required in prismatic/models/action_heads.py prismatic/models/projectors.py \
    prismatic/models/film_vit_wrapper.py prismatic/models/backbones/llm/prompting/__init__.py; do
    [[ -f "${required}" ]] || {
        printf 'Incomplete OpenVLA source: missing %s. See README.md environment prerequisites.\n' "${required}" >&2
        exit 1
    }
done
"${PYTHON}" scripts/check_flying_hand_data.py --data-root "${DATA_ROOT}"
extra_args=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    extra_args+=(--wandb_entity "${WANDB_ENTITY}")
fi
exec "${PYTHON}" -m torch.distributed.run --standalone --nnodes 1 \
    --nproc_per_node "${NPROC_PER_NODE:-8}" vla-scripts/finetune.py \
    --vla_path "${VLA_PATH}" \
    --dataset_backend flying_hand_lerobot \
    --dataset_name flying_hand_direct_all_tasks --data_root_dir "${DATA_ROOT}" \
    --run_root_dir "${RUN_ROOT}" --run_id_override "${RUN_ID}" \
    --num_images_in_input 2 --use_proprio True \
    --use_l1_regression True --use_diffusion False --use_film False \
    --use_lora True --lora_rank 32 --merge_lora_during_training False \
    --batch_size "${BATCH_SIZE:-5}" --grad_accumulation_steps 1 \
    --learning_rate 0.0005 --max_steps "${MAX_STEPS:-80000}" \
    --lr_warmup_steps "${WARMUP_STEPS:-4000}" \
    --num_steps_before_decay "${DECAY_STEPS:-64000}" --save_freq "${SAVE_FREQ:-1000}" \
    --image_aug False --use_val_set False --wandb_project "${WANDB_PROJECT}" \
    "${extra_args[@]}" "$@"
