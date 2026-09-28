#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd -- "${REPO_ROOT}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/scripts/configs/vlajepa_flying_hand_ft.yaml}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/lerobot/flying_hand_direct_all_tasks}"
VLM_PATH="${VLM_PATH:-${REPO_ROOT}/checkpoints/Qwen3-VL-2B-Instruct}"
VJEPA_PATH="${VJEPA_PATH:-${REPO_ROOT}/checkpoints/vjepa2-vitl-fpc64-256}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
RESUME_FROM="${RESUME_FROM:-}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs}"
RUN_ID="${RUN_ID:-vlajepa_flying_hand_$(date -u +%Y%m%dT%H%M%S)_$$}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-11}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-100}"
MAX_STEPS="${MAX_STEPS:-null}"
WARMUP_STEPS="${WARMUP_STEPS:-273}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-flying-hand-vla-jepa}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
cd "${REPO_ROOT}"

if [[ "${1:-}" == "--help" ]]; then
  printf '%s\n' \
    'Usage: VLM_PATH=/local/Qwen3-VL VJEPA_PATH=/local/V-JEPA2 bash scripts/train_flying_hand.sh' \
    '       bash scripts/train_flying_hand.sh --dry-run' \
    'Environment: PYTHON_BIN DATA_ROOT CONFIG_FILE RUN_ROOT RUN_ID NUM_GPUS(default 8)' \
    '             BATCH_SIZE(default 11/device, global 88) NUM_WORKERS(default 4)' \
    '             EPOCHS(default 100) MAX_STEPS(optional override, total target) WARMUP_STEPS(default 273)' \
    '             RESUME_FROM(full state directory; state follows existing save_interval)' \
    '             WANDB_MODE(default online) WANDB_PROJECT WANDB_ENTITY'
  exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_flying_hand.py" \
    --config "${CONFIG_FILE}" --data-root "${DATA_ROOT}" "$@"
fi

for path in "${DATA_ROOT}/meta/info.json" "${VLM_PATH}" "${VJEPA_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'Missing local training resource: %s\nSet DATA_ROOT, VLM_PATH and VJEPA_PATH before launch.\n' "${path}" >&2
    exit 1
  fi
done
if [[ -z "${RESUME_FROM}" && -n "${PRETRAINED_CHECKPOINT}" && ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
  printf 'Missing pretrained checkpoint: %s\n' "${PRETRAINED_CHECKPOINT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/${RUN_ID}/logs"
printf 'VLA-JEPA Flying-Hand run: %s\nW&B: %s\n' "${RUN_ROOT}/${RUN_ID}" "${WANDB_MODE}"
checkpoint_args=()
if [[ -n "${RESUME_FROM}" ]]; then
  if [[ ! -f "${RESUME_FROM}/trainer_state.json" ]]; then
    printf 'Missing complete resume state: %s\n' "${RESUME_FROM}" >&2
    exit 1
  fi
  checkpoint_args=(--trainer.resume_from_checkpoint "${RESUME_FROM}")
elif [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
  checkpoint_args=(--trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
fi
"${PYTHON_BIN}" -m accelerate.commands.launch \
  --config_file "${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero2.yaml" \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MASTER_PORT:-29500}" \
  "${REPO_ROOT}/starVLA/training/train_starvla.py" \
  --config_yaml "${CONFIG_FILE}" \
  --run_root_dir "${RUN_ROOT}" --run_id "${RUN_ID}" \
  --framework.qwenvl.base_vlm "${VLM_PATH}" \
  --framework.vj2_model.base_encoder "${VJEPA_PATH}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}" \
  --trainer.epochs "${EPOCHS}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.num_warmup_steps "${WARMUP_STEPS}" \
  "${checkpoint_args[@]}" "$@" 2>&1 | tee "${RUN_ROOT}/${RUN_ID}/logs/train.log"
