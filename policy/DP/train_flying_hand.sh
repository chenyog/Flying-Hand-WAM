#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  Single task, one GPU:
    train_flying_hand.sh --task-config TASK_CONFIG [options]

  All tasks, one or more GPUs:
    train_flying_hand.sh --all-tasks [options]

Options:
  --task-config PATH   Task path under data/flying_hand (mutually exclusive with --all-tasks)
  --all-tasks          Merge the selected number of episodes from every task
  --gpus IDS           GPU ids, for example 0 or 0,1,2,3 (default: 0)
  --episodes N         Episodes per task for --all-tasks, total episodes otherwise (default: 100)
  --seed N             Training seed (default: 0)
  --epochs N           Number of epochs (default: 600)
  --batch-size N       Batch size per GPU (default: 32)
  --force-convert      Rebuild the Zarr dataset even when it already exists
  -h, --help           Show this help

Examples:
  train_flying_hand.sh --all-tasks --gpus 0,1,2,3,4,5,6,7 --episodes 100 --epochs 600 --batch-size 32
  train_flying_hand.sh --task-config blocks_ranking_rgb/flying_hand_clean --gpus 0 --episodes 100

For DDP, batch-size is per GPU and global batch size is batch-size times the GPU count.
EOF
}

ALL_TASKS=0
TASK_CONFIG=""
GPU_IDS="0"
EPISODES="100"
SEED="0"
EPOCHS="600"
BATCH_SIZE="32"
FORCE_CONVERT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all-tasks)
      ALL_TASKS=1
      shift
      ;;
    --task-config)
      [[ $# -ge 2 ]] || { echo "--task-config requires a value" >&2; exit 2; }
      TASK_CONFIG="$2"
      shift 2
      ;;
    --gpus)
      [[ $# -ge 2 ]] || { echo "--gpus requires a value" >&2; exit 2; }
      GPU_IDS="$2"
      shift 2
      ;;
    --episodes)
      [[ $# -ge 2 ]] || { echo "--episodes requires a value" >&2; exit 2; }
      EPISODES="$2"
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || { echo "--seed requires a value" >&2; exit 2; }
      SEED="$2"
      shift 2
      ;;
    --epochs)
      [[ $# -ge 2 ]] || { echo "--epochs requires a value" >&2; exit 2; }
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      [[ $# -ge 2 ]] || { echo "--batch-size requires a value" >&2; exit 2; }
      BATCH_SIZE="$2"
      shift 2
      ;;
    --force-convert)
      FORCE_CONVERT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if (( ALL_TASKS == 1 )) && [[ -n "${TASK_CONFIG}" ]]; then
  echo "--all-tasks and --task-config are mutually exclusive" >&2
  exit 2
fi
if (( ALL_TASKS == 0 )) && [[ -z "${TASK_CONFIG}" ]]; then
  echo "Specify --all-tasks or --task-config" >&2
  usage >&2
  exit 2
fi
if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "--gpus must be a comma-separated list, for example 0,1,2,3" >&2
  exit 2
fi
if ! [[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "episodes, epochs, and batch-size must be positive integers" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
TORCHRUN_BIN="${REPO_ROOT}/.venv/bin/torchrun"
DATA_ROOT="${REPO_ROOT}/data/flying_hand"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  TORCHRUN_BIN="torchrun"
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Flying-Hand data root does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
GPU_COUNT="${#GPU_ARRAY[@]}"

if (( ALL_TASKS == 1 )); then
  ZARR_PATH="${PROJECT_ROOT}/data/flying_hand-all-tasks-${EPISODES}.zarr"
  CONVERT_ARGS=(
    --data-root "${DATA_ROOT}"
    --output "${ZARR_PATH}"
    --num-episodes "${EPISODES}"
    --all-tasks
  )
  RUN_NAME="flying_hand_all_tasks"
else
  TASK_DATA_ROOT="${DATA_ROOT}/${TASK_CONFIG}"
  if [[ ! -d "${TASK_DATA_ROOT}/data" ]]; then
    echo "Dataset directory does not exist: ${TASK_DATA_ROOT}/data" >&2
    exit 1
  fi
  SAFE_CONFIG="${TASK_CONFIG//\//_}"
  ZARR_PATH="${PROJECT_ROOT}/data/flying_hand-${SAFE_CONFIG}-${EPISODES}.zarr"
  CONVERT_ARGS=(
    --data-root "${TASK_DATA_ROOT}"
    --output "${ZARR_PATH}"
    --num-episodes "${EPISODES}"
  )
  RUN_NAME="flying_hand_${SAFE_CONFIG}"
fi

# Check array metadata, rather than only the directory, so an interrupted
# conversion is not mistaken for a complete dataset.
zarr_complete() {
  [[ -f "${ZARR_PATH}/data/head_camera/.zarray" ]] && \
  [[ -f "${ZARR_PATH}/data/wrist_camera/.zarray" ]] && \
  [[ -f "${ZARR_PATH}/data/state/.zarray" ]] && \
  [[ -f "${ZARR_PATH}/data/action/.zarray" ]] && \
  [[ -f "${ZARR_PATH}/meta/episode_ends/.zarray" ]]
}

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if (( FORCE_CONVERT == 1 )) || ! zarr_complete; then
  "${PYTHON_BIN}" process_flying_hand_data.py "${CONVERT_ARGS[@]}"
else
  echo "Reusing existing Zarr dataset: ${ZARR_PATH}"
  echo "Use --force-convert to rebuild it from HDF5."
fi

RUN_ID="${RUN_NAME}_$(date -u +%Y%m%d_%H%M%S)"
TRAIN_ARGS=(
  --config-name=robot_dp_5.yaml
  task.name=flying_hand
  task.dataset.zarr_path="${ZARR_PATH}"
  dataloader.batch_size="${BATCH_SIZE}"
  val_dataloader.batch_size="${BATCH_SIZE}"
  training.seed="${SEED}"
  training.num_epochs="${EPOCHS}"
  training.resume=False
  exp_name="${RUN_NAME}_dp"
  expert_data_num="${EPISODES}"
  hydra.run.dir="data/outputs/${RUN_ID}"
  hydra.output_subdir=null
)

if (( ALL_TASKS == 0 )); then
  TRAIN_ARGS+=(setting="${TASK_CONFIG}")
else
  TRAIN_ARGS+=(setting=all_tasks)
fi

if (( GPU_COUNT == 1 )); then
  "${PYTHON_BIN}" train.py "${TRAIN_ARGS[@]}" training.device=cuda:0
else
  "${TORCHRUN_BIN}" \
    --standalone \
    --nproc_per_node="${GPU_COUNT}" \
    train.py "${TRAIN_ARGS[@]}"
fi
