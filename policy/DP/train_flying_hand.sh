#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  train_flying_hand.sh [options]

Train one DP model on all selected Flying-Hand datasets. The converter uses
all available episodes from every selected directory.

Options:
  --all-tasks          Explicitly select the all-task mode (default behavior)
  --dataset-dir PATH   Raw dataset dir containing data/episode*.hdf5; repeat for multiple datasets
  --run-name NAME      Optional run-name prefix; a UTC timestamp is always appended
  --gpus IDS           GPU ids, for example 0 or 0,1,2,3 (default: 0)
  --seed N             Training seed (default: 0)
  --epochs N           Full-dataloader epochs (default: 8000, official image recipe)
  --batch-size N       Batch size per GPU (default: 64 divided by GPU count)
  --force-convert      Rebuild the Zarr dataset even when it already exists
  -h, --help           Show this help

Examples:
  train_flying_hand.sh --gpus 0,1,2,3
  train_flying_hand.sh --dataset-dir /data/set_a --dataset-dir /data/set_b --gpus 0,1

The official Diffusion Policy image recipe uses global batch 64. An explicit
--batch-size is still per GPU, so global batch is batch-size times GPU count.
EOF
}

GPU_IDS="0"
SEED="0"
EPOCHS="8000"
BATCH_SIZE=""
FORCE_CONVERT=0
DATASET_DIRS=()
RUN_NAME_OVERRIDE=""
ALL_TASKS_EXPLICIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all-tasks)
      ALL_TASKS_EXPLICIT=1
      shift
      ;;
    --dataset-dir)
      [[ $# -ge 2 ]] || { echo "--dataset-dir requires a value" >&2; exit 2; }
      DATASET_DIRS+=("$2")
      shift 2
      ;;
    --run-name)
      [[ $# -ge 2 ]] || { echo "--run-name requires a value" >&2; exit 2; }
      RUN_NAME_OVERRIDE="$2"
      shift 2
      ;;
    --gpus)
      [[ $# -ge 2 ]] || { echo "--gpus requires a value" >&2; exit 2; }
      GPU_IDS="$2"
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

if (( ALL_TASKS_EXPLICIT == 1 && ${#DATASET_DIRS[@]} > 0 )); then
  echo "--all-tasks and --dataset-dir cannot be combined" >&2
  exit 2
fi
if [[ -n "${RUN_NAME_OVERRIDE}" ]] && [[ ! "${RUN_NAME_OVERRIDE}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--run-name may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "--gpus must be a comma-separated list, for example 0,1,2,3" >&2
  exit 2
fi
if ! [[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || \
   { [[ -n "${BATCH_SIZE}" ]] && ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; }; then
  echo "epochs and batch-size must be positive integers" >&2
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
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
GPU_COUNT="${#GPU_ARRAY[@]}"
REFERENCE_GLOBAL_BATCH_SIZE=64
if [[ -z "${BATCH_SIZE}" ]]; then
  if (( REFERENCE_GLOBAL_BATCH_SIZE % GPU_COUNT != 0 )); then
    echo "Cannot divide official global batch 64 across ${GPU_COUNT} GPUs; pass --batch-size explicitly" >&2
    exit 2
  fi
  BATCH_SIZE=$((REFERENCE_GLOBAL_BATCH_SIZE / GPU_COUNT))
fi
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * GPU_COUNT))

if (( ${#DATASET_DIRS[@]} > 0 )); then
  DATASET_FINGERPRINT="$("${PYTHON_BIN}" - "${DATASET_DIRS[@]}" <<'PY'
import hashlib
import pathlib
import sys

resolved = [str(pathlib.Path(item).expanduser().resolve()) for item in sys.argv[1:]]
print(hashlib.sha256("\n".join(resolved).encode()).hexdigest()[:12])
PY
)"
  ZARR_PATH="${PROJECT_ROOT}/data/flying_hand-multi-${DATASET_FINGERPRINT}.zarr"
  CONVERT_ARGS=(--output "${ZARR_PATH}")
  for dataset_dir in "${DATASET_DIRS[@]}"; do
    if [[ ! -d "${dataset_dir}" ]]; then
      echo "Dataset directory does not exist: ${dataset_dir}" >&2
      exit 1
    fi
    ABS_DATASET_DIR="$(cd "${dataset_dir}" && pwd)"
    CONVERT_ARGS+=(--dataset-dir "${ABS_DATASET_DIR}")
  done
  RUN_NAME="flying_hand_multi_${#DATASET_DIRS[@]}"
else
  if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Flying-Hand data root does not exist: ${DATA_ROOT}" >&2
    exit 1
  fi
  ZARR_PATH="${PROJECT_ROOT}/data/flying_hand-all-tasks.zarr"
  CONVERT_ARGS=(
    --data-root "${DATA_ROOT}"
    --output "${ZARR_PATH}"
    --all-tasks
  )
  RUN_NAME="flying_hand_all_tasks"
fi

# Check array metadata, rather than only the directory, so an interrupted
# conversion is not mistaken for a complete dataset.
zarr_complete() {
  [[ -f "${ZARR_PATH}/data/head_camera/.zarray" ]] && \
  [[ -f "${ZARR_PATH}/data/wrist_camera/.zarray" ]] && \
  [[ -f "${ZARR_PATH}/data/state/.zarray" ]] && \
  [[ -f "${ZARR_PATH}/data/action/.zarray" ]] && \
  [[ -f "${ZARR_PATH}/meta/episode_ends/.zarray" ]] && \
  [[ -f "${ZARR_PATH}/meta/episode_task_ids/.zarray" ]]
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

RUN_PREFIX="${RUN_NAME_OVERRIDE:-${RUN_NAME}}"
RUN_ID="${RUN_PREFIX}_$(date -u +%Y%m%dT%H%M%S_%6NZ)"
TRAIN_ARGS=(
  --config-name=robot_dp_5.yaml
  task.name=flying_hand
  task.dataset.zarr_path="${ZARR_PATH}"
  dataloader.batch_size="${BATCH_SIZE}"
  val_dataloader.batch_size="${BATCH_SIZE}"
  training.seed="${SEED}"
  training.num_epochs="${EPOCHS}"
  training.resume=False
  exp_name="${RUN_ID}"
  hydra.run.dir="data/outputs/${RUN_ID}"
  hydra.output_subdir=null
)

echo "[launch] run_id=${RUN_ID} output_dir=${PROJECT_ROOT}/data/outputs/${RUN_ID} datasets=${#DATASET_DIRS[@]} per_gpu_batch=${BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE}"

if (( GPU_COUNT == 1 )); then
  "${PYTHON_BIN}" train.py "${TRAIN_ARGS[@]}" training.device=cuda:0
else
  "${TORCHRUN_BIN}" \
    --standalone \
    --nproc_per_node="${GPU_COUNT}" \
    train.py "${TRAIN_ARGS[@]}"
fi
