#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  eval.sh TASK TASK_CONFIG CKPT_SETTING EXPERT_DATA_NUM SEED GPU_ID [options]

Options:
  --checkpoint-dir DIR       ACT directory containing policy_last.ckpt and dataset_stats.pkl
  --episodes N               Number of evaluation episodes (default: 100)
  --output-dir DIR           Evaluation result directory
  --instruction-type TYPE    seen or unseen (default: seen)
  --eval-video-log BOOL      true saves replay histories and renders videos; false disables it
  --temporal-agg BOOL        Enable ACT temporal aggregation (default: true)

The six positional arguments remain compatible with the existing ACT launcher.
For the all-task Flying-Hand model, pass --checkpoint-dir explicitly.
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
    usage
    exit 0
fi
if [[ $# -lt 6 ]]; then
    usage >&2
    exit 2
fi

policy_name="ACT"
task_name="$1"
task_config="$2"
ckpt_setting="$3"
expert_data_num="$4"
seed="$5"
gpu_id="$6"

episodes=100
output_dir=""
checkpoint_dir=""
instruction_type="seen"
eval_video_log=""
temporal_agg="true"

shift 6
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir|--ckpt-dir)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            checkpoint_dir="$2"; shift 2 ;;
        --episodes)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            episodes="$2"; shift 2 ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            output_dir="$2"; shift 2 ;;
        --instruction-type)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            instruction_type="$2"; shift 2 ;;
        --eval-video-log)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            eval_video_log="$2"; shift 2 ;;
        --temporal-agg)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            temporal_agg="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2 ;;
    esac
done

if [[ ! "${episodes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--episodes must be a positive integer, got: ${episodes}" >&2
    exit 2
fi
if [[ "${instruction_type}" != "seen" && "${instruction_type}" != "unseen" ]]; then
    echo "--instruction-type must be seen or unseen, got: ${instruction_type}" >&2
    exit 2
fi
for bool_value in "${temporal_agg}" ${eval_video_log:+"${eval_video_log}"}; do
    if [[ "${bool_value}" != "true" && "${bool_value}" != "false" ]]; then
        echo "Boolean options must be true or false, got: ${bool_value}" >&2
        exit 2
    fi
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
    python_bin="python"
fi

if [[ -z "${checkpoint_dir}" ]]; then
    checkpoint_dir="${repo_root}/policy/ACT/act_ckpt/act-${task_name}/${ckpt_setting}-${expert_data_num}"
elif [[ "${checkpoint_dir}" != /* ]]; then
    checkpoint_dir="${repo_root}/${checkpoint_dir}"
fi
if [[ ! -f "${checkpoint_dir}/policy_last.ckpt" ]]; then
    echo "ACT checkpoint not found: ${checkpoint_dir}/policy_last.ckpt" >&2
    exit 1
fi
if [[ ! -f "${checkpoint_dir}/dataset_stats.pkl" ]]; then
    echo "ACT normalization stats not found: ${checkpoint_dir}/dataset_stats.pkl" >&2
    exit 1
fi

eval_args=(--config "policy/${policy_name}/deploy_policy.yml")
if [[ -n "${output_dir}" ]]; then
    eval_args+=(--eval_output_dir "${output_dir}")
fi
overrides=(
    --task_name "${task_name}"
    --task_config "${task_config}"
    --ckpt_setting "${ckpt_setting}"
    --ckpt_dir "${checkpoint_dir}"
    --seed "${seed}"
    --instruction_type "${instruction_type}"
    --temporal_agg "${temporal_agg}"
    --eval_num_episodes "${episodes}"
)
if [[ -n "${eval_video_log}" ]]; then
    overrides+=(--eval_video_log "${eval_video_log}")
fi

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo "ACT checkpoint: ${checkpoint_dir}"
cd "${repo_root}"
CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONWARNINGS=ignore::UserWarning \
    "${python_bin}" script/eval_policy.py "${eval_args[@]}" \
    --overrides "${overrides[@]}"
