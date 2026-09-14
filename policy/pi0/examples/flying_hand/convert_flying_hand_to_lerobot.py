"""Convert RoboTwin Flying-Hand HDF5 episodes to a local LeRobot dataset.

Example:
  uv run examples/flying_hand/convert_flying_hand_to_lerobot.py \
    --input-root /data/Flying-Hand-WAM/data/flying_hand \
    --setting flying_hand_clean --repo-id flying_hand_all_tasks --all-tasks
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SHARED_UTILS = PROJECT_ROOT / "envs" / "utils"
if str(SHARED_UTILS) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS))

from rgb_codec import RGB_JPEG_CODEC  # noqa: E402
from rgb_codec import decode_hdf5_rgb  # noqa: E402
from rgb_codec import require_current_rgb_jpeg_codec  # noqa: E402


@dataclass(frozen=True)
class EpisodeSpec:
    task_name: str
    episode_path: Path
    instruction: str


def read_instruction(path: Path, instruction_type: str, fallback: str | None = None) -> str:
    """Read the requested instruction without silently changing its domain."""
    if not path.is_file():
        if fallback is not None:
            return fallback
        raise FileNotFoundError(f"Missing instruction file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get(instruction_type)
    if isinstance(values, str):
        values = [values]
    if not values:
        if fallback is not None:
            return fallback
        raise ValueError(f"Instruction file {path} has no non-empty {instruction_type!r} entry")
    instruction = str(values[0]).strip()
    if not instruction:
        if fallback is not None:
            return fallback
        raise ValueError(f"Instruction file {path} has an empty {instruction_type!r} entry")
    return instruction


def discover_episodes(
    input_root: Path,
    setting: str,
    instruction_type: str,
    *,
    all_tasks: bool,
    expected_episodes_per_task: int | None,
    allow_missing_instructions: bool,
) -> tuple[list[EpisodeSpec], dict[str, int]]:
    """Validate all source episodes before creating a destination dataset."""
    if all_tasks:
        task_roots = sorted(path for path in input_root.iterdir() if (path / setting).is_dir())
    else:
        task_roots = [input_root]
    if not task_roots:
        raise FileNotFoundError(f"No Flying-Hand task directories found under {input_root}")

    specs: list[EpisodeSpec] = []
    episode_counts: dict[str, int] = {}
    for task_root in task_roots:
        setting_root = task_root / setting if (task_root / setting).is_dir() else task_root
        data_root = setting_root / "data"
        if not data_root.is_dir():
            raise FileNotFoundError(f"Missing HDF5 data directory: {data_root}")
        episode_paths = sorted(
            data_root.glob("episode*.hdf5"),
            key=lambda path: int(path.stem.removeprefix("episode")),
        )
        if not episode_paths:
            raise FileNotFoundError(f"No HDF5 episodes found in {data_root}")
        episode_indices = [int(path.stem.removeprefix("episode")) for path in episode_paths]
        if episode_indices != list(range(len(episode_paths))):
            raise ValueError(
                f"Episode indices in {data_root} must be contiguous from 0; got "
                f"{episode_indices[:3]}...{episode_indices[-3:]}"
            )
        if expected_episodes_per_task is not None and len(episode_paths) != expected_episodes_per_task:
            raise ValueError(
                f"Expected {expected_episodes_per_task} episodes for {task_root.name}, " f"found {len(episode_paths)}"
            )

        fallback = task_root.name.replace("_", " ") if allow_missing_instructions else None
        for episode_path in episode_paths:
            with h5py.File(episode_path, "r") as root:
                require_current_rgb_jpeg_codec(root)
            instruction = read_instruction(
                setting_root / "instructions" / f"{episode_path.stem}.json",
                instruction_type,
                fallback=fallback,
            )
            specs.append(EpisodeSpec(task_root.name, episode_path, instruction))
        episode_counts[task_root.name] = len(episode_paths)
    return specs, episode_counts


def create_dataset(
    repo_id: str,
    fps: float,
    *,
    overwrite: bool,
    image_writer_processes: int,
    image_writer_threads: int,
) -> LeRobotDataset:
    output = Path(HF_LEROBOT_HOME) / repo_id
    if overwrite and output.exists():
        shutil.rmtree(output)
    features = {
        "observation.state": {"dtype": "float32", "shape": (5,), "names": [["x", "y", "z", "yaw", "grasp"]]},
        "action": {"dtype": "float32", "shape": (5,), "names": [["x", "y", "z", "yaw", "grasp"]]},
        "observation.images.head_camera": {
            "dtype": "image",
            "shape": (3, 360, 480),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist_camera": {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="flying_hand",
        features=features,
        use_videos=False,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )


def add_episode(dataset: LeRobotDataset, episode_path: Path, instruction: str) -> int:
    with h5py.File(episode_path, "r") as root:
        require_current_rgb_jpeg_codec(root)
        state = np.asarray(root["flying_hand/actual_state"], dtype=np.float32)
        target = np.asarray(root["flying_hand/target_state"], dtype=np.float32)
        head = root["observation/head_camera/rgb"]
        wrist = root["observation/wrist_camera/rgb"]
        if state.shape != target.shape or state.ndim != 2 or state.shape[1] != 5:
            raise ValueError(f"Invalid Flying-Hand state/action shape in {episode_path}: {state.shape}, {target.shape}")
        if len(head) != len(state) or len(wrist) != len(state):
            raise ValueError(f"Image/state length mismatch in {episode_path}")
        if len(state) < 2:
            raise ValueError(f"Episode must have at least two states: {episode_path}")
        # Preserve one LeRobot frame per 20 Hz source record.  The final frame
        # is terminal-padded because target[t+1] does not exist there.
        action = np.concatenate([target[1:], target[-1:]], axis=0)
        for index in range(len(state)):
            dataset.add_frame(
                {
                    "observation.state": state[index],
                    "action": action[index],
                    "observation.images.head_camera": decode_hdf5_rgb(head[index]),
                    "observation.images.wrist_camera": decode_hdf5_rgb(wrist[index]),
                    "task": instruction,
                }
            )
    dataset.save_episode()
    return len(state)


def write_conversion_contract(
    output: Path,
    *,
    input_root: Path,
    setting: str,
    instruction_type: str,
    fps: int,
    episode_counts: dict[str, int],
    num_frames: int,
) -> None:
    contract = {
        "input_root": str(input_root.resolve()),
        "setting": setting,
        "instruction_type": instruction_type,
        "fps": fps,
        "state": "flying_hand/actual_state[t]",
        "action": "flying_hand/target_state[t+1] (terminal frame repeats target[-1])",
        "rgb_output": "canonical_rgb",
        "required_source_rgb_codec": RGB_JPEG_CODEC,
        "episodes_by_task": episode_counts,
        "total_episodes": sum(episode_counts.values()),
        "total_frames": num_frames,
    }
    path = output / "meta" / "flying_hand_conversion.json"
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def convert_episode_specs(
    repo_id: str,
    fps: int,
    episode_specs: list[EpisodeSpec],
    *,
    image_writer_processes: int,
    image_writer_threads: int,
) -> tuple[str, int]:
    """Convert one independent episode group using the official LeRobot writer."""
    dataset = create_dataset(
        repo_id,
        fps,
        overwrite=False,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )
    num_frames = 0
    try:
        for index, spec in enumerate(episode_specs, start=1):
            num_frames += add_episode(dataset, spec.episode_path, spec.instruction)
            if index % 25 == 0 or index == len(episode_specs):
                print(f"Part {repo_id}: {index}/{len(episode_specs)} episodes, {num_frames} frames")
    finally:
        dataset.stop_image_writer()
    return repo_id, num_frames


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def rewrite_part_parquet(
    source: Path,
    destination: Path,
    episode_offset: int,
    frame_offset: int,
    task_index_map: dict[int, int],
) -> None:
    """Reindex one independent part parquet into the combined dataset."""
    table = pq.read_table(source)

    def replace_column(name: str, values: np.ndarray) -> None:
        nonlocal table
        column_index = table.schema.get_field_index(name)
        table = table.set_column(column_index, table.schema.field(column_index), pa.array(values))

    episode_index = table.column("episode_index").to_numpy(zero_copy_only=False).astype(np.int64)
    global_index = table.column("index").to_numpy(zero_copy_only=False).astype(np.int64)
    local_task_index = table.column("task_index").to_numpy(zero_copy_only=False).astype(np.int64)
    replace_column("episode_index", episode_index + episode_offset)
    replace_column("index", global_index + frame_offset)
    replace_column(
        "task_index",
        np.asarray([task_index_map[int(value)] for value in local_task_index], dtype=np.int64),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination)


def merge_part_datasets(
    output: Path,
    part_roots: list[Path],
    *,
    merge_workers: int,
) -> tuple[int, int, int]:
    """Merge task-local LeRobot datasets while preserving embedded RGB bytes."""
    if output.exists():
        raise FileExistsError(f"Combined output already exists: {output}")
    (output / "meta").mkdir(parents=True)

    global_task_to_index: dict[str, int] = {}
    task_rows: list[dict] = []
    episode_rows: list[dict] = []
    episode_stat_rows: list[dict] = []
    parquet_jobs: list[tuple[Path, Path, int, int, dict[int, int]]] = []
    episode_offset = 0
    frame_offset = 0
    base_info: dict | None = None

    for part_root in part_roots:
        info = json.loads((part_root / "meta" / "info.json").read_text(encoding="utf-8"))
        if base_info is None:
            base_info = info
        elif info["features"] != base_info["features"] or info["fps"] != base_info["fps"]:
            raise ValueError(f"Incompatible LeRobot part metadata: {part_root}")

        local_task_map: dict[int, int] = {}
        for row in read_jsonl(part_root / "meta" / "tasks.jsonl"):
            task = str(row["task"])
            if task not in global_task_to_index:
                global_task_to_index[task] = len(global_task_to_index)
                task_rows.append({"task_index": global_task_to_index[task], "task": task})
            local_task_map[int(row["task_index"])] = global_task_to_index[task]

        local_episodes = read_jsonl(part_root / "meta" / "episodes.jsonl")
        local_stats = read_jsonl(part_root / "meta" / "episodes_stats.jsonl")
        if len(local_episodes) != int(info["total_episodes"]) or len(local_stats) != len(local_episodes):
            raise ValueError(f"Incomplete LeRobot part metadata: {part_root}")
        for row in local_episodes:
            row["episode_index"] = int(row["episode_index"]) + episode_offset
            episode_rows.append(row)
        for row in local_stats:
            row["episode_index"] = int(row["episode_index"]) + episode_offset
            episode_stat_rows.append(row)

        for local_episode_index in range(int(info["total_episodes"])):
            global_episode_index = episode_offset + local_episode_index
            source = part_root / info["data_path"].format(
                episode_chunk=local_episode_index // int(info["chunks_size"]),
                episode_index=local_episode_index,
            )
            destination = output / base_info["data_path"].format(
                episode_chunk=global_episode_index // int(base_info["chunks_size"]),
                episode_index=global_episode_index,
            )
            if not source.is_file():
                raise FileNotFoundError(f"Missing part parquet: {source}")
            parquet_jobs.append((source, destination, episode_offset, frame_offset, local_task_map))
        episode_offset += int(info["total_episodes"])
        frame_offset += int(info["total_frames"])

    assert base_info is not None
    with ProcessPoolExecutor(max_workers=merge_workers) as executor:
        list(executor.map(lambda_args_rewrite_part_parquet, parquet_jobs))

    base_info.update(
        {
            "total_episodes": episode_offset,
            "total_frames": frame_offset,
            "total_tasks": len(task_rows),
            "total_videos": 0,
            "total_chunks": (episode_offset + int(base_info["chunks_size"]) - 1) // int(base_info["chunks_size"]),
            "splits": {"train": f"0:{episode_offset}"},
        }
    )
    (output / "meta" / "info.json").write_text(json.dumps(base_info, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output / "meta" / "tasks.jsonl", task_rows)
    write_jsonl(output / "meta" / "episodes.jsonl", episode_rows)
    write_jsonl(output / "meta" / "episodes_stats.jsonl", episode_stat_rows)
    return episode_offset, frame_offset, len(task_rows)


def lambda_args_rewrite_part_parquet(args: tuple[Path, Path, int, int, dict[int, int]]) -> None:
    """Picklable adapter for ProcessPoolExecutor.map."""
    rewrite_part_parquet(*args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="data/flying_hand or one task directory")
    parser.add_argument("--setting", default="flying_hand_clean")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo id under HF_LEROBOT_HOME")
    parser.add_argument("--instruction-type", default="seen", choices=("seen", "unseen"))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--expected-episodes-per-task",
        type=int,
        help="Fail before conversion unless each selected task has exactly this many episodes",
    )
    parser.add_argument(
        "--expected-tasks",
        type=int,
        help="Fail before conversion unless exactly this many task directories are selected",
    )
    parser.add_argument(
        "--allow-missing-instructions",
        action="store_true",
        help="Use the task directory name only when an episode instruction is missing (strict by default)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of LeRobot image-writer processes (default: 8)",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=2,
        help="Image-writer threads in each worker process (default: 2)",
    )
    parser.add_argument(
        "--task-processes",
        type=int,
        default=1,
        help="Convert independent tasks concurrently before merging them (default: 1)",
    )
    parser.add_argument(
        "--merge-workers",
        type=int,
        default=16,
        help="Processes used to reindex part parquets during a parallel-task merge (default: 16)",
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep temporary per-task LeRobot datasets after a successful merge",
    )
    args = parser.parse_args()

    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.threads_per_worker <= 0:
        parser.error("--threads-per-worker must be positive")
    if args.task_processes <= 0:
        parser.error("--task-processes must be positive")
    if args.merge_workers <= 0:
        parser.error("--merge-workers must be positive")
    if args.task_processes > 1 and not args.all_tasks:
        parser.error("--task-processes greater than 1 requires --all-tasks")
    if args.fps <= 0 or not args.fps.is_integer():
        parser.error("--fps must be a positive integer")
    if args.expected_episodes_per_task is not None and args.expected_episodes_per_task <= 0:
        parser.error("--expected-episodes-per-task must be positive")
    if args.expected_tasks is not None and args.expected_tasks <= 0:
        parser.error("--expected-tasks must be positive")

    episode_specs, episode_counts = discover_episodes(
        args.input_root,
        args.setting,
        args.instruction_type,
        all_tasks=args.all_tasks,
        expected_episodes_per_task=args.expected_episodes_per_task,
        allow_missing_instructions=args.allow_missing_instructions,
    )
    if args.expected_tasks is not None and len(episode_counts) != args.expected_tasks:
        raise ValueError(f"Expected {args.expected_tasks} tasks, found {len(episode_counts)}")
    print(
        f"Preflight passed: {len(episode_counts)} tasks, {len(episode_specs)} episodes, "
        f"instruction_type={args.instruction_type}, fps={int(args.fps)}"
    )

    print(
        "LeRobot image writer: "
        f"{args.workers} processes x {args.threads_per_worker} threads"
        + (f" per task, {args.task_processes} concurrent tasks" if args.task_processes > 1 else "")
    )
    output = Path(HF_LEROBOT_HOME) / args.repo_id
    if args.task_processes > 1:
        if args.overwrite and output.exists():
            shutil.rmtree(output)
        elif output.exists():
            raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")

        grouped_specs = {
            task_name: [spec for spec in episode_specs if spec.task_name == task_name] for task_name in episode_counts
        }
        part_repo_ids = [
            f"{args.repo_id}__part_{index:02d}_{task_name}" for index, task_name in enumerate(grouped_specs)
        ]
        part_roots = [Path(HF_LEROBOT_HOME) / part_repo_id for part_repo_id in part_repo_ids]
        for part_root in part_roots:
            if part_root.exists():
                if args.overwrite:
                    shutil.rmtree(part_root)
                else:
                    raise FileExistsError(f"Temporary part already exists: {part_root}; pass --overwrite to replace it")

        futures = []
        with ProcessPoolExecutor(max_workers=min(args.task_processes, len(grouped_specs))) as executor:
            for part_repo_id, specs in zip(part_repo_ids, grouped_specs.values(), strict=True):
                futures.append(
                    executor.submit(
                        convert_episode_specs,
                        part_repo_id,
                        int(args.fps),
                        specs,
                        image_writer_processes=args.workers,
                        image_writer_threads=args.threads_per_worker,
                    )
                )
            part_results = [future.result() for future in futures]
        num_frames = sum(result[1] for result in part_results)
        total_episodes, merged_frames, total_tasks = merge_part_datasets(
            output,
            part_roots,
            merge_workers=args.merge_workers,
        )
        if total_episodes != len(episode_specs) or merged_frames != num_frames:
            raise ValueError(
                f"Merged count mismatch: episodes={total_episodes}/{len(episode_specs)}, "
                f"frames={merged_frames}/{num_frames}"
            )
        print(
            f"Merged {len(part_roots)} task datasets: tasks={total_tasks}, episodes={total_episodes}, frames={num_frames}"
        )
        if not args.keep_parts:
            for part_root in part_roots:
                shutil.rmtree(part_root)
        converted_by_task = episode_counts
    else:
        dataset = create_dataset(
            args.repo_id,
            int(args.fps),
            overwrite=args.overwrite,
            image_writer_processes=args.workers,
            image_writer_threads=args.threads_per_worker,
        )
        converted_by_task = {task_name: 0 for task_name in episode_counts}
        num_frames = 0
        try:
            for index, spec in enumerate(episode_specs, start=1):
                num_frames += add_episode(dataset, spec.episode_path, spec.instruction)
                converted_by_task[spec.task_name] += 1
                if converted_by_task[spec.task_name] == episode_counts[spec.task_name]:
                    print(f"Converted {spec.task_name}: {converted_by_task[spec.task_name]} episodes")
                elif index % 25 == 0:
                    print(f"Progress: {index}/{len(episode_specs)} episodes, {num_frames} frames")
        finally:
            dataset.stop_image_writer()

    write_conversion_contract(
        output,
        input_root=args.input_root,
        setting=args.setting,
        instruction_type=args.instruction_type,
        fps=int(args.fps),
        episode_counts=converted_by_task,
        num_frames=num_frames,
    )
    print(f"Saved LeRobot dataset to {output}: {len(episode_specs)} episodes, {num_frames} frames")


if __name__ == "__main__":
    main()
