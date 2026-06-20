#!/usr/bin/env python3
"""Convert Flying-Hand HDF5 data to the local LeRobot format used by FastWAM."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CHUNK_SIZE = 1000
CAMERAS = ("head_camera", "wrist_camera")
STATE_NAMES = ["x", "y", "z", "yaw", "grasp"]


def default_robotwin_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_fastwam_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Flying-Hand HDF5 episodes to LeRobot format.")
    parser.add_argument("--task-id", default=None, help="Task id, e.g. move_bottle.")
    parser.add_argument("--all-tasks", action="store_true", help="Convert all task directories under data/flying_hand.")
    parser.add_argument("--setting", default="flying_hand_clean", help="Dataset setting directory.")
    parser.add_argument("--instruction-type", choices=["seen", "unseen"], default="seen")
    parser.add_argument("--robotwin-root", type=Path, default=default_robotwin_root())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_fastwam_root() / "data" / "flying_hand_lerobot",
        help="Output root. Each task is written under output-root/task_id.",
    )
    parser.add_argument("--copy-videos", action="store_true", help="Copy videos instead of symlinking them.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output task directory first.")
    parser.add_argument(
        "--stats-action-size",
        type=int,
        default=32,
        help="Action horizon used for normalization stats. This should match num_frames - 1.",
    )
    parser.add_argument(
        "--stats-val-set-proportion",
        type=float,
        default=0.0,
        help="Validation split proportion used by FastWAM when computing train normalization stats.",
    )
    parser.add_argument(
        "--stats-seed",
        type=int,
        default=42,
        help="Episode split seed used by FastWAM when computing train normalization stats.",
    )
    return parser.parse_args()


def sorted_episode_paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("episode*.hdf5"), key=lambda p: int(p.stem.removeprefix("episode")))
    if not paths:
        raise FileNotFoundError(f"No episode*.hdf5 files found in {data_dir}")
    return paths


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_task(task_id: str, episode_index: int, scene_info: dict, instruction_payload: dict, instruction_type: str) -> str:
    templates = instruction_payload[instruction_type]
    task = str(templates[episode_index % len(templates)])
    info = scene_info[f"episode_{episode_index}"]["info"]
    for key, value in info.items():
        value = str(value)
        task = task.replace(str(key), value)
        stripped = str(key).strip("{}")
        task = task.replace("{" + stripped + "}", value)
    return task


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    finally:
        cap.release()
    if width <= 0 or height <= 0 or fps <= 0 or frames <= 0:
        raise RuntimeError(f"Failed to read video metadata: {path}")
    return {"width": width, "height": height, "fps": fps, "frames": frames}


def feature_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def sliding_window_with_replication(values: np.ndarray, window_size: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"`values` must be 2D [T,D], got shape {values.shape}")
    if window_size <= 0:
        raise ValueError(f"`window_size` must be positive, got {window_size}")
    length = values.shape[0]
    indices = np.arange(length)[:, None] + np.arange(window_size)[None, :]
    indices = np.minimum(indices, length - 1)
    return values[indices]


def _empty_stats_bucket() -> dict[str, list[np.ndarray]]:
    return {
        "min": [],
        "max": [],
        "mean": [],
        "var": [],
        "q01": [],
        "q99": [],
    }


def _append_episode_stats(bucket: dict[str, list[np.ndarray]], values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"`values` must be 3D [B,T,D], got shape {values.shape}")
    bucket["min"].append(values.min(axis=0))
    bucket["max"].append(values.max(axis=0))
    bucket["mean"].append(values.mean(axis=0))
    bucket["var"].append(values.var(axis=0, ddof=1))
    bucket["q01"].append(np.quantile(values, 0.01, axis=0).astype(np.float32))
    bucket["q99"].append(np.quantile(values, 0.99, axis=0).astype(np.float32))


def _aggregate_bucket(bucket: dict[str, list[np.ndarray]]) -> dict:
    mins = np.stack(bucket["min"], axis=0)
    maxs = np.stack(bucket["max"], axis=0)
    means = np.stack(bucket["mean"], axis=0)
    vars_ = np.stack(bucket["var"], axis=0)
    q01s = np.stack(bucket["q01"], axis=0)
    q99s = np.stack(bucket["q99"], axis=0)

    stepwise_min = mins.min(axis=0)
    stepwise_max = maxs.max(axis=0)
    stepwise_q01 = q01s.min(axis=0)
    stepwise_q99 = q99s.max(axis=0)
    stepwise_mean = means.mean(axis=0)
    stepwise_std = np.sqrt((vars_ + (means - stepwise_mean[None, :, :]) ** 2).mean(axis=0))

    global_mean = means.mean(axis=(0, 1))
    global_std = np.sqrt((vars_ + (means - global_mean[None, None, :]) ** 2).mean(axis=(0, 1)))

    return {
        "stepwise_min": stepwise_min.tolist(),
        "stepwise_max": stepwise_max.tolist(),
        "global_min": stepwise_min.min(axis=0).tolist(),
        "global_max": stepwise_max.max(axis=0).tolist(),
        "stepwise_q01": stepwise_q01.tolist(),
        "stepwise_q99": stepwise_q99.tolist(),
        "global_q01": stepwise_q01.min(axis=0).tolist(),
        "global_q99": stepwise_q99.max(axis=0).tolist(),
        "stepwise_mean": stepwise_mean.tolist(),
        "stepwise_std": stepwise_std.tolist(),
        "global_mean": global_mean.tolist(),
        "global_std": global_std.tolist(),
    }


def train_episode_paths(episode_paths: list[Path], val_set_proportion: float, seed: int) -> list[Path]:
    if val_set_proportion < 1e-6:
        return episode_paths
    episode_indices = list(range(len(episode_paths)))
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_indices)
    split_idx = int(len(episode_indices) * (1 - val_set_proportion))
    selected = sorted(episode_indices[:split_idx])
    return [episode_paths[i] for i in selected]


def write_fastwam_dataset_stats(task_ids: list[str], args: argparse.Namespace) -> Path:
    state_stats = defaultdict(_empty_stats_bucket)
    action_stats = defaultdict(_empty_stats_bucket)
    num_episodes = 0
    num_transition = 0

    for task_id in task_ids:
        data_dir = args.robotwin_root / "data" / "flying_hand" / task_id / args.setting / "data"
        episode_paths = train_episode_paths(
            sorted_episode_paths(data_dir),
            val_set_proportion=float(args.stats_val_set_proportion),
            seed=int(args.stats_seed),
        )
        for h5_path in episode_paths:
            with h5py.File(h5_path, "r") as f:
                actual = f["flying_hand/actual_state"][:].astype(np.float32)
                target = f["flying_hand/target_state"][:].astype(np.float32)
            if actual.shape != target.shape or actual.ndim != 2 or actual.shape[1] != 5:
                raise ValueError(f"Invalid state shapes in {h5_path}: actual={actual.shape} target={target.shape}")

            action = np.concatenate([target[1:], target[-1:]], axis=0).astype(np.float32)
            state_windows = actual[:, None, :]
            action_windows = sliding_window_with_replication(action, int(args.stats_action_size))
            _append_episode_stats(state_stats["default"], state_windows)
            _append_episode_stats(action_stats["default"], action_windows)
            num_episodes += 1
            num_transition += int(actual.shape[0])

    if num_episodes == 0:
        raise ValueError("No episodes selected for dataset stats.")

    stats = {
        "state": {"default": _aggregate_bucket(state_stats["default"])},
        "action": {"default": _aggregate_bucket(action_stats["default"])},
        "num_episodes": num_episodes,
        "num_transition": num_transition,
    }
    output_path = args.output_root / "dataset_stats.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"Wrote FastWAM dataset stats: episodes={num_episodes} "
        f"frames={num_transition} action_size={args.stats_action_size} -> {output_path}"
    )
    return output_path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parquet_table(rows: list[dict]) -> pa.Table:
    return pa.table(
        {
            "observation.state": pa.array([r["observation.state"] for r in rows], type=pa.list_(pa.float32(), 5)),
            "action": pa.array([r["action"] for r in rows], type=pa.list_(pa.float32(), 5)),
            "timestamp": pa.array([r["timestamp"] for r in rows], type=pa.float32()),
            "frame_index": pa.array([r["frame_index"] for r in rows], type=pa.int64()),
            "episode_index": pa.array([r["episode_index"] for r in rows], type=pa.int64()),
            "index": pa.array([r["index"] for r in rows], type=pa.int64()),
            "task_index": pa.array([r["task_index"] for r in rows], type=pa.int64()),
        }
    )


def link_or_copy(src: Path, dst: Path, copy_videos: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_videos:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def info_json(total_episodes: int, total_frames: int, total_tasks: int, fps: int, video_meta: dict) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [5],
            "names": [STATE_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": [5],
            "names": [STATE_NAMES],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for camera in CAMERAS:
        meta = video_meta[camera]
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": [meta["height"], meta["width"], 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": meta["height"],
                "video.width": meta["width"],
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    return {
        "codebase_version": "v2.1",
        "robot_type": "flying_hand",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(CAMERAS),
        "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def convert_task(task_id: str, args: argparse.Namespace) -> Path:
    src_root = args.robotwin_root / "data" / "flying_hand" / task_id / args.setting
    data_dir = src_root / "data"
    scene_info_path = src_root / "scene_info.json"
    instruction_path = args.robotwin_root / "description" / "task_instruction" / "flying_hand" / f"{task_id}.json"
    out_root = args.output_root / task_id

    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    scene_info = load_json(scene_info_path)
    instruction_payload = load_json(instruction_path)
    episode_paths = sorted_episode_paths(data_dir)

    first_ep = int(episode_paths[0].stem.removeprefix("episode"))
    video_meta = {
        camera: video_info(src_root / "video" / camera / f"episode{first_ep}.mp4")
        for camera in CAMERAS
    }
    fps_values = {round(meta["fps"]) for meta in video_meta.values()}
    if len(fps_values) != 1:
        raise ValueError(f"Camera FPS mismatch in {src_root}: {video_meta}")
    fps = int(next(iter(fps_values)))

    task_to_index: dict[str, int] = {}
    task_rows: list[dict] = []
    episode_rows: list[dict] = []
    episode_stat_rows: list[dict] = []
    total_frames = 0
    global_index = 0

    for h5_path in episode_paths:
        ep = int(h5_path.stem.removeprefix("episode"))
        chunk = ep // CHUNK_SIZE
        task = episode_task(task_id, ep, scene_info, instruction_payload, args.instruction_type)
        if task not in task_to_index:
            task_to_index[task] = len(task_to_index)
            task_rows.append({"task_index": task_to_index[task], "task": task})
        task_index = task_to_index[task]

        with h5py.File(h5_path, "r") as f:
            actual = f["flying_hand/actual_state"][:].astype(np.float32)
            target = f["flying_hand/target_state"][:].astype(np.float32)
        if actual.shape != target.shape or actual.ndim != 2 or actual.shape[1] != 5:
            raise ValueError(f"Invalid state shapes in {h5_path}: actual={actual.shape}, target={target.shape}")
        length = int(actual.shape[0])

        for camera in CAMERAS:
            src_video = src_root / "video" / camera / f"episode{ep}.mp4"
            meta = video_info(src_video)
            if meta["frames"] != length:
                raise ValueError(f"Frame count mismatch for {src_video}: video={meta['frames']} hdf5={length}")
            if int(round(meta["fps"])) != fps:
                raise ValueError(f"FPS mismatch for {src_video}: {meta['fps']} vs dataset fps {fps}")
            dst_video = out_root / f"videos/chunk-{chunk:03d}/observation.images.{camera}/episode_{ep:06d}.mp4"
            link_or_copy(src_video, dst_video, args.copy_videos)

        action = np.concatenate([target[1:], target[-1:]], axis=0).astype(np.float32)
        rows = []
        for frame_idx in range(length):
            rows.append(
                {
                    "observation.state": actual[frame_idx].tolist(),
                    "action": action[frame_idx].tolist(),
                    "timestamp": np.float32(frame_idx / fps),
                    "frame_index": frame_idx,
                    "episode_index": ep,
                    "index": global_index,
                    "task_index": task_index,
                }
            )
            global_index += 1

        parquet_path = out_root / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(parquet_table(rows), parquet_path)

        episode_rows.append({"episode_index": ep, "tasks": [task], "length": length})
        episode_stat_rows.append(
            {
                "episode_index": ep,
                "stats": {
                    "observation.state": feature_stats(actual),
                    "action": feature_stats(action),
                },
            }
        )
        total_frames += length

    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(
        json.dumps(info_json(len(episode_paths), total_frames, len(task_rows), fps, video_meta), indent=2),
        encoding="utf-8",
    )
    write_jsonl(meta_dir / "tasks.jsonl", task_rows)
    write_jsonl(meta_dir / "episodes.jsonl", episode_rows)
    write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stat_rows)

    print(f"Converted {task_id}: episodes={len(episode_paths)} frames={total_frames} tasks={len(task_rows)} -> {out_root}")
    return out_root


def main() -> None:
    args = parse_args()
    if args.all_tasks == (args.task_id is not None):
        raise ValueError("Pass exactly one of --task-id or --all-tasks.")

    if args.all_tasks:
        task_ids = sorted(p.name for p in (args.robotwin_root / "data" / "flying_hand").iterdir() if p.is_dir())
    else:
        task_ids = [args.task_id]

    for task_id in task_ids:
        convert_task(task_id, args)

    write_fastwam_dataset_stats(task_ids, args)


if __name__ == "__main__":
    main()
