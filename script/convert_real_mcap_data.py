#!/usr/bin/env python3
"""Convert real Flying-Hand MCAP demonstrations to the project HDF5 layout.

The active MCAP files are classified from the task/trajectory identifiers in
``Flying-Hand-WAM-Deployment/WAM/collect_data/tasks/*.yaml``.  Each output
episode preserves the recorded 30 Hz robot-state frames and contains the same
state/action and camera names used by the simulated Flying-Hand datasets.  The third-view
camera becomes ``head_camera`` and the URDF-mounted mid camera becomes
``wrist_camera``; the additional real front camera is retained as
``front_camera``.

Run this script with its optional MCAP dependencies, for example::

    uv run --with mcap --with mcap-ros1-support \
      python script/convert_real_mcap_data.py
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable

import h5py
from mcap.reader import make_reader
from mcap_ros1.decoder import DecoderFactory
import numpy as np
import yaml


RGB_JPEG_CODEC_ATTR = "rgb_jpeg_codec"
RGB_JPEG_CODEC = "opencv_bgr_from_rgb_v2"
SETTING = "flying_hand_real"

CAMERA_TOPICS = {
    "head_camera": "/camera/third_view/compressed",
    "wrist_camera": "/camera/mid/compressed",
    "front_camera": "/camera/front/compressed",
}
STATE_TOPICS = {
    "robot_state": "/wam/robot_state",
    "reference": "/wam/reference",
    "actuator_state": "/wam/actuator_state",
    "actuator_command": "/wam/actuator_command",
}
REQUIRED_TOPICS = frozenset((*CAMERA_TOPICS.values(), *STATE_TOPICS.values()))
MAX_SOURCE_GAP_SECONDS = 1.0
MAX_ALIGNMENT_ERROR_SECONDS = 0.5
MIN_EFFECTIVE_SOURCE_RATE_HZ = 20.0

# A trajectory suffix such as ``_1`` denotes a second shelf position, not a
# different object.  These labels are the values substituted into the task
# description from the authoritative Deployment YAML.
OBJECT_LABELS = {
    "nongfu_spring_green": "the green Nongfu Spring bottle",
    "nongfu_spring_red": "the red Nongfu Spring bottle",
    "tea": "the tea bottle",
    "black": "the black ring",
    "default": "the bottle",
}


@dataclass(frozen=True)
class TaskConfig:
    name: str
    description: str
    trajectories: tuple[str, ...]
    path: str


@dataclass(frozen=True)
class EpisodeSpec:
    source: str
    task: str
    trajectory: str
    episode: int
    instruction: str
    config_path: str


@dataclass(frozen=True)
class EpisodeResult:
    task: str
    trajectory: str
    episode: int
    source: str
    frames: int
    duration_seconds: float
    source_rate_hz: float
    max_alignment_error_seconds: float
    instruction: str


@dataclass(frozen=True)
class EpisodeQuality:
    source: str
    accepted: bool
    reasons: tuple[str, ...]
    topic_counts: dict[str, int]
    effective_rates_hz: dict[str, float]
    max_gaps_seconds: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/Flying-Hand-WAM/data/runs"),
        help="Directory containing active MCAP files. Subdirectories are excluded.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/Flying-Hand-WAM/data/flying_hand_real"),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("/data/Flying-Hand-WAM-Deployment/WAM/collect_data/tasks"),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate and print classification; do not create output.",
    )
    return parser.parse_args()


def load_task_configs(config_dir: Path) -> dict[str, TaskConfig]:
    configs: dict[str, TaskConfig] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Task YAML must contain a mapping: {path}")
        name = str(payload.get("name", "")).strip()
        description = str(payload.get("description", "")).strip()
        trajectories = tuple(
            str(item.get("id", "")).strip()
            for item in payload.get("trajectories", [])
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        )
        if not name or not description or not trajectories:
            raise ValueError(
                f"Task YAML requires name, description, and trajectory ids: {path}"
            )
        if name in configs:
            raise ValueError(f"Duplicate task name {name!r}: {path}")
        configs[name] = TaskConfig(name, description, trajectories, str(path.resolve()))
    if not configs:
        raise FileNotFoundError(f"No task YAML files found in {config_dir}")
    return configs


def strip_position_suffix(trajectory: str) -> str:
    return re.sub(r"_\d+$", "", trajectory)


def instruction_values(task: str, trajectory: str) -> dict[str, str]:
    object_id = strip_position_suffix(trajectory)
    label = OBJECT_LABELS.get(object_id, object_id.replace("_", " "))
    if task in {"move_bottle", "adjust_bottle", "shake_bottle"}:
        return {"A": label, "B": "target location"}
    if task == "thread_ring_rod":
        return {"A": label, "B": "the target rod"}
    return {"A": label, "B": "the target location"}


def make_instruction(config: TaskConfig, trajectory: str) -> str:
    try:
        return config.description.format(**instruction_values(config.name, trajectory))
    except KeyError as error:
        raise ValueError(
            f"Unsupported placeholder {error} in description from {config.path}"
        ) from error


def match_source(path: Path, configs: dict[str, TaskConfig]) -> tuple[str, str]:
    stem = path.stem
    matches: list[tuple[int, str, str]] = []
    for config in configs.values():
        for trajectory in config.trajectories:
            prefix = f"{config.name}_{trajectory}_"
            if stem.startswith(prefix):
                remainder = stem[len(prefix) :]
                if re.fullmatch(r"\d{8}_\d{6}(?:_\d+)?", remainder):
                    matches.append((len(prefix), config.name, trajectory))
    if not matches:
        raise ValueError(
            f"MCAP filename does not match a configured task/trajectory: {path.name}"
        )
    matches.sort(reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        raise ValueError(f"Ambiguous task/trajectory filename: {path.name}")
    _, task, trajectory = matches[0]
    return task, trajectory


def discover_episodes(input_root: Path, configs: dict[str, TaskConfig]) -> list[EpisodeSpec]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    paths = sorted(input_root.glob("*.mcap"))
    if not paths:
        raise FileNotFoundError(f"No active MCAP files found directly in {input_root}")

    grouped: dict[str, list[tuple[Path, str]]] = {}
    for path in paths:
        task, trajectory = match_source(path, configs)
        grouped.setdefault(task, []).append((path, trajectory))

    specs: list[EpisodeSpec] = []
    for task in sorted(grouped):
        config = configs[task]
        for episode, (path, trajectory) in enumerate(sorted(grouped[task])):
            specs.append(
                EpisodeSpec(
                    source=str(path.resolve()),
                    task=task,
                    trajectory=trajectory,
                    episode=episode,
                    instruction=make_instruction(config, trajectory),
                    config_path=config.path,
                )
            )
    return specs


def message_time_seconds(message: Any) -> float:
    return float(message.publish_time) / 1_000_000_000.0


def decode_mcap(path: Path) -> dict[str, list[tuple[float, Any]]]:
    streams: dict[str, list[tuple[float, Any]]] = {topic: [] for topic in REQUIRED_TOPICS}
    with path.open("rb") as source:
        reader = make_reader(source, decoder_factories=[DecoderFactory()])
        header = reader.get_header()
        if header.profile != "ros1" or header.library != "wam-direct-mcap":
            raise ValueError(
                f"Unsupported MCAP contract in {path}: "
                f"profile={header.profile!r}, library={header.library!r}"
            )
        for _schema, channel, message, decoded in reader.iter_decoded_messages(
            topics=list(REQUIRED_TOPICS)
        ):
            streams[channel.topic].append((message_time_seconds(message), decoded))

    missing = [topic for topic, values in streams.items() if not values]
    if missing:
        raise ValueError(f"Missing required MCAP topics in {path}: {missing}")
    for topic, values in streams.items():
        times = [item[0] for item in values]
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError(f"Non-monotonic timestamps for {topic} in {path}")
    return streams


def assess_episode_quality(spec: EpisodeSpec) -> EpisodeQuality:
    times: dict[str, list[float]] = {topic: [] for topic in REQUIRED_TOPICS}
    path = Path(spec.source)
    with path.open("rb") as source:
        reader = make_reader(source)
        header = reader.get_header()
        if header.profile != "ros1" or header.library != "wam-direct-mcap":
            return EpisodeQuality(
                source=spec.source,
                accepted=False,
                reasons=(
                    f"unsupported contract profile={header.profile!r}, library={header.library!r}",
                ),
                topic_counts={topic: 0 for topic in REQUIRED_TOPICS},
                effective_rates_hz={topic: 0.0 for topic in REQUIRED_TOPICS},
                max_gaps_seconds={topic: math.inf for topic in REQUIRED_TOPICS},
            )
        for _schema, channel, message in reader.iter_messages(topics=list(REQUIRED_TOPICS)):
            times[channel.topic].append(message_time_seconds(message))

    reasons: list[str] = []
    counts = {topic: len(values) for topic, values in times.items()}
    missing = sorted(topic for topic, count in counts.items() if not count)
    if missing:
        reasons.append(f"missing required topics: {missing}")

    rates: dict[str, float] = {}
    max_gaps: dict[str, float] = {}
    for topic, values in times.items():
        if len(values) < 2 or values[-1] <= values[0]:
            rates[topic] = 0.0
            max_gaps[topic] = math.inf
            continue
        deltas = [b - a for a, b in zip(values, values[1:])]
        if any(delta < 0 for delta in deltas):
            reasons.append(f"non-monotonic timestamps: {topic}")
        rates[topic] = (len(values) - 1) / (values[-1] - values[0])
        max_gaps[topic] = max(deltas)

    if rates:
        slowest_topic = min(rates, key=rates.get)
        if rates[slowest_topic] + 1e-6 < MIN_EFFECTIVE_SOURCE_RATE_HZ:
            reasons.append(
                f"{slowest_topic} effective rate {rates[slowest_topic]:.3f} Hz "
                f"is below the accepted source rate {MIN_EFFECTIVE_SOURCE_RATE_HZ:g} Hz"
            )
        widest_topic = max(max_gaps, key=max_gaps.get)
        if max_gaps[widest_topic] > MAX_SOURCE_GAP_SECONDS:
            reasons.append(
                f"{widest_topic} has a {max_gaps[widest_topic]:.3f} s gap "
                f"> {MAX_SOURCE_GAP_SECONDS:g} s"
            )
    return EpisodeQuality(
        source=spec.source,
        accepted=not reasons,
        reasons=tuple(reasons),
        topic_counts=counts,
        effective_rates_hz=rates,
        max_gaps_seconds=max_gaps,
    )


def reindex_episodes(specs: list[EpisodeSpec]) -> list[EpisodeSpec]:
    next_index: dict[str, int] = {}
    result: list[EpisodeSpec] = []
    for spec in specs:
        episode = next_index.get(spec.task, 0)
        result.append(replace(spec, episode=episode))
        next_index[spec.task] = episode + 1
    return result


def nearest_indices(source_times: list[float], target_times: np.ndarray) -> tuple[np.ndarray, float]:
    indices: list[int] = []
    max_error = 0.0
    for target in target_times:
        right = bisect_left(source_times, float(target))
        candidates = [min(right, len(source_times) - 1)]
        if right > 0:
            candidates.append(right - 1)
        index = min(candidates, key=lambda item: abs(source_times[item] - target))
        error = abs(source_times[index] - target)
        max_error = max(max_error, error)
        indices.append(index)
    return np.asarray(indices, dtype=np.int64), max_error


def quaternion_yaw(value: Any) -> float:
    x, y, z, w = float(value.x), float(value.y), float(value.z), float(value.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_matrix(value: Any) -> np.ndarray:
    x, y, z, w = float(value.x), float(value.y), float(value.z), float(value.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("Encountered a zero-norm robot-state quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def vector3(value: Any) -> np.ndarray:
    return np.asarray([value.x, value.y, value.z], dtype=np.float64)


def relative_pose(
    position: Any,
    orientation: Any,
    origin_position: np.ndarray,
    origin_rotation: np.ndarray,
    origin_yaw: float,
) -> np.ndarray:
    relative_position = origin_rotation.T @ (vector3(position) - origin_position)
    return np.asarray(
        [*relative_position, wrap_angle(quaternion_yaw(orientation) - origin_yaw)],
        dtype=np.float32,
    )


def write_jpeg_dataset(group: h5py.Group, name: str, frames: Iterable[bytes]) -> None:
    encoded = [bytes(frame) for frame in frames]
    if not encoded:
        raise ValueError(f"Cannot write an empty image stream: {group.name}/{name}")
    max_size = max(map(len, encoded))
    group.create_dataset(name, data=encoded, dtype=f"S{max_size}")


def write_video(frames: Iterable[bytes], output: Path, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-framerate",
        f"{fps:g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{fps:g}",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame in frames:
            process.stdin.write(bytes(frame))
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {output}: {stderr.strip()}")


def sample_rate(values: list[tuple[float, Any]]) -> float:
    times = np.asarray([item[0] for item in values], dtype=np.float64)
    deltas = np.diff(times)
    deltas = deltas[deltas > 0]
    if not len(deltas):
        return 0.0
    return float(1.0 / np.median(deltas))


def hardlink_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def convert_episode(spec: EpisodeSpec, output_root: str, fps: float) -> EpisodeResult:
    source = Path(spec.source)
    streams = decode_mcap(source)
    start = max(values[0][0] for values in streams.values())
    end = min(values[-1][0] for values in streams.values())
    if end <= start:
        raise ValueError(f"No common time range across required topics in {source}")
    timeline = np.asarray(
        [
            timestamp
            for timestamp, _message in streams[STATE_TOPICS["robot_state"]]
            if start <= timestamp <= end
        ],
        dtype=np.float64,
    )
    frame_count = len(timeline)
    if frame_count < 2:
        raise ValueError(f"Episode has fewer than two aligned robot-state frames: {source}")

    sampled: dict[str, list[Any]] = {}
    max_alignment_error = 0.0
    for topic, values in streams.items():
        source_times = [item[0] for item in values]
        indices, error = nearest_indices(source_times, timeline)
        sampled[topic] = [values[int(index)][1] for index in indices]
        max_alignment_error = max(max_alignment_error, error)
    if max_alignment_error > MAX_ALIGNMENT_ERROR_SECONDS:
        raise ValueError(
            f"A required stream is too far from the {fps:g} Hz timeline in {source}: "
            f"{max_alignment_error:.6f} s > {MAX_ALIGNMENT_ERROR_SECONDS:g} s"
        )

    robot_messages = sampled[STATE_TOPICS["robot_state"]]
    reference_messages = sampled[STATE_TOPICS["reference"]]
    command_messages = sampled[STATE_TOPICS["actuator_command"]]
    actuator_messages = sampled[STATE_TOPICS["actuator_state"]]
    if any(not message.poses for message in reference_messages):
        raise ValueError(f"An empty /wam/reference Path was found in {source}")

    first_robot = robot_messages[0]
    origin_position = vector3(first_robot.position)
    origin_rotation = quaternion_matrix(first_robot.attitude)
    origin_yaw = quaternion_yaw(first_robot.attitude)
    command_grasp = np.asarray(
        [float(message.reference_position) <= -5.5 for message in command_messages],
        dtype=np.float32,
    )
    actual = np.stack(
        [
            np.append(
                relative_pose(
                    message.position,
                    message.attitude,
                    origin_position,
                    origin_rotation,
                    origin_yaw,
                ),
                command_grasp[index],
            )
            for index, message in enumerate(robot_messages)
        ]
    ).astype(np.float32)
    target = np.stack(
        [
            np.append(
                relative_pose(
                    message.poses[0].pose.position,
                    message.poses[0].pose.orientation,
                    origin_position,
                    origin_rotation,
                    origin_yaw,
                ),
                command_grasp[index],
            )
            for index, message in enumerate(reference_messages)
        ]
    ).astype(np.float32)

    camera_frames = {
        camera: [bytes(message.data) for message in sampled[topic]]
        for camera, topic in CAMERA_TOPICS.items()
    }
    root = Path(output_root) / spec.task / SETTING
    stem = f"episode{spec.episode}"
    data_path = root / "data" / f"{stem}.hdf5"
    instruction_path = root / "instructions" / f"{stem}.json"
    metadata_path = root / "metadata" / f"{stem}.json"
    mcap_path = root / "_mcap" / f"{stem}.mcap"
    for directory in (data_path.parent, instruction_path.parent, metadata_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_path, "w") as output:
        output.attrs[RGB_JPEG_CODEC_ATTR] = RGB_JPEG_CODEC
        output.attrs["fps"] = float(fps)
        output.attrs["task"] = spec.task
        output.attrs["trajectory_id"] = spec.trajectory
        output.attrs["source_mcap"] = source.name
        flying_hand = output.create_group("flying_hand")
        flying_hand.create_dataset("actual_state", data=actual)
        flying_hand.create_dataset("target_state", data=target)
        observation = output.create_group("observation")
        for camera, frames in camera_frames.items():
            write_jpeg_dataset(observation.create_group(camera), "rgb", frames)
        real = output.create_group("real")
        real.create_dataset("timestamp", data=timeline - timeline[0])
        real.create_dataset(
            "actuator_position",
            data=np.asarray([message.position for message in actuator_messages], dtype=np.float32),
        )
        real.create_dataset(
            "actuator_command_position",
            data=np.asarray(
                [message.reference_position for message in command_messages], dtype=np.float32
            ),
        )

    instruction_payload = {"seen": [spec.instruction], "unseen": []}
    instruction_path.write_text(
        json.dumps(instruction_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    source_storage = hardlink_or_copy(source, mcap_path)
    for camera, frames in camera_frames.items():
        write_video(frames, root / "video" / camera / f"{stem}.mp4", fps)

    source_rates = [sample_rate(streams[topic]) for topic in REQUIRED_TOPICS]
    source_rate_hz = float(np.median(source_rates))
    metadata = {
        "episode": spec.episode,
        "task": spec.task,
        "trajectory_id": spec.trajectory,
        "instruction": spec.instruction,
        "instruction_source": {
            "yaml": spec.config_path,
            "field": "description",
            "substitutions": instruction_values(spec.task, spec.trajectory),
        },
        "source_mcap": str(source.resolve()),
        "stored_mcap": str(mcap_path.relative_to(Path(output_root))),
        "stored_mcap_method": source_storage,
        "source_rate_hz_median": source_rate_hz,
        "timestamp_basis": "mcap_publish_time",
        "output_fps": fps,
        "frames": frame_count,
        "duration_seconds": float(timeline[-1] - timeline[0]),
        "max_alignment_error_seconds": max_alignment_error,
        "camera_mapping": CAMERA_TOPICS,
        "state_contract": {
            "actual_state": "initial-body-relative /wam/robot_state [x,y,z,yaw] plus commanded grasp",
            "target_state": "initial-body-relative first pose of /wam/reference [x,y,z,yaw] plus commanded grasp",
            "grasp": "1 when /wam/actuator_command.reference_position <= -5.5, else 0",
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return EpisodeResult(
        task=spec.task,
        trajectory=spec.trajectory,
        episode=spec.episode,
        source=str(source.resolve()),
        frames=frame_count,
        duration_seconds=float(timeline[-1] - timeline[0]),
        source_rate_hz=source_rate_hz,
        max_alignment_error_seconds=max_alignment_error,
        instruction=spec.instruction,
    )


def validate_output(output_root: Path, specs: list[EpisodeSpec], fps: float) -> None:
    by_task: dict[str, list[EpisodeSpec]] = {}
    for spec in specs:
        by_task.setdefault(spec.task, []).append(spec)
    for task, task_specs in by_task.items():
        root = output_root / task / SETTING
        for directory, suffix in (
            ("data", ".hdf5"),
            ("instructions", ".json"),
            ("metadata", ".json"),
            ("_mcap", ".mcap"),
        ):
            paths = sorted((root / directory).glob(f"episode*{suffix}"))
            if len(paths) != len(task_specs):
                raise ValueError(
                    f"Output count mismatch for {task}/{directory}: "
                    f"expected {len(task_specs)}, found {len(paths)}"
                )
        for camera in CAMERA_TOPICS:
            videos = list((root / "video" / camera).glob("episode*.mp4"))
            if len(videos) != len(task_specs):
                raise ValueError(
                    f"Output video count mismatch for {task}/{camera}: "
                    f"expected {len(task_specs)}, found {len(videos)}"
                )
        for spec in task_specs:
            path = root / "data" / f"episode{spec.episode}.hdf5"
            with h5py.File(path, "r") as episode:
                if episode.attrs.get(RGB_JPEG_CODEC_ATTR) != RGB_JPEG_CODEC:
                    raise ValueError(f"Invalid RGB codec attribute: {path}")
                if not math.isclose(float(episode.attrs["fps"]), fps):
                    raise ValueError(f"Invalid fps attribute: {path}")
                state = episode["flying_hand/actual_state"]
                target = episode["flying_hand/target_state"]
                if state.shape != target.shape or len(state.shape) != 2 or state.shape[1] != 5:
                    raise ValueError(f"Invalid state/action shape in {path}")
                for camera in CAMERA_TOPICS:
                    if len(episode[f"observation/{camera}/rgb"]) != len(state):
                        raise ValueError(f"Image/state length mismatch in {path} for {camera}")


def print_classification(specs: list[EpisodeSpec], archived_count: int) -> None:
    counts: dict[str, dict[str, int]] = {}
    for spec in specs:
        counts.setdefault(spec.task, {})
        counts[spec.task][spec.trajectory] = counts[spec.task].get(spec.trajectory, 0) + 1
    print(json.dumps(counts, indent=2, sort_keys=True))
    print(f"active episodes: {len(specs)}")
    print(f"archived MCAP files excluded from conversion: {archived_count}")


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    configs = load_task_configs(args.config_dir.resolve())
    specs = discover_episodes(args.input_root.resolve(), configs)
    archived_count = len(list((args.input_root / "old").glob("*.mcap")))
    print_classification(specs, archived_count)
    if args.dry_run:
        return

    print("Running MCAP topic/rate quality preflight...", flush=True)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(specs))) as executor:
        qualities = list(executor.map(assess_episode_quality, specs))
    quality_by_source = {quality.source: quality for quality in qualities}
    rejected_specs = [spec for spec in specs if not quality_by_source[spec.source].accepted]
    specs = reindex_episodes(
        [spec for spec in specs if quality_by_source[spec.source].accepted]
    )
    if rejected_specs:
        print(
            f"Rejecting {len(rejected_specs)} MCAP files that fail the source-quality floor:",
            file=sys.stderr,
        )
        for spec in rejected_specs:
            quality = quality_by_source[spec.source]
            print(
                f"  {Path(spec.source).name}: {'; '.join(quality.reasons)}",
                file=sys.stderr,
            )
    if not specs:
        raise ValueError("No MCAP episodes passed the topic/rate quality preflight")

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite converted data: {output_root}"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = output_root.with_name(f".{output_root.name}.partial-{timestamp}")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)

    results: list[EpisodeResult] = []
    try:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(specs))) as executor:
            futures = {
                executor.submit(convert_episode, spec, str(staging), float(args.fps)): spec
                for spec in specs
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                spec = futures[future]
                result = future.result()
                results.append(result)
                print(
                    f"[{completed}/{len(specs)}] {result.task}/episode{result.episode} "
                    f"<- {Path(spec.source).name}: {result.frames} frames",
                    flush=True,
                )

        results.sort(key=lambda item: (item.task, item.episode))
        manifest = {
            "format": "flying_hand_real_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_root": str(args.input_root.resolve()),
            "config_dir": str(args.config_dir.resolve()),
            "setting": SETTING,
            "fps": float(args.fps),
            "camera_mapping": CAMERA_TOPICS,
            "archived_mcap_excluded": archived_count,
            "rejected_active_mcap": [
                {
                    "task": spec.task,
                    "trajectory": spec.trajectory,
                    **asdict(quality_by_source[spec.source]),
                }
                for spec in rejected_specs
            ],
            "episodes": [asdict(result) for result in results],
        }
        (staging / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_output(staging, specs, float(args.fps))
        staging.rename(output_root)
    except Exception:
        print(
            f"Conversion failed. Partial output was preserved for diagnosis at: {staging}",
            file=sys.stderr,
        )
        raise
    print(f"Converted and validated {len(results)} episodes at: {output_root}")


if __name__ == "__main__":
    main()
