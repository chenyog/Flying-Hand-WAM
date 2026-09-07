"""Convert Flying-Hand RobotWin HDF5 episodes to a Diffusion Policy Zarr dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import zarr


def episode_paths(
    data_root: Path,
    num_episodes: int | None = None,
    all_tasks: bool = False,
) -> list[Path]:
    if all_tasks:
        paths = sorted(
            data_root.glob("*/**/data/episode*.hdf5"),
            key=lambda path: (
                str(path.parent.parent.parent.relative_to(data_root)),
                int(path.stem.removeprefix("episode")),
            ),
        )
        if not paths:
            raise FileNotFoundError(
                f"No Flying-Hand episodes found below task root: {data_root}"
            )
        if num_episodes is not None:
            paths = [
                path
                for path in paths
                if int(path.stem.removeprefix("episode")) < num_episodes
            ]
    else:
        if num_episodes is None or num_episodes <= 0:
            raise ValueError("num_episodes must be positive for a single task root")
        paths = [data_root / "data" / f"episode{i}.hdf5" for i in range(num_episodes)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} episode file(s), first missing file: {missing[0]}"
        )
    if not paths:
        raise ValueError("No episodes selected")
    return paths


def episode_length(path: Path) -> int:
    with h5py.File(path, "r") as f:
        actual = f["flying_hand/actual_state"]
        target = f["flying_hand/target_state"]
        head = f["observation/head_camera/rgb"]
        wrist = f["observation/wrist_camera/rgb"]
        if actual.ndim != 2 or actual.shape[1] != 5:
            raise ValueError(f"{path}: expected actual_state shape [T, 5], got {actual.shape}")
        if target.shape != actual.shape:
            raise ValueError(f"{path}: actual/target shape mismatch: {actual.shape} vs {target.shape}")
        if not (head.shape[0] == wrist.shape[0] == actual.shape[0]):
            raise ValueError(f"{path}: observation/state frame count mismatch")
        return max(int(actual.shape[0]) - 1, 0)


def decode_image(encoded: np.ndarray, path: Path, index: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{path}: failed to decode image at frame {index}")
    return image


def convert(
    data_root: Path,
    output: Path,
    num_episodes: int | None = None,
    all_tasks: bool = False,
) -> None:
    paths = episode_paths(data_root, num_episodes, all_tasks=all_tasks)
    lengths = [episode_length(path) for path in paths]
    total = sum(lengths)
    if total == 0:
        raise ValueError("No transitions found in the selected episodes")

    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(output), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    head = data.create_dataset(
        "head_camera", shape=(total, 3, 360, 480), chunks=(1, 3, 360, 480),
        dtype="uint8", compressor=compressor,
    )
    wrist = data.create_dataset(
        "wrist_camera", shape=(total, 3, 480, 640), chunks=(1, 3, 480, 640),
        dtype="uint8", compressor=compressor,
    )
    state = data.create_dataset(
        "state", shape=(total, 5), chunks=(1024, 5), dtype="float32", compressor=compressor,
    )
    action = data.create_dataset(
        "action", shape=(total, 5), chunks=(1024, 5), dtype="float32", compressor=compressor,
    )
    ends = np.cumsum(np.asarray(lengths, dtype=np.int64))
    meta.create_dataset("episode_ends", data=ends, dtype="int64", compressor=compressor)

    offset = 0
    for episode_index, (path, length) in enumerate(zip(paths, lengths)):
        print(f"processing episode {episode_index + 1}/{len(paths)}: {path.name}")
        with h5py.File(path, "r") as f:
            actual = f["flying_hand/actual_state"][:].astype(np.float32)
            target = f["flying_hand/target_state"][:].astype(np.float32)
            head_encoded = f["observation/head_camera/rgb"]
            wrist_encoded = f["observation/wrist_camera/rgb"]
            for frame in range(length):
                head[offset + frame] = np.moveaxis(
                    decode_image(head_encoded[frame], path, frame), -1, 0
                )
                wrist[offset + frame] = np.moveaxis(
                    decode_image(wrist_encoded[frame], path, frame), -1, 0
                )
            state[offset:offset + length] = actual[:-1]
            action[offset:offset + length] = target[1:]
        offset += length

    print(f"wrote {total} transitions from {len(paths)} episodes to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Episodes per task for --all-tasks, or total episodes for one task",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Recursively merge data_root/*/*/data/episode*.hdf5",
    )
    args = parser.parse_args()
    if args.num_episodes is not None and args.num_episodes <= 0:
        parser.error("--num-episodes must be positive")
    if not args.all_tasks and args.num_episodes is None:
        parser.error("--num-episodes is required unless --all-tasks is used")
    convert(
        args.data_root,
        args.output,
        args.num_episodes,
        all_tasks=args.all_tasks,
    )


if __name__ == "__main__":
    main()
