"""Convert Flying-Hand RobotWin HDF5 episodes to a Diffusion Policy Zarr dataset."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import zarr

SHARED_UTILS = Path(__file__).resolve().parents[2] / "envs" / "utils"
if str(SHARED_UTILS) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS))

from rgb_codec import decode_hdf5_rgb, require_current_rgb_jpeg_codec


def episode_number(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"Unexpected episode filename: {path}")
    return int(match.group(1))


def episode_paths(
    data_roots: Path | Sequence[Path],
    all_tasks: bool = False,
) -> list[Path]:
    if isinstance(data_roots, Path):
        data_roots = [data_roots]
    else:
        data_roots = list(data_roots)
    if not data_roots:
        raise ValueError("At least one dataset directory is required")

    if all_tasks:
        if len(data_roots) != 1:
            raise ValueError("--all-tasks accepts one task root; use repeated --dataset-dir instead")
        data_root = data_roots[0]
        paths = sorted(
            data_root.glob("*/**/data/episode*.hdf5"),
            key=lambda path: (
                str(path.parent.parent.parent.relative_to(data_root)),
                episode_number(path),
            ),
        )
        if not paths:
            raise FileNotFoundError(
                f"No Flying-Hand episodes found below task root: {data_root}"
            )
    else:
        paths = []
        for data_root in data_roots:
            episode_dir = data_root if data_root.name == "data" else data_root / "data"
            episodes = sorted(
                episode_dir.glob("episode*.hdf5"),
                key=episode_number,
            )
            if not episodes:
                raise FileNotFoundError(
                    f"No episode*.hdf5 files found in {episode_dir}"
                )
            paths.extend(episodes)
    if not paths:
        raise ValueError("No episodes selected")
    return paths


def episode_task_names(
    data_roots: Path | Sequence[Path],
    paths: Sequence[Path],
    all_tasks: bool = False,
) -> list[str]:
    """Return one stable source-task label for every selected episode."""
    roots = [data_roots] if isinstance(data_roots, Path) else list(data_roots)
    if all_tasks:
        data_root = roots[0]
        return [path.relative_to(data_root).parts[0] for path in paths]

    episode_dir_to_name = {}
    for data_root in roots:
        dataset_root = data_root.parent if data_root.name == "data" else data_root
        episode_dir = dataset_root / "data"
        if dataset_root.parent == dataset_root:
            task_name = dataset_root.name
        else:
            task_name = f"{dataset_root.parent.name}/{dataset_root.name}"
        episode_dir_to_name[episode_dir.resolve()] = task_name

    result = []
    for path in paths:
        episode_dir = path.parent.resolve()
        if episode_dir not in episode_dir_to_name:
            raise ValueError(f"Cannot determine source task for episode: {path}")
        result.append(episode_dir_to_name[episode_dir])
    return result


def episode_length(path: Path) -> int:
    with h5py.File(path, "r") as f:
        require_current_rgb_jpeg_codec(f)
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
    try:
        return decode_hdf5_rgb(encoded)
    except ValueError as exc:
        raise ValueError(f"{path}: failed to decode image at frame {index}") from exc


def write_episode_images(job: tuple[Path, Path, int, int]) -> tuple[Path, int]:
    """Write one episode's image chunks; every job owns a disjoint frame range."""
    output, path, offset, length = job
    root = zarr.open_group(str(output), mode="r+")
    head = root["data/head_camera"]
    wrist = root["data/wrist_camera"]
    with h5py.File(path, "r") as f:
        require_current_rgb_jpeg_codec(f)
        head_encoded = f["observation/head_camera/rgb"]
        wrist_encoded = f["observation/wrist_camera/rgb"]
        for frame in range(length):
            # Image arrays use one complete frame per Zarr chunk, so separate
            # episode workers never read-modify-write the same chunk key.
            head[offset + frame] = np.moveaxis(
                decode_image(head_encoded[frame], path, frame), -1, 0
            )
            wrist[offset + frame] = np.moveaxis(
                decode_image(wrist_encoded[frame], path, frame), -1, 0
            )
    return path, length


def convert(
    data_roots: Path | Sequence[Path],
    output: Path,
    all_tasks: bool = False,
    workers: int = 1,
) -> None:
    paths = episode_paths(data_roots, all_tasks=all_tasks)
    task_names_by_episode = episode_task_names(data_roots, paths, all_tasks=all_tasks)
    task_names = sorted(set(task_names_by_episode))
    task_name_to_id = {name: index for index, name in enumerate(task_names)}
    episode_task_ids = np.asarray(
        [task_name_to_id[name] for name in task_names_by_episode],
        dtype=np.int32,
    )
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
    meta.create_dataset(
        "episode_task_ids",
        data=episode_task_ids,
        dtype="int32",
        compressor=compressor,
    )
    meta.attrs["task_names"] = task_names

    # State/action chunks span episode boundaries, so only the parent process
    # writes them.  Image chunks are exactly one frame and are populated later
    # by workers with non-overlapping episode ranges.
    offsets = np.concatenate(([0], ends[:-1])).astype(np.int64)
    for episode_index, (path, length) in enumerate(zip(paths, lengths)):
        print(f"processing state/action {episode_index + 1}/{len(paths)}: {path}")
        offset = int(offsets[episode_index])
        with h5py.File(path, "r") as f:
            actual = f["flying_hand/actual_state"][:].astype(np.float32)
            target = f["flying_hand/target_state"][:].astype(np.float32)
            state[offset : offset + length] = actual[:-1]
            action[offset : offset + length] = target[1:]

    image_jobs = [
        (output, path, int(offset), length)
        for path, offset, length in zip(paths, offsets, lengths)
    ]
    if workers == 1:
        results = map(write_episode_images, image_jobs)
        for episode_index, (path, _) in enumerate(results, start=1):
            print(f"processed images {episode_index}/{len(paths)}: {path}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for episode_index, (path, _) in enumerate(
                executor.map(write_episode_images, image_jobs),
                start=1,
            ):
                print(f"processed images {episode_index}/{len(paths)}: {path}")

    print(
        f"wrote {total} transitions from {len(paths)} episodes "
        f"across {len(task_names)} task(s) to {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--dataset-dir",
        action="append",
        type=Path,
        default=[],
        help="Dataset directory containing data/episode*.hdf5; repeat to concatenate datasets",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Recursively merge data_root/*/*/data/episode*.hdf5",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel episode image writers; state/action always use the parent process (default: 1)",
    )
    args = parser.parse_args()
    if args.data_root is not None and args.dataset_dir:
        parser.error("--data-root and --dataset-dir cannot be combined")
    if args.data_root is None and not args.dataset_dir:
        parser.error("provide --data-root or at least one --dataset-dir")
    if args.all_tasks and args.dataset_dir:
        parser.error("--all-tasks cannot be combined with --dataset-dir")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    convert(
        args.dataset_dir or args.data_root,
        args.output,
        all_tasks=args.all_tasks,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
