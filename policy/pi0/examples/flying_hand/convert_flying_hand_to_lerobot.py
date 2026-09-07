"""Convert RoboTwin Flying-Hand HDF5 episodes to a local LeRobot dataset.

Example:
  uv run examples/flying_hand/convert_flying_hand_to_lerobot.py \
    --input-root /data/Flying-Hand-WAM/data/flying_hand \
    --setting flying_hand_clean --repo-id flying_hand_all_tasks --all-tasks
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def decode_rgb(value: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode an RGB frame from HDF5")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_instruction(path: Path, instruction_type: str, fallback: str) -> str:
    if not path.is_file():
        return fallback
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get(instruction_type) or payload.get("seen") or payload.get("instructions")
    if isinstance(values, str):
        return values
    if not values:
        return fallback
    return str(values[0])


def create_dataset(repo_id: str, fps: float, *, overwrite: bool) -> LeRobotDataset:
    output = Path(HF_LEROBOT_HOME) / repo_id
    if overwrite and output.exists():
        shutil.rmtree(output)
    features = {
        "observation.state": {"dtype": "float32", "shape": (5,), "names": [["x", "y", "z", "yaw", "grasp"]]},
        "action": {"dtype": "float32", "shape": (5,), "names": [["x", "y", "z", "yaw", "grasp"]]},
        "observation.images.head_camera": {
            "dtype": "image", "shape": (3, 360, 480), "names": ["channels", "height", "width"]
        },
        "observation.images.wrist_camera": {
            "dtype": "image", "shape": (3, 480, 640), "names": ["channels", "height", "width"]
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="flying_hand",
        features=features,
        use_videos=False,
        image_writer_processes=1,
        image_writer_threads=4,
    )


def add_episode(dataset: LeRobotDataset, episode_path: Path, instruction: str) -> None:
    with h5py.File(episode_path, "r") as root:
        state = np.asarray(root["flying_hand/actual_state"], dtype=np.float32)
        target = np.asarray(root["flying_hand/target_state"], dtype=np.float32)
        head = root["observation/head_camera/rgb"]
        wrist = root["observation/wrist_camera/rgb"]
        if state.shape != target.shape or state.ndim != 2 or state.shape[1] != 5:
            raise ValueError(f"Invalid Flying-Hand state/action shape in {episode_path}: {state.shape}, {target.shape}")
        if len(head) != len(state) or len(wrist) != len(state):
            raise ValueError(f"Image/state length mismatch in {episode_path}")
        action = np.concatenate([target[1:], target[-1:]], axis=0)
        for index in range(len(state)):
            dataset.add_frame({
                "observation.state": state[index],
                "action": action[index],
                "observation.images.head_camera": decode_rgb(head[index]),
                "observation.images.wrist_camera": decode_rgb(wrist[index]),
                "task": instruction,
            })
    dataset.save_episode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="data/flying_hand or one task directory")
    parser.add_argument("--setting", default="flying_hand_clean")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo id under HF_LEROBOT_HOME")
    parser.add_argument("--instruction-type", default="seen", choices=("seen", "unseen"))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.all_tasks:
        task_roots = sorted(path for path in args.input_root.iterdir() if (path / args.setting).is_dir())
    else:
        task_roots = [args.input_root]
    if not task_roots:
        raise FileNotFoundError(f"No Flying-Hand task directories found under {args.input_root}")

    dataset = create_dataset(args.repo_id, args.fps, overwrite=args.overwrite)
    for task_root in task_roots:
        setting_root = task_root / args.setting if (task_root / args.setting).is_dir() else task_root
        data_root = setting_root / "data"
        if not data_root.is_dir():
            raise FileNotFoundError(f"Missing HDF5 data directory: {data_root}")
        episode_paths = sorted(data_root.glob("episode*.hdf5"), key=lambda p: int(p.stem.removeprefix("episode")))
        for episode_path in episode_paths:
            instruction = read_instruction(
                setting_root / "instructions" / f"{episode_path.stem}.json",
                args.instruction_type,
                fallback=task_root.name.replace("_", " "),
            )
            add_episode(dataset, episode_path, instruction)
        print(f"Converted {task_root.name}: {len(episode_paths)} episodes")
    print(f"Saved LeRobot dataset to {Path(HF_LEROBOT_HOME) / args.repo_id}")


if __name__ == "__main__":
    main()
