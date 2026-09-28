"""Dataset-only preflight: no model imports, GPU initialization or downloads."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_reader():
    path = Path(__file__).resolve().parents[1] / "prismatic/vla/datasets/flying_hand.py"
    spec = importlib.util.spec_from_file_location("flying_hand_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--all-episodes", action="store_true", help="Validate all numeric frames and decode both endpoint images per episode")
    parser.add_argument("--source-root", type=Path, help="Also compare one original HDF5 episode from every task")
    args = parser.parse_args()
    reader = load_reader()
    source = reader.FlyingHandSource(args.data_root)
    # Use local sample statistics for shape tests; training computes full-dataset statistics.
    task_first_episodes = []
    offset = 0
    for task, count in sorted(source.conversion["episodes_by_task"].items()):
        task_first_episodes.append((task, source.episodes[offset]))
        offset += count
    episodes = source.episodes if args.all_episodes else [episode for _, episode in task_first_episodes]
    for episode in episodes:
        data = source.read_episode(episode)
        stats = {}
        for key, column in (("action", "action"), ("proprio", "observation.state")):
            values = data[column]
            stats[key] = {"q01": values.min(0), "q99": values.max(0), "min": values.min(0), "max": values.max(0)}
        for frame in (0, episode["length"] - 1):
            sample = source.sample(data, frame, 8, stats)
            assert sample["action"].shape == (8, 5)
            assert sample["observation"]["proprio"].shape == (1, 5)
            assert sample["observation"]["image_primary"].shape == (1, 360, 480, 3)
            assert sample["observation"]["image_wrist"].shape == (1, 480, 640, 3)
            assert np.isfinite(sample["action"]).all()
        np.testing.assert_array_equal(sample["action"], np.repeat(sample["action"][:1], 8, axis=0))
    if args.source_root:
        import h5py

        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "envs/utils"))
        from rgb_codec import decode_hdf5_rgb, require_current_rgb_jpeg_codec

        for task, episode in task_first_episodes:
            raw_root = args.source_root / task / "flying_hand_direct_clean"
            instruction = json.loads((raw_root / "instructions/episode0.json").read_text())["seen"][0]
            data = source.read_episode(episode)
            with h5py.File(raw_root / "data/episode0.hdf5", "r") as raw:
                require_current_rgb_jpeg_codec(raw)
                np.testing.assert_array_equal(data["observation.state"], raw["flying_hand/actual_state"][:])
                targets = raw["flying_hand/target_state"][:]
                np.testing.assert_array_equal(data["action"], np.concatenate([targets[1:], targets[-1:]]))
                assert all(source.tasks[index] == instruction for index in data["task_index"])
                for frame in (0, len(targets) - 1):
                    sample = source.sample(data, frame, 8, stats)
                    for camera, target in (("head_camera", "image_primary"), ("wrist_camera", "image_wrist")):
                        rgb = decode_hdf5_rgb(raw[f"observation/{camera}/rgb"][frame])
                        np.testing.assert_array_equal(sample["observation"][target][0], rgb)
    print(json.dumps({
        "dataset": reader.DATASET_NAME, "tasks": sorted(reader.TASKS),
        "episodes": len(source.episodes), "frames": source.info["total_frames"],
        "episodes_checked": len(episodes), "fps": 20, "action_chunk": [8, 5],
        "proprio": [5], "cameras": list(reader.CAMERAS),
        "source_episodes_compared": len(task_first_episodes) if args.source_root else 0, "status": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
