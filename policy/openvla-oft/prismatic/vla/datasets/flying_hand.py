"""Local LeRobot v2.1 reader; no LeRobot, TensorFlow, model or network dependency.

Read this module by filename for dataset-only validation without importing prismatic.
Actions in the converted dataset already refer to target_state[t+1].
"""

import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info


DATASET_NAME = "flying_hand_direct_all_tasks"
TASKS = {
    "blocks_ranking_rgb", "blocks_ranking_size", "move_bottle", "move_can_pot",
    "place_can_basket", "place_fruit_skillet", "shake_bottle", "stack_blocks_two",
    "stack_blocks_two_size", "thread_ring_rod",
}
CAMERAS = ("observation.images.head_camera", "observation.images.wrist_camera")
FIELDS = ["x", "y", "z", "yaw", "grasp"]


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def chunk_indices(start, length, chunk_size):
    """Absolute targets are padded by holding the final target; never cross episodes."""
    return np.minimum(np.arange(start, start + chunk_size), length - 1)


def normalize(values, stats):
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    out = np.clip(2 * (values - low) / (high - low + 1e-8) - 1, -1, 1)
    out = np.where(np.asarray(stats["min"]) == np.asarray(stats["max"]), 0, out)
    return out.astype(np.float32)


class FlyingHandSource:
    """Validate metadata and read episode tables with embedded RGB image bytes."""

    def __init__(self, root):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        conversion = json.loads((self.root / "meta/flying_hand_conversion.json").read_text())
        self.conversion = conversion
        if self.info["codebase_version"] != "v2.1" or self.info["fps"] != 20:
            raise ValueError("Flying-Hand requires LeRobot v2.1 at 20 Hz")
        if conversion["setting"] != "flying_hand_direct_clean" or set(conversion["episodes_by_task"]) != TASKS:
            raise ValueError("Expected all 10 flying_hand_direct_clean tasks")
        if conversion.get("instruction_type") != "seen" or conversion.get("rgb_output") != "canonical_rgb":
            raise ValueError("Expected seen instructions and canonical RGB conversion")
        if (conversion.get("state") != "flying_hand/actual_state[t]" or
                conversion.get("action") != "flying_hand/target_state[t+1] (terminal frame repeats target[-1])"):
            raise ValueError("Unexpected state/action time alignment in conversion manifest")
        for key in ("observation.state", "action"):
            feature = self.info["features"][key]
            if feature["shape"] != [5] or feature["names"] not in (FIELDS, [FIELDS]):
                raise ValueError(f"Invalid {key} layout: {feature}")
        for key, shape in zip(CAMERAS, ([3, 360, 480], [3, 480, 640])):
            if self.info["features"][key]["dtype"] != "image" or self.info["features"][key]["shape"] != shape:
                raise ValueError(f"Expected embedded RGB images for {key}")
        self.episodes = read_jsonl(self.root / "meta/episodes.jsonl")
        self.tasks = {row["task_index"]: row["task"] for row in read_jsonl(self.root / "meta/tasks.jsonl")}
        if len(self.episodes) != self.info["total_episodes"]:
            raise ValueError("Episode count does not match metadata")
        if sum(row["length"] for row in self.episodes) != self.info["total_frames"]:
            raise ValueError("Frame count does not match metadata")
        if sum(conversion["episodes_by_task"].values()) != len(self.episodes):
            raise ValueError("Task episode counts do not match metadata")
        if any(count <= 0 for count in conversion["episodes_by_task"].values()):
            raise ValueError("Every task must contain episodes")
        ids = [row["episode_index"] for row in self.episodes]
        if len(set(ids)) != len(ids) or any(row["length"] <= 0 for row in self.episodes):
            raise ValueError("Duplicate or empty episodes")
        for episode in self.episodes:
            if not self.episode_path(episode).is_file():
                raise FileNotFoundError(self.episode_path(episode))

    def episode_path(self, episode):
        index = episode["episode_index"]
        return self.root / self.info["data_path"].format(
            episode_chunk=index // self.info["chunks_size"], episode_index=index
        )

    def read_episode(self, episode, images=True):
        columns = ["action", "observation.state", "timestamp", "frame_index", "episode_index", "task_index"]
        table = pq.read_table(self.episode_path(episode), columns=columns + (list(CAMERAS) if images else []))
        if len(table) != episode["length"]:
            raise ValueError(f"Wrong episode length: {self.episode_path(episode)}")
        result = {key: table[key].to_pylist() for key in columns}
        for key in ("action", "observation.state"):
            result[key] = np.asarray(result[key], dtype=np.float32)
            if result[key].shape != (len(table), 5) or not np.isfinite(result[key]).all():
                raise ValueError(f"Invalid {key} in episode {episode['episode_index']}")
        if not np.allclose(result["timestamp"], np.arange(len(table)) / 20, atol=1e-4):
            raise ValueError("Timestamps are not contiguous at 20 Hz")
        if result["frame_index"] != list(range(len(table))):
            raise ValueError("Frame indices are not contiguous")
        if any(index != episode["episode_index"] for index in result["episode_index"]):
            raise ValueError("Episode index mismatch")
        if any(index not in self.tasks or not self.tasks[index].strip() for index in result["task_index"]):
            raise ValueError("Missing language instruction")
        if images:
            for key in CAMERAS:
                result[key] = table[key]  # Keep compressed data; decode only selected frames.
        return result

    def statistics(self):
        arrays = {"action": [], "proprio": []}
        for episode in self.episodes:
            data = self.read_episode(episode, images=False)
            arrays["action"].append(data["action"])
            arrays["proprio"].append(data["observation.state"])
        stats = {"num_transitions": self.info["total_frames"], "num_trajectories": len(self.episodes)}
        for key, pieces in arrays.items():
            values = np.concatenate(pieces)
            stats[key] = {
                "mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
                "min": values.min(0).tolist(), "max": values.max(0).tolist(),
                "q01": np.quantile(values, 0.01, axis=0).tolist(),
                "q99": np.quantile(values, 0.99, axis=0).tolist(), "mask": [True] * 5,
            }
        return {DATASET_NAME: stats}

    def sample(self, data, index, chunk_size, statistics):
        observation = {"proprio": normalize(data["observation.state"][index:index + 1], statistics["proprio"])}
        for source, target in zip(CAMERAS, ("image_primary", "image_wrist")):
            encoded = data[source][index].as_py()
            if not encoded.get("bytes"):
                raise ValueError("Expected image bytes embedded in Parquet")
            with Image.open(io.BytesIO(encoded["bytes"])) as image:
                rgb = np.asarray(image.convert("RGB"))
            expected = self.info["features"][source]["shape"][1:] + [3]
            if list(rgb.shape) != expected:
                raise ValueError(f"Wrong image shape for {source}: {rgb.shape}")
            observation[target] = rgb[None]
        indices = chunk_indices(index, len(data["action"]), chunk_size)
        return {
            "dataset_name": DATASET_NAME,
            "observation": observation,
            "action": normalize(data["action"][indices], statistics["action"]),
            "task": {"language_instruction": self.tasks[data["task_index"][index]]},
        }


class FlyingHandDataset(IterableDataset):
    """Infinite episode/frame shuffle with disjoint rank/worker partitions per epoch."""

    def __init__(self, root, batch_transform, chunk_size=8, rank=0, world_size=1, seed=0):
        super().__init__()
        if chunk_size != 8:
            raise ValueError("Flying-Hand model constants require an 8-action chunk")
        self.source = FlyingHandSource(root)
        self.batch_transform, self.chunk_size = batch_transform, chunk_size
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.dataset_statistics = self.source.statistics()

    def __len__(self):
        return self.source.info["total_frames"]

    def __iter__(self):
        worker = get_worker_info()
        workers, worker_id = (worker.num_workers, worker.id) if worker else (1, 0)
        shards, shard = self.world_size * workers, self.rank * workers + worker_id
        if len(self.source.episodes) < shards:
            raise ValueError("More data workers than episodes")
        epoch = 0
        while True:
            order = np.random.default_rng(self.seed + epoch).permutation(len(self.source.episodes))
            rng = np.random.default_rng(self.seed + epoch * shards + shard)
            for episode_index in order[shard::shards]:
                data = self.source.read_episode(self.source.episodes[episode_index])
                for frame in rng.permutation(len(data["action"])):
                    yield self.batch_transform(self.source.sample(
                        data, frame, self.chunk_size, self.dataset_statistics[DATASET_NAME]
                    ))
            epoch += 1
