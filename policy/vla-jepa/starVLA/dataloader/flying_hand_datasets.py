"""Read-only LeRobot v2.1 embedded-image adapter for VLA-JEPA.

The regular GR00T reader requires MP4s, modality.json and mutable index caches.
This adapter consumes the unified Flying-Hand export directly and emits the
same image/video/state/action/lang sample interface, without those requirements.
"""

import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

from .gr00t_lerobot.flying_hand_config import FlyingHandDataConfig
from .gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from .gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG


def collate_fn(batch):
    return batch


class FlyingHandDataset(Dataset):
    def __init__(self, data_cfg, action_horizon=8, video_horizon=8):
        mixture = DATASET_NAMED_MIXTURES[data_cfg.data_mix]
        if mixture != [("", 1.0, "flying_hand")]:
            raise ValueError("FlyingHandDataset requires the unified Flying-Hand mixture")
        self.root = Path(data_cfg.data_root_dir)
        self.data_config = FlyingHandDataConfig(list(range(video_horizon)), list(range(action_horizon)))
        self.tag = ROBOT_TYPE_TO_EMBODIMENT_TAG["flying_hand"].value
        self.action_horizon = action_horizon
        self.video_horizon = video_horizon
        if action_horizon < 1 or video_horizon < 2 or video_horizon % 2:
            raise ValueError("Action horizon must be positive; video horizon must be positive and even")
        self.resolution_size = int(data_cfg.get("resolution_size", 224))
        self.video_resolution_size = int(data_cfg.get("video_resolution_size", 256))
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info["codebase_version"] != "v2.1" or self.info["fps"] != 20:
            raise ValueError("Expected Flying-Hand LeRobot v2.1 at 20 Hz")
        for key in ("observation.state", "action"):
            feature = self.info["features"][key]
            names = feature.get("names")
            if feature["shape"] != [5] or names not in (self.data_config.vector_names, [self.data_config.vector_names]):
                raise ValueError(f"Unexpected Flying-Hand vector contract for {key}: {feature}")
        for key in self.data_config.image_keys:
            if self.info["features"][key]["dtype"] != "image":
                raise ValueError(f"Expected embedded LeRobot image feature: {key}")
        self.episodes = [json.loads(line) for line in (self.root / "meta/episodes.jsonl").read_text().splitlines()]
        self.tasks = {row["task_index"]: row["task"] for row in (
            json.loads(line) for line in (self.root / "meta/tasks.jsonl").read_text().splitlines()
        )}
        self.ends = np.cumsum([episode["length"] for episode in self.episodes])
        self.statistics = self._statistics()
        self._cached_episode = None
        self._cached_table = None

    def episode_path(self, episode):
        index = episode["episode_index"]
        return self.root / self.info["data_path"].format(
            episode_chunk=index // self.info["chunks_size"], episode_index=index,
        )

    def _statistics(self):
        # Read only 5D columns, never image payloads. Exact dataset-wide quantiles
        # keep training and BaseFramework.unnormalize_actions consistent.
        arrays = {"observation.state": [], "action": []}
        for episode in self.episodes:
            table = pq.read_table(self.episode_path(episode), columns=list(arrays))
            for key in arrays:
                arrays[key].append(np.asarray(table[key].to_pylist(), dtype=np.float32))
        result = {}
        for key, parts in arrays.items():
            values = np.concatenate(parts)
            stats = {
                "min": values.min(0), "max": values.max(0),
                "mean": values.mean(0), "std": values.std(0),
                "q01": np.quantile(values, 0.01, axis=0),
                "q99": np.quantile(values, 0.99, axis=0),
            }
            result["state" if key == "observation.state" else key] = {
                name: value.tolist() for name, value in stats.items()
            }
        result["action"]["mask"] = [True, True, True, True, False]
        result["action"]["binary_mask"] = [False, False, False, False, True]
        result["num_trajectories"] = len(self.episodes)
        result["num_transitions"] = int(self.ends[-1])
        return {self.tag: result}

    def save_dataset_statistics(self, path):
        Path(path).write_text(json.dumps(self.statistics, indent=2) + "\n")

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_pos = int(np.searchsorted(self.ends, index, side="right"))
        episode = self.episodes[episode_pos]
        start = index - (int(self.ends[episode_pos - 1]) if episode_pos else 0)
        if self._cached_episode != episode_pos:
            self._cached_table = pq.read_table(self.episode_path(episode))
            self._cached_episode = episode_pos
        table = self._cached_table
        # Absolute commands and frames repeat at the terminal boundary.
        action_indices = np.minimum(start + np.arange(self.action_horizon), episode["length"] - 1)
        video_indices = np.minimum(start + np.arange(self.video_horizon), episode["length"] - 1)
        action = np.asarray(table["action"].take(action_indices).to_pylist(), dtype=np.float32)
        stats = self.statistics[self.tag]["action"]
        low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        scale = high[:4] - low[:4]
        valid = scale != 0
        normalized = action[:, :4].copy()
        normalized[:, valid] = 2 * (normalized[:, valid] - low[:4][valid]) / scale[valid] - 1
        action[:, :4] = np.clip(normalized, -1, 1)
        state = np.asarray([table["observation.state"][start].as_py()], dtype=np.float32)
        videos, images = [], []
        for key in self.data_config.image_keys:
            frames = []
            for frame_index in video_indices:
                encoded = table[key][int(frame_index)].as_py()
                source = io.BytesIO(encoded["bytes"]) if encoded.get("bytes") is not None else self.root / encoded["path"]
                with Image.open(source) as raw:
                    rgb = raw.convert("RGB")
                    if not frames:
                        images.append(rgb.resize((self.resolution_size,) * 2, Image.Resampling.BILINEAR))
                    frames.append(np.asarray(rgb.resize((self.video_resolution_size,) * 2, Image.Resampling.BILINEAR)))
            videos.append(np.stack(frames))
        # The export fixes each episode to the first seen instruction.
        task_index = table["task_index"][start].as_py()
        language = self.tasks[task_index]
        if not language:
            raise ValueError(f"Empty instruction in episode {episode['episode_index']}")
        return dict(image=images, video=np.stack(videos), state=state, action=action, lang=language)
