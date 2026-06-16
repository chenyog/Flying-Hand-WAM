import hashlib
import json
import os
from pathlib import Path
from io import BytesIO

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json, save_dataset_stats_to_json
from fastwam.utils import misc
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


def _require_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`h5py` is required to read RoboTwin flying-hand HDF5 data. "
            "Install h5py in the active FastWAM Python environment."
        ) from exc
    return h5py


class FlyingHandHDF5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        action_video_freq_ratio=4,
        video_size=[384, 320],
        camera_keys=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.01,
        is_training_set=False,
        instruction_json=None,
        instruction_type="seen",
        scene_info_name="scene_info.json",
    ):
        self.dataset_dirs = [Path(str(p)).expanduser() for p in dataset_dirs]
        self.shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        self.num_frames = int(num_frames)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.video_size = video_size
        self.camera_keys = list(camera_keys or ["head_camera", "wrist_camera"])
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = int(context_len)
        self.instruction_json = Path(str(instruction_json)).expanduser() if instruction_json else None
        self.instruction_type = str(instruction_type)
        self.is_training_set = bool(is_training_set)
        self.video_sample_indices = list(range(0, self.num_frames, self.action_video_freq_ratio))
        if (self.num_frames - 1) % self.action_video_freq_ratio:
            raise ValueError("num_frames-1 must be divisible by action_video_freq_ratio.")
        if ((self.num_frames - 1) // self.action_video_freq_ratio) % 4:
            raise ValueError("video frames must be divisible by 4 for tokenization.")
        self.episodes = self._episodes()
        self.indices = self._indices(float(val_set_proportion))
        self.prompts = self._prompts(scene_info_name)
        self.processor = None

        if processor is None:
            return
        processor = instantiate(processor) if isinstance(processor, DictConfig) else processor
        if pretrained_norm_stats:
            stats = load_dataset_stats_from_json(pretrained_norm_stats)
            logger.info("Using dataset stats: %s", pretrained_norm_stats)
        else:
            if not self.is_training_set:
                raise ValueError("pretrained_norm_stats must be provided for validation/test sets.")
            stats = self.get_dataset_stats(processor)
        if PartialState().is_main_process:
            save_dataset_stats_to_json(stats, os.path.join(misc.get_work_dir(), "dataset_stats.json"))
        processor.set_normalizer_from_stats(stats)
        self.set_processor(processor)

    def _episodes(self):
        eps = []
        h5py = _require_h5py()
        for root in self.dataset_dirs:
            data_dir = root / "data"
            if not data_dir.is_dir():
                raise FileNotFoundError(f"Missing flying-hand data dir: {data_dir}")
            for path in sorted(data_dir.glob("episode*.hdf5"), key=lambda p: int(p.stem.removeprefix("episode"))):
                with h5py.File(path, "r") as f:
                    for key in (
                        "observation/head_camera/rgb",
                        "observation/wrist_camera/rgb",
                        "flying_hand/state",
                    ):
                        if key not in f:
                            raise KeyError(f"Missing `{key}` in {path}")
                    length = int(f["flying_hand/state"].shape[0])
                eps.append({"path": path, "root": root, "episode": int(path.stem.removeprefix("episode")), "length": length})
        if not eps:
            raise FileNotFoundError(f"No episode*.hdf5 found under {self.dataset_dirs}")
        return eps

    def _indices(self, val_set_proportion):
        order = np.arange(len(self.episodes))
        np.random.default_rng(42).shuffle(order)
        split = int(len(order) * (1 - val_set_proportion))
        picked = set(order[:split] if self.is_training_set else order[split:])
        return [
            (ep_idx, start)
            for ep_idx, ep in enumerate(self.episodes)
            if ep_idx in picked
            for start in range(0, ep["length"] - self.num_frames + 1)
        ]

    def _prompts(self, scene_info_name):
        if self.instruction_json is None:
            return {id(ep): DEFAULT_PROMPT.format(task="Use the flying hand to grab the bottle and pull it out from the board.") for ep in self.episodes}
        payload = json.loads(self.instruction_json.read_text(encoding="utf-8"))
        templates = payload[self.instruction_type]
        prompts = {}
        for ep in self.episodes:
            info = json.loads((ep["root"] / scene_info_name).read_text(encoding="utf-8"))[f"episode_{ep['episode']}"]["info"]
            task = templates[ep["episode"] % len(templates)]
            for key, value in info.items():
                task = task.replace(key, value)
            prompts[id(ep)] = DEFAULT_PROMPT.format(task=task)
        return prompts

    @staticmethod
    def _state(f):
        return torch.from_numpy(f["flying_hand/state"][:]).float()

    @staticmethod
    def _decode(raw):
        img = Image.open(BytesIO(bytes(raw).rstrip(b"\0"))).convert("RGB")
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)

    def _video(self, f, start):
        frames = []
        for i in [start + j for j in self.video_sample_indices]:
            head = transforms_F.resize(self._decode(f["observation/head_camera/rgb"][i]), [256, 320], antialias=True)
            wrist = transforms_F.resize(self._decode(f["observation/wrist_camera/rgb"][i]), [128, 320], antialias=True)
            frames.append(torch.cat([head, wrist], dim=-2))
        video = torch.stack(frames).float()
        if list(video.shape[-2:]) != list(self.video_size):
            video = transforms_F.resize(video, self.video_size, antialias=True)
        return (video * (2.0 / 255.0) - 1.0).permute(1, 0, 2, 3)

    def _context(self, prompt):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        path = Path(self.text_embedding_cache_dir) / f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}.t5_len{self.context_len}.wan22ti2v5b.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing text embedding cache: {path}. Run scripts/precompute_text_embeds.py first.")
        payload = torch.load(path, map_location="cpu")
        return payload["context"], torch.ones_like(payload["mask"].bool())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start = self.indices[idx]
        ep = self.episodes[ep_idx]
        h5py = _require_h5py()
        with h5py.File(ep["path"], "r") as f:
            state = self._state(f)[start:start + self.num_frames]
            action = state[1:]
            proprio = state[:-1]
            sample = {
                "video": self._video(f, start),
                "prompt": self.prompts[id(ep)],
                "image_is_pad": torch.zeros(len(self.video_sample_indices), dtype=torch.bool),
                "action_is_pad": torch.zeros(self.num_frames - 1, dtype=torch.bool),
                "proprio_is_pad": torch.zeros(self.num_frames - 1, dtype=torch.bool),
            }
        if self.processor is None:
            sample["action"] = action
            sample["proprio"] = proprio
            sample["action_dim_is_pad"] = torch.zeros(action.shape[-1], dtype=torch.bool)
            sample["proprio_dim_is_pad"] = torch.zeros(proprio.shape[-1], dtype=torch.bool)
        else:
            batch = self.processor.action_state_transform({
                "action": {"default": action},
                "state": {"default": proprio},
            })
            batch = self.processor.normalizer.forward(batch)
            batch = self.processor.action_state_merger.forward(batch)
            sample["action"] = batch["action"]
            sample["proprio"] = batch["state"]
            sample["action_dim_is_pad"] = batch["action_dim_is_pad"]
            sample["proprio_dim_is_pad"] = batch["state_dim_is_pad"]
        sample["context"], sample["context_mask"] = self._context(sample["prompt"])
        return sample

    def set_processor(self, processor):
        self.processor = processor.train() if self.is_training_set else processor.eval()
        return self

    def get_dataset_stats(self, preprocessor):
        action = []
        state = []
        h5py = _require_h5py()
        for ep in self.episodes:
            with h5py.File(ep["path"], "r") as f:
                x = self._state(f)
            state.append(x[:-1])
            action.append(x[1:])
        return self._stats(torch.cat(state), torch.stack(action), preprocessor)

    def _stats(self, state, action, preprocessor):
        batch = preprocessor.action_state_transform({
            "state": {"default": state},
            "action": {"default": action.reshape(-1, action.shape[-1])},
        })
        return {
            "num_episodes": len(self.episodes),
            "num_transition": int(sum(ep["length"] - 1 for ep in self.episodes)),
            "state": {"default": self._field_stats(batch["state"]["default"], None)},
            "action": {"default": self._field_stats(batch["action"]["default"], action)},
        }

    @staticmethod
    def _field_stats(x, step_x):
        y = x.unsqueeze(0) if step_x is None else step_x
        return {
            "global_min": x.amin(0),
            "global_max": x.amax(0),
            "global_mean": x.mean(0),
            "global_std": x.std(0),
            "global_q01": torch.quantile(x, 0.01, dim=0),
            "global_q99": torch.quantile(x, 0.99, dim=0),
            "stepwise_min": y.amin(0),
            "stepwise_max": y.amax(0),
            "stepwise_mean": y.mean(0),
            "stepwise_std": y.std(0, unbiased=False),
            "stepwise_q01": torch.quantile(y, 0.01, dim=0),
            "stepwise_q99": torch.quantile(y, 0.99, dim=0),
        }
