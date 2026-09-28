"""Check real samples without loading/downloading models or initializing W&B."""

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.flying_hand_datasets import FlyingHandDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.datasets.vla_data.data_root_dir = str(args.data_root)
    action_cfg = cfg.framework.action_model
    assert action_cfg.action_dim == action_cfg.state_dim == 5
    assert action_cfg.action_horizon == action_cfg.future_action_window_size + 1
    assert action_cfg.action_horizon == cfg.datasets.vla_data.action_horizon
    assert cfg.trainer.repeated_diffusion_steps == action_cfg.repeated_diffusion_steps == 8
    dataset = FlyingHandDataset(cfg.datasets.vla_data, action_cfg.action_horizon, cfg.framework.vj2_model.num_frames)
    expected_video = (2, cfg.framework.vj2_model.num_frames, cfg.datasets.vla_data.video_resolution_size, cfg.datasets.vla_data.video_resolution_size, 3)
    # Check the beginning plus an episode boundary where the chunk must repeat.
    for index in (0, int(dataset.ends[0]) - 1):
        sample = dataset[index]
        assert len(sample["image"]) == 2
        assert all(image.mode == "RGB" for image in sample["image"])
        assert sample["video"].shape == expected_video
        assert sample["state"].shape == (1, 5)
        assert sample["action"].shape == (action_cfg.action_horizon, 5)
        assert np.isfinite(sample["action"]).all() and np.isfinite(sample["state"]).all()
        assert sample["lang"] in dataset.episodes[0]["tasks"]
        assert np.all((sample["action"][:, 4] >= 0) & (sample["action"][:, 4] <= 1))
        if index:
            assert np.all(sample["action"] == sample["action"][0])
            assert np.all(sample["video"] == sample["video"][:, :1])
    print(json.dumps({
        "episodes": len(dataset.episodes), "frames": len(dataset), "fps": dataset.info["fps"],
        "instruction_entries": len(dataset.tasks), "image_views": len(sample["image"]),
        "video_shape": list(sample["video"].shape), "state_shape": list(sample["state"].shape),
        "action_shape": list(sample["action"].shape), "instruction": sample["lang"],
        "terminal_padding": "repeat_last_passed", "model_loaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
