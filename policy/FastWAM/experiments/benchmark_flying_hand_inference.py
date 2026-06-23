#!/usr/bin/env python3
"""Benchmark one FastWAM Flying-Hand policy inference call."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.robotwin.deploy_policy import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure FastWAM Flying-Hand infer_action latency.")
    parser.add_argument(
        "--ckpt",
        default=ROOT / "runs/flying_hand_1cam320_1e-4/2026-06-17_20-14-30/checkpoints/weights/step_012000.pt",
        type=Path,
    )
    parser.add_argument("--dataset-stats-path", default=ROOT / "data/flying_hand_lerobot/dataset_stats.json", type=Path)
    parser.add_argument("--config-name", default="sim_flying_hand")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default="move the bottle")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

    policy = get_model(
        {
            "sim_cfg_name": args.config_name,
            "ckpt_setting": str(args.ckpt),
            "dataset_stats_path": str(args.dataset_stats_path),
            "device": args.device,
            "num_inference_steps": args.num_inference_steps,
            "action_horizon": args.action_horizon,
            "seed": args.seed,
        }
    )
    obs = {
        "observation": {
            "wrist_camera": {"rgb": np.zeros((480, 640, 3), dtype=np.uint8)},
            "head_camera": {"rgb": np.zeros((480, 640, 3), dtype=np.uint8)},
        },
        "flying_hand": {"actual_state": np.zeros(5, dtype=np.float32)},
    }

    times = []
    total = args.warmup + args.iters
    for i in range(total):
        if torch.cuda.is_available() and str(policy.model.device).startswith("cuda"):
            torch.cuda.synchronize(policy.model.device)
        t0 = time.perf_counter()
        action = policy._infer_action_chunk(obs, args.task)
        if torch.cuda.is_available() and str(policy.model.device).startswith("cuda"):
            torch.cuda.synchronize(policy.model.device)
        dt = time.perf_counter() - t0
        if i >= args.warmup:
            times.append(dt)
        print(f"{'warmup' if i < args.warmup else 'iter'} {i + 1:02d}/{total}: {dt:.4f}s")

    arr = np.asarray(times, dtype=np.float64)
    print(
        "\n".join(
            [
                f"action_shape={tuple(action.shape)}",
                f"iters={args.iters} warmup={args.warmup} num_inference_steps={args.num_inference_steps}",
                f"mean={arr.mean():.4f}s median={np.median(arr):.4f}s p90={np.quantile(arr, 0.9):.4f}s",
                f"min={arr.min():.4f}s max={arr.max():.4f}s",
            ]
        )
    )


if __name__ == "__main__":
    main()
