"""Render 20 Hz Flying-Hand videos from lightweight evaluation histories.

Large evaluations save actions and state trajectories instead of encoding RGB
inside the policy process.  This tool reconstructs the same seeded scene and
replays the executed action chunks through the shared Flying-Hand executor.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import load_task_class
from envs.flying_hand.eval_control import execute_action_chunk, reset_execution_state


def _history_paths(inputs: list[Path]) -> list[Path]:
    paths = []
    for value in inputs:
        value = value.expanduser().resolve()
        if value.is_dir():
            paths.extend(value.rglob("episode*_history.npz"))
        elif value.suffix == ".json":
            paths.append(value.with_suffix(".npz"))
        else:
            paths.append(value)
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    def episode_key(path: Path):
        match = re.search(r"episode(\d+)_history$", path.stem)
        if match:
            return (str(path.parent), 0, int(match.group(1)))
        return (str(path.parent), 1, path.stem)

    return sorted(unique, key=episode_key)


def _open_video_writers(output_dir: Path, episode: int, video_sizes: dict, fps: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    for camera_name, video_size in video_sizes.items():
        safe_name = camera_name.replace("/", "_")
        output_path = output_dir / f"episode{episode}_{safe_name}.mp4"
        writers[camera_name] = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", str(video_size), "-framerate", str(fps),
                "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264",
                "-crf", "23", str(output_path),
            ],
            stdin=subprocess.PIPE,
        )
    return writers


def rerender(history_path: Path, output_dir: Path | None) -> None:
    if not history_path.is_file():
        raise FileNotFoundError(history_path)
    metadata_path = history_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    format_version = int(metadata.get("format_version", 0))
    if format_version not in {1, 2}:
        raise ValueError(f"Unsupported history format in {metadata_path}")
    if format_version >= 2:
        render_contract = {
            "render_shader": "rt",
            "render_samples_per_pixel": 32,
            "render_path_depth": 8,
            "render_denoiser": "oidn",
        }
        mismatches = {
            key: metadata.get(key)
            for key, expected in render_contract.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"History does not match the Flying-Hand RT/OIDN contract: {mismatches}"
            )

    task_config_path = PROJECT_ROOT / "task_config" / f"{metadata['task_config']}.yml"
    task_args = yaml.safe_load(task_config_path.read_text(encoding="utf-8"))
    task_args.update({
        "task_name": metadata["task_name"],
        "task_config": metadata["task_config"],
        "task_namespace": "flying_hand",
        "eval_mode": True,
        "eval_video_save_dir": str(output_dir or history_path.parent),
        "timestep": float(metadata["timestep"]),
        "save_freq": int(metadata["save_freq"]),
        "enable_dynamics": bool(metadata["enable_dynamics"]),
        "render_shader": str(metadata["render_shader"]),
        "render_samples_per_pixel": int(metadata["render_samples_per_pixel"]),
        "render_path_depth": int(metadata["render_path_depth"]),
        "render_denoiser": str(metadata["render_denoiser"]),
        "render_freq": 0,
    })

    task_class = load_task_class(f"flying_hand/{metadata['task_name']}")
    task_env = task_class()
    final_output_dir = (output_dir or history_path.parent).resolve()
    writers = None
    try:
        task_env.setup_demo(
            now_ep_num=int(metadata["episode"]),
            seed=int(metadata["seed"]),
            is_test=True,
            **task_args,
        )
        task_env.set_instruction(str(metadata["instruction"]))
        task_env.eval_video_path = str(final_output_dir)
        task_env.eval_video_frame_limit = None
        task_env.eval_video_frames_written = 0
        writers = _open_video_writers(
            final_output_dir,
            int(metadata["episode"]),
            metadata["video_sizes"],
            float(metadata["fps"]),
        )
        task_env._set_eval_video_ffmpeg(writers)

        replay_model = SimpleNamespace()
        reset_execution_state(replay_model)
        task_env.capture_flying_hand_eval_waypoints = True
        # Re-record the lightweight state trajectory while replaying. A full
        # replay render is accepted only when the seeded simulator execution
        # matches the trajectory saved by the policy process.
        task_env.record_flying_hand_eval_history = True
        task_env.flying_hand_eval_action_chunks = []
        task_env.flying_hand_eval_actual_states = [
            np.asarray(task_env._get_flying_hand_actual_state(), dtype=np.float32)
        ]
        task_env.flying_hand_eval_target_states = [
            np.asarray(task_env._get_flying_hand_target_state(), dtype=np.float32)
        ]

        # Initial scene frame, followed by exactly one frame per 20 Hz action.
        task_env.get_obs()
        with np.load(history_path) as history:
            actions = np.asarray(history["actions"], dtype=np.float32)
            chunk_lengths = np.asarray(history["chunk_lengths"], dtype=np.int64)
            expected_actual = np.asarray(history["actual_states"], dtype=np.float32)
            expected_target = np.asarray(history["target_states"], dtype=np.float32)
        if int(chunk_lengths.sum()) != len(actions):
            raise ValueError(f"Corrupt chunk lengths in {history_path}")
        offset = 0
        for chunk_length in chunk_lengths:
            if task_env.eval_video_ffmpeg is None:
                break
            chunk = actions[offset:offset + int(chunk_length)]
            execute_action_chunk(task_env, replay_model, chunk)
            offset += int(chunk_length)
        replay_actual = np.asarray(
            task_env.flying_hand_eval_actual_states, dtype=np.float32
        )
        replay_target = np.asarray(
            task_env.flying_hand_eval_target_states, dtype=np.float32
        )
        if replay_actual.shape != expected_actual.shape:
            raise ValueError(
                "Replay actual-state length mismatch: "
                f"saved={expected_actual.shape}, replay={replay_actual.shape}"
            )
        if replay_target.shape != expected_target.shape:
            raise ValueError(
                "Replay target-state length mismatch: "
                f"saved={expected_target.shape}, replay={replay_target.shape}"
            )
        np.testing.assert_allclose(
            replay_actual,
            expected_actual,
            rtol=1.0e-5,
            atol=1.0e-5,
            err_msg="Replay actual states diverged from policy evaluation",
        )
        np.testing.assert_allclose(
            replay_target,
            expected_target,
            rtol=1.0e-5,
            atol=1.0e-5,
            err_msg="Replay target states diverged from policy evaluation",
        )
    finally:
        if getattr(task_env, "eval_video_ffmpeg", None) is not None:
            task_env._del_eval_video_ffmpeg()
        task_env.close_env()
    print(
        f"Rendered {metadata['task_name']} episode {metadata['episode']} at "
        f"{metadata['fps']} Hz to {final_output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "history",
        nargs="+",
        type=Path,
        help="episode*_history.npz file(s), matching JSON file(s), or directories",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = _history_paths(args.history)
    if not paths:
        parser.error("No episode*_history.npz files found")
    for path in paths:
        rerender(path, args.output_dir)


if __name__ == "__main__":
    main()
