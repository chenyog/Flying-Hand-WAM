import logging
import os
import sys
import time
import inspect
import hashlib
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

logger = logging.getLogger(__name__)
CAMERA_KEYS = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            return cfg_path.relative_to(configs_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}") from exc
    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        return compose(
            config_name=config_name,
            overrides=[] if _is_none_like(sim_task) else [f"task={str(sim_task)}"],
        )


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resolve_optional_path(path_value: Any) -> Optional[Path]:
    if _is_none_like(path_value):
        return None
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _get_observation_rgb(obs_data: Dict[str, Any], config_key: str) -> np.ndarray:
    obs_key = CAMERA_KEYS.get(config_key, config_key)
    camera_data = obs_data[obs_key]
    image = np.asarray(camera_data["rgb"])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image [H,W,3] for `{obs_key}`, got shape {image.shape}")
    return image


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        image_meta: list[Dict[str, Any]],
        concat_multi_camera: Optional[str],
        video_size: list[int],
        text_embedding_cache_dir: Optional[Path],
        context_len: int,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = text_embedding_cache_dir is None

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self.image_meta = image_meta
        self.concat_multi_camera = concat_multi_camera
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = int(context_len)
        if len(video_size) != 2:
            raise ValueError(f"`video_size` must be [H,W], got: {video_size}")
        self.video_size = [int(video_size[0]), int(video_size[1])]
        self._model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        self._model_dtype = model_dtype
        self._lazy_text_encoder = None
        self._lazy_tokenizer = None

        # Keep each replan window contiguous so grasp edges and waypoint
        # derivatives are evaluated in the model's original 20 Hz sequence.
        self.pending_actions: deque[np.ndarray] = deque()
        self.attached_actor = None
        self.attached_pose = None
        self.grasp_commanded = False
        self.gripper_state = "open"
        self.grasp_diagnostics = []
        self.action_diagnostics = self._empty_action_diagnostics()
        self.waypoint_diagnostics = []
        self.flight_diagnostics = self._empty_flight_diagnostics()
        self._waypoint_reference_position = None
        self._waypoint_reference_velocity = np.zeros(3)
        self._waypoint_reference_orientation = None
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )

    def _text_cache_path(self, prompt: str) -> Path:
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return self.text_embedding_cache_dir / f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"

    def _save_text_context_cache(
        self,
        cache_path: Path,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.parent / f".{cache_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
        torch.save(
            {
                "context": context.detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                "mask": context_mask.detach().to(device="cpu", dtype=torch.bool).contiguous(),
            },
            tmp_path,
        )
        os.replace(tmp_path, cache_path)

    def _load_lazy_text_encoder(self):
        if self._lazy_text_encoder is not None and self._lazy_tokenizer is not None:
            return self._lazy_text_encoder, self._lazy_tokenizer

        model_id = str(self._model_cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"))
        tokenizer_model_id = str(self._model_cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"))
        redirect_common_files = bool(self._model_cfg.get("redirect_common_files", True))
        _, text_config, _, tokenizer_config = _resolve_configs(
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            redirect_common_files=redirect_common_files,
        )
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()
        self._lazy_text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=self._model_dtype,
            device="cpu",
        ).eval()
        self._lazy_tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=self.context_len,
            clean="whitespace",
        )
        return self._lazy_text_encoder, self._lazy_tokenizer

    def _encode_text_context(self, prompt: str, cache_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model.text_encoder is not None and self.model.tokenizer is not None:
            context, context_mask = self.model.encode_prompt(prompt)
            context = context.squeeze(0).detach().to(device="cpu", dtype=torch.bfloat16)
            context_mask = context_mask.squeeze(0).detach().to(device="cpu", dtype=torch.bool)
        else:
            text_encoder, tokenizer = self._load_lazy_text_encoder()
            ids, context_mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
            context_mask = context_mask.to(dtype=torch.bool)
            with torch.no_grad():
                context = text_encoder(ids, context_mask)[0].detach().to(device="cpu", dtype=torch.bfloat16)
            context_mask = context_mask[0].detach().to(device="cpu", dtype=torch.bool)
        self._save_text_context_cache(cache_path, context, context_mask)
        logger.info("Encoded and cached missing text embedding: %s", cache_path)
        return context, context_mask

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cache_path = self._text_cache_path(prompt)
        if not cache_path.exists():
            context, context_mask = self._encode_text_context(prompt, cache_path)
        else:
            payload = torch.load(cache_path, map_location="cpu")
            context = payload["context"]
            context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(f"Cached `context` must be 2D [L,D], got {tuple(context.shape)} in {cache_path}")
        if context_mask.ndim != 1:
            raise ValueError(f"Cached `mask` must be 1D [L], got {tuple(context_mask.shape)} in {cache_path}")
        if context.shape[0] != self.context_len or context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached text len mismatch: expected {self.context_len}, "
                f"got context={context.shape[0]} mask={context_mask.shape[0]} in {cache_path}"
            )
        context[~context_mask] = 0.0
        return context, torch.ones_like(context_mask)

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    @staticmethod
    def _flying_hand_relative_xyzyaw_to_world_pose(task_env, action: np.ndarray):
        import sapien

        relative_pose = sapien.Pose(
            action[:3].tolist(),
            [np.cos(action[3] / 2), 0, 0, np.sin(action[3] / 2)],
        )
        initial_imu_odom_pose = task_env.initial_imu_odom_pose
        initial_root_pose = task_env.flying_hand_initial_pose
        root_to_imu_initial = initial_root_pose.inv() * initial_imu_odom_pose
        return initial_imu_odom_pose * relative_pose * root_to_imu_initial.inv()

    def _advance_flying_hand_reference(
        self,
        task_env,
        target_pose,
        target_velocity,
        target_acceleration,
        carried_actor=None,
        carried_pose=None,
    ) -> None:
        from envs.flying_hand import planner

        if carried_actor is not None:
            planner.begin_isolated_carry(task_env, carried_actor)
        task_env.flying_hand_ref_pose = target_pose
        if task_env.enable_dynamics:
            hand_pose, hand_v = task_env.flying_hand_dynamics.step(
                target_pose,
                target_velocity,
                target_acceleration,
                task_env.is_grasping,
            )
            task_env.flying_hand.set_root_pose(hand_pose)
            task_env.flying_hand.set_root_linear_velocity(hand_v.tolist())
            task_env.flying_hand.set_root_angular_velocity(
                task_env.flying_hand_dynamics.w.tolist()
            )
        else:
            hand_pose, hand_v = target_pose, np.asarray(target_velocity, dtype=float)
            planner.set_pose(task_env, hand_pose, hand_v)
        if carried_actor is not None:
            planner.set_isolated_carried_actor_target(
                task_env,
                carried_actor,
                hand_pose * carried_pose,
            )
        planner.step(
            task_env,
            1,
            save_freq=None,
            step_callback=self._record_flight_sample,
        )

    def _build_image_array(self, observation: Dict[str, Any]) -> np.ndarray:
        obs_data = observation["observation"]
        camera_images = []
        for meta in self.image_meta:
            key = str(meta["key"])
            shape = list(meta["shape"])
            camera_images.append(_resize_rgb(_get_observation_rgb(obs_data, key), (int(shape[2]), int(shape[1]))))

        if self.concat_multi_camera == "robotwin":
            if len(camera_images) != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {len(camera_images)}"
                )
            top = _resize_rgb(camera_images[0], (320, 256))
            left = _resize_rgb(camera_images[1], (160, 128))
            right = _resize_rgb(camera_images[2], (160, 128))
            image = np.concatenate([top, np.concatenate([left, right], axis=1)], axis=0)
        elif len(camera_images) > 1:
            if self.concat_multi_camera == "horizontal":
                image = np.concatenate(camera_images, axis=1)
            elif self.concat_multi_camera == "vertical":
                image = np.concatenate(camera_images, axis=0)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin, or null."
                )
        else:
            image = camera_images[0]

        target_h, target_w = self.video_size
        if image.shape[:2] != (target_h, target_w):
            image = _resize_rgb(image, (target_w, target_h))
        return image

    def _build_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        image = self._build_image_array(observation)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_image_tensor(observation)
        state_vector = (
            observation["flying_hand"]["actual_state"]
            if int(self.processor.shape_meta["action"][0]["shape"]) == 5
            else observation["joint_action"]["vector"]
        )
        proprio = self._normalize_state(np.asarray(state_vector, dtype=np.float32))

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if self.text_embedding_cache_dir is None:
            infer_kwargs["prompt"] = prompt
        else:
            context, context_mask = self._get_cached_text_context(prompt)
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = context
            infer_kwargs["context_mask"] = context_mask
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(self, task_env, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        # In this evaluator inference is synchronous: SAPIEN is paused while
        # the model runs.  Wall-clock inference latency therefore must not add
        # extra open-loop simulation actions.
        n = min(action_chunk.shape[0], self.replan_steps)
        self.pending_actions.append(np.asarray(action_chunk[:n], dtype=np.float32))

    @staticmethod
    def _empty_flight_diagnostics() -> Dict[str, Any]:
        return {
            "samples": 0,
            "max_abs_roll_rad": 0.0,
            "max_abs_pitch_rad": 0.0,
            "max_abs_pitch_rate_rad_s": 0.0,
            "max_abs_bodyrate_rad_s": [0.0, 0.0, 0.0],
            "rotor_saturation_samples": 0,
            "large_pitch_excursions": 0,
            "large_pitch_events": [],
            "_large_pitch_active": False,
            "max_position_error_m": 0.0,
        }

    @staticmethod
    def _empty_action_diagnostics() -> Dict[str, Any]:
        return {
            "samples": 0,
            "grasp_min": None,
            "grasp_max": None,
            "above_close_threshold": 0,
            "below_open_threshold": 0,
            "chunk_grasp_ranges": [],
        }

    def _record_action_chunk(self, task_env, actions: np.ndarray) -> None:
        values = np.asarray(actions[:, 4], dtype=float)
        if values.size == 0:
            return
        close_threshold = float(task_env.flying_hand_grasp_validation["close_threshold"])
        open_threshold = float(task_env.flying_hand_grasp_validation["open_threshold"])
        diagnostics = self.action_diagnostics
        diagnostics["samples"] += int(values.size)
        chunk_min = float(values.min())
        chunk_max = float(values.max())
        diagnostics["grasp_min"] = (
            chunk_min
            if diagnostics["grasp_min"] is None
            else min(diagnostics["grasp_min"], chunk_min)
        )
        diagnostics["grasp_max"] = (
            chunk_max
            if diagnostics["grasp_max"] is None
            else max(diagnostics["grasp_max"], chunk_max)
        )
        diagnostics["above_close_threshold"] += int(np.count_nonzero(values >= close_threshold))
        diagnostics["below_open_threshold"] += int(np.count_nonzero(values <= open_threshold))
        diagnostics["chunk_grasp_ranges"].append([chunk_min, chunk_max])

    def _record_flight_sample(self, task_env) -> None:
        q = np.asarray(task_env.flying_hand.get_root_pose().q, dtype=float)
        angular_velocity = np.asarray(
            task_env.flying_hand.get_root_angular_velocity(),
            dtype=float,
        )
        w, x, y, z = q
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        diagnostics = self.flight_diagnostics
        diagnostics["samples"] += 1
        diagnostics["max_abs_roll_rad"] = max(diagnostics["max_abs_roll_rad"], abs(float(roll)))
        diagnostics["max_abs_pitch_rad"] = max(diagnostics["max_abs_pitch_rad"], abs(float(pitch)))
        pitch_rate = abs(float(angular_velocity[1]))
        diagnostics["max_abs_pitch_rate_rad_s"] = max(
            diagnostics["max_abs_pitch_rate_rad_s"],
            pitch_rate,
        )
        diagnostics["max_abs_bodyrate_rad_s"] = np.maximum(
            np.asarray(diagnostics["max_abs_bodyrate_rad_s"], dtype=float),
            np.abs(angular_velocity),
        ).tolist()
        dynamics = getattr(task_env, "flying_hand_dynamics", None)
        dynamics_debug = getattr(dynamics, "debug", {})
        rotor_thrust = np.asarray(dynamics_debug.get("rotor_thrust", []), dtype=float)
        if rotor_thrust.size and dynamics is not None:
            at_min = rotor_thrust <= dynamics.thrust_min + 1e-8
            at_max = rotor_thrust >= dynamics.thrust_max - 1e-8
            diagnostics["rotor_saturation_samples"] += int(np.any(at_min | at_max))
        # Count distinct entries into a large-pitch region. A lower exit
        # threshold prevents one noisy excursion from being counted repeatedly.
        if not diagnostics["_large_pitch_active"] and abs(pitch) >= np.deg2rad(15.0):
            diagnostics["large_pitch_excursions"] += 1
            diagnostics["_large_pitch_active"] = True
            if len(diagnostics["large_pitch_events"]) < 20:
                actual_pose = task_env.flying_hand.get_root_pose()
                ref_pose = task_env.flying_hand_ref_pose
                diagnostics["large_pitch_events"].append({
                    "sample": int(diagnostics["samples"]),
                    "action_step": int(task_env.take_action_cnt),
                    "roll_deg": float(np.degrees(roll)),
                    "pitch_deg": float(np.degrees(pitch)),
                    "bodyrate_rad_s": angular_velocity.tolist(),
                    "actual_position": np.asarray(actual_pose.p, dtype=float).tolist(),
                    "reference_position": (
                        np.asarray(ref_pose.p, dtype=float).tolist()
                        if ref_pose is not None
                        else None
                    ),
                    "grasping": bool(task_env.is_grasping),
                    "attached_actor": (
                        self.attached_actor.get_name()
                        if self.attached_actor is not None
                        else None
                    ),
                    "rotor_thrust": rotor_thrust.tolist(),
                    "torque_command": np.asarray(
                        dynamics_debug.get("torque_command", []),
                        dtype=float,
                    ).tolist(),
                    "torque_applied": np.asarray(
                        dynamics_debug.get("torque_applied", []),
                        dtype=float,
                    ).tolist(),
                    "desired_bodyrates": np.asarray(
                        dynamics_debug.get("desired_bodyrates", []),
                        dtype=float,
                    ).tolist(),
                    "torque_l1": np.asarray(
                        dynamics_debug.get("torque_l1", []),
                        dtype=float,
                    ).tolist(),
                })
        elif diagnostics["_large_pitch_active"] and abs(pitch) <= np.deg2rad(10.0):
            diagnostics["_large_pitch_active"] = False
        # The outer evaluator observes only once per replan chunk. Capture
        # simulation-time frames here so videos show waypoint and gripper
        # motion rather than only the replan observations.
        if (
            task_env.eval_video_ffmpeg is not None
            and task_env.flying_hand_save_step % int(task_env.save_freq) == 0
        ):
            task_env.get_obs()
        if task_env.flying_hand_ref_pose is not None:
            position_error = np.linalg.norm(
                np.asarray(task_env.flying_hand_ref_pose.p, dtype=float)
                - np.asarray(task_env.flying_hand.get_root_pose().p, dtype=float)
            )
            diagnostics["max_position_error_m"] = max(
                diagnostics["max_position_error_m"],
                float(position_error),
            )

    def _grasp_states(self, task_env, actions: np.ndarray) -> np.ndarray:
        config = task_env.flying_hand_grasp_validation
        close_threshold = float(config["close_threshold"])
        open_threshold = float(config["open_threshold"])
        state = bool(self.grasp_commanded)
        states = []
        for value in np.asarray(actions[:, 4], dtype=float):
            if state:
                state = value > open_threshold
            else:
                state = value >= close_threshold
            states.append(state)
        return np.asarray(states, dtype=bool)

    def _attach_actor_in_grasp_space(self, task_env, event: Dict[str, Any]) -> bool:
        from envs.utils.actor_utils import Actor

        candidates = []
        actor_diagnostics = []
        for actor in task_env.get_flying_hand_grasp_candidates():
            if not isinstance(actor, Actor):
                continue
            diagnostic = task_env.get_flying_hand_grasp_diagnostic(actor)
            actor_diagnostics.append(diagnostic)
            if diagnostic["eligible"]:
                candidates.append((np.linalg.norm(diagnostic["actor_center_u"]), actor))
        event["actors"] = actor_diagnostics
        if not candidates:
            event["attachment"] = "rejected_box_center_outside_grasp_region"
            return False

        _, self.attached_actor = min(candidates, key=lambda item: item[0])
        hand_pose = task_env.flying_hand.get_root_pose()
        self.attached_pose = hand_pose.inv() * self.attached_actor.get_pose()
        event["attachment"] = "attached"
        event["attached_actor"] = self.attached_actor.get_name()
        return True

    def _apply_flying_hand_grasp_command(
        self,
        task_env,
        grasp: bool,
        *,
        action_step: int,
    ) -> None:
        """Apply actor attachment or release immediately at a 20 Hz action edge."""
        event = {
            "event": "close" if grasp else "open",
            "action_step": int(action_step),
            "completed": True,
            "command_latency_seconds": 0.0,
        }
        self.grasp_diagnostics.append(event)
        self.grasp_commanded = bool(grasp)
        if grasp:
            task_env.set_flying_hand_gripper(
                task_env.flying_hand_config["gripper"]["close_qpos"],
                is_grasp=True,
            )
            event["gripper_close_commanded"] = True
            if self._attach_actor_in_grasp_space(task_env, event):
                self.gripper_state = "grasping_attached"
            else:
                self.gripper_state = "grasping_empty"
        else:
            event["released"] = self.attached_actor is not None
            task_env.set_flying_hand_gripper(
                task_env.flying_hand_config["gripper"]["open_qpos"],
                is_grasp=False,
            )
            self.attached_actor = None
            self.attached_pose = None
            self.gripper_state = "open"
            event["gripper_open_commanded"] = True

    @staticmethod
    def _limit_vector_norm(vector: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= limit or norm == 0.0:
            return vector
        return vector * (limit / norm)

    def _reset_waypoint_reference(self, task_env, pose=None) -> None:
        pose = task_env.flying_hand.get_root_pose() if pose is None else pose
        self._waypoint_reference_position = np.asarray(pose.p, dtype=float).copy()
        self._waypoint_reference_velocity = np.zeros(3)
        self._waypoint_reference_orientation = np.asarray(pose.q, dtype=float).copy()

    @staticmethod
    def _slerp_towards(
        current: np.ndarray,
        target: np.ndarray,
        max_angle: float,
    ) -> np.ndarray:
        """Move a unit quaternion toward ``target`` by at most ``max_angle``."""
        current = np.asarray(current, dtype=float)
        target = np.asarray(target, dtype=float)
        current /= np.linalg.norm(current)
        target /= np.linalg.norm(target)
        dot = float(np.dot(current, target))
        if dot < 0.0:
            target = -target
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        angle = 2.0 * float(np.arccos(dot))
        if angle <= max_angle or angle < 1.0e-9:
            return target
        fraction = max_angle / angle
        half_angle = 0.5 * angle
        sin_half_angle = float(np.sin(half_angle))
        if sin_half_angle < 1.0e-9:
            result = (1.0 - fraction) * current + fraction * target
        else:
            result = (
                np.sin((1.0 - fraction) * half_angle) / sin_half_angle * current
                + np.sin(fraction * half_angle) / sin_half_angle * target
            )
        return result / np.linalg.norm(result)

    def _track_flying_hand_waypoints(
        self,
        task_env,
        actions: np.ndarray,
        grasp_states: Optional[np.ndarray] = None,
    ) -> None:
        """Track each 20 Hz model waypoint through a causal, non-optimizing limiter."""
        import sapien

        if len(actions) == 0:
            return
        steps = int(task_env.save_freq)
        sim_dt = float(task_env.sim_timestep)
        action_dt = steps * sim_dt
        limits = task_env.flying_hand_waypoint_tracking
        max_velocity = float(limits["max_velocity"])
        max_acceleration = float(limits["max_acceleration"])
        max_yaw_rate = float(limits["max_yaw_rate"])
        poses = [
            self._flying_hand_relative_xyzyaw_to_world_pose(task_env, action)
            for action in actions
        ]
        if self._waypoint_reference_position is None:
            self._reset_waypoint_reference(task_env)
        previous_raw_position = np.asarray(
            task_env.flying_hand.get_root_pose().p,
            dtype=float,
        )
        max_waypoint_spacing = 0.0
        max_reference_velocity = 0.0
        max_reference_acceleration = 0.0
        max_reference_lag = 0.0
        if grasp_states is not None and len(grasp_states) != len(actions):
            raise ValueError("grasp_states must contain one state per waypoint")
        for action_index, target_pose in enumerate(poses):
            if (
                grasp_states is not None
                and bool(grasp_states[action_index]) != self.grasp_commanded
            ):
                self._apply_flying_hand_grasp_command(
                    task_env,
                    bool(grasp_states[action_index]),
                    action_step=int(task_env.take_action_cnt + action_index),
                )
            target_position = np.asarray(target_pose.p, dtype=float)
            max_waypoint_spacing = max(
                max_waypoint_spacing,
                float(np.linalg.norm(target_position - previous_raw_position)),
            )
            previous_raw_position = target_position
            target_orientation = np.asarray(target_pose.q, dtype=float)
            for substep in range(steps):
                remaining_time = max((steps - substep) * sim_dt, sim_dt)
                desired_velocity = self._limit_vector_norm(
                    (target_position - self._waypoint_reference_position) / remaining_time,
                    max_velocity,
                )
                velocity_change = self._limit_vector_norm(
                    desired_velocity - self._waypoint_reference_velocity,
                    max_acceleration * sim_dt,
                )
                self._waypoint_reference_velocity += velocity_change
                self._waypoint_reference_velocity = self._limit_vector_norm(
                    self._waypoint_reference_velocity,
                    max_velocity,
                )
                self._waypoint_reference_position += self._waypoint_reference_velocity * sim_dt
                self._waypoint_reference_orientation = self._slerp_towards(
                    self._waypoint_reference_orientation,
                    target_orientation,
                    max_yaw_rate * sim_dt,
                )
                reference_pose = sapien.Pose(
                    self._waypoint_reference_position.tolist(),
                    self._waypoint_reference_orientation.tolist(),
                )
                self._advance_flying_hand_reference(
                    task_env,
                    reference_pose,
                    # Velocity is internal to the position-rate limiter only.
                    # The flight controller intentionally receives zero desired
                    # velocity/acceleration and tracks the bounded pose target.
                    np.zeros(3),
                    np.zeros(3),
                    self.attached_actor,
                    self.attached_pose,
                )
                max_reference_velocity = max(
                    max_reference_velocity,
                    float(np.linalg.norm(self._waypoint_reference_velocity)),
                )
                max_reference_acceleration = max(
                    max_reference_acceleration,
                    float(np.linalg.norm(velocity_change) / sim_dt),
                )
                max_reference_lag = max(
                    max_reference_lag,
                    float(np.linalg.norm(target_position - self._waypoint_reference_position)),
                )

        self.waypoint_diagnostics.append({
            "segments": int(len(actions)),
            "samples": int(len(actions) * steps),
            "duration_seconds": float(len(actions) * action_dt),
            "max_waypoint_spacing_m": max_waypoint_spacing,
            "max_reference_velocity_mps": max_reference_velocity,
            "max_reference_acceleration_mps2": max_reference_acceleration,
            "max_reference_lag_m": max_reference_lag,
            "interpolation": "causal_slew_limited_position_reference",
        })

    def _execute_flying_hand_waypoint_chunk(self, task_env, actions: np.ndarray) -> int:
        """Track a full fixed-cadence chunk and apply gripper edges immediately."""
        if len(actions) == 0:
            return 0
        states = self._grasp_states(task_env, actions)
        self._record_action_chunk(task_env, actions)
        self._track_flying_hand_waypoints(
            task_env,
            actions,
            grasp_states=states,
        )
        return int(len(actions))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(task_env=task_env, observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        if action.ndim != 2 or action.shape[1] != 5:
            raise ValueError(
                f"Expected flying-hand waypoint chunk [T, 5], got {tuple(action.shape)}"
            )
        consumed_actions = self._execute_flying_hand_waypoint_chunk(task_env, action)
        task_env.take_action_cnt += consumed_actions
        task_env.eval_success = task_env.check_success()
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def get_episode_diagnostics(self) -> Dict[str, Any]:
        flight = dict(self.flight_diagnostics)
        flight.pop("_large_pitch_active", None)
        flight["max_abs_roll_deg"] = float(np.degrees(flight["max_abs_roll_rad"]))
        flight["max_abs_pitch_deg"] = float(np.degrees(flight["max_abs_pitch_rad"]))
        return {
            "flight": flight,
            "actions": dict(self.action_diagnostics),
            "gripper": {
                "state": self.gripper_state,
                "transition_pending": False,
            },
            "grasp_events": list(self.grasp_diagnostics),
            "waypoint_tracking": list(self.waypoint_diagnostics),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self.attached_actor = None
        self.attached_pose = None
        self.grasp_commanded = False
        self.gripper_state = "open"
        self.grasp_diagnostics = []
        self.action_diagnostics = self._empty_action_diagnostics()
        self.waypoint_diagnostics = []
        self.flight_diagnostics = self._empty_flight_diagnostics()
        self._waypoint_reference_position = None
        self._waypoint_reference_velocity = np.zeros(3)
        self._waypoint_reference_orientation = None
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))
    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    text_embedding_cache_dir = _resolve_optional_path(cfg.data.train.get("text_embedding_cache_dir"))

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        image_meta=OmegaConf.to_container(cfg.data.train.shape_meta.images, resolve=True),
        concat_multi_camera=cfg.data.train.get("concat_multi_camera", None),
        video_size=OmegaConf.to_container(cfg.data.train.video_size, resolve=True),
        text_embedding_cache_dir=text_embedding_cache_dir,
        context_len=int(cfg.data.train.get("context_len", 128)),
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
