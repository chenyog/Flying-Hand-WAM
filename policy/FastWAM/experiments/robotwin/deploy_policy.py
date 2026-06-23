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

        self.pending_actions: deque[np.ndarray] = deque()
        self.attached_actor = None
        self.attached_pose = None
        self.grip_steps = 0
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

    @staticmethod
    def _track_flying_hand_world_pose(task_env, target_pose, steps: int, carried_actor=None, carried_pose=None) -> None:
        from envs.flying_hand import planner

        zero = np.zeros(3)
        for _ in range(steps):
            task_env.flying_hand_ref_pose = target_pose
            if task_env.enable_dynamics:
                hand_pose, hand_v = task_env.flying_hand_dynamics.step(target_pose, zero, zero, task_env.is_grasping)
                task_env.flying_hand.set_root_pose(hand_pose)
                task_env.flying_hand.set_root_linear_velocity(hand_v.tolist())
                task_env.flying_hand.set_root_angular_velocity(task_env.flying_hand_dynamics.w.tolist())
            else:
                hand_pose, hand_v = target_pose, zero
                planner.set_pose(task_env, hand_pose, hand_v)
            carried_pose_fn = None
            if carried_actor is not None:
                actor_pose = hand_pose * carried_pose
                carried_pose_fn = lambda actor=carried_actor, pose=actor_pose, vel=hand_v: (actor, pose, vel)
            planner.step(task_env, 1, save_freq=None, carried_pose_fn=carried_pose_fn)

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

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

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
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        if action.shape[0] == 5:
            steps = int(task_env.save_freq)
            grasp = action[4] >= 0.5
            task_env.set_flying_hand_gripper(
                task_env.flying_hand_config["gripper"]["close_qpos" if grasp else "open_qpos"],
                is_grasp=grasp,
            )
            if grasp:
                self.grip_steps += 1
                if self.attached_actor is None and self.grip_steps >= int(np.ceil(task_env.grasp_hold_seconds / (steps * task_env.sim_timestep))):
                    from envs.utils.actor_utils import Actor

                    hand_pose = task_env.flying_hand.get_root_pose()
                    self.attached_actor = min(
                        [actor for actor in task_env.task_actors if type(actor) is Actor],
                        key=lambda actor: np.linalg.norm(actor.get_pose().p - hand_pose.p),
                    )
                    self.attached_pose = hand_pose.inv() * self.attached_actor.get_pose()
            else:
                self.grip_steps = 0
                self.attached_actor = None
                self.attached_pose = None
            target_pose = self._flying_hand_relative_xyzyaw_to_world_pose(task_env, action)
            self._track_flying_hand_world_pose(task_env, target_pose, steps, self.attached_actor, self.attached_pose)
            task_env.take_action_cnt += 1
            task_env.eval_success = task_env.check_success()
        else:
            task_env.take_action(action, action_type="qpos")
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

    def reset(self) -> None:
        self.pending_actions.clear()
        self.attached_actor = None
        self.attached_pose = None
        self.grip_steps = 0
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
