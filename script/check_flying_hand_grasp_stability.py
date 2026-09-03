#!/usr/bin/env python3
"""Run Flying-Hand grasp tasks and report object and link stability per seed.

The link monitor separates commanded gripper closing, static post-close hold,
and object transport.  Several Flying-Hand tasks explicitly carry an object
after closure, so task success by itself is not evidence that the object was
held stably by contact or that the articulation links remained stable.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script.view_task import load_task_args
from envs.flying_hand import planner


DEFAULT_TASKS = (
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "move_bottle",
    "move_can_pot",
    "place_can_basket",
    "place_fruit_skillet",
    "shake_bottle",
    "stack_blocks_two",
    "stack_blocks_two_size",
    "thread_ring_rod",
)


def _json_value(value):
    """Convert NumPy scalar/array values in diagnostics to JSON values."""
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _minco_planner_summary(plans):
    """Build compact, aggregate MINCO diagnostics for one task execution."""
    plans = _json_value(plans or [])

    def number(mapping, key, default=0.0):
        value = mapping.get(key, default)
        return float(value) if value is not None else float(default)

    def duration_total(plan, key):
        values = plan.get(key, [])
        if isinstance(values, (int, float)):
            return float(values)
        return sum(float(value) for value in (values or []))

    def metric(plan, key):
        return number(plan.get("metrics", {}), key)

    optimization_failed_plans = [
        plan for plan in plans if not bool(plan.get("success", False))
    ]
    reach_plans = [plan for plan in plans if "reach_succeeded" in plan]
    reach_failed_plans = [
        plan for plan in reach_plans if not bool(plan.get("reach_succeeded", False))
    ]
    failed_phase_names = list(dict.fromkeys(
        [plan.get("phase") for plan in optimization_failed_plans + reach_failed_plans]
    ))
    return {
        "phase_count": len(plans),
        "optimizer_backends": sorted({
            str(plan.get("backend", "unknown")) for plan in plans
        }),
        "total_optimization_wall_time_s": sum(
            number(plan, "optimization_wall_time_seconds") for plan in plans
        ),
        "total_initial_flight_time_s": sum(duration_total(plan, "initial_times") for plan in plans),
        "total_optimized_flight_time_s": sum(metric(plan, "total_time_s") for plan in plans),
        "total_iterations": sum(int(number(plan, "iterations")) for plan in plans),
        "total_function_evaluations": sum(int(number(plan, "function_evaluations")) for plan in plans),
        "any_failed": bool(failed_phase_names),
        "failed_phase_count": len(failed_phase_names),
        "failed_phases": failed_phase_names,
        "optimizer_failed_phase_count": len(optimization_failed_plans),
        "reach_checked_phase_count": len(reach_plans),
        "reach_failed_phase_count": len(reach_failed_plans),
        "max_start_position_error_m": max((number(plan, "start_position_error_m") for plan in plans), default=0.0),
        "max_start_yaw_error_rad": max((number(plan, "start_yaw_error_rad") for plan in plans), default=0.0),
        "max_reach_position_error_m": max((number(plan, "reach_position_error_m") for plan in reach_plans), default=0.0),
        "max_reach_yaw_error_rad": max((number(plan, "reach_yaw_error_rad") for plan in reach_plans), default=0.0),
        "total_reach_wait_time_s": sum(number(plan, "reach_wait_time_s") for plan in reach_plans),
        "max_planned_velocity_mps": max((metric(plan, "max_velocity_mps") for plan in plans), default=0.0),
        "max_planned_acceleration_mps2": max((metric(plan, "max_acceleration_mps2") for plan in plans), default=0.0),
        "max_planned_yaw_rate_rad_s": max((metric(plan, "max_yaw_rate_rad_s") for plan in plans), default=0.0),
    }


def _position(actor):
    return np.asarray(actor.get_pose().p, dtype=float)


def _linear_speed(actor):
    body = actor.actor
    components = (
        body.get_components()
        if hasattr(body, "get_components")
        else body.get_links()
        if hasattr(body, "get_links")
        else []
    )
    speeds = []
    for component in components:
        get_velocity = getattr(component, "get_linear_velocity", None)
        if get_velocity is not None:
            speeds.append(float(np.linalg.norm(get_velocity())))
    return max(speeds, default=0.0)


class GraspMonitor:
    """Collect one record for every physical closing event in a task."""

    def __init__(self):
        self.events = []
        self.active = None

    @staticmethod
    def _is_closing(env):
        return (
            env.is_grasping
            and env.flying_hand_gripper_steps > 0
            and env.flying_hand_gripper_step < env.flying_hand_gripper_steps
        )

    def before_step(self, env):
        closing = self._is_closing(env)
        if closing and self.active is None:
            initial_positions = {
                actor.get_name(): _position(actor)
                for actor in env.task_actors
            }
            initial_bottoms = {
                actor.get_name(): float(env._get_actor_world_bounds(actor)[0][2])
                for actor in env.task_actors
            }
            self.active = {
                "initial_positions": initial_positions,
                "initial_bottoms": initial_bottoms,
                "actors": {
                    actor.get_name(): {
                        "max_speed_mps": 0.0,
                        "max_displacement_m": 0.0,
                        "min_bottom_z_m": initial_bottoms[actor.get_name()],
                    }
                    for actor in env.task_actors
                },
                "max_speed_mps": 0.0,
                "max_displacement_m": 0.0,
                "max_hand_root_speed_mps": 0.0,
                "min_bottom_z_m": float("inf"),
                "steps": 0,
            }
        elif not closing and self.active is not None:
            target = np.asarray(env.flying_hand_gripper_qpos, dtype=float)
            actual = env.flying_hand.get_qpos()[env.flying_hand_gripper_joint_indices]
            error = np.abs(target - actual)
            motion = np.abs(target - np.asarray(env.flying_hand_gripper_start_qpos, dtype=float))
            normalized_error = np.divide(error, motion, out=np.zeros_like(error), where=motion > 1e-8)
            self.active["max_joint_target_error"] = float(np.max(error))
            self.active["max_normalized_joint_target_error"] = float(np.max(normalized_error))
            self.events.append(self.active)
            self.active = None

    def after_step(self, env):
        if self.active is None:
            return
        self.active["steps"] += 1
        self.active["max_hand_root_speed_mps"] = max(
            self.active["max_hand_root_speed_mps"],
            float(np.linalg.norm(env.flying_hand.get_root_linear_velocity())),
        )
        for actor in env.task_actors:
            name = actor.get_name()
            position = _position(actor)
            initial = self.active["initial_positions"][name]
            speed = _linear_speed(actor)
            displacement = float(np.linalg.norm(position - initial))
            bottom_z = float(env._get_actor_world_bounds(actor)[0][2])
            actor_metrics = self.active["actors"][name]
            actor_metrics["max_speed_mps"] = max(actor_metrics["max_speed_mps"], speed)
            actor_metrics["max_displacement_m"] = max(actor_metrics["max_displacement_m"], displacement)
            actor_metrics["min_bottom_z_m"] = min(actor_metrics["min_bottom_z_m"], bottom_z)
            self.active["max_speed_mps"] = max(self.active["max_speed_mps"], speed)
            self.active["max_displacement_m"] = max(
                self.active["max_displacement_m"],
                displacement,
            )
            self.active["min_bottom_z_m"] = min(self.active["min_bottom_z_m"], bottom_z)

    def finish(self, env):
        # Handles a task that ends immediately after the final closing step.
        self.before_step(env)


class LinkJitterMonitor:
    """Measure link motion relative to the flying-hand root while grasping.

    World velocity is not a useful jitter signal because the complete flying
    hand intentionally translates and rotates during transport.  For every
    link we therefore subtract the rigid-body velocity induced by the root and
    evaluate only the residual motion.  Closing, post-close holding, and
    carrying are reported separately because link motion is commanded during
    closing but should be nearly static after the fingers reach the object.
    """

    PHASES = ("closing", "holding", "carrying")

    def __init__(self):
        self.events = []
        self.traces = []
        self.active = None

    @staticmethod
    def _phase(env, carrying=False):
        if not env.is_grasping:
            return None
        if GraspMonitor._is_closing(env):
            return "closing"
        return "carrying" if carrying else "holding"

    @staticmethod
    def _link_metric():
        return {
            "samples": 0,
            "linear_speed_sq_sum": 0.0,
            "angular_speed_sq_sum": 0.0,
            "linear_accel_sq_sum": 0.0,
            "angular_accel_sq_sum": 0.0,
            "accel_samples": 0,
            "max_relative_linear_speed_mps": 0.0,
            "max_relative_angular_speed_rad_s": 0.0,
            "max_relative_linear_accel_mps2": 0.0,
            "max_relative_angular_accel_rad_s2": 0.0,
            "velocity_reversal_count": 0,
            "local_position_min": None,
            "local_position_max": None,
        }

    def _start(self, env):
        links = env.flying_hand.get_links()
        joints = env.flying_hand.get_active_joints()
        link_names = [link.get_name() for link in links]
        joint_names = [joint.get_name() for joint in joints]
        self.active = {
            "link_names": link_names,
            "joint_names": joint_names,
            "phases": {
                phase: {
                    "samples": 0,
                    "links": {name: self._link_metric() for name in link_names},
                    "joint_qpos_min": None,
                    "joint_qpos_max": None,
                    "joint_qvel_abs_max": np.zeros(len(joint_names), dtype=float),
                    "joint_qvel_sq_sum": np.zeros(len(joint_names), dtype=float),
                }
                for phase in self.PHASES
            },
            "previous": {phase: {} for phase in self.PHASES},
            "trace": {
                "sim_step": [],
                "phase": [],
                "relative_linear_velocity": [],
                "relative_angular_velocity": [],
                "local_position": [],
                "joint_qpos": [],
                "joint_qvel": [],
            },
        }

    def before_step(self, env, carrying=False):
        phase = self._phase(env, carrying=carrying)
        if phase is not None and self.active is None:
            self._start(env)
        elif phase is None and self.active is not None:
            self._finish_event()

    def after_step(self, env, carrying=False):
        phase = self._phase(env, carrying=carrying)
        if phase is None or self.active is None:
            return

        links = env.flying_hand.get_links()
        root = env.imu_odom_link
        root_pose = root.get_pose()
        root_transform_inv = np.linalg.inv(root_pose.to_transformation_matrix())
        root_position = np.asarray(root_pose.p, dtype=float)
        root_linear_velocity = np.asarray(root.get_linear_velocity(), dtype=float)
        root_angular_velocity = np.asarray(root.get_angular_velocity(), dtype=float)
        dt = float(env.sim_timestep)

        relative_linear_velocity = []
        relative_angular_velocity = []
        local_positions = []
        phase_data = self.active["phases"][phase]
        phase_data["samples"] += 1

        for link in links:
            name = link.get_name()
            link_pose = link.get_pose()
            link_position = np.asarray(link_pose.p, dtype=float)
            rigid_linear_velocity = root_linear_velocity + np.cross(
                root_angular_velocity,
                link_position - root_position,
            )
            residual_linear = np.asarray(link.get_linear_velocity(), dtype=float) - rigid_linear_velocity
            residual_angular = np.asarray(link.get_angular_velocity(), dtype=float) - root_angular_velocity
            local_position = (root_transform_inv @ np.r_[link_position, 1.0])[:3]

            linear_speed = float(np.linalg.norm(residual_linear))
            angular_speed = float(np.linalg.norm(residual_angular))
            metrics = phase_data["links"][name]
            metrics["samples"] += 1
            metrics["linear_speed_sq_sum"] += linear_speed ** 2
            metrics["angular_speed_sq_sum"] += angular_speed ** 2
            metrics["max_relative_linear_speed_mps"] = max(
                metrics["max_relative_linear_speed_mps"], linear_speed,
            )
            metrics["max_relative_angular_speed_rad_s"] = max(
                metrics["max_relative_angular_speed_rad_s"], angular_speed,
            )
            if metrics["local_position_min"] is None:
                metrics["local_position_min"] = local_position.copy()
                metrics["local_position_max"] = local_position.copy()
            else:
                metrics["local_position_min"] = np.minimum(metrics["local_position_min"], local_position)
                metrics["local_position_max"] = np.maximum(metrics["local_position_max"], local_position)

            previous = self.active["previous"][phase].get(name)
            if previous is not None:
                linear_accel = float(np.linalg.norm(residual_linear - previous[0]) / dt)
                angular_accel = float(np.linalg.norm(residual_angular - previous[1]) / dt)
                metrics["accel_samples"] += 1
                metrics["linear_accel_sq_sum"] += linear_accel ** 2
                metrics["angular_accel_sq_sum"] += angular_accel ** 2
                metrics["max_relative_linear_accel_mps2"] = max(
                    metrics["max_relative_linear_accel_mps2"], linear_accel,
                )
                metrics["max_relative_angular_accel_rad_s2"] = max(
                    metrics["max_relative_angular_accel_rad_s2"], angular_accel,
                )
                if (
                    np.dot(residual_linear, previous[0]) < 0.0
                    and linear_speed > 1e-3
                    and np.linalg.norm(previous[0]) > 1e-3
                ):
                    metrics["velocity_reversal_count"] += 1
            self.active["previous"][phase][name] = (residual_linear.copy(), residual_angular.copy())
            relative_linear_velocity.append(residual_linear)
            relative_angular_velocity.append(residual_angular)
            local_positions.append(local_position)

        qpos = np.asarray(env.flying_hand.get_qpos(), dtype=float)
        qvel = np.asarray(env.flying_hand.get_qvel(), dtype=float)
        if phase_data["joint_qpos_min"] is None:
            phase_data["joint_qpos_min"] = qpos.copy()
            phase_data["joint_qpos_max"] = qpos.copy()
        else:
            phase_data["joint_qpos_min"] = np.minimum(phase_data["joint_qpos_min"], qpos)
            phase_data["joint_qpos_max"] = np.maximum(phase_data["joint_qpos_max"], qpos)
        phase_data["joint_qvel_abs_max"] = np.maximum(phase_data["joint_qvel_abs_max"], np.abs(qvel))
        phase_data["joint_qvel_sq_sum"] += qvel ** 2

        trace = self.active["trace"]
        trace["sim_step"].append(int(env.flying_hand_save_step))
        trace["phase"].append(self.PHASES.index(phase))
        trace["relative_linear_velocity"].append(relative_linear_velocity)
        trace["relative_angular_velocity"].append(relative_angular_velocity)
        trace["local_position"].append(local_positions)
        trace["joint_qpos"].append(qpos)
        trace["joint_qvel"].append(qvel)

    def _finish_event(self):
        event = {
            "link_names": self.active["link_names"],
            "joint_names": self.active["joint_names"],
            "phases": {},
        }
        for phase, phase_data in self.active["phases"].items():
            samples = phase_data["samples"]
            links = {}
            for name, metrics in phase_data["links"].items():
                link_samples = metrics["samples"]
                accel_samples = metrics["accel_samples"]
                local_span = (
                    float(np.linalg.norm(metrics["local_position_max"] - metrics["local_position_min"]))
                    if metrics["local_position_min"] is not None else 0.0
                )
                links[name] = {
                    "samples": link_samples,
                    "rms_relative_linear_speed_mps": (
                        float(np.sqrt(metrics["linear_speed_sq_sum"] / link_samples)) if link_samples else 0.0
                    ),
                    "rms_relative_angular_speed_rad_s": (
                        float(np.sqrt(metrics["angular_speed_sq_sum"] / link_samples)) if link_samples else 0.0
                    ),
                    "rms_relative_linear_accel_mps2": (
                        float(np.sqrt(metrics["linear_accel_sq_sum"] / accel_samples)) if accel_samples else 0.0
                    ),
                    "rms_relative_angular_accel_rad_s2": (
                        float(np.sqrt(metrics["angular_accel_sq_sum"] / accel_samples)) if accel_samples else 0.0
                    ),
                    "max_relative_linear_speed_mps": metrics["max_relative_linear_speed_mps"],
                    "max_relative_angular_speed_rad_s": metrics["max_relative_angular_speed_rad_s"],
                    "max_relative_linear_accel_mps2": metrics["max_relative_linear_accel_mps2"],
                    "max_relative_angular_accel_rad_s2": metrics["max_relative_angular_accel_rad_s2"],
                    "local_position_span_m": local_span,
                    "velocity_reversal_count": metrics["velocity_reversal_count"],
                }
            qpos_span = (
                phase_data["joint_qpos_max"] - phase_data["joint_qpos_min"]
                if phase_data["joint_qpos_min"] is not None else np.zeros(len(event["joint_names"]))
            )
            qvel_rms = (
                np.sqrt(phase_data["joint_qvel_sq_sum"] / samples)
                if samples else np.zeros(len(event["joint_names"]))
            )
            event["phases"][phase] = {
                "samples": samples,
                "links": links,
                "joint_qpos_span": dict(zip(event["joint_names"], qpos_span.tolist())),
                "joint_qvel_abs_max": dict(zip(event["joint_names"], phase_data["joint_qvel_abs_max"].tolist())),
                "joint_qvel_rms": dict(zip(event["joint_names"], qvel_rms.tolist())),
            }
        trace = {
            "link_names": np.asarray(self.active["link_names"]),
            "joint_names": np.asarray(self.active["joint_names"]),
            **{key: np.asarray(value) for key, value in self.active["trace"].items()},
        }
        self.events.append(event)
        self.traces.append(trace)
        self.active = None

    def finish(self, env):
        self.before_step(env)
        if self.active is not None:
            self._finish_event()


def _class_for_task(task_name):
    module = importlib.import_module(f"envs.flying_hand.{task_name}")
    return getattr(module, task_name)


def _classify(event, ground_height):
    unstable = []
    review = []
    if event["max_speed_mps"] > 0.20:
        unstable.append("closure_speed_gt_0.20_mps")
    elif event["max_speed_mps"] > 0.08:
        review.append("closure_speed_gt_0.08_mps")
    if event["max_displacement_m"] > 0.04:
        unstable.append("closure_displacement_gt_0.04_m")
    elif event["max_displacement_m"] > 0.02:
        review.append("closure_displacement_gt_0.02_m")
    ground_boundary = ground_height + 0.03
    dropped_during_closure = [
        name
        for name, metrics in event["actors"].items()
        if event["initial_bottoms"][name] > ground_boundary
        and metrics["min_bottom_z_m"] <= ground_boundary
    ]
    already_grounded = [
        name
        for name, bottom in event["initial_bottoms"].items()
        if bottom <= ground_boundary
    ]
    if dropped_during_closure:
        unstable.append("object_reached_ground_during_closure:" + ",".join(sorted(dropped_during_closure)))
    if already_grounded:
        review.append("object_already_grounded_before_closure:" + ",".join(sorted(already_grounded)))
    if event.get("max_normalized_joint_target_error", 0.0) > 0.40:
        review.append("joint_close_error_gt_40_percent")
    return unstable, review


def _save_link_traces(path, traces):
    arrays = {"event_count": np.asarray(len(traces), dtype=np.int32)}
    for event_index, trace in enumerate(traces):
        for key, value in trace.items():
            arrays[f"event_{event_index}_{key}"] = value
    np.savez_compressed(path, **arrays)


class TaskVideoRecorder:
    """Stream selected task cameras to one H.264 video per camera."""

    def __init__(self, output_dir, stride_steps, fps):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stride_steps = int(stride_steps)
        self.fps = float(fps)
        self.processes = {}
        self.paths = {}
        self.frame_count = 0
        self.last_step = None

    def _start_process(self, camera_name, frame):
        height, width = frame.shape[:2]
        safe_camera_name = camera_name.replace("/", "_")
        path = self.output_dir / f"{safe_camera_name}.mp4"
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(self.fps),
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        self.processes[camera_name] = process
        self.paths[camera_name] = str(path)
        return process

    def capture(self, env, force=False):
        step_index = int(env.flying_hand_save_step)
        if self.last_step == step_index:
            return
        if not force and step_index % self.stride_steps:
            return
        observation = env.get_obs()["observation"]
        wrote_frame = False
        for camera_name, camera_observation in observation.items():
            if "rgb" not in camera_observation:
                continue
            frame = np.asarray(camera_observation["rgb"])
            if frame.ndim != 3 or frame.shape[2] < 3:
                raise ValueError(
                    f"Camera {camera_name!r} returned invalid RGB shape {frame.shape}"
                )
            frame = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)
            process = self.processes.get(camera_name)
            if process is None:
                process = self._start_process(camera_name, frame)
            process.stdin.write(frame.tobytes())
            wrote_frame = True
        if wrote_frame:
            self.frame_count += 1
            self.last_step = step_index

    def close(self):
        errors = []
        for process in self.processes.values():
            process.stdin.close()
        for camera_name, process in self.processes.items():
            return_code = process.wait()
            if return_code:
                errors.append(f"{camera_name}: ffmpeg exited with {return_code}")
        if errors:
            raise RuntimeError("; ".join(errors))


def _empty_link_jitter_summary():
    return {
        "samples": 0,
        "worst_linear_speed": {"link": None, "value_mps": 0.0},
        "worst_linear_rms": {"link": None, "value_mps": 0.0},
        "worst_angular_speed": {"link": None, "value_rad_s": 0.0},
        "worst_angular_rms": {"link": None, "value_rad_s": 0.0},
        "worst_local_position_span": {"link": None, "value_m": 0.0},
        "worst_joint_qvel_rms": {"joint": None, "value": 0.0},
    }


def _summarize_link_jitter(events):
    summary = {
        "phases": {
            phase: _empty_link_jitter_summary()
            for phase in LinkJitterMonitor.PHASES
        },
    }
    for event in events:
        for phase in LinkJitterMonitor.PHASES:
            phase_data = event["phases"][phase]
            phase_summary = summary["phases"][phase]
            phase_summary["samples"] += phase_data["samples"]
            for link, metrics in phase_data["links"].items():
                candidates = (
                    ("worst_linear_speed", "max_relative_linear_speed_mps", "value_mps"),
                    ("worst_linear_rms", "rms_relative_linear_speed_mps", "value_mps"),
                    ("worst_angular_speed", "max_relative_angular_speed_rad_s", "value_rad_s"),
                    ("worst_angular_rms", "rms_relative_angular_speed_rad_s", "value_rad_s"),
                    ("worst_local_position_span", "local_position_span_m", "value_m"),
                )
                for summary_key, metric_key, value_key in candidates:
                    if metrics[metric_key] > phase_summary[summary_key][value_key]:
                        phase_summary[summary_key] = {"link": link, value_key: metrics[metric_key]}
            for joint, value in phase_data["joint_qvel_rms"].items():
                if value > phase_summary["worst_joint_qvel_rms"]["value"]:
                    phase_summary["worst_joint_qvel_rms"] = {"joint": joint, "value": value}
    return summary


def run_once(
    task_name,
    task_config,
    seed,
    trace_path=None,
    video_dir=None,
    video_stride_steps=None,
):
    started = time.monotonic()
    result = {
        "task": task_name,
        "task_config": task_config,
        "seed": seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "minco_plans": [],
        "planner_summary": _minco_planner_summary([]),
        "rod_physics": {},
    }
    task = None
    original_step = planner.step
    monitor = GraspMonitor()
    link_monitor = LinkJitterMonitor()
    video_recorder = None
    rod_physics = {
        "source_rod": {"samples": 0, "disabled_samples": 0},
        "target_rod": {"samples": 0, "disabled_samples": 0},
    }

    def sample_rod_physics():
        if task is None:
            return
        for attribute, counts in rod_physics.items():
            rod = getattr(task, attribute, None)
            if rod is None:
                continue
            components = [
                component
                for component in rod.components
                if hasattr(component, "is_enabled")
            ]
            if not components:
                continue
            counts["samples"] += 1
            counts["disabled_samples"] += int(
                any(not bool(component.is_enabled) for component in components)
            )

    def monitored_step(env, n, save_freq=-1):
        for _ in range(n):
            sample_rod_physics()
            monitor.before_step(env)
            carrying = bool(getattr(env, "_flying_hand_carrying", False))
            link_monitor.before_step(env, carrying=carrying)
            original_step(env, 1, save_freq=save_freq)
            monitor.after_step(env)
            link_monitor.after_step(env, carrying=carrying)
            if video_recorder is not None:
                video_recorder.capture(env)

    try:
        args = load_task_args(task_name, task_config, 0)
        args["seed"] = seed
        task = _class_for_task(task_name)()
        planner.step = monitored_step
        task.setup_demo(**args)
        if video_dir is not None:
            stride_steps = int(video_stride_steps or task.save_freq)
            if stride_steps <= 0:
                raise ValueError("video_stride_steps must be positive")
            video_recorder = TaskVideoRecorder(
                video_dir,
                stride_steps,
                fps=1.0 / (float(task.sim_timestep) * stride_steps),
            )
            video_recorder.capture(task, force=True)
        task.play_once()
        monitor.finish(task)
        link_monitor.finish(task)
        result["task_success"] = bool(task.check_success())
        result["task_failed_flag"] = bool(task.task_failed)
        result["rod_physics"] = {
            name: counts
            for name, counts in rod_physics.items()
            if counts["samples"] > 0
        }
        result["events"] = []
        all_unstable = []
        all_review = []
        for index, event in enumerate(monitor.events):
            unstable, review = _classify(event, task.ground_height)
            event = {
                key: value
                for key, value in event.items()
                if key not in {"initial_positions", "initial_bottoms"}
            }
            event["unstable_reasons"] = unstable
            event["review_reasons"] = review
            event["event_index"] = index
            result["events"].append(event)
            all_unstable.extend(unstable)
            all_review.extend(review)
        if not result["events"]:
            all_review.append("no_grasp_closure_event_observed")
        if not result["task_success"]:
            all_review.append("task_check_success_false")
        result["unstable_reasons"] = sorted(set(all_unstable))
        result["review_reasons"] = sorted(set(all_review))
        result["link_jitter_events"] = link_monitor.events
        result["link_jitter_summary"] = _summarize_link_jitter(link_monitor.events)
        if trace_path is not None:
            _save_link_traces(trace_path, link_monitor.traces)
            result["link_jitter_trace"] = str(trace_path)
        result["status"] = "unstable" if result["unstable_reasons"] else "review" if result["review_reasons"] else "stable"
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        planner.step = original_step
        if task is not None:
            if video_recorder is not None:
                try:
                    video_recorder.capture(task, force=True)
                    video_recorder.close()
                    result["videos"] = video_recorder.paths
                    result["video_frame_count"] = video_recorder.frame_count
                except Exception as exc:
                    result.setdefault(
                        "video_error", f"{type(exc).__name__}: {exc}"
                    )
            result["minco_plans"] = _json_value(getattr(task, "minco_plan_diagnostics", []))
            result["planner_summary"] = _minco_planner_summary(result["minco_plans"])
            try:
                task.close_env()
            except Exception as exc:
                result.setdefault("close_error", f"{type(exc).__name__}: {exc}")
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", default="flying_hand_clean")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--render-videos", action="store_true")
    parser.add_argument("--video-stride-steps", type=int)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args()

    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        parser.error("worker index must satisfy 0 <= worker-index < worker-count")
    if args.render_videos and args.output_dir is None:
        parser.error("--render-videos requires --output-dir")
    if args.video_stride_steps is not None and args.video_stride_steps <= 0:
        parser.error("--video-stride-steps must be positive")

    jobs = [(task_name, seed) for task_name in args.tasks for seed in args.seeds]
    jobs = [job for index, job in enumerate(jobs) if index % args.worker_count == args.worker_index]
    output_file = None
    trace_dir = None
    if args.output_dir is not None:
        workers_dir = args.output_dir / "workers"
        trace_dir = args.output_dir / "link_traces"
        workers_dir.mkdir(parents=True, exist_ok=True)
        trace_dir.mkdir(parents=True, exist_ok=True)
        output_file = workers_dir / f"worker_{args.worker_index:02d}.jsonl"

    for task_name, seed in jobs:
        trace_path = trace_dir / f"{task_name}__seed_{seed:03d}.npz" if trace_dir is not None else None
        result = run_once(
            task_name,
            args.task_config,
            seed,
            trace_path=trace_path,
            video_dir=(
                args.output_dir / "videos" / f"{task_name}__seed_{seed:03d}"
                if args.render_videos
                else None
            ),
            video_stride_steps=args.video_stride_steps,
        )
        line = json.dumps(result, sort_keys=True)
        print(line, flush=True)
        if output_file is not None:
            with output_file.open("a", encoding="utf-8") as file:
                file.write(line + "\n")


if __name__ == "__main__":
    main()
