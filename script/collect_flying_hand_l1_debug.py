#!/usr/bin/env python3
"""Collect replayable Flying-Hand L1 diagnostics for every task and seed.

The collector deliberately records numerical state rather than RGB frames.  One
compressed ``.npz`` file contains every 5 ms simulation step, including the
reference/actual pose, gripper progress, L1 estimates, angular velocity, and
rotor-allocation residual.  This makes grasp-transition failures inspectable
without generating a large image dataset.
"""

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.flying_hand import planner
from script.view_task import load_task_args


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
        "max_planned_path_deviation_m": max((metric(plan, "max_path_deviation_m") for plan in plans), default=0.0),
    }


def _task_class(task_name):
    module = importlib.import_module(f"envs.flying_hand.{task_name}")
    return getattr(module, task_name)


def _deep_update(target, update):
    for key, value in update.items():
        if isinstance(value, dict):
            _deep_update(target.setdefault(key, {}), value)
        else:
            target[key] = value


def _as_vector(value, size, fill=0.0):
    if value is None:
        return np.full(size, fill, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.size != size:
        return np.full(size, fill, dtype=np.float32)
    return value


class L1TrajectoryRecorder:
    """Sample post-physics state from every planner simulation step."""

    def __init__(self):
        self.rows = []

    def sample(self, env):
        dynamics = env.flying_hand_dynamics
        debug = dynamics.debug
        pose = env.flying_hand.get_root_pose()
        ref_pose = env.flying_hand_ref_pose or pose
        close_steps = max(int(env.flying_hand_gripper_steps), 1)
        close_progress = float(env.flying_hand_gripper_step) / close_steps
        allocation_grasped = bool(
            env.is_grasping
            and env.flying_hand_gripper_step
            >= env.flying_hand_gripper_steps * env.flying_hand_gripper_prismatic_stage_ratio
        )
        self.rows.append({
            "sim_step": int(env.flying_hand_save_step),
            "is_grasping": float(env.is_grasping),
            "allocation_grasped": float(allocation_grasped),
            "gripper_progress": close_progress,
            "gripper_qpos": np.asarray(
                env.flying_hand.get_qpos()[env.flying_hand_gripper_joint_indices], dtype=np.float32,
            ),
            "ref_p": _as_vector(ref_pose.p, 3),
            "ref_q": _as_vector(ref_pose.q, 4),
            "actual_p": _as_vector(pose.p, 3),
            "actual_q": _as_vector(pose.q, 4),
            "actual_v": _as_vector(env.flying_hand.get_root_linear_velocity(), 3),
            "actual_w": _as_vector(env.flying_hand.get_root_angular_velocity(), 3),
            "model_p": _as_vector(dynamics.p, 3),
            "model_v": _as_vector(dynamics.v, 3),
            "model_w": _as_vector(dynamics.w, 3),
            "force_l1": _as_vector(debug.get("force_l1"), 3),
            "torque_l1": _as_vector(debug.get("torque_l1"), 3),
            "w_hat": _as_vector(debug.get("w_hat"), 3),
            "desired_bodyrates": _as_vector(debug.get("desired_bodyrates"), 3),
            "desired_angular_acceleration": _as_vector(debug.get("desired_angular_acceleration"), 3),
            "torque_command": _as_vector(debug.get("torque_command"), 3),
            "torque_applied": _as_vector(debug.get("torque_applied"), 3),
            "torque_allocation_error": _as_vector(debug.get("torque_allocation_error"), 3),
            "rotor_thrust": _as_vector(debug.get("rotor_thrust"), 4),
        })

    def arrays(self):
        if not self.rows:
            return {"sim_step": np.empty(0, dtype=np.int32)}
        return {
            key: np.asarray([row[key] for row in self.rows])
            for key in self.rows[0]
        }


def _metrics(arrays, cfg):
    if arrays["sim_step"].size == 0:
        return {"samples": 0}
    torque_l1 = arrays["torque_l1"]
    w = arrays["model_w"]
    allocation_error = arrays["torque_allocation_error"]
    grasped = arrays["allocation_grasped"] > 0.5
    torque_bound = np.asarray(cfg["dynamics"]["estimator"]["torque_bound"], dtype=float)
    rotor_min = np.asarray(cfg["dynamics"]["limits"]["rotor_thrusts_min"], dtype=float)
    rotor_max = np.asarray(cfg["dynamics"]["limits"]["rotor_thrusts_max"], dtype=float)
    rotor = arrays["rotor_thrust"]
    saturating = np.logical_or(np.isclose(rotor, rotor_min, atol=1e-5), np.isclose(rotor, rotor_max, atol=1e-5))

    def max_norm(values, mask=None):
        if mask is not None:
            values = values[mask]
        return float(np.max(np.linalg.norm(values, axis=1))) if len(values) else 0.0

    return {
        "samples": int(len(arrays["sim_step"])),
        "grasped_samples": int(np.count_nonzero(grasped)),
        "max_model_angular_speed_rad_s": max_norm(w),
        "max_grasped_angular_speed_rad_s": max_norm(w, grasped),
        "max_torque_l1_norm_nm": max_norm(torque_l1),
        "max_grasped_torque_l1_norm_nm": max_norm(torque_l1, grasped),
        "max_allocation_torque_error_nm": max_norm(allocation_error),
        "max_grasped_allocation_torque_error_nm": max_norm(allocation_error, grasped),
        "torque_l1_bound_hit_fraction": float(np.mean(np.isclose(np.abs(torque_l1), torque_bound, atol=1e-6))),
        "rotor_limit_hit_fraction": float(np.mean(saturating)),
    }


def run_episode(task_name, task_config, seed, output_dir, dynamics_patch):
    started = time.monotonic()
    result = {
        "task": task_name,
        "task_config": task_config,
        "seed": seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "minco_plans": [],
        "planner_summary": _minco_planner_summary([]),
    }
    task = None
    recorder = L1TrajectoryRecorder()
    original_step = planner.step

    def recorded_step(env, n, save_freq=-1, carried_pose_fn=None):
        for _ in range(n):
            original_step(env, 1, save_freq=save_freq, carried_pose_fn=carried_pose_fn)
            recorder.sample(env)

    try:
        args = load_task_args(task_name, task_config, render_freq=0)
        args.update(seed=seed, now_ep_num=seed, collect_data=False, save_data=False, need_plan=True, eval_mode=False)
        task = _task_class(task_name)()
        if dynamics_patch:
            load_config = task._load_flying_hand_config

            def load_patched_config():
                config = load_config()
                _deep_update(config["dynamics"], dynamics_patch)
                return config

            task._load_flying_hand_config = load_patched_config
        planner.step = recorded_step
        task.setup_demo(**args)
        task.play_once()
        arrays = recorder.arrays()
        trajectory_path = output_dir / "trajectories" / f"{task_name}__seed_{seed:03d}.npz"
        minco_plans = _json_value(task.minco_plan_diagnostics)
        planner_summary = _minco_planner_summary(minco_plans)
        metadata = {
            "task": task_name,
            "task_config": task_config,
            "seed": seed,
            "effective_task_args": _json_value(args),
            "flying_hand_config": _json_value(task.flying_hand_config),
            "dynamics_patch": _json_value(dynamics_patch),
            "sim_timestep_s": task.sim_timestep,
            "save_freq": task.save_freq,
            "control_mass_kg": task.flying_hand_dynamics.mass,
            "control_inertia_diag_kg_m2": np.diag(task.flying_hand_dynamics.j).tolist(),
            "estimator": task.flying_hand_config["dynamics"]["estimator"],
            "minco_plans": minco_plans,
            "planner_summary": planner_summary,
        }
        np.savez_compressed(trajectory_path, metadata=json.dumps(metadata), **arrays)
        result["trajectory"] = str(trajectory_path)
        result["task_success"] = bool(task.check_success())
        result["task_failed_flag"] = bool(task.task_failed)
        result["metrics"] = _metrics(arrays, task.flying_hand_config)
        result["status"] = "success" if result["task_success"] else "task_failed"
    except Exception as exc:
        result.update({
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        planner.step = original_step
        if task is not None:
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument(
        "--dynamics-patch",
        default="{}",
        help="JSON object merged into the dynamics section for an experiment only.",
    )
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count:
        raise SystemExit("worker-index must be in [0, worker-count)")
    try:
        dynamics_patch = json.loads(args.dynamics_patch)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid --dynamics-patch JSON: {exc}") from exc
    if not isinstance(dynamics_patch, dict):
        raise SystemExit("--dynamics-patch must be a JSON object")

    output_dir = args.output_dir.resolve()
    (output_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (output_dir / "workers").mkdir(parents=True, exist_ok=True)
    jobs = [(task, seed) for task in args.tasks for seed in args.seeds]
    worker_jobs = [job for index, job in enumerate(jobs) if index % args.worker_count == args.worker_index]
    worker_path = output_dir / "workers" / f"worker_{args.worker_index:02d}.jsonl"
    with worker_path.open("w", encoding="utf-8") as stream:
        for task_name, seed in worker_jobs:
            result = run_episode(task_name, args.task_config, seed, output_dir, dynamics_patch)
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
