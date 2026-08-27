#!/usr/bin/env python3
"""Run Flying-Hand grasp tasks and report closure-stage stability per seed.

The monitor only evaluates the physical gripper-closing interval.  Several
Flying-Hand tasks explicitly carry an object after closure, so task success by
itself is not evidence that the object was held stably by contact.
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
    "thread_tape_rod",
)


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


def run_once(task_name, task_config, seed):
    started = time.monotonic()
    result = {
        "task": task_name,
        "task_config": task_config,
        "seed": seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    task = None
    original_step = planner.step
    monitor = GraspMonitor()

    def monitored_step(env, n, save_freq=-1, carried_pose_fn=None):
        for _ in range(n):
            monitor.before_step(env)
            original_step(env, 1, save_freq=save_freq, carried_pose_fn=carried_pose_fn)
            monitor.after_step(env)

    try:
        args = load_task_args(task_name, task_config, 0)
        args["seed"] = seed
        task = _class_for_task(task_name)()
        planner.step = monitored_step
        task.setup_demo(**args)
        task.play_once()
        monitor.finish(task)
        result["task_success"] = bool(task.check_success())
        result["task_failed_flag"] = bool(task.task_failed)
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
    args = parser.parse_args()

    for task_name in args.tasks:
        for seed in args.seeds:
            print(json.dumps(run_once(task_name, args.task_config, seed), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
