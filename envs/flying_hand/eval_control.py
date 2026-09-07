"""Shared Flying-Hand evaluation control path.

Policy implementations are intentionally kept out of this module.  Every
policy adapter produces a ``[T, 5]`` chunk with the same semantics:
``[x, y, z, yaw, grasp]``.  This module owns the conversion from that chunk to
the simulator reference, derivative limiting, grasp transitions, and episode
accounting.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def is_flying_hand_observation(observation: dict[str, Any]) -> bool:
    return (
        isinstance(observation, dict)
        and "flying_hand" in observation
        and "actual_state" in observation["flying_hand"]
    )


def canonical_action_chunk(actions: Any) -> np.ndarray:
    """Return a finite ``[T, 5]`` float32 Flying-Hand action chunk."""
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    elif array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"Expected one action batch, got shape {array.shape}")
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 5:
        raise ValueError(f"Expected Flying-Hand action chunk [T, 5], got {array.shape}")
    if array.shape[0] == 0:
        return array
    if not np.isfinite(array).all():
        raise ValueError("Flying-Hand action chunk contains NaN or infinity")
    return array


def relative_xyzyaw_to_world_pose(task_env: Any, action: np.ndarray):
    import sapien

    action = np.asarray(action, dtype=np.float32)
    if action.shape != (5,):
        raise ValueError(f"Expected Flying-Hand action shape (5,), got {action.shape}")
    relative_pose = sapien.Pose(
        action[:3].tolist(),
        [np.cos(float(action[3]) / 2.0), 0, 0, np.sin(float(action[3]) / 2.0)],
    )
    root_to_imu_initial = task_env.flying_hand_initial_pose.inv() * task_env.initial_imu_odom_pose
    return task_env.initial_imu_odom_pose * relative_pose * root_to_imu_initial.inv()


def _limit_vector_norm(value: np.ndarray, maximum: float) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= maximum or norm < 1e-12:
        return value
    return value * (maximum / norm)


def _waypoint_dt(task_env: Any) -> float:
    """Return the configured 20 Hz waypoint period."""
    config = getattr(task_env, "flying_hand_trajectory_execution", {})
    waypoint_hz = float(config.get("waypoint_hz", 0.0))
    if waypoint_hz > 0.0:
        return 1.0 / waypoint_hz
    return max(
        float(getattr(task_env, "save_freq", 1))
        * float(getattr(task_env, "sim_timestep", 1.0)),
        np.finfo(float).eps,
    )


def finite_difference_actions(task_env: Any, actions: Any) -> tuple[np.ndarray, np.ndarray]:
    """Estimate waypoint velocity and acceleration using the task cadence."""
    actions = canonical_action_chunk(actions)
    positions = np.stack(
        [np.asarray(relative_xyzyaw_to_world_pose(task_env, action).p, dtype=float)
         for action in actions]
    ) if len(actions) else np.empty((0, 3), dtype=float)
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    if len(positions) < 2:
        return velocities, accelerations

    dt = _waypoint_dt(task_env)
    velocities[0] = (positions[1] - positions[0]) / dt
    velocities[-1] = (positions[-1] - positions[-2]) / dt
    if len(positions) > 2:
        velocities[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)

    limits = task_env.flying_hand_waypoint_tracking
    velocities = np.stack([
        _limit_vector_norm(value, float(limits["max_velocity"])) for value in velocities
    ])
    accelerations[0] = (velocities[1] - velocities[0]) / dt
    accelerations[-1] = (velocities[-1] - velocities[-2]) / dt
    if len(positions) > 2:
        accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2.0 * dt)
    accelerations = np.stack([
        _limit_vector_norm(value, float(limits["max_acceleration"])) for value in accelerations
    ])
    return velocities, accelerations


def _slerp_towards(current: np.ndarray, target: np.ndarray, max_angle: float) -> np.ndarray:
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
    if angle <= max_angle or angle < 1e-9:
        return target
    fraction = max_angle / angle
    half_angle = 0.5 * angle
    sin_half_angle = float(np.sin(half_angle))
    if sin_half_angle < 1e-9:
        result = (1.0 - fraction) * current + fraction * target
    else:
        result = (
            np.sin((1.0 - fraction) * half_angle) / sin_half_angle * current
            + np.sin(fraction * half_angle) / sin_half_angle * target
        )
    return result / np.linalg.norm(result)


def reset_reference(model: Any) -> None:
    """Reset the one reference state owned by the shared executor."""
    model._flying_hand_reference_position = None
    model._flying_hand_reference_velocity = np.zeros(3, dtype=float)
    model._flying_hand_reference_acceleration = np.zeros(3, dtype=float)
    model._flying_hand_reference_orientation = None


def _ensure_reference(task_env: Any, model: Any) -> tuple[str, str, str, str]:
    names = (
        "_flying_hand_reference_position",
        "_flying_hand_reference_velocity",
        "_flying_hand_reference_acceleration",
        "_flying_hand_reference_orientation",
    )
    position_name, velocity_name, acceleration_name, orientation_name = names
    if getattr(model, position_name, None) is None:
        pose = task_env.flying_hand.get_root_pose()
        setattr(model, position_name, np.asarray(pose.p, dtype=float).copy())
        setattr(model, velocity_name, np.zeros(3, dtype=float))
        setattr(model, acceleration_name, np.zeros(3, dtype=float))
        setattr(model, orientation_name, np.asarray(pose.q, dtype=float).copy())
    return names


def _start_generic_carry(task_env: Any, model: Any) -> bool:
    from envs.flying_hand import planner

    candidates = []
    for actor in task_env.get_flying_hand_grasp_candidates():
        diagnostic = task_env.get_flying_hand_grasp_diagnostic(actor)
        if diagnostic.get("eligible"):
            candidates.append((float(np.linalg.norm(diagnostic["actor_center_u"])), actor))
    if not candidates:
        return False
    _, actor = min(candidates, key=lambda item: item[0])
    planner.begin_isolated_carry(task_env, actor)
    model.attached_actor = actor
    model.attached_pose = task_env.flying_hand.get_root_pose().inv() * actor.get_pose()
    return True


def _apply_grasp_edge(task_env: Any, model: Any, grasp: bool, action_step: int) -> bool:
    previous = bool(getattr(model, "grasp_commanded", False))
    if bool(grasp) == previous:
        return previous
    hook = getattr(model, "_apply_flying_hand_grasp_command", None)
    if callable(hook):
        hook(task_env, bool(grasp), action_step=int(action_step))
        return bool(getattr(model, "grasp_commanded", grasp))

    model.grasp_commanded = bool(grasp)
    if grasp:
        attached = _start_generic_carry(task_env, model)
        if attached:
            task_env.set_flying_hand_gripper(
                task_env.flying_hand_config["gripper"]["close_qpos"], is_grasp=True
            )
            return True
        task_env.set_flying_hand_gripper(
            task_env.flying_hand_config["gripper"]["open_qpos"], is_grasp=False
        )
        model.grasp_commanded = False
        return False

    task_env.set_flying_hand_gripper(
        task_env.flying_hand_config["gripper"]["open_qpos"], is_grasp=False
    )
    model.attached_actor = None
    model.attached_pose = None
    return False


def _build_linear_reference_samples(task_env: Any, poses: list[Any]):
    """Sample adjacent 20 Hz waypoints linearly at the simulator rate.

    The model waypoint cadence is independent from the simulator timestep.
    The position is continuous, velocity is constant within a segment, and
    feed-forward acceleration is zero.  The dynamics controller remains the
    only component that turns this reference into a physical state.
    """
    if len(poses) < 2:
        return []
    import sapien

    sim_dt = float(task_env.sim_timestep)
    segment_duration = _waypoint_dt(task_env)
    sample_count = max(1, int(round(segment_duration / sim_dt)))
    samples = []
    for start, end in zip(poses[:-1], poses[1:]):
        start_position = np.asarray(start.p, dtype=float)
        end_position = np.asarray(end.p, dtype=float)
        velocity = (end_position - start_position) / segment_duration
        start_yaw = float(np.arctan2(
            2.0 * (start.q[0] * start.q[3] + start.q[1] * start.q[2]),
            1.0 - 2.0 * (start.q[2] ** 2 + start.q[3] ** 2),
        ))
        end_yaw = float(np.arctan2(
            2.0 * (end.q[0] * end.q[3] + end.q[1] * end.q[2]),
            1.0 - 2.0 * (end.q[2] ** 2 + end.q[3] ** 2),
        ))
        yaw_delta = (end_yaw - start_yaw + np.pi) % (2.0 * np.pi) - np.pi
        for sample_index in range(sample_count):
            fraction = (sample_index + 1) / sample_count
            position = (1.0 - fraction) * start_position + fraction * end_position
            yaw = start_yaw + fraction * yaw_delta
            orientation = np.asarray(
                [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)],
                dtype=float,
            )
            samples.append(
                (
                    sapien.Pose(position.tolist(), orientation.tolist()),
                    np.asarray(velocity, dtype=float),
                    np.zeros(3, dtype=float),
                )
            )
    return samples


def _constrain_policy_target_pose(task_env: Any, previous_pose: Any, target_pose: Any):
    """Limit one policy waypoint to one control-period of reachable motion.

    A learned action can be a large outlier even when adjacent training
    targets are smooth. Feeding that outlier directly into the 200 Hz
    reference creates an unreachable segment. Retaining the direction while
    limiting the per-waypoint translation/yaw is the same incremental target
    contract used by the arm executors.
    """
    import sapien

    previous = np.asarray(previous_pose.p, dtype=float)
    target = np.asarray(target_pose.p, dtype=float)
    limits = task_env.flying_hand_waypoint_tracking
    action_dt = _waypoint_dt(task_env)
    max_distance = float(limits["max_velocity"]) * action_dt
    delta = target - previous
    distance = float(np.linalg.norm(delta))
    if distance > max_distance > 0.0:
        target = previous + delta * (max_distance / distance)

    previous_yaw = float(np.arctan2(
        2.0 * (previous_pose.q[0] * previous_pose.q[3] + previous_pose.q[1] * previous_pose.q[2]),
        1.0 - 2.0 * (previous_pose.q[2] ** 2 + previous_pose.q[3] ** 2),
    ))
    target_yaw = float(np.arctan2(
        2.0 * (target_pose.q[0] * target_pose.q[3] + target_pose.q[1] * target_pose.q[2]),
        1.0 - 2.0 * (target_pose.q[2] ** 2 + target_pose.q[3] ** 2),
    ))
    yaw_delta = (target_yaw - previous_yaw + np.pi) % (2.0 * np.pi) - np.pi
    max_yaw_delta = float(limits["max_yaw_rate"]) * action_dt
    if abs(yaw_delta) > max_yaw_delta > 0.0:
        target_yaw = previous_yaw + np.sign(yaw_delta) * max_yaw_delta
    orientation = np.asarray(
        [np.cos(0.5 * target_yaw), 0.0, 0.0, np.sin(0.5 * target_yaw)],
        dtype=float,
    )
    return sapien.Pose(target.tolist(), orientation.tolist())


def _step_dynamics_safely(
    task_env: Any,
    reference_pose: Any,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    grasped: bool,
):
    """Advance analytical dynamics without forwarding a numerical explosion.

    The learned policy is allowed to be wrong, but one bad target must not
    turn into an unbounded SAPIEN root pose. The recovery is deliberately
    local to policy evaluation; expert planner execution is unchanged.
    Resetting the observer at the reference pose also prevents a
    stale estimator residual from re-launching the body on the next step.
    """
    dynamics = task_env.flying_hand_dynamics
    hand_pose, hand_v = dynamics.step(
        reference_pose,
        velocity,
        acceleration,
        grasped,
    )
    limits = task_env.flying_hand_waypoint_tracking
    max_error = max(0.25, float(limits["max_velocity"]) * 0.75)
    position_error = float(
        np.linalg.norm(np.asarray(hand_pose.p, dtype=float) - np.asarray(reference_pose.p, dtype=float))
    )
    finite = (
        np.isfinite(np.asarray(hand_pose.p, dtype=float)).all()
        and np.isfinite(np.asarray(hand_pose.q, dtype=float)).all()
        and np.isfinite(np.asarray(hand_v, dtype=float)).all()
        and np.isfinite(np.asarray(dynamics.w, dtype=float)).all()
    )
    if not finite or position_error > max_error:
        dynamics.sync(reference_pose)
        hand_pose = reference_pose
        hand_v = np.zeros(3, dtype=float)
        recoveries = int(getattr(task_env, "flying_hand_dynamics_recoveries", 0)) + 1
        task_env.flying_hand_dynamics_recoveries = recoveries
    return hand_pose, np.asarray(hand_v, dtype=float)


def execute_action_chunk(task_env: Any, model: Any, actions: Any) -> int:
    """Execute one complete policy chunk and update common eval accounting."""
    actions = canonical_action_chunk(actions)
    if len(actions) == 0:
        return 0
    if int(getattr(task_env, "take_action_cnt", 0)) == 0:
        task_env.flying_hand_dynamics_recoveries = 0
    if not hasattr(task_env, "take_action_cnt"):
        task_env.take_action_cnt = 0
    if callable(getattr(model, "_record_action_chunk", None)):
        model._record_action_chunk(task_env, actions)
    grasp_config = getattr(task_env, "flying_hand_grasp_validation", {})
    close_threshold = float(grasp_config.get("close_threshold", 0.50))
    open_threshold = float(grasp_config.get("open_threshold", 0.40))
    grasp_state = bool(getattr(model, "grasp_commanded", False))
    target_poses = []
    previous_target_pose = task_env.flying_hand.get_root_pose()
    for action in actions:
        pose = relative_xyzyaw_to_world_pose(task_env, action)
        target_filter = getattr(task_env, "filter_flying_hand_policy_target", None)
        if callable(target_filter):
            current = getattr(task_env, "flying_hand_ref_pose", None)
            if current is None:
                current = task_env.flying_hand.get_root_pose()
            filtered = target_filter(
                pose,
                current,
                carried_actor=getattr(model, "attached_actor", None),
            )
            pose = filtered[0] if isinstance(filtered, tuple) else filtered
        pose = _constrain_policy_target_pose(task_env, previous_target_pose, pose)
        target_poses.append(pose)
        previous_target_pose = pose

    poses = [task_env.flying_hand.get_root_pose(), *target_poses]
    # Both modes consume the same time-aligned linear reference.  The mode
    # switch below only selects kinematic playback versus FlyingHandDynamics.
    samples = _build_linear_reference_samples(task_env, poses)
    samples_per_waypoint = max(
        1,
        int(np.ceil(len(samples) / max(len(actions), 1))),
    )
    sample_index = 0
    for index, action in enumerate(actions):
        grasp_state = (
            float(action[4]) > open_threshold
            if grasp_state
            else float(action[4]) >= close_threshold
        )
        # A grasp edge is associated with the target waypoint, while the
        # pose itself is followed by the dense trajectory samples below.
        _apply_grasp_edge(
            task_env,
            model,
            grasp_state,
            int(getattr(task_env, "take_action_cnt", 0) + index),
        )
        segment_end = min(
            len(samples),
            sample_index + samples_per_waypoint,
        )
        for reference_pose, velocity, acceleration in samples[sample_index:segment_end]:
            task_env.flying_hand_ref_pose = reference_pose
            if bool(getattr(task_env, "enable_dynamics", False)):
                hand_pose, hand_v = _step_dynamics_safely(
                    task_env,
                    reference_pose,
                    velocity,
                    acceleration,
                    bool(getattr(task_env, "is_grasping", False))
                    and int(getattr(task_env, "flying_hand_gripper_step", 0))
                    >= int(getattr(task_env, "flying_hand_gripper_steps", 0)),
                )
                task_env.flying_hand.set_root_pose(hand_pose)
                task_env.flying_hand.set_root_linear_velocity(hand_v.tolist())
                task_env.flying_hand.set_root_angular_velocity(
                    task_env.flying_hand_dynamics.w.tolist()
                )
                applied_pose = hand_pose
            else:
                from envs.flying_hand import planner

                planner.set_pose(task_env, reference_pose, velocity)
                applied_pose = reference_pose
            if getattr(model, "attached_actor", None) is not None:
                from envs.flying_hand import planner

                planner.set_isolated_carried_actor_target(
                    task_env,
                    model.attached_actor,
                    applied_pose * model.attached_pose,
                )
            from envs.flying_hand import planner

            planner.step(task_env, 1, save_freq=None)
            sample_hook = getattr(model, "_record_flight_sample", None)
            if callable(sample_hook):
                sample_hook(task_env)
        sample_index = segment_end
    task_env.take_action_cnt += int(len(actions))
    check_success = getattr(task_env, "check_success", None)
    if callable(check_success):
        task_env.eval_success = check_success()
    return int(len(actions))
