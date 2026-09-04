from dataclasses import asdict, dataclass

import numpy as np
import sapien

try:
    from . import _minco_cpp
except ImportError as error:
    _minco_cpp = None
    _MINCO_CPP_IMPORT_ERROR = error
else:
    _MINCO_CPP_IMPORT_ERROR = None


def _require_minco_cpp():
    if _minco_cpp is None:
        raise RuntimeError(
            "The Flying-Hand C++ MINCO module is not built. Run "
            "`.venv/bin/python script/build_flying_hand_minco_cpp.py` "
            "from the repository root."
        ) from _MINCO_CPP_IMPORT_ERROR


def _dynamic_components(actor):
    return [
        component
        for component in actor.actor.components
        if isinstance(component, sapien.physx.PhysxRigidDynamicComponent)
    ]


def _collision_shapes(entity):
    shapes = []
    for component in entity.components:
        get_collision_shapes = getattr(component, "get_collision_shapes", None)
        if get_collision_shapes is not None:
            shapes.extend(get_collision_shapes())
    return shapes


def restore_released_actor_collisions(env):
    """Restore collisions between a released actor and the opened gripper."""
    state = env._released_actor_collision_state
    if state is None:
        return
    for shape, groups in state["collision_groups"]:
        shape.set_collision_groups(groups)
    env._released_actor_collision_state = None


def _suppress_released_actor_gripper_collisions(env, actor):
    """Keep release physical while an opening gripper still surrounds the actor."""
    restore_released_actor_collisions(env)
    hand_shapes = [
        shape
        for link in env.flying_hand.get_links()
        for shape in link.get_collision_shapes()
    ]
    signatures = {
        tuple(shape.get_collision_groups()[2:4])
        for shape in hand_shapes
    }
    if len(signatures) != 1:
        raise RuntimeError(
            "Flying-hand collision shapes do not share one self-collision filter"
        )
    ignore_mask, filter_id = signatures.pop()
    if ignore_mask == 0:
        raise RuntimeError("Flying-hand self-collision filter is not configured")

    collision_groups = []
    for shape in _collision_shapes(actor.actor):
        groups = shape.get_collision_groups()
        collision_groups.append((shape, list(groups)))
        groups[2] |= ignore_mask
        groups[3] = filter_id
        shape.set_collision_groups(groups)
    env._released_actor_collision_state = {
        "actor": actor,
        "collision_groups": collision_groups,
    }


def set_pose(env, pose, vel=None):
    env.flying_hand_ref_pose = pose
    env.flying_hand.set_root_pose(pose)
    vel = (np.zeros(3) if vel is None else vel).tolist()
    env.flying_hand.set_root_linear_velocity(vel)
    env.flying_hand.set_root_angular_velocity([0, 0, 0])


def _begin_isolated_carry(env, actor):
    """Temporarily remove a scripted carried actor from PhysX simulation."""
    restore_released_actor_collisions(env)
    state = env._isolated_carried_actor_state
    if state is not None:
        if state["actor"] is actor:
            return
        raise RuntimeError("An isolated carried actor is already active")

    components = _dynamic_components(actor)
    if not components:
        raise ValueError(f"Carried actor {actor.get_name()!r} has no dynamic component")
    excluded_components = []
    for entity in env._get_isolated_carry_exclusions(actor):
        for component in entity.components:
            if isinstance(
                component,
                (
                    sapien.physx.PhysxRigidDynamicComponent,
                    sapien.physx.PhysxRigidStaticComponent,
                ),
            ):
                excluded_components.append((component, bool(component.is_enabled)))

    env._isolated_carried_actor_state = {
        "actor": actor,
        "components": components,
        "enabled": [bool(component.is_enabled) for component in components],
        "excluded_components": excluded_components,
    }
    for component in components:
        component.disable()
    for component, _ in excluded_components:
        component.disable()


def begin_isolated_carry(env, actor):
    """Start the supported isolated-carry mode for an already validated actor."""
    _begin_isolated_carry(env, actor)


def _set_isolated_actor_pose(actor, pose):
    actor.actor.set_pose(pose)
    for component in _dynamic_components(actor):
        component.set_entity_pose(pose)


def set_isolated_carried_actor_target(env, actor, pose):
    """Set an isolated actor to an exact scripted endpoint before release."""
    state = env._isolated_carried_actor_state
    if state is None or state["actor"] is not actor:
        raise RuntimeError("The requested actor is not the active isolated carried actor")
    _set_isolated_actor_pose(actor, pose)


def restore_isolated_carried_actor(env, suppress_gripper_collisions=False):
    """Restore an isolated actor at its scripted pose with zero velocity.

    During release, the actor immediately rejoins PhysX so gravity and support
    contacts take effect.  Its collision with the still-opening gripper can be
    suppressed independently until the open command finishes.
    """
    state = env._isolated_carried_actor_state
    if state is None:
        return

    pose = state["actor"].get_pose()
    if suppress_gripper_collisions:
        _suppress_released_actor_gripper_collisions(env, state["actor"])
    for component, was_enabled in state["excluded_components"]:
        if was_enabled:
            component.enable()
    for component, was_enabled in zip(state["components"], state["enabled"]):
        component.set_entity_pose(pose)
        component.set_linear_velocity([0, 0, 0])
        component.set_angular_velocity([0, 0, 0])
        if was_enabled:
            component.enable()
            component.wake_up()
    env._isolated_carried_actor_state = None


def step(env, n, save_freq=-1, step_callback=None):
    for _ in range(n):
        env.apply_flying_hand_gripper_qpos()
        env.scene.step()
        if env.enable_dynamics:
            # The floating articulation root is governed by the analytical
            # flight dynamics. PhysX is still required for gripper joints and
            # actor contacts, but it must not integrate the same root a second
            # time (especially with collision impulses) after the controller
            # has already advanced it for this simulation step.
            dynamics = env.flying_hand_dynamics
            env.flying_hand.set_root_pose(sapien.Pose(dynamics.p.tolist(), dynamics.q.tolist()))
            env.flying_hand.set_root_linear_velocity(dynamics.v.tolist())
            env.flying_hand.set_root_angular_velocity(dynamics.w.tolist())
        if (
            env._released_actor_collision_state is not None
            and not env.is_grasping
            and env.flying_hand_gripper_step >= env.flying_hand_gripper_steps
        ):
            restore_released_actor_collisions(env)
        env._task_objects_safe()
        if step_callback is not None:
            step_callback(env)
        env.flying_hand_save_step += 1
        if env.render_freq and env.flying_hand_save_step % env.render_freq == 0:
            env._update_render()
            env.viewer.render()
        env._save_flying_hand_frame(save_freq)


def hold(env, pose, steps, save_freq=-1, step_callback=None):
    for _ in range(steps):
        if env.enable_dynamics:
            env.flying_hand_ref_pose = pose
            hand_pose, hand_v = env.flying_hand_dynamics.step(
                pose,
                np.zeros(3),
                np.zeros(3),
                env.is_grasping
                and env.flying_hand_gripper_step >= env.flying_hand_gripper_steps,
            )
            env.flying_hand.set_root_pose(hand_pose)
            env.flying_hand.set_root_linear_velocity(hand_v.tolist())
            env.flying_hand.set_root_angular_velocity(env.flying_hand_dynamics.w.tolist())
        else:
            set_pose(env, pose)
        step(env, 1, save_freq=save_freq, step_callback=step_callback)


def _position_coefficients(points, times):
    _require_minco_cpp()
    return np.asarray(
        _minco_cpp.generate_position_coefficients(points, times),
        dtype=float,
    )


def sample(coeff, t):
    x = np.array([1, t, t**2, t**3, t**4, t**5])
    v = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
    a = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3])
    return x @ coeff, v @ coeff, a @ coeff


def slerp(q0, q1, t):
    q0, q1 = np.array(q0, dtype=float), np.array(q1, dtype=float)
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


@dataclass(frozen=True)
class MincoOptimizationConfig:
    """Runtime subset of the deployment TrajOpt parameters.

    Task geometry is kept fixed.  Only positive segment durations are
    optimized, matching ``fix_p: true`` in the deployment planner.
    """

    enabled: bool = True
    segment_per_distance: float = 0.5
    min_piece_num: int = 2
    plan_max_vel: float = 0.6
    plan_max_acc: float = 0.8
    plan_max_dyaw: float = 0.5
    K: int = 8
    rhoT: float = 10.0
    rhoV: float = 1000.0
    rhoA: float = 1000.0
    rhoDYaw: float = 1000.0
    rhoYawAlignmentAngle: float = 1000.0
    max_iteration: int = 300

    @classmethod
    def from_mapping(cls, values):
        values = {} if values is None else dict(values)
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unsupported MINCO optimization parameters: {unknown}")
        config = cls(**values)
        positive = {
            "segment_per_distance": config.segment_per_distance,
            "plan_max_vel": config.plan_max_vel,
            "plan_max_acc": config.plan_max_acc,
            "plan_max_dyaw": config.plan_max_dyaw,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"MINCO optimization parameters must be positive: {invalid}")
        weights = {
            "rhoT": config.rhoT,
            "rhoV": config.rhoV,
            "rhoA": config.rhoA,
            "rhoDYaw": config.rhoDYaw,
            "rhoYawAlignmentAngle": config.rhoYawAlignmentAngle,
        }
        invalid_weights = [name for name, value in weights.items() if value < 0.0]
        if invalid_weights:
            raise ValueError(
                f"MINCO optimization penalty weights must be non-negative: {invalid_weights}"
            )
        if config.min_piece_num < 1 or config.K < 1 or config.max_iteration < 1:
            raise ValueError(
                "MINCO min_piece_num, K and max_iteration must be positive integers"
            )
        return config


@dataclass(frozen=True)
class MincoOptimizationResult:
    times: np.ndarray
    initial_times: np.ndarray
    success: bool
    used_fallback: bool
    status: int
    message: str
    iterations: int
    function_evaluations: int
    initial_cost: float
    final_cost: float
    metrics: dict
    backend: str
    optimization_wall_time_seconds: float
    safety_time_scale: float = 1.0
    limits_satisfied: bool = True

    def as_dict(self):
        result = asdict(self)
        result["times"] = self.times.tolist()
        result["initial_times"] = self.initial_times.tolist()
        return result


@dataclass(frozen=True)
class GoalGraspPlannerConfig:
    """Reach-gating parameters used by deployment goal_grasp.cpp."""

    position_tolerance: float = 0.05
    yaw_tolerance: float = 0.1
    reach_timeout_seconds: float = 3.0

    @classmethod
    def from_mapping(cls, values):
        values = {} if values is None else dict(values)
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unsupported goal-grasp planner parameters: {unknown}")
        config = cls(**values)
        if (
            config.position_tolerance <= 0.0
            or config.yaw_tolerance <= 0.0
            or config.reach_timeout_seconds <= 0.0
        ):
            raise ValueError("Goal-grasp tolerances and timeout must be positive")
        return config


def _quaternion_yaw(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _angle_error(start, end):
    return float((end - start + np.pi) % (2.0 * np.pi) - np.pi)


def _minco_yaw_points(poses):
    start = _quaternion_yaw(poses[0].q)
    terminal = start + _angle_error(start, _quaternion_yaw(poses[-1].q))
    return np.linspace(start, terminal, len(poses))


def _yaw_coefficients(yaw_points, times, rates=None):
    _require_minco_cpp()
    rates = np.zeros(2) if rates is None else np.asarray(
        [rates[0], rates[-1]], dtype=float
    )
    return np.asarray(
        _minco_cpp.generate_yaw_coefficients(
            yaw_points,
            times,
            float(rates[0]),
            float(rates[1]),
        ),
        dtype=float,
    )


def sample_yaw(coeff, time):
    yaw = coeff @ np.array([1.0, time, time**2, time**3])
    yaw_rate = coeff @ np.array([0.0, 1.0, 2.0 * time, 3.0 * time**2])
    yaw_acceleration = coeff @ np.array([0.0, 0.0, 2.0, 6.0 * time])
    return float(yaw), float(yaw_rate), float(yaw_acceleration)


def _trajectory_metrics(
    points,
    yaw_points,
    times,
    samples_per_piece,
    yaw_rates=None,
):
    """Sample runtime trajectory limits without duplicating the C++ objective."""
    coefficients = _position_coefficients(points, times)
    yaw_coefficients = _yaw_coefficients(yaw_points, times, rates=yaw_rates)
    max_velocity = 0.0
    max_acceleration = 0.0
    max_yaw_rate = 0.0

    for coefficient, yaw_coefficient, duration in zip(
        coefficients, yaw_coefficients, times
    ):
        step_time = duration / samples_per_piece
        for sample_index in range(samples_per_piece + 1):
            sample_time = sample_index * step_time
            _, velocity, acceleration = sample(coefficient, sample_time)
            _, yaw_rate, _ = sample_yaw(yaw_coefficient, sample_time)
            velocity_norm = float(np.linalg.norm(velocity))
            acceleration_norm = float(np.linalg.norm(acceleration))
            max_velocity = max(max_velocity, velocity_norm)
            max_acceleration = max(max_acceleration, acceleration_norm)
            max_yaw_rate = max(max_yaw_rate, abs(yaw_rate))
    return {
        "max_velocity_mps": max_velocity,
        "max_acceleration_mps2": max_acceleration,
        "max_yaw_rate_rad_s": max_yaw_rate,
        "total_time_s": float(np.sum(times)),
    }


def _retime_to_hard_limits(points, yaw_points, times, config, yaw_rates=None):
    """Uniformly stretch a MINCO trajectory until sampled limits are satisfied.

    The C++ objective uses smooth penalties for velocity, acceleration, and yaw
    rate.  Policy-generated waypoints can be far outside the training cadence,
    so a final deterministic retiming pass prevents a finite penalty from
    accepting an unsafe high-acceleration trajectory.
    """
    safe_times = np.asarray(times, dtype=float).copy()
    total_scale = 1.0
    for _ in range(4):
        metrics = _trajectory_metrics(
            points,
            yaw_points,
            safe_times,
            config.K,
            yaw_rates=yaw_rates,
        )
        scale = max(
            1.0,
            metrics["max_velocity_mps"] / config.plan_max_vel,
            np.sqrt(metrics["max_acceleration_mps2"] / config.plan_max_acc),
            metrics["max_yaw_rate_rad_s"] / config.plan_max_dyaw,
        )
        if scale <= 1.0 + 1e-6:
            return safe_times, metrics, total_scale, True
        # A small margin avoids landing immediately outside a sampled limit due
        # to floating-point roundoff when coefficients are regenerated.
        scale *= 1.01
        safe_times *= scale
        total_scale *= scale

    metrics = _trajectory_metrics(
        points,
        yaw_points,
        safe_times,
        config.K,
        yaw_rates=yaw_rates,
    )
    limits_satisfied = bool(
        metrics["max_velocity_mps"] <= config.plan_max_vel * (1.0 + 1e-6)
        and metrics["max_acceleration_mps2"] <= config.plan_max_acc * (1.0 + 1e-6)
        and metrics["max_yaw_rate_rad_s"] <= config.plan_max_dyaw * (1.0 + 1e-6)
    )
    return safe_times, metrics, total_scale, limits_satisfied


class MincoTimeOptimizer:
    """Optimize positive MINCO segment durations while fixing all waypoints."""

    def __init__(self, config):
        self.config = config

    def _initial_times(self, poses, initial_yaw_rate=0.0):
        points = np.asarray([pose.p for pose in poses], dtype=float)
        distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
        yaw_points = _minco_yaw_points(poses)
        yaw_rates = np.array([initial_yaw_rate, 0.0], dtype=float)
        # Deployment getInitTraj() initializes time from xyz distance only;
        # yaw-rate is a soft objective penalty, not an initial hard bound.
        times = distances / self.config.plan_max_vel
        times = np.maximum(times, 1e-3)
        for attempt in range(3):
            metrics = _trajectory_metrics(
                points,
                yaw_points,
                times,
                self.config.K,
                yaw_rates=yaw_rates,
            )
            if (
                metrics["max_velocity_mps"] <= self.config.plan_max_vel
                and metrics["max_acceleration_mps2"] <= self.config.plan_max_acc
            ):
                break
            if attempt < 2:
                times *= 1.5
        return times

    def optimize(self, poses, initial_yaw_rate=0.0):
        initial_times = self._initial_times(poses, initial_yaw_rate)
        points = np.asarray([pose.p for pose in poses], dtype=float)
        yaw_points = _minco_yaw_points(poses)
        yaw_rates = np.array([initial_yaw_rate, 0.0], dtype=float)
        initial_metrics = _trajectory_metrics(
            points,
            yaw_points,
            initial_times,
            self.config.K,
            yaw_rates=yaw_rates,
        )
        if not self.config.enabled:
            return MincoOptimizationResult(
                times=initial_times,
                initial_times=initial_times,
                success=True,
                used_fallback=False,
                status=0,
                message="MINCO time optimization disabled",
                iterations=0,
                function_evaluations=1,
                initial_cost=0.0,
                final_cost=0.0,
                metrics=initial_metrics,
                backend="disabled",
                optimization_wall_time_seconds=0.0,
            )

        result = _minco_cpp.optimize_times(
            points,
            yaw_points,
            initial_times,
            float(initial_yaw_rate),
            asdict(self.config),
        )
        candidate_times = np.asarray(result["times"], dtype=float)
        candidate_valid = (
            candidate_times.shape == initial_times.shape
            and np.all(np.isfinite(candidate_times))
            and np.all(candidate_times > 0.0)
        )
        success = bool(
            result["success"]
            and candidate_valid
            and np.isfinite(float(result["final_cost"]))
        )
        if success:
            final_times = candidate_times
            final_cost = float(result["final_cost"])
        else:
            final_times = initial_times
            final_cost = float(result["initial_cost"])
        final_times, final_metrics, safety_time_scale, limits_satisfied = (
            _retime_to_hard_limits(
                points,
                yaw_points,
                final_times,
                self.config,
                yaw_rates=yaw_rates,
            )
        )
        if not limits_satisfied:
            success = False
            result["message"] = (
                f"{result['message']}; hard trajectory limits remain violated "
                "after safety retiming"
            )
        return MincoOptimizationResult(
            times=final_times,
            initial_times=initial_times,
            success=success,
            used_fallback=not success,
            status=int(result["status"]),
            message=str(result["message"]),
            iterations=int(result["iterations"]),
            function_evaluations=int(result["function_evaluations"]),
            initial_cost=float(result["initial_cost"]),
            final_cost=float(final_cost),
            metrics=final_metrics,
            backend=str(_minco_cpp.backend_name),
            optimization_wall_time_seconds=float(result["wall_time_seconds"]),
            safety_time_scale=safety_time_scale,
            limits_satisfied=limits_satisfied,
        )


def _densify_plan(poses, config):
    """Subdivide every task waypoint leg without moving its key endpoints.

    ``segment_per_distance`` retains the deployment planner's distance-based
    subdivision, while ``min_piece_num`` guarantees enough pieces globally.
    """
    dense_poses = [poses[0]]
    for start, end in zip(poses[:-1], poses[1:]):
        distance = float(np.linalg.norm(np.asarray(end.p) - np.asarray(start.p)))
        # C++ std::round() rounds positive half-integers away from zero,
        # whereas Python round() uses bankers rounding.
        piece_count = max(
            1,
            int(np.floor(distance / config.segment_per_distance + 0.5)),
        )
        for index in range(1, piece_count + 1):
            ratio = index / piece_count
            position = (1.0 - ratio) * np.asarray(start.p) + ratio * np.asarray(end.p)
            orientation = slerp(start.q, end.q, ratio)
            dense_poses.append(sapien.Pose(position.tolist(), orientation.tolist()))

    while len(dense_poses) - 1 < config.min_piece_num:
        distances = [
            np.linalg.norm(np.asarray(end.p) - np.asarray(start.p))
            for start, end in zip(dense_poses[:-1], dense_poses[1:])
        ]
        split_index = int(np.argmax(distances))
        start = dense_poses[split_index]
        end = dense_poses[split_index + 1]
        midpoint = sapien.Pose(
            ((np.asarray(start.p) + np.asarray(end.p)) * 0.5).tolist(),
            slerp(start.q, end.q, 0.5).tolist(),
        )
        dense_poses.insert(split_index + 1, midpoint)
    return dense_poses


def plan_and_move_minco(
    env,
    poses,
    time_hints,
    save_freq=-1,
    carried_actor=None,
    carried_pose=None,
    phase_name=None,
    step_callback=None,
):
    """Optimize task-path segment times, then execute the resulting MINCO path."""
    if len(poses) < 2:
        raise ValueError("A planned MINCO move requires at least two poses")
    config = env.minco_time_optimizer.config
    time_hints = np.asarray(time_hints, dtype=float)
    if time_hints.shape != (len(poses) - 1,) or np.any(time_hints <= 0.0):
        raise ValueError("time_hints must contain one positive value per input segment")
    requested_start = poses[0]
    actual_start = env.flying_hand.get_root_pose()
    poses = [actual_start, *poses[1:]]
    start_position_error = float(
        np.linalg.norm(np.asarray(actual_start.p) - np.asarray(requested_start.p))
    )
    start_yaw_error = abs(
        _angle_error(_quaternion_yaw(actual_start.q), _quaternion_yaw(requested_start.q))
    )
    measured_initial_yaw_rate = (
        float(env.flying_hand_dynamics.w[2]) if env.enable_dynamics else 0.0
    )
    # A disturbed vehicle can enter replanning above the configured reference
    # yaw-rate limit. Such a boundary derivative cannot be repaired by time
    # scaling, so bound the new reference while retaining the measured value in
    # diagnostics. The controller then decelerates from the measured state.
    initial_yaw_rate = float(np.clip(
        measured_initial_yaw_rate,
        -config.plan_max_dyaw,
        config.plan_max_dyaw,
    ))
    dense_poses = _densify_plan(poses, config)
    result = env.minco_time_optimizer.optimize(
        dense_poses,
        initial_yaw_rate=initial_yaw_rate,
    )
    phase_index = len(env.minco_plan_diagnostics)
    diagnostic = result.as_dict()
    diagnostic.update({
        "phase": phase_name or f"move_{phase_index:03d}",
        "input_waypoint_count": len(poses),
        "optimized_waypoint_count": len(dense_poses),
        "inserted_waypoint_count": len(dense_poses) - len(poses),
        "waypoints_xyz": [np.asarray(pose.p, dtype=float).tolist() for pose in poses],
        "optimized_waypoints_xyz": [
            np.asarray(pose.p, dtype=float).tolist() for pose in dense_poses
        ],
        "legacy_time_hints": time_hints.tolist(),
        "start_position_error_m": start_position_error,
        "start_yaw_error_rad": start_yaw_error,
        "measured_initial_yaw_rate_rad_s": measured_initial_yaw_rate,
        "initial_yaw_rate_rad_s": initial_yaw_rate,
    })
    env.minco_plan_diagnostics.append(diagnostic)
    if not result.success:
        raise RuntimeError(
            f"MINCO time optimization failed for {diagnostic['phase']}: {result.message}"
        )
    move_minco(
        env,
        dense_poses,
        times=result.times,
        save_freq=save_freq,
        carried_actor=carried_actor,
        carried_pose=carried_pose,
        yaw_coefficients=_yaw_coefficients(
            _minco_yaw_points(dense_poses),
            result.times,
            rates=[initial_yaw_rate, 0.0],
        ),
        step_callback=step_callback,
    )
    return result


class TaskMotionPlanner:
    """Task-level motion/gripper state machine modelled after goal_grasp.cpp."""

    def __init__(self, env, save_freq):
        self.env = env
        self.save_freq = save_freq

    def _pose_errors(self, target_pose):
        actual_pose = self.env.flying_hand.get_root_pose()
        position_error = float(
            np.linalg.norm(np.asarray(actual_pose.p) - np.asarray(target_pose.p))
        )
        yaw_error = abs(
            _angle_error(_quaternion_yaw(actual_pose.q), _quaternion_yaw(target_pose.q))
        )
        return position_error, yaw_error

    def _wait_until_reached(self, target_pose, diagnostic):
        config = self.env.goal_grasp_planner_config
        timeout_steps = self.env._seconds_to_steps(config.reach_timeout_seconds)
        reached = False
        wait_steps = 0
        position_error = yaw_error = float("inf")
        for wait_steps in range(timeout_steps + 1):
            position_error, yaw_error = self._pose_errors(target_pose)
            if (
                position_error <= config.position_tolerance
                and yaw_error <= config.yaw_tolerance
            ):
                reached = True
                break
            if wait_steps < timeout_steps:
                hold(self.env, target_pose, 1, save_freq=self.save_freq)

        diagnostic.update({
            "reach_succeeded": reached,
            "reach_wait_steps": wait_steps,
            "reach_wait_time_s": wait_steps * self.env.sim_timestep,
            "reach_position_error_m": position_error,
            "reach_yaw_error_rad": yaw_error,
        })
        if not reached:
            raise RuntimeError(
                "Flying hand failed to reach "
                f"{diagnostic['phase']} within {config.reach_timeout_seconds:.3f}s: "
                f"position error {position_error:.4f}m, yaw error {yaw_error:.4f}rad"
            )

    def move(
        self,
        poses,
        time_hints,
        *,
        phase_name=None,
        carried_actor=None,
        carried_pose=None,
        gripper_after_reach=None,
        gripper_qpos=None,
    ):
        result = plan_and_move_minco(
            self.env,
            poses,
            time_hints,
            save_freq=self.save_freq,
            carried_actor=carried_actor,
            carried_pose=carried_pose,
            phase_name=phase_name,
        )
        if gripper_after_reach is not None:
            self.set_gripper(
                poses[-1],
                gripper_after_reach,
                qpos=gripper_qpos,
            )
        return result

    def set_gripper(self, pose, action, qpos=None):
        if action not in {"close", "open"}:
            raise ValueError("gripper action must be 'close' or 'open'")
        if not self.env.minco_plan_diagnostics:
            raise RuntimeError("A gripper command requires a preceding planned move")
        diagnostic = self.env.minco_plan_diagnostics[-1]
        diagnostic["reach_action"] = action
        self._wait_until_reached(pose, diagnostic)
        is_grasp = action == "close"
        if qpos is None:
            qpos = self.env.flying_hand_config["gripper"][
                "close_qpos" if is_grasp else "open_qpos"
            ]
        self.env.set_flying_hand_gripper(qpos, is_grasp=is_grasp)
        seconds = (
            self.env.grasp_hold_seconds
            if is_grasp
            else self.env.release_hold_seconds
        )
        hold(
            self.env,
            pose,
            self.env._seconds_to_steps(seconds),
            save_freq=self.save_freq,
        )

    def hold(self, pose, seconds):
        hold(
            self.env,
            pose,
            self.env._seconds_to_steps(seconds),
            save_freq=self.save_freq,
        )


def move_minco(
    env,
    poses,
    times,
    save_freq=-1,
    carried_actor=None,
    carried_pose=None,
    yaw_coefficients=None,
    step_callback=None,
):
    if carried_actor is not None:
        _begin_isolated_carry(env, carried_actor)

    ps = np.array([pose.p for pose in poses], dtype=float)
    times = np.asarray(times, dtype=float)
    if (
        times.shape != (len(poses) - 1,)
        or np.any(~np.isfinite(times))
        or np.any(times <= 0.0)
    ):
        raise ValueError("times must contain one finite positive duration per segment")
    coeffs = _position_coefficients(ps, times)
    carried_pose = carried_pose if carried_pose is not None else (
        poses[0].inv() * carried_actor.get_pose() if carried_actor is not None else None
    )
    env._flying_hand_carrying = carried_actor is not None
    try:
        if yaw_coefficients is not None and len(yaw_coefficients) != len(times):
            raise ValueError("yaw_coefficients must contain one polynomial per segment")
        for segment_index, (coeff, T, start, end) in enumerate(
            zip(coeffs, times, poses[:-1], poses[1:])
        ):
            for idx in range(max(1, int(np.ceil(T / env.sim_timestep)))):
                t = min((idx + 1) * env.sim_timestep, T)
                p, v, a = sample(coeff, t)
                if yaw_coefficients is None:
                    orientation = slerp(start.q, end.q, t / T)
                else:
                    yaw, _, _ = sample_yaw(yaw_coefficients[segment_index], t)
                    orientation = np.array([
                        np.cos(0.5 * yaw),
                        0.0,
                        0.0,
                        np.sin(0.5 * yaw),
                    ])
                ref_pose = sapien.Pose(p.tolist(), orientation.tolist())
                env.flying_hand_ref_pose = ref_pose
                if env.enable_dynamics:
                    hand_pose, hand_v = env.flying_hand_dynamics.step(
                        ref_pose,
                        v,
                        a,
                        env.is_grasping
                        and env.flying_hand_gripper_step >= env.flying_hand_gripper_steps,
                    )
                    env.flying_hand.set_root_pose(hand_pose)
                    env.flying_hand.set_root_linear_velocity(hand_v.tolist())
                    env.flying_hand.set_root_angular_velocity(env.flying_hand_dynamics.w.tolist())
                else:
                    hand_pose, hand_v = ref_pose, v
                    set_pose(env, hand_pose, hand_v)
                if carried_actor is not None:
                    _set_isolated_actor_pose(carried_actor, hand_pose * carried_pose)
                step(
                    env,
                    1,
                    save_freq=save_freq,
                    step_callback=step_callback,
                )
    finally:
        env._flying_hand_carrying = False
