from dataclasses import asdict, dataclass

import numpy as np
import sapien
from scipy.optimize import minimize


def _dynamic_components(actor):
    return [
        component
        for component in actor.actor.components
        if isinstance(component, sapien.physx.PhysxRigidDynamicComponent)
    ]


def set_pose(env, pose, vel=None):
    env.flying_hand_ref_pose = pose
    env.flying_hand.set_root_pose(pose)
    vel = (np.zeros(3) if vel is None else vel).tolist()
    env.flying_hand.set_root_linear_velocity(vel)
    env.flying_hand.set_root_angular_velocity([0, 0, 0])


def set_actor_pose(actor, pose, vel=None):
    actor.actor.set_pose(pose)
    vel = (np.zeros(3) if vel is None else vel).tolist()
    for component in actor.actor.components:
        if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
            component.set_entity_pose(pose)
            component.set_linear_velocity(vel)
            component.set_angular_velocity([0, 0, 0])


def _begin_isolated_carry(env, actor):
    """Temporarily remove a scripted carried actor from PhysX simulation."""
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


def restore_isolated_carried_actor(env):
    """Restore an isolated actor at its scripted pose with zero velocity."""
    state = env._isolated_carried_actor_state
    if state is None:
        return

    pose = state["actor"].get_pose()
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


def step(env, n, save_freq=-1, carried_pose_fn=None):
    for _ in range(n):
        env.apply_flying_hand_gripper_qpos()
        if carried_pose_fn is not None:
            actor, pose, vel = carried_pose_fn()
            set_actor_pose(actor, pose, vel)
        env.scene.step()
        env._task_objects_safe()
        env.flying_hand_save_step += 1
        if env.render_freq and env.flying_hand_save_step % env.render_freq == 0:
            env._update_render()
            env.viewer.render()
        env._save_flying_hand_frame(save_freq)


def hold(env, pose, steps, save_freq=-1):
    for _ in range(steps):
        if env.enable_dynamics:
            env.flying_hand_ref_pose = pose
            hand_pose, hand_v = env.flying_hand_dynamics.step(
                pose,
                np.zeros(3),
                np.zeros(3),
                env.is_grasping and env.flying_hand_gripper_step >= env.flying_hand_gripper_steps * env.flying_hand_gripper_prismatic_stage_ratio,
            )
            env.flying_hand.set_root_pose(hand_pose)
            env.flying_hand.set_root_linear_velocity(hand_v.tolist())
            env.flying_hand.set_root_angular_velocity(env.flying_hand_dynamics.w.tolist())
        else:
            set_pose(env, pose)
        if (
            not env.is_grasping
            and env._isolated_carried_actor_state is not None
            and env.flying_hand_gripper_step + 1 >= env.flying_hand_gripper_steps
        ):
            restore_isolated_carried_actor(env)
        step(env, 1, save_freq=save_freq)
    if (
        not env.is_grasping
        and env.flying_hand_gripper_step >= env.flying_hand_gripper_steps
    ):
        restore_isolated_carried_actor(env)


def minco(points, times, vels=None, accs=None):
    p = np.asarray(points, dtype=float)
    t = np.asarray(times, dtype=float)
    n = len(t)
    v = np.zeros((2, 3)) if vels is None else np.asarray([vels[0], vels[-1]], dtype=float)
    a = np.zeros((2, 3)) if accs is None else np.asarray([accs[0], accs[-1]], dtype=float)
    A = np.zeros((6 * n, 6 * n))
    b = np.zeros((6 * n, 3))
    A[0, 0] = A[1, 1] = 1
    A[2, 2] = 2
    b[:3] = [p[0], v[0], a[0]]
    for i, T in enumerate(t[:-1]):
        j = 6 * i
        A[j + 3, j + 3:j + 6] = [6, 24 * T, 60 * T**2]
        A[j + 3, j + 9] = -6
        A[j + 4, j + 4:j + 6] = [24, 120 * T]
        A[j + 4, j + 10] = -24
        A[j + 5, j:j + 6] = [1, T, T**2, T**3, T**4, T**5]
        A[j + 6, j:j + 7] = [1, T, T**2, T**3, T**4, T**5, -1]
        A[j + 7, j + 1:j + 8] = [1, 2*T, 3*T**2, 4*T**3, 5*T**4, 0, -1]
        A[j + 8, j + 2:j + 9] = [2, 6*T, 12*T**2, 20*T**3, 0, 0, -2]
        b[j + 5] = p[i + 1]
    T = t[-1]
    j = 6 * n
    A[j - 3, j - 6:j] = [1, T, T**2, T**3, T**4, T**5]
    A[j - 2, j - 5:j] = [1, 2*T, 3*T**2, 4*T**3, 5*T**4]
    A[j - 1, j - 4:j] = [2, 6*T, 12*T**2, 20*T**3]
    b[j - 3:j] = [p[-1], v[1], a[1]]
    return np.linalg.solve(A, b).reshape(n, 6, 3)


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
    min_piece_num_per_key_segment: int = 2
    min_piece_num: int = 2
    plan_max_vel: float = 0.6
    plan_max_acc: float = 0.8
    plan_max_dyaw: float = 0.5
    plan_max_path_deviation: float = 0.03
    K: int = 8
    rhoT: float = 10.0
    rhoV: float = 1000.0
    rhoA: float = 1000.0
    rhoDYaw: float = 1000.0
    rhoYawAlignmentAngle: float = 1000.0
    rhoPathDeviation: float = 1000.0
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
            "plan_max_path_deviation": config.plan_max_path_deviation,
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
            "rhoPathDeviation": config.rhoPathDeviation,
        }
        invalid_weights = [name for name, value in weights.items() if value < 0.0]
        if invalid_weights:
            raise ValueError(
                f"MINCO optimization penalty weights must be non-negative: {invalid_weights}"
            )
        if (
            config.min_piece_num_per_key_segment < 1
            or config.min_piece_num < 1
            or config.K < 1
            or config.max_iteration < 1
        ):
            raise ValueError(
                "MINCO min_piece_num_per_key_segment, min_piece_num, K and "
                "max_iteration must be positive integers"
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


def _smoothed_l1(value, mu=0.01):
    """Deployment planner's C2-smoothed positive L1 penalty."""
    if value < 0.0:
        return 0.0
    if value > mu:
        return value - 0.5 * mu
    ratio = value / mu
    return (mu - 0.5 * value) * ratio**3


def _forward_time(raw_time):
    raw_time = np.asarray(raw_time, dtype=float)
    positive = raw_time > 0.0
    duration = np.empty_like(raw_time)
    duration[positive] = (0.5 * raw_time[positive] + 1.0) * raw_time[positive] + 1.0
    denominator = (0.5 * raw_time[~positive] - 1.0) * raw_time[~positive] + 1.0
    duration[~positive] = 1.0 / denominator
    return duration


def _backward_time(duration):
    duration = np.asarray(duration, dtype=float)
    raw_time = np.empty_like(duration)
    large = duration > 1.0
    raw_time[large] = np.sqrt(2.0 * duration[large] - 1.0) - 1.0
    raw_time[~large] = 1.0 - np.sqrt(2.0 / duration[~large] - 1.0)
    return raw_time


def _segment_rotation_angle(start_q, end_q):
    start_q = np.asarray(start_q, dtype=float)
    end_q = np.asarray(end_q, dtype=float)
    start_q /= np.linalg.norm(start_q)
    end_q /= np.linalg.norm(end_q)
    return 2.0 * np.arccos(np.clip(abs(float(np.dot(start_q, end_q))), 0.0, 1.0))


def _quaternion_yaw(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=float)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _angle_error(start, end):
    return float((end - start + np.pi) % (2.0 * np.pi) - np.pi)


def _minco_yaw_points(poses):
    start = _quaternion_yaw(poses[0].q)
    terminal = start + _angle_error(start, _quaternion_yaw(poses[-1].q))
    return np.linspace(start, terminal, len(poses))


def minco_yaw(yaw_points, times, rates=None):
    yaw_points = np.asarray(yaw_points, dtype=float)
    times = np.asarray(times, dtype=float)
    piece_count = len(times)
    if yaw_points.shape != (piece_count + 1,):
        raise ValueError("yaw_points must contain one value per MINCO waypoint")
    rates = np.zeros(2) if rates is None else np.asarray([rates[0], rates[-1]], dtype=float)
    matrix = np.zeros((4 * piece_count, 4 * piece_count))
    target = np.zeros(4 * piece_count)
    matrix[0, 0] = matrix[1, 1] = 1.0
    target[:2] = [yaw_points[0], rates[0]]
    for index, duration in enumerate(times[:-1]):
        row = 4 * index
        matrix[row + 2, row + 2:row + 4] = [2.0, 6.0 * duration]
        matrix[row + 2, row + 6] = -2.0
        matrix[row + 3, row:row + 4] = [1.0, duration, duration**2, duration**3]
        matrix[row + 4, row:row + 5] = [1.0, duration, duration**2, duration**3, -1.0]
        matrix[row + 5, row + 1:row + 6] = [1.0, 2.0 * duration, 3.0 * duration**2, 0.0, -1.0]
        target[row + 3] = yaw_points[index + 1]
    duration = times[-1]
    row = 4 * piece_count
    matrix[row - 2, row - 4:row] = [1.0, duration, duration**2, duration**3]
    matrix[row - 1, row - 3:row] = [1.0, 2.0 * duration, 3.0 * duration**2]
    target[row - 2:row] = [yaw_points[-1], rates[1]]
    return np.linalg.solve(matrix, target).reshape(piece_count, 4)


def sample_yaw(coeff, time):
    yaw = coeff @ np.array([1.0, time, time**2, time**3])
    yaw_rate = coeff @ np.array([0.0, 1.0, 2.0 * time, 3.0 * time**2])
    yaw_acceleration = coeff @ np.array([0.0, 0.0, 2.0, 6.0 * time])
    return float(yaw), float(yaw_rate), float(yaw_acceleration)


def _jerk_energy(coeff, duration):
    a = 6.0 * coeff[3]
    b = 24.0 * coeff[4]
    c = 60.0 * coeff[5]
    return float(
        np.dot(a, a) * duration
        + np.dot(a, b) * duration**2
        + (np.dot(b, b) + 2.0 * np.dot(a, c)) * duration**3 / 3.0
        + np.dot(b, c) * duration**4 / 2.0
        + np.dot(c, c) * duration**5 / 5.0
    )


def _point_segment_distance(point, start, end):
    segment = end - start
    squared_length = float(np.dot(segment, segment))
    if squared_length <= 1e-16:
        return float(np.linalg.norm(point - start))
    ratio = float(np.clip(np.dot(point - start, segment) / squared_length, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + ratio * segment)))


def _minco_cost(
    points,
    yaw_points,
    times,
    config,
    yaw_rates=None,
    path_deviation_limit=None,
):
    coefficients = minco(points, times)
    yaw_coefficients = minco_yaw(yaw_points, times, rates=yaw_rates)
    cost_jerk = 0.0
    cost_yaw_acceleration = 0.0
    cost_velocity = 0.0
    cost_acceleration = 0.0
    cost_yaw_rate = 0.0
    cost_yaw_alignment = 0.0
    cost_path_deviation = 0.0
    max_velocity = 0.0
    max_acceleration = 0.0
    max_yaw_rate = 0.0
    max_path_deviation = 0.0

    target_yaw = yaw_points[-1]
    for segment_index, (coefficient, yaw_coefficient, duration) in enumerate(
        zip(coefficients, yaw_coefficients, times)
    ):
        cost_jerk += _jerk_energy(coefficient, duration)
        yaw_c2, yaw_c3 = yaw_coefficient[2:4]
        cost_yaw_acceleration += (
            4.0 * yaw_c2**2 * duration
            + 12.0 * yaw_c2 * yaw_c3 * duration**2
            + 12.0 * yaw_c3**2 * duration**3
        )
        step_time = duration / config.K
        for sample_index in range(config.K + 1):
            sample_time = sample_index * step_time
            trapezoid_weight = 0.5 if sample_index in (0, config.K) else 1.0
            position, velocity, acceleration = sample(coefficient, sample_time)
            yaw, yaw_rate, _ = sample_yaw(yaw_coefficient, sample_time)
            velocity_norm = float(np.linalg.norm(velocity))
            acceleration_norm = float(np.linalg.norm(acceleration))
            max_velocity = max(max_velocity, velocity_norm)
            max_acceleration = max(max_acceleration, acceleration_norm)
            max_yaw_rate = max(max_yaw_rate, abs(yaw_rate))
            path_deviation = _point_segment_distance(
                position,
                points[segment_index],
                points[segment_index + 1],
            )
            max_path_deviation = max(max_path_deviation, path_deviation)
            integration_weight = trapezoid_weight * step_time
            cost_velocity += integration_weight * config.rhoV * _smoothed_l1(
                velocity_norm**2 - config.plan_max_vel**2
            )
            cost_acceleration += integration_weight * config.rhoA * _smoothed_l1(
                acceleration_norm**2 - config.plan_max_acc**2
            )
            cost_yaw_rate += integration_weight * config.rhoDYaw * _smoothed_l1(
                yaw_rate**2 - config.plan_max_dyaw**2
            )
            cost_yaw_alignment += (
                integration_weight
                * config.rhoYawAlignmentAngle
                * _angle_error(yaw, target_yaw) ** 2
            )
            if path_deviation_limit is not None:
                cost_path_deviation += (
                    integration_weight
                    * config.rhoPathDeviation
                    * _smoothed_l1(path_deviation - path_deviation_limit)
                )

    cost_time = config.rhoT * float(np.sum(times))
    metrics = {
        "cost_time": cost_time,
        "cost_jerk": cost_jerk,
        "cost_yaw_acceleration": cost_yaw_acceleration,
        "cost_velocity": cost_velocity,
        "cost_acceleration": cost_acceleration,
        "cost_yaw_rate": cost_yaw_rate,
        "cost_yaw_alignment": cost_yaw_alignment,
        "cost_path_deviation": cost_path_deviation,
        "max_velocity_mps": max_velocity,
        "max_acceleration_mps2": max_acceleration,
        "max_yaw_rate_rad_s": max_yaw_rate,
        "max_path_deviation_m": max_path_deviation,
        "total_time_s": float(np.sum(times)),
    }
    return sum(metrics[key] for key in (
        "cost_time",
        "cost_jerk",
        "cost_yaw_acceleration",
        "cost_velocity",
        "cost_acceleration",
        "cost_yaw_rate",
        "cost_yaw_alignment",
        "cost_path_deviation",
    )), metrics


class MincoTimeOptimizer:
    """Optimize positive MINCO segment durations while fixing all waypoints."""

    def __init__(self, config):
        self.config = config

    def _initial_times(
        self,
        poses,
        initial_yaw_rate=0.0,
        constrain_path_deviation=False,
    ):
        points = np.asarray([pose.p for pose in poses], dtype=float)
        distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
        yaw_points = _minco_yaw_points(poses)
        yaw_rates = np.array([initial_yaw_rate, 0.0], dtype=float)
        # Deployment getInitTraj() initializes time from xyz distance only;
        # yaw-rate is a soft objective penalty, not an initial hard bound.
        times = distances / self.config.plan_max_vel
        times = np.maximum(times, 1e-3)
        for attempt in range(3):
            _, metrics = _minco_cost(
                points,
                yaw_points,
                times,
                self.config,
                yaw_rates=yaw_rates,
                path_deviation_limit=(
                    self.config.plan_max_path_deviation
                    if constrain_path_deviation
                    else None
                ),
            )
            if (
                metrics["max_velocity_mps"] <= self.config.plan_max_vel
                and metrics["max_acceleration_mps2"] <= self.config.plan_max_acc
            ):
                break
            if attempt < 2:
                times *= 1.5
        return times

    def optimize(
        self,
        poses,
        initial_yaw_rate=0.0,
        constrain_path_deviation=False,
    ):
        initial_times = self._initial_times(
            poses,
            initial_yaw_rate,
            constrain_path_deviation,
        )
        points = np.asarray([pose.p for pose in poses], dtype=float)
        yaw_points = _minco_yaw_points(poses)
        yaw_rates = np.array([initial_yaw_rate, 0.0], dtype=float)
        initial_cost, initial_metrics = _minco_cost(
            points,
            yaw_points,
            initial_times,
            self.config,
            yaw_rates=yaw_rates,
            path_deviation_limit=(
                self.config.plan_max_path_deviation
                if constrain_path_deviation
                else None
            ),
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
                initial_cost=initial_cost,
                final_cost=initial_cost,
                metrics=initial_metrics,
            )

        raw_initial = _backward_time(initial_times)

        def objective(raw_time):
            try:
                value, _ = _minco_cost(
                    points,
                    yaw_points,
                    _forward_time(raw_time),
                    self.config,
                    yaw_rates=yaw_rates,
                    path_deviation_limit=(
                        self.config.plan_max_path_deviation
                        if constrain_path_deviation
                        else None
                    ),
                )
                return value if np.isfinite(value) else 1e30
            except np.linalg.LinAlgError:
                return 1e30

        result = minimize(
            objective,
            raw_initial,
            method="L-BFGS-B",
            options={
                "maxiter": self.config.max_iteration,
                "ftol": 1e-8,
                "maxls": 40,
            },
        )
        candidate_times = _forward_time(result.x)
        candidate_cost, candidate_metrics = _minco_cost(
            points,
            yaw_points,
            candidate_times,
            self.config,
            yaw_rates=yaw_rates,
            path_deviation_limit=(
                self.config.plan_max_path_deviation
                if constrain_path_deviation
                else None
            ),
        )
        success = bool(result.success and np.isfinite(candidate_cost))
        if success:
            final_times = candidate_times
            final_cost = candidate_cost
            final_metrics = candidate_metrics
        else:
            final_times = initial_times
            final_cost = initial_cost
            final_metrics = initial_metrics
        return MincoOptimizationResult(
            times=final_times,
            initial_times=initial_times,
            success=success,
            used_fallback=not success,
            status=int(result.status),
            message=str(result.message),
            iterations=int(result.nit),
            function_evaluations=int(result.nfev),
            initial_cost=float(initial_cost),
            final_cost=float(final_cost),
            metrics=final_metrics,
        )


def _densify_plan(poses, config):
    """Subdivide every task waypoint leg without moving its key endpoints.

    ``segment_per_distance`` retains the deployment planner's distance-based
    subdivision.  The simulation-only per-key-segment minimum additionally
    prevents a long quintic span from joining two otherwise nearby collision-
    clearance waypoints without an intermediate positional constraint.
    """
    dense_poses = [poses[0]]
    for start, end in zip(poses[:-1], poses[1:]):
        distance = float(np.linalg.norm(np.asarray(end.p) - np.asarray(start.p)))
        # C++ std::round() rounds positive half-integers away from zero,
        # whereas Python round() uses bankers rounding.
        piece_count = max(
            config.min_piece_num_per_key_segment,
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
    constrain_path_deviation=False,
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
    initial_yaw_rate = (
        float(env.flying_hand_dynamics.w[2]) if env.enable_dynamics else 0.0
    )
    dense_poses = _densify_plan(poses, config)
    result = env.minco_time_optimizer.optimize(
        dense_poses,
        initial_yaw_rate=initial_yaw_rate,
        constrain_path_deviation=constrain_path_deviation,
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
        "initial_yaw_rate_rad_s": initial_yaw_rate,
        "path_deviation_constrained": bool(constrain_path_deviation),
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
        yaw_coefficients=minco_yaw(
            _minco_yaw_points(dense_poses),
            result.times,
            rates=[initial_yaw_rate, 0.0],
        ),
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
        constrain_path_deviation=False,
    ):
        result = plan_and_move_minco(
            self.env,
            poses,
            time_hints,
            save_freq=self.save_freq,
            carried_actor=carried_actor,
            carried_pose=carried_pose,
            phase_name=phase_name,
            constrain_path_deviation=constrain_path_deviation,
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
    times=None,
    duration=None,
    steps=None,
    vels=None,
    accs=None,
    save_freq=-1,
    carried_actor=None,
    carried_pose=None,
    yaw_coefficients=None,
):
    carry_mode = env.flying_hand_carry_mode
    if carried_actor is not None and carry_mode == "isolated_set_actor_pose":
        _begin_isolated_carry(env, carried_actor)

    if carried_actor is not None and env._flying_hand_grasp_needs_settle:
        settle_steps = int(round(env.post_grasp_settle_seconds / env.sim_timestep))
        if settle_steps > 0:
            hold(env, poses[0], settle_steps, save_freq=save_freq)
        env._flying_hand_grasp_needs_settle = False

    ps = np.array([pose.p for pose in poses], dtype=float)
    if times is None:
        duration = (steps or 80) * env.sim_timestep if duration is None else duration
        dist = np.linalg.norm(ps[1:] - ps[:-1], axis=1)
        times = np.full(len(poses) - 1, duration / (len(poses) - 1)) if dist.sum() == 0 else duration * dist / dist.sum()
    coeffs = minco(ps, times, vels, accs)
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
                        env.is_grasping and env.flying_hand_gripper_step >= env.flying_hand_gripper_steps * env.flying_hand_gripper_prismatic_stage_ratio,
                    )
                    env.flying_hand.set_root_pose(hand_pose)
                    env.flying_hand.set_root_linear_velocity(hand_v.tolist())
                    env.flying_hand.set_root_angular_velocity(env.flying_hand_dynamics.w.tolist())
                else:
                    hand_pose, hand_v = ref_pose, v
                    set_pose(env, hand_pose, hand_v)
                carried_pose_fn = None
                if carried_actor is not None and carry_mode == "set_actor_pose":
                    actor_pose = hand_pose * carried_pose
                    carried_pose_fn = lambda actor=carried_actor, pose=actor_pose, vel=hand_v: (actor, pose, vel)
                elif carried_actor is not None and carry_mode == "isolated_set_actor_pose":
                    _set_isolated_actor_pose(carried_actor, hand_pose * carried_pose)
                step(env, 1, save_freq=save_freq, carried_pose_fn=carried_pose_fn)
    finally:
        env._flying_hand_carrying = False


def move_linear(env, start, end, duration=None, steps=None, save_freq=-1, carried_actor=None, carried_pose=None):
    move_minco(env, [start, end], duration=duration, steps=steps, save_freq=save_freq, carried_actor=carried_actor, carried_pose=carried_pose)
