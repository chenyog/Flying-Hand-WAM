import numpy as np
import sapien


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


def step(env, n, save_freq=-1, carried_pose_fn=None):
    for _ in range(n):
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
            hand_pose, hand_v = env.flying_hand_dynamics.step(pose, np.zeros(3), np.zeros(3), env.is_grasping)
            env.flying_hand.set_root_pose(hand_pose)
            env.flying_hand.set_root_linear_velocity(hand_v.tolist())
            env.flying_hand.set_root_angular_velocity(env.flying_hand_dynamics.w.tolist())
        else:
            set_pose(env, pose)
        step(env, 1, save_freq=save_freq)


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


def move_minco(env, poses, times=None, duration=None, steps=None, vels=None, accs=None, save_freq=-1, carried_actor=None, carried_pose=None):
    ps = np.array([pose.p for pose in poses], dtype=float)
    if times is None:
        duration = (steps or 80) * env.sim_timestep if duration is None else duration
        dist = np.linalg.norm(ps[1:] - ps[:-1], axis=1)
        times = np.full(len(poses) - 1, duration / (len(poses) - 1)) if dist.sum() == 0 else duration * dist / dist.sum()
    coeffs = minco(ps, times, vels, accs)
    carried_pose = carried_pose if carried_pose is not None else (
        poses[0].inv() * carried_actor.get_pose() if carried_actor is not None else None
    )
    for coeff, T, start, end in zip(coeffs, times, poses[:-1], poses[1:]):
        for idx in range(max(1, int(np.ceil(T / env.sim_timestep)))):
            t = min((idx + 1) * env.sim_timestep, T)
            p, v, a = sample(coeff, t)
            ref_pose = sapien.Pose(p.tolist(), slerp(start.q, end.q, t / T).tolist())
            env.flying_hand_ref_pose = ref_pose
            if env.enable_dynamics:
                hand_pose, hand_v = env.flying_hand_dynamics.step(ref_pose, v, a, env.is_grasping)
                env.flying_hand.set_root_pose(hand_pose)
                env.flying_hand.set_root_linear_velocity(hand_v.tolist())
                env.flying_hand.set_root_angular_velocity(env.flying_hand_dynamics.w.tolist())
            else:
                hand_pose, hand_v = ref_pose, v
                set_pose(env, hand_pose, hand_v)
            carried_pose_fn = None
            if carried_actor is not None:
                actor_pose = hand_pose * carried_pose
                carried_pose_fn = lambda actor=carried_actor, pose=actor_pose, vel=hand_v: (actor, pose, vel)
            step(env, 1, save_freq=save_freq, carried_pose_fn=carried_pose_fn)


def move_linear(env, start, end, duration=None, steps=None, save_freq=-1, carried_actor=None, carried_pose=None):
    move_minco(env, [start, end], duration=duration, steps=steps, save_freq=save_freq, carried_actor=carried_actor, carried_pose=carried_pose)
