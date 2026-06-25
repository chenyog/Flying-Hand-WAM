import numpy as np
import sapien
import transforms3d as t3d


G = 9.81


def _vec(v):
    if isinstance(v, dict):
        return np.array([v["x"], v["y"], v["z"]], dtype=float)
    return np.array(v, dtype=float)


def _unit(v, fallback=None):
    n = np.linalg.norm(v)
    if n > 1e-8:
        return v / n
    if fallback is not None:
        return np.array(fallback, dtype=float)
    ret = np.zeros_like(v, dtype=float)
    ret[-1] = 1.0
    return ret


def _qmul(a, b):
    return np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ], dtype=float)


def _qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float) / np.dot(q, q)


def _yaw(q):
    return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def _att_from_z_yaw(z, yaw):
    z = _unit(z)
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y = _unit(np.cross(z, x))
    x = _unit(np.cross(y, z))
    return t3d.quaternions.mat2quat(np.column_stack([x, y, z]))


def _clip_rotor_thrusts(rotor_thrust, thrust_target, thrust_min, thrust_max):
    rotor_thrust = np.clip(rotor_thrust, thrust_min, thrust_max)
    thrust_target = np.clip(thrust_target, thrust_min.sum(), thrust_max.sum())
    for _ in range(4):
        delta = thrust_target - rotor_thrust.sum()
        if abs(delta) < 1e-6:
            break
        if delta > 0:
            room = thrust_max - rotor_thrust
        else:
            room = rotor_thrust - thrust_min
        active = room > 1e-6
        if not np.any(active):
            break
        step = min(abs(delta), room[active].sum()) * np.sign(delta)
        rotor_thrust[active] += step * room[active] / room[active].sum()
    return rotor_thrust


class FlyingHandDynamics:
    def __init__(self, cfg, dt):
        self.dt = dt
        self.cfg = cfg
        p = cfg["nominal"]
        self.mass = float(p["mass"])
        self.j = np.diag(_vec(p["inertia"]))
        self.j_inv = np.linalg.inv(self.j)
        self.k_pos = _vec(cfg["control"]["k_pos"])
        self.k_vel = _vec(cfg["control"]["k_vel"])
        self.k_att = _vec(cfg["control"]["k_att"])
        self.k_omega = _vec(cfg["control"]["k_omega"])
        self.torque_max = _vec(cfg["limits"]["torque_max"])
        self.bodyrates_max = _vec(cfg["limits"]["bodyrates_max"])
        self.thrust_min = np.array(cfg["limits"]["rotor_thrusts_min"], dtype=float)
        self.thrust_max = np.array(cfg["limits"]["rotor_thrusts_max"], dtype=float)
        self.tilt_max = float(cfg["limits"]["tilt_angle_max"])
        self.ct = float(cfg["rotor"]["thrust_coeff"])
        self.cq = float(cfg["rotor"]["moment_coeff"])
        e = cfg["estimator"]
        self.estimator_enabled = bool(e["enabled"])
        self.as_v = float(e["as_v"]) if self.estimator_enabled else -5.0
        self.as_w = float(e["as_omega"]) if self.estimator_enabled else -10.0
        self.force_lpf = np.exp(-float(e["thrust_cutoff"]) * dt) if self.estimator_enabled else 0.0
        self.torque_lpf1 = np.exp(-float(e["torque_cutoff1"]) * dt) if self.estimator_enabled else 0.0
        self.torque_lpf2 = np.exp(-float(e["torque_cutoff2"]) * dt) if self.estimator_enabled else 0.0
        self.force_bound = _vec(e["force_bound"]) if self.estimator_enabled else np.zeros(3)
        self.torque_bound = _vec(e["torque_bound"]) if self.estimator_enabled else np.zeros(3)
        self.rot = {
            "free": np.array(cfg["free"]["rotor_position"], dtype=float).T,
            "grasped": np.array(cfg["grasped"]["rotor_position"], dtype=float).T,
        }
        self.reset(sapien.Pose())

    def reset(self, pose):
        self.p = np.array(pose.p, dtype=float)
        self.v = np.zeros(3)
        self.q = np.array(pose.q, dtype=float)
        self.w = np.zeros(3)
        self.force_l1 = np.zeros(3)
        self.torque_l1 = np.zeros(3)
        self.v_hat = self.v.copy()
        self.w_hat = self.w.copy()
        self.prev_v = self.v.copy()
        self.prev_w = self.w.copy()
        self.prev_q = self.q.copy()
        self.prev_thrust = 0.0
        self.prev_torque = np.zeros(3)
        self.force_flt = np.zeros(3)
        self.torque_flt1 = np.zeros(3)
        self.torque_flt2 = np.zeros(3)
        self.ref_pose = pose
        self.debug = {}

    def sync(self, pose):
        self.reset(pose)

    def allocation(self, grasped):
        r = self.rot["grasped" if grasped else "free"]
        c = self.cq / self.ct
        return np.array([
            [1.0, 1.0, 1.0, 1.0],
            [r[1, 0], r[1, 1], r[1, 2], r[1, 3]],
            [-r[0, 0], -r[0, 1], -r[0, 2], -r[0, 3]],
            [-c, -c, c, c],
        ], dtype=float)

    def step(self, ref_pose, ref_vel, ref_acc, grasped):
        self.ref_pose = ref_pose
        ref_p = np.array(ref_pose.p, dtype=float)
        ref_v = np.array(ref_vel, dtype=float)
        ref_a = np.array(ref_acc, dtype=float)
        r = t3d.quaternions.quat2mat(self.q)
        if self.estimator_enabled:
            pr = t3d.quaternions.quat2mat(self.prev_q)
            b = pr / self.mass
            self.v_hat += (
                np.array([0.0, 0.0, -G])
                + b @ (self.force_l1 + np.array([0.0, 0.0, self.prev_thrust]))
                + self.as_v * (self.v_hat - self.prev_v)
            ) * self.dt
            ev = self.v_hat - self.v
            phi = self.as_v * np.exp(self.as_v * self.dt) / (np.exp(self.as_v * self.dt) - 1.0)
            self.force_flt = (
                self.force_lpf * self.force_flt
                - (1 - self.force_lpf) * np.linalg.solve(b, phi * ev)
            )
            self.force_l1 = np.clip(self.force_flt, -self.force_bound, self.force_bound)

            self.w_hat += (
                -self.j_inv @ np.cross(self.prev_w, self.j @ self.prev_w)
                + self.j_inv @ (self.prev_torque + self.torque_l1)
                + self.as_w * (self.w_hat - self.prev_w)
            ) * self.dt
            ew = self.w_hat - self.w
            phi = self.as_w * np.exp(self.as_w * self.dt) / (np.exp(self.as_w * self.dt) - 1.0)
            self.torque_flt1 = (
                self.torque_lpf1 * self.torque_flt1
                - (1 - self.torque_lpf1) * np.linalg.solve(self.j_inv, phi * ew)
            )
            self.torque_flt2 = self.torque_lpf2 * self.torque_flt2 + (1 - self.torque_lpf2) * self.torque_flt1
            self.torque_l1 = np.clip(self.torque_flt2, -self.torque_bound, self.torque_bound)

        a_des = ref_a + self.k_pos * (ref_p - self.p) + self.k_vel * (ref_v - self.v)
        f = self.mass * (a_des + np.array([0.0, 0.0, G])) - r @ self.force_l1
        if self.tilt_max > 0:
            z = _unit(f)
            tilt = np.arccos(np.clip(z[2], -1.0, 1.0))
            if tilt > self.tilt_max:
                xy = _unit(z[:2], [1.0, 0.0])
                z = np.r_[np.sin(self.tilt_max) * xy, np.cos(self.tilt_max)]
                f = np.linalg.norm(f) * z

        q_des = _att_from_z_yaw(f, _yaw(ref_pose.q))
        qe = _qmul(_qinv(self.q), q_des)
        if qe[0] < 0:
            qe *= -1
        torque = np.clip(self.k_att * (2 * qe[1:]) - self.k_omega * self.w - self.torque_l1, -self.torque_max, self.torque_max)
        thrust = max(float(f.dot(r[:, 2])), 0.0)
        u = np.r_[thrust, torque]
        g = self.allocation(grasped)
        rot_thrust = _clip_rotor_thrusts(np.linalg.solve(g, u), thrust, self.thrust_min, self.thrust_max)
        thrust, torque = (g @ rot_thrust)[0], (g @ rot_thrust)[1:]
        self.prev_v = self.v.copy()
        self.prev_w = self.w.copy()
        self.prev_q = self.q.copy()
        self.prev_thrust = thrust
        self.prev_torque = torque.copy()

        self.v += (r @ np.array([0.0, 0.0, thrust]) / self.mass - np.array([0.0, 0.0, G])) * self.dt
        self.p += self.v * self.dt
        self.w += self.j_inv @ (torque - np.cross(self.w, self.j @ self.w)) * self.dt
        self.w = np.clip(self.w, -self.bodyrates_max, self.bodyrates_max)
        self.q = _qmul(self.q, np.r_[1.0, 0.5 * self.w * self.dt])
        self.q /= np.linalg.norm(self.q)
        self.debug = {
            "mode": "grasped" if grasped else "free",
            "rotor_position": self.rot["grasped" if grasped else "free"].copy(),
            "allocation": g.copy(),
            "rotor_thrust": rot_thrust.copy(),
            "estimator_enabled": self.estimator_enabled,
            "force_l1": self.force_l1.copy(),
            "torque_l1": self.torque_l1.copy(),
            "v_hat": self.v_hat.copy(),
            "w_hat": self.w_hat.copy(),
        }
        return sapien.Pose(self.p.tolist(), self.q.tolist()), self.v.copy()
