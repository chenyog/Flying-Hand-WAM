"""Seeded, one-shot Flying-Hand perturbations for policy evaluation only."""
import zlib
import numpy as np


def validate_config(config=None):
    defaults = dict(enabled=False, position_m=0.05, force_n=2.0,
                    duration_s=0.2, start_fraction=0.2, end_fraction=0.4)
    if config is not None:
        if not isinstance(config, dict) or set(config) - set(defaults):
            raise ValueError('Unknown or invalid eval_perturbation configuration')
        defaults.update(config)
    if not isinstance(defaults['enabled'], bool):
        raise ValueError('eval_perturbation.enabled must be boolean')
    for key in set(defaults) - {'enabled'}:
        defaults[key] = float(defaults[key])
        if not np.isfinite(defaults[key]):
            raise ValueError(f'Nonfinite perturbation parameter: {key}')
    if not 0 < defaults['start_fraction'] <= defaults['end_fraction'] < 1:
        raise ValueError('Perturbation window must be inside (0, 1)')
    if any(defaults[k] <= 0 for k in ('position_m', 'force_n', 'duration_s')):
        raise ValueError('Perturbation magnitudes and duration must be positive')
    return defaults


class EvalPerturbation:
    def __init__(self, config, task, seed, dt, horizon_s, dynamics):
        self.config = validate_config(config)
        self.dt = float(dt)
        self.tick_count = 0
        self.dynamics = bool(dynamics)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(task.encode()), 19283]))
        fraction = rng.uniform(self.config['start_fraction'], self.config['end_fraction'])
        self.trigger_tick = max(1, int(round(horizon_s * fraction / self.dt)))
        self.duration_ticks = max(1, int(round(self.config['duration_s'] / self.dt)))
        angle = rng.uniform(0, 2 * np.pi)
        self.direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        self.record = dict(seed=int(seed), task=task, mode='force' if dynamics else 'position',
            config=self.config, scheduled_time_s=self.trigger_tick*self.dt,
            direction_world=self.direction.tolist(), triggered=False, force_ticks=0)

    def sample(self, pose):
        """One call per policy simulation step; no policy RNG is consumed."""
        offset, force = np.zeros(3), np.zeros(3)
        if self.config['enabled'] and self.tick_count == self.trigger_tick:
            self.record.update(triggered=True, actual_time_s=self.tick_count*self.dt,
                               hand_position_before=np.asarray(pose.p).tolist())
            if not self.dynamics:
                offset = self.direction * self.config['position_m']
                self.record['position_offset_world_m'] = offset.tolist()
        if (self.config['enabled'] and self.dynamics
                and self.trigger_tick <= self.tick_count < self.trigger_tick+self.duration_ticks):
            force = self.direction * self.config['force_n']
            self.record['force_ticks'] += 1
            self.record['force_world_n'] = force.tolist()
        self.tick_count += 1
        return offset, force
