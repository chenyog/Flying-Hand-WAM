import unittest
from types import SimpleNamespace
import numpy as np
from envs.flying_hand.eval_perturbation import EvalPerturbation, validate_config


class PerturbationTest(unittest.TestCase):
    def create(self, dynamics=False, enabled=True):
        return EvalPerturbation(dict(enabled=enabled), 'move_bottle', 100000, .005, 35, dynamics)

    def test_seed_and_rng_isolation(self):
        np.random.seed(12)
        expected = np.random.random()
        np.random.seed(12)
        a, b = self.create(), self.create()
        self.assertEqual(np.random.random(), expected)
        self.assertEqual(a.trigger_tick, b.trigger_tick)
        np.testing.assert_equal(a.direction, b.direction)
        self.assertTrue(7 <= a.trigger_tick*.005 <= 14)

    def test_direct_single_offset(self):
        p = self.create()
        samples = [p.sample(SimpleNamespace(p=np.zeros(3)))[0] for _ in range(7000)]
        nonzero = [v for v in samples if np.any(v)]
        self.assertEqual(len(nonzero), 1)
        self.assertAlmostEqual(np.linalg.norm(nonzero[0]), .05)
        self.assertEqual(nonzero[0][2], 0)

    def test_force_pulse(self):
        p = self.create(dynamics=True)
        samples = [p.sample(SimpleNamespace(p=np.zeros(3))) for _ in range(7000)]
        self.assertTrue(all(not np.any(offset) for offset, _ in samples))
        forces = [force for _, force in samples if np.any(force)]
        self.assertEqual(len(forces), 40)
        self.assertAlmostEqual(np.linalg.norm(forces[0]), 2)

    def test_disabled_and_early_end(self):
        p = self.create(enabled=False)
        for _ in range(7000):
            offset, force = p.sample(SimpleNamespace(p=np.zeros(3)))
            self.assertFalse(np.any(offset) or np.any(force))
        self.assertFalse(p.record['triggered'])
        self.assertFalse(self.create().record['triggered'])

    def test_invalid(self):
        for config in [dict(force_n=-1), dict(start_fraction=.8,end_fraction=.2),dict(position_m=float('nan'))]:
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_force_enters_actual_dynamics(self):
        import sapien
        import yaml
        from pathlib import Path
        from envs.flying_hand.dynamics import FlyingHandDynamics
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'envs/flying_hand/_base_config.yml').read_text())['dynamics']
        a, b = FlyingHandDynamics(cfg, .005), FlyingHandDynamics(cfg, .005)
        for model in (a, b):
            model.reset(sapien.Pose())
        _, va = a.step(sapien.Pose(), np.zeros(3), np.zeros(3), False)
        _, vb = b.step(sapien.Pose(), np.zeros(3), np.zeros(3), False, external_force_world=[2, 0, 0])
        np.testing.assert_allclose(vb-va, [2/a.mass*.005, 0, 0], atol=1e-8)

    def test_direct_offset_survives_remaining_chunk(self):
        from unittest import mock
        import sapien
        from tests.test_flying_hand_policy_execution import FlyingHandPolicyExecutionTest
        from envs.flying_hand.eval_control import execute_action_chunk
        fixture = FlyingHandPolicyExecutionTest()
        fixture.setUp()
        env = fixture.env
        env.enable_dynamics = False
        env.eval_perturbation = self.create()
        env.eval_perturbation.trigger_tick = 1
        env.eval_perturbation.direction = np.array([1., 0., 0.])
        positions = []
        def set_pose(task, pose, velocity):
            positions.append(np.asarray(pose.p).copy())
            task.flying_hand.set_root_pose(pose)
        with mock.patch('envs.flying_hand.planner.set_pose', side_effect=set_pose), mock.patch('envs.flying_hand.planner.step'):
            execute_action_chunk(env, fixture.model, np.zeros((2, 5)))
            np.testing.assert_allclose(np.array(positions)[1:, 0], .05, atol=1e-7)
            boundary = len(positions)
            execute_action_chunk(env, fixture.model, np.zeros((1, 5)))
        self.assertGreater(positions[boundary][0], .04)
        self.assertLess(positions[-1][0], .05)


if __name__ == '__main__':
    unittest.main()
