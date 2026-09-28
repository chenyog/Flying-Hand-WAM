import unittest

import numpy as np

from envs.flying_hand.shake_bottle import shake_bottle


class _Pose:
    def __init__(self, xyz):
        self.p = np.asarray(xyz, dtype=float)


class _Bottle:
    def __init__(self, xyz):
        self.pose = _Pose(xyz)

    def get_pose(self):
        return self.pose


class ShakeBottleSuccessTest(unittest.TestCase):
    def setUp(self):
        self.env = object.__new__(shake_bottle)
        self.env.bottle_initial_pose = _Pose([1.0, 0.2, 0.8])
        self.env.bottle = _Bottle(self.env.bottle_initial_pose.p.copy())
        self.env._task_objects_safe = lambda: True
        self.env._reset_shake_progress()

    def sample(self, *, y, z=0.9):
        self.env.bottle.pose = _Pose([1.0, y, z])
        self.env.update_flying_hand_policy_progress()

    def test_one_sided_excursion_is_not_a_shake_cycle(self):
        for y in [0.31, 0.34, 0.31, 0.2, 0.33]:
            self.sample(y=y)

        self.assertEqual(self.env.shake_extreme_visits, 1)
        self.assertEqual(self.env.completed_shake_cycles, 0)
        self.assertFalse(self.env.check_success())

    def test_one_complete_cycle_is_not_enough(self):
        for y in [0.29, 0.11]:
            self.sample(y=y)

        self.assertEqual(self.env.completed_shake_cycles, 1)
        self.assertFalse(self.env.check_success())

    def test_two_actual_alternating_cycles_succeed_while_lifted(self):
        for y in [0.29, 0.11, 0.29, 0.11]:
            self.sample(y=y)

        self.assertEqual(self.env.shake_extreme_visits, 4)
        self.assertEqual(self.env.completed_shake_cycles, 2)
        self.assertTrue(self.env.check_success())

    def test_small_alternating_motion_has_no_amplitude_threshold(self):
        for y in [0.205, 0.195, 0.205, 0.195]:
            self.sample(y=y)

        self.assertEqual(self.env.completed_shake_cycles, 2)
        self.assertTrue(self.env.check_success())

    def test_no_lateral_motion_is_not_a_shake_cycle(self):
        for _ in range(4):
            self.sample(y=0.2)

        self.assertEqual(self.env.completed_shake_cycles, 0)
        self.assertFalse(self.env.check_success())

    def test_three_actual_alternating_cycles_succeed_while_lifted(self):
        for y in [0.31, 0.09, 0.31, 0.09, 0.31, 0.09]:
            self.sample(y=y)

        self.assertAlmostEqual(self.env.max_shake_y, 0.11)
        self.assertAlmostEqual(self.env.min_shake_displacement_y, -0.11)
        self.assertAlmostEqual(self.env.max_shake_displacement_y, 0.11)
        self.assertEqual(self.env.shake_extreme_visits, 6)
        self.assertEqual(self.env.completed_shake_cycles, 3)
        self.assertTrue(self.env.check_success())

    def test_motion_before_lift_does_not_count(self):
        for y in [0.31, 0.09, 0.31, 0.09, 0.31, 0.09]:
            self.sample(y=y, z=0.81)
        self.sample(y=0.2, z=0.9)

        self.assertEqual(self.env.shake_extreme_visits, 0)
        self.assertFalse(self.env.check_success())

    def test_bottle_must_remain_lifted_at_success_check(self):
        for y in [0.31, 0.09, 0.31, 0.09, 0.31, 0.09]:
            self.sample(y=y)
        self.env.bottle.pose = _Pose([1.0, 0.2, 0.82])

        self.assertEqual(self.env.completed_shake_cycles, 3)
        self.assertFalse(self.env.check_success())

    def test_episode_reset_clears_all_trajectory_metrics(self):
        for y in [0.31, 0.09]:
            self.sample(y=y)
        self.env._reset_shake_progress()

        self.assertEqual(self.env.max_shake_y, 0.0)
        self.assertEqual(self.env.min_shake_displacement_y, 0.0)
        self.assertEqual(self.env.max_shake_displacement_y, 0.0)
        self.assertEqual(self.env.shake_extreme_visits, 0)
        self.assertEqual(self.env.completed_shake_cycles, 0)
        self.assertEqual(self.env._last_shake_side, 0)


if __name__ == "__main__":
    unittest.main()
