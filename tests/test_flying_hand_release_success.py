import unittest
from unittest import mock

import numpy as np
import sapien

from envs.flying_hand.blocks_ranking_rgb import blocks_ranking_rgb
from envs.flying_hand.blocks_ranking_size import blocks_ranking_size
from envs.flying_hand.place_can_basket import place_can_basket


class _Actor:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)

    def get_pose(self):
        return type("Pose", (), {"p": self.position})()


class FlyingHandReleaseSuccessTest(unittest.TestCase):
    def test_basket_expert_releases_without_snapping_can(self):
        env = object.__new__(place_can_basket)
        env.can = mock.Mock()
        # Deliberately retain an offset and non-spawn orientation at release.
        carried_pose = sapien.Pose([0.03, -0.02, 0.10], [0.98, 0, 0, 0.2])
        env.can.get_pose.return_value = carried_pose
        env.basket = mock.Mock()
        env.flying_hand = mock.Mock()
        env.flying_hand.get_root_pose.return_value = sapien.Pose()
        env.flying_hand_initial_pose = sapien.Pose()
        env.start_flying_hand_record = mock.Mock(return_value=10)
        env.finish_flying_hand_record = mock.Mock()
        env._get_flying_hand_pose = mock.Mock(return_value=sapien.Pose([0.5, 0.2, 0.8]))
        env._get_actor_world_bounds = mock.Mock(side_effect=AssertionError('No release alignment'))
        env.can_name, env.can_id = 'can', 0
        env.basket_name, env.basket_id = 'basket', 0
        env.info = {}
        motion = mock.Mock()
        released = []

        def record_release(pose, command):
            self.assertEqual(command, 'open')
            released.append(env.can.get_pose())

        motion.set_gripper.side_effect = record_release
        with mock.patch('envs.flying_hand.planner.TaskMotionPlanner', return_value=motion), mock.patch(
            'envs.flying_hand.planner.set_isolated_carried_actor_target'
        ) as snap:
            env.play_once()

        snap.assert_not_called()
        env._get_actor_world_bounds.assert_not_called()
        self.assertEqual(len(released), 1)
        np.testing.assert_allclose(released[0].p, carried_pose.p)
        np.testing.assert_allclose(released[0].q, carried_pose.q)
        self.assertEqual(motion.move.call_count, 3)
        motion.hold.assert_called_once()

    def test_place_can_basket_requires_release(self):
        env = object.__new__(place_can_basket)
        env.can = _Actor([0.0, 0.0, 0.6])
        env.basket = _Actor([0.0, 0.0, 0.6])
        env.can_name = "can"
        env.basket_name = "basket"
        env.basket_start_z = 0.6
        env._task_objects_safe = lambda: True
        env.check_actors_contact = lambda *_: True
        env._get_actor_world_bounds = lambda _: np.asarray(
            [[-0.1, -0.1, 0.5], [0.1, 0.1, 0.7]]
        )

        env.is_grasping = True
        self.assertFalse(env.check_success())
        env.is_grasping = False
        self.assertTrue(env.check_success())

    def test_rgb_ranking_requires_release(self):
        env = object.__new__(blocks_ranking_rgb)
        positions = [[0.0, y, 0.6] for y in (-0.2, 0.0, 0.2)]
        env.blocks = [_Actor(position) for position in positions]
        env.target_centers = [np.asarray(position) for position in positions]
        env._task_objects_safe = lambda: True

        env.is_grasping = True
        self.assertFalse(env.check_success())
        env.is_grasping = False
        self.assertTrue(env.check_success())

    def test_size_ranking_requires_release(self):
        env = object.__new__(blocks_ranking_size)
        positions = [[0.0, y, 0.6] for y in (0.2, 0.0, -0.2)]
        env.blocks = [_Actor(position) for position in positions]
        env.target_centers = [np.asarray(position) for position in positions]
        env._task_objects_safe = lambda: True

        env.is_grasping = True
        self.assertFalse(env.check_success())
        env.is_grasping = False
        self.assertTrue(env.check_success())


if __name__ == "__main__":
    unittest.main()
