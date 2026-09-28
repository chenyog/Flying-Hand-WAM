"""Exercise the real layout generator with lightweight actor stubs (no renderer)."""
import unittest
from unittest.mock import patch

import numpy as np

from envs.flying_hand.move_can_pot import move_can_pot


class Actor:
    def __init__(self, half):
        self.half = np.asarray(half)
        self.center = np.zeros(3)
        self.config = {}


class MoveCanPotLayoutTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.get_state()

    def tearDown(self):
        np.random.set_state(self.rng)

    def test_distinct_heights_for_many_seeds(self):
        env = object.__new__(move_can_pot)
        env.board_slots = [(0, .5), (0, .5), (0, 1.2), (0, 1.9)]
        for seed in range(1000):
            np.random.seed(seed)
            env.pot_slot_id = seed % len(env.board_slots)
            can_slot = env._choose_can_slot()
            self.assertNotEqual(can_slot, env.pot_slot_id)
            self.assertNotEqual(env.board_slots[can_slot][1], env.board_slots[env.pot_slot_id][1])

    def test_requires_two_heights(self):
        env = object.__new__(move_can_pot)
        env.board_slots = [(0, .5), (.3, .5)]
        env.pot_slot_id = 0
        with self.assertRaisesRegex(ValueError, "distinct shelf heights"):
            env._choose_can_slot()

    def test_load_actors_and_goal_remain_on_pot_shelf(self):
        for seed in range(200):
            np.random.seed(seed)
            env = object.__new__(move_can_pot)
            env.scene = None
            env.board_slots = [(0, .5), (.1, 1.2), (-.1, 1.9)]
            env.shelf_xy_areas = {i: [] for i in range(3)}
            env.shelf_length = .5
            env.shelf_width = 1.2
            env._reset_board_slots = lambda: None
            env._board_front_x = lambda: 1.
            env._object_x = lambda actor: actor.center[0]
            env.add_task_objects = lambda *args: None
            env.add_prohibit_area = lambda *args, **kwargs: None
            env._get_actor_world_bounds = lambda actor: np.array([
                actor.center - actor.half, actor.center + actor.half])

            def place(actor, slot, y=None):
                sy, z = env.board_slots[slot]
                actor.center = np.array([.6, sy if y is None else y, z + actor.half[2]])
                return actor

            env._place_actor_on_shelf = place
            env._create_board_actor = lambda name, model, slot, **kwargs: place(Actor([.03, .03, .05]), slot)
            with patch("envs.flying_hand.move_can_pot.create_sapien_urdf_obj",
                       side_effect=lambda **kwargs: Actor([.1, .1, .1])):
                env.load_actors()
            self.assertNotEqual(env.can_slot_id, env.pot_slot_id)
            can_bottom = env._get_actor_world_bounds(env.can)[0][2]
            pot_bottom = env._get_actor_world_bounds(env.pot)[0][2]
            self.assertGreater(abs(can_bottom - pot_bottom), .5)
            self.assertAlmostEqual(env.target_center[2] - env.can.half[2], pot_bottom)
            env._task_objects_safe = lambda: True
            env.can.center = env.target_center.copy()
            env.is_grasping = False
            self.assertTrue(env.check_success())
            env.is_grasping = True
            self.assertFalse(env.check_success())


if __name__ == "__main__":
    unittest.main()
