import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import sapien

from envs.flying_hand.eval_control import (
    canonical_action_chunk,
    execute_action_chunk,
    finite_difference_actions,
    reset_reference,
)


class _Hand:
    def __init__(self):
        self.pose = sapien.Pose()

    def get_root_pose(self):
        return self.pose

    def set_root_pose(self, pose):
        self.pose = pose

    def set_root_linear_velocity(self, velocity):
        pass

    def set_root_angular_velocity(self, velocity):
        pass


class _Dynamics:
    def __init__(self):
        self.references = []
        self.w = np.zeros(3)

    def step(self, pose, velocity, acceleration, grasped):
        self.references.append(
            (
                np.asarray(pose.p, dtype=float),
                np.asarray(velocity, dtype=float).copy(),
                np.asarray(acceleration, dtype=float).copy(),
            )
        )
        return pose, np.asarray(velocity, dtype=float)


class FlyingHandPolicyExecutionTest(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(
            grasp_commanded=False,
            attached_actor=None,
            attached_pose=None,
        )
        self.env = SimpleNamespace(
            flying_hand=_Hand(),
            flying_hand_dynamics=_Dynamics(),
            flying_hand_ref_pose=sapien.Pose(),
            flying_hand_initial_pose=sapien.Pose(),
            initial_imu_odom_pose=sapien.Pose(),
            flying_hand_waypoint_tracking={
                "max_velocity": 0.6,
                "max_acceleration": 0.8,
                "max_jerk": 2.5,
                "max_yaw_rate": 0.5,
            },
            enable_dynamics=True,
            save_freq=10,
            sim_timestep=0.005,
            is_grasping=False,
            flying_hand_gripper_step=0,
            flying_hand_gripper_steps=0,
        )

    def test_bounded_reference_derivatives_are_passed_to_dynamics(self):
        with mock.patch("envs.flying_hand.planner.step"):
            execute_action_chunk(
                self.env,
                self.model,
                np.array([0.1, 0.0, 0.0, 0.0, 0.0]),
            )

        references = self.env.flying_hand_dynamics.references
        self.assertEqual(len(references), 10)
        velocities = np.stack([reference[1] for reference in references])
        accelerations = np.stack([reference[2] for reference in references])
        positions = np.stack([reference[0] for reference in references])
        self.assertGreater(float(np.linalg.norm(velocities[-1])), 0.0)
        self.assertLessEqual(float(np.linalg.norm(velocities, axis=1).max()), 0.6 + 1e-12)
        self.assertLessEqual(float(np.linalg.norm(accelerations, axis=1).max()), 0.8 + 1e-12)
        acceleration_changes = np.diff(
            np.vstack([np.zeros((1, 3)), accelerations]),
            axis=0,
        )
        self.assertLessEqual(
            float(np.linalg.norm(acceleration_changes, axis=1).max()),
            2.5 * self.env.sim_timestep + 1e-12,
        )
        self.assertTrue(np.all(np.diff(positions[:, 0]) > 0.0))

    def test_action_chunk_uses_endpoint_and_central_differences(self):
        actions = np.array(
            [
                [0.00, 0.0, 0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0, 0.0, 0.0],
                [0.04, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        velocities, accelerations = finite_difference_actions(
            self.env, actions
        )

        np.testing.assert_allclose(velocities[:, 0], [0.2, 0.4, 0.6], atol=1e-7)
        self.assertLessEqual(
            float(np.linalg.norm(accelerations, axis=1).max()),
            self.env.flying_hand_waypoint_tracking["max_acceleration"] + 1e-12,
        )

    def test_reset_clears_reference_history(self):
        self.model._flying_hand_reference_position = np.ones(3)
        self.model._flying_hand_reference_velocity = np.ones(3)
        self.model._flying_hand_reference_acceleration = np.ones(3)
        self.model._flying_hand_reference_orientation = np.ones(4)

        reset_reference(self.model)

        self.assertIsNone(self.model._flying_hand_reference_position)
        np.testing.assert_array_equal(
            self.model._flying_hand_reference_velocity,
            np.zeros(3),
        )
        np.testing.assert_array_equal(
            self.model._flying_hand_reference_acceleration,
            np.zeros(3),
        )
        self.assertIsNone(self.model._flying_hand_reference_orientation)

    def test_shared_executor_counts_chunk_once_in_direct_mode(self):
        self.env.enable_dynamics = False
        self.env.take_action_cnt = 0
        self.env.eval_success = False
        self.env.check_success = lambda: False
        actions = canonical_action_chunk(
            np.asarray([[0.01, 0.0, 0.0, 0.0, 0.0], [0.02, 0.0, 0.0, 0.0, 0.0]])
        )
        with mock.patch("envs.flying_hand.planner.set_pose"), mock.patch(
            "envs.flying_hand.planner.step"
        ):
            consumed = execute_action_chunk(self.env, self.model, actions)
        self.assertEqual(consumed, 2)
        self.assertEqual(self.env.take_action_cnt, 2)

    def test_shared_finite_difference_matches_task_cadence(self):
        actions = np.asarray(
            [[0.00, 0.0, 0.0, 0.0, 0.0], [0.01, 0.0, 0.0, 0.0, 0.0], [0.04, 0.0, 0.0, 0.0, 0.0]]
        )
        velocities, accelerations = finite_difference_actions(self.env, actions)
        np.testing.assert_allclose(velocities[:, 0], [0.2, 0.4, 0.6], atol=1e-7)
        self.assertEqual(velocities.shape, (3, 3))
        self.assertEqual(accelerations.shape, (3, 3))

    def test_shared_dynamics_executor_passes_limited_references(self):
        self.env.take_action_cnt = 0
        self.env.eval_success = False
        self.env.check_success = lambda: False
        actions = np.asarray(
            [[0.00, 0.0, 0.0, 0.0, 0.0], [0.04, 0.0, 0.0, 0.0, 0.0]]
        )
        with mock.patch("envs.flying_hand.planner.step"):
            consumed = execute_action_chunk(self.env, self.model, actions)
        self.assertEqual(consumed, 2)
        self.assertEqual(self.env.take_action_cnt, 2)
        references = self.env.flying_hand_dynamics.references
        # MINCO may uniformly stretch an infeasible 50 ms segment; the
        # resulting dense trajectory therefore contains at least the nominal
        # 20 physics samples for two waypoints.
        self.assertGreaterEqual(len(references), 20)
        self.assertLessEqual(
            max(np.linalg.norm(reference[1]) for reference in references), 0.6 + 1e-12
        )
        self.assertLessEqual(
            max(np.linalg.norm(reference[2]) for reference in references), 0.8 + 1e-6
        )

    def test_shared_executor_does_not_apply_task_level_replan_limit(self):
        self.env.enable_dynamics = False
        self.env.replan_steps = 2
        self.env.take_action_cnt = 0
        self.env.eval_success = False
        self.env.check_success = lambda: False
        actions = np.asarray(
            [[0.01 * index, 0.0, 0.0, 0.0, 0.0] for index in range(1, 5)],
            dtype=np.float32,
        )
        with mock.patch("envs.flying_hand.planner.step"):
            consumed = execute_action_chunk(self.env, self.model, actions)
        self.assertEqual(consumed, 4)
        self.assertEqual(self.env.take_action_cnt, 4)

    def test_deferred_history_records_executed_window_only(self):
        self.env.enable_dynamics = False
        self.env.replan_steps = 2
        self.env.take_action_cnt = 0
        self.env.eval_success = False
        self.env.check_success = lambda: False
        self.env.record_flying_hand_eval_history = True
        self.env.flying_hand_eval_action_chunks = []
        self.env.flying_hand_eval_actual_states = []
        self.env.flying_hand_eval_target_states = []
        self.env._get_flying_hand_actual_state = lambda: np.zeros(5, dtype=np.float32)
        self.env._get_flying_hand_target_state = lambda: np.zeros(5, dtype=np.float32)
        actions = np.zeros((4, 5), dtype=np.float32)
        with mock.patch("envs.flying_hand.planner.step"):
            execute_action_chunk(self.env, self.model, actions)
        self.assertEqual([len(chunk) for chunk in self.env.flying_hand_eval_action_chunks], [4])
        self.assertEqual(len(self.env.flying_hand_eval_actual_states), 4)
        self.assertEqual(len(self.env.flying_hand_eval_target_states), 4)

    def test_episode_boundary_only_shortens_final_policy_chunk(self):
        self.env.enable_dynamics = False
        self.env.replan_steps = 1
        self.env.step_lim = 5
        self.env.take_action_cnt = 3
        self.env.eval_success = False
        self.env.check_success = lambda: False
        actions = np.zeros((4, 5), dtype=np.float32)
        with mock.patch("envs.flying_hand.planner.step"):
            consumed = execute_action_chunk(self.env, self.model, actions)
        self.assertEqual(consumed, 2)
        self.assertEqual(self.env.take_action_cnt, 5)

    def test_grasp_edge_is_applied_after_reaching_target_waypoint(self):
        self.env.enable_dynamics = False
        self.env.take_action_cnt = 0
        self.env.eval_success = False
        self.env.check_success = lambda: False
        edge_positions = []

        def apply_grasp(task_env, grasp, *, action_step):
            edge_positions.append(np.asarray(task_env.flying_hand.get_root_pose().p).copy())
            self.model.grasp_commanded = grasp

        self.model._apply_flying_hand_grasp_command = apply_grasp
        with mock.patch("envs.flying_hand.planner.step"):
            execute_action_chunk(
                self.env,
                self.model,
                np.asarray([[0.01, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            )
        self.assertEqual(len(edge_positions), 1)
        self.assertAlmostEqual(float(edge_positions[0][0]), 0.01, places=6)


if __name__ == "__main__":
    unittest.main()
