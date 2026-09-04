#!/usr/bin/env python3
"""Unit tests for policy-time Flying-Hand grasp eligibility and execution."""

import unittest
from unittest import mock
from pathlib import Path
import sys

import numpy as np
import sapien

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.flying_hand._base_task import FlyingHandBaseTask
from envs.flying_hand.blocks_ranking_rgb import blocks_ranking_rgb
from envs.flying_hand import planner
from envs.utils.create_actor import UnStableError
from policy.FastWAM.experiments.robotwin.deploy_policy import WorldActionRobotWinPolicy


class _Named:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class _Actor:
    config = {
        "center": [0.0, 0.0, 0.0],
        "extents": [0.04, 0.04, 0.04],
        "scale": [1.0, 1.0, 1.0],
    }

    def __init__(self, name, position):
        self.actor = _Named(name)
        self.pose = sapien.Pose(position)

    def get_name(self):
        return self.actor.get_name()

    def get_pose(self):
        return self.pose


class _Hand:
    def __init__(self):
        self.link = _Named("left_down_link")

    def get_root_pose(self):
        return sapien.Pose()

    def get_links(self):
        return [self.link]


def _environment(actor):
    env = FlyingHandBaseTask.__new__(FlyingHandBaseTask)
    env.flying_hand = _Hand()
    env.flying_hand_config = {"root": {"u_center_offset": [0.0, 0.0, 0.0]}}
    env.flying_hand_grasp_validation = {
        "center_bounds_min": np.array([0.04, -0.08, -0.12]),
        "center_bounds_max": np.array([0.16, 0.08, 0.06]),
    }
    return env


class FlyingHandGraspValidationTest(unittest.TestCase):
    def test_blocks_ranking_policy_filter_climbs_before_horizontal_approach(self):
        block = _Actor("green block", [0.70, 0.0, 1.00])
        env = blocks_ranking_rgb.__new__(blocks_ranking_rgb)
        env.blocks = [block]
        env.flying_hand_config = {"root": {"u_center_offset": [0.0, 0.0, 0.0]}}
        target = sapien.Pose([0.60, 0.0, 0.90])
        current = sapien.Pose([0.10, -0.20, 0.80])

        filtered, diagnostic = env.filter_flying_hand_policy_target(
            target,
            current,
        )

        np.testing.assert_allclose(filtered.p[:2], current.p[:2])
        self.assertAlmostEqual(filtered.p[2], 1.09)
        self.assertTrue(diagnostic["vertical_first"])
        self.assertEqual(diagnostic["actor"], "green block")

    def test_blocks_ranking_policy_filter_preserves_safe_target(self):
        block = _Actor("green block", [0.70, 0.0, 1.00])
        env = blocks_ranking_rgb.__new__(blocks_ranking_rgb)
        env.blocks = [block]
        env.flying_hand_config = {"root": {"u_center_offset": [0.0, 0.0, 0.0]}}
        target = sapien.Pose([0.60, 0.0, 1.12])

        filtered, diagnostic = env.filter_flying_hand_policy_target(
            target,
            sapien.Pose([0.50, 0.0, 1.10]),
        )

        np.testing.assert_allclose(filtered.p, target.p)
        self.assertIsNone(diagnostic)

    def test_blocks_ranking_policy_filter_protects_neighbor_while_carrying(self):
        carried = _Actor("green block", [0.70, 0.0, 1.00])
        neighbor = _Actor("blue block", [0.70, 0.28, 1.09])
        env = blocks_ranking_rgb.__new__(blocks_ranking_rgb)
        env.blocks = [carried, neighbor]
        env.flying_hand_config = {"root": {"u_center_offset": [0.0, 0.0, 0.0]}}

        filtered, diagnostic = env.filter_flying_hand_policy_target(
            sapien.Pose([0.60, 0.02, 0.90]),
            sapien.Pose([0.10, -0.20, 0.80]),
            carried_actor=carried,
        )

        self.assertAlmostEqual(filtered.p[2], 1.18, places=6)
        self.assertEqual(diagnostic["actor"], "blue block")

    def test_expert_isolated_carry_rejects_invalid_box_center(self):
        actor = _Actor("knocked block", [0.10, 0.20, -0.04])
        env = _environment(actor)
        env._isolated_carried_actor_state = None
        env.flying_hand_grasp_diagnostics = []
        env.plan_success = True
        env.task_failed = False

        with self.assertRaises(UnStableError):
            planner.begin_isolated_carry(env, actor)

        self.assertFalse(env.plan_success)
        self.assertTrue(env.task_failed)
        self.assertIsNone(env._isolated_carried_actor_state)
        self.assertEqual(
            env.flying_hand_grasp_diagnostics[-1]["stage"],
            "isolated_carry_start",
        )

    def test_scripted_follow_stops_if_target_box_center_leaves_region(self):
        actor = _Actor("block", [0.10, 0.0, -0.04])
        env = _environment(actor)
        env._released_actor_collision_state = None
        env.flying_hand_grasp_diagnostics = []
        env.plan_success = True
        env.task_failed = False
        env._isolated_carried_actor_state = {
            "actor": actor,
            "components": [],
            "enabled": [],
            "excluded_components": [],
        }

        with self.assertRaises(UnStableError):
            planner.set_isolated_carried_actor_target(
                env,
                actor,
                sapien.Pose([0.10, 0.10, -0.04]),
            )

        self.assertFalse(env.plan_success)
        self.assertTrue(env.task_failed)
        self.assertIsNone(env._isolated_carried_actor_state)
        self.assertEqual(
            env.flying_hand_grasp_diagnostics[-1]["stage"],
            "isolated_carry_follow",
        )

    def test_task_can_exclude_destination_container_from_grasp_candidates(self):
        env = FlyingHandBaseTask.__new__(FlyingHandBaseTask)
        target = object()
        destination = object()
        env.task_actors = [target, destination]
        env.graspable_actors = [target]

        self.assertEqual(env.get_flying_hand_grasp_candidates(), (target,))

    def test_close_edge_validates_and_attaches_immediately(self):
        policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
        policy.grasp_commanded = False
        policy.gripper_state = "open"
        policy.grasp_diagnostics = []
        policy.attached_actor = None
        policy.attached_pose = None

        captures = []

        def reject_capture(task_env, event):
            captures.append(event)
            event["attachment"] = "rejected_box_center_outside_grasp_region"
            return False

        policy._attach_actor_in_grasp_space = reject_capture

        env = type("Env", (), {})()
        env.take_action_cnt = 7
        env.is_grasping = False
        env.flying_hand_config = {
            "gripper": {
                "close_qpos": [1.0],
                "open_qpos": [0.0],
            },
        }
        commands = []

        def set_gripper(qpos, is_grasp):
            commands.append((list(qpos), bool(is_grasp)))
            env.is_grasping = bool(is_grasp)

        env.set_flying_hand_gripper = set_gripper
        with mock.patch("envs.flying_hand.planner.hold") as hold:
            policy._apply_flying_hand_grasp_command(
                env,
                True,
                action_step=7,
            )

        self.assertEqual(commands, [([0.0], False)])
        hold.assert_not_called()
        self.assertFalse(env.is_grasping)
        self.assertEqual(len(captures), 1)
        self.assertEqual(policy.gripper_state, "grasping_empty")
        self.assertTrue(policy.grasp_diagnostics[-1]["completed"])
        self.assertEqual(policy.grasp_diagnostics[-1]["command_latency_seconds"], 0.0)
        self.assertEqual(
            policy.grasp_diagnostics[-1]["attachment"],
            "rejected_box_center_outside_grasp_region",
        )
        self.assertFalse(
            policy.grasp_diagnostics[-1]["gripper_close_commanded"]
        )
        self.assertTrue(policy.grasp_diagnostics[-1]["gripper_kept_open"])

    def test_open_edge_releases_immediately(self):
        policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
        policy.grasp_commanded = True
        policy.gripper_state = "grasping_attached"
        policy.grasp_diagnostics = []
        policy.attached_actor = object()
        policy.attached_pose = sapien.Pose()

        env = type("Env", (), {})()
        env.take_action_cnt = 11
        env.is_grasping = True
        env.flying_hand_config = {
            "gripper": {
                "close_qpos": [1.0],
                "open_qpos": [0.0],
            },
        }
        events = []

        def set_gripper(qpos, is_grasp):
            events.append(("command", list(qpos), bool(is_grasp)))
            env.is_grasping = bool(is_grasp)

        env.set_flying_hand_gripper = set_gripper
        with mock.patch("envs.flying_hand.planner.hold") as hold:
            policy._apply_flying_hand_grasp_command(
                env,
                False,
                action_step=11,
            )

        self.assertEqual(events[0], ("command", [0.0], False))
        hold.assert_not_called()
        self.assertIsNone(policy.attached_actor)
        self.assertEqual(policy.gripper_state, "open")
        self.assertTrue(policy.grasp_diagnostics[-1]["completed"])
        self.assertTrue(policy.grasp_diagnostics[-1]["gripper_open_commanded"])
        self.assertEqual(policy.grasp_diagnostics[-1]["command_latency_seconds"], 0.0)

    def test_close_then_open_edges_apply_immediately_without_pending_state(self):
        policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
        policy.grasp_commanded = False
        policy.gripper_state = "open"
        policy.grasp_diagnostics = []
        policy.attached_actor = None
        policy.attached_pose = None
        attached_actor = object()

        def attach(task_env, event):
            policy.attached_actor = attached_actor
            policy.attached_pose = sapien.Pose()
            event["attachment"] = "attached"
            return True

        policy._attach_actor_in_grasp_space = attach

        env = type("Env", (), {})()
        env.is_grasping = False
        env.flying_hand_config = {
            "gripper": {
                "close_qpos": [1.0],
                "open_qpos": [0.0],
            },
        }

        def set_gripper(qpos, is_grasp):
            env.is_grasping = bool(is_grasp)

        env.set_flying_hand_gripper = set_gripper
        policy._apply_flying_hand_grasp_command(env, True, action_step=4)
        self.assertIs(policy.attached_actor, attached_actor)
        self.assertEqual(policy.gripper_state, "grasping_attached")

        policy._apply_flying_hand_grasp_command(env, False, action_step=6)
        close_event, open_event = policy.grasp_diagnostics
        self.assertEqual(close_event["command_latency_seconds"], 0.0)
        self.assertEqual(open_event["command_latency_seconds"], 0.0)
        self.assertTrue(open_event["released"])
        self.assertIsNone(policy.attached_actor)
        self.assertEqual(policy.gripper_state, "open")

    def test_actor_box_center_inside_region_is_eligible(self):
        actor = _Actor("block", [0.10, 0.0, -0.04])
        diagnostic = _environment(actor).get_flying_hand_grasp_diagnostic(actor)

        self.assertTrue(diagnostic["eligible"])
        self.assertTrue(diagnostic["center_inside"])

    def test_box_overlap_does_not_count_when_center_is_outside(self):
        actor = _Actor("block", [0.17, 0.0, -0.04])
        diagnostic = _environment(actor).get_flying_hand_grasp_diagnostic(actor)

        self.assertLess(diagnostic["actor_bounds_min_u"][0], 0.16)
        self.assertFalse(diagnostic["center_inside"])
        self.assertFalse(diagnostic["eligible"])

    def test_knocked_actor_with_center_outside_region_is_rejected(self):
        actor = _Actor("block", [0.10, 0.30, -0.04])
        diagnostic = _environment(actor).get_flying_hand_grasp_diagnostic(actor)

        self.assertFalse(diagnostic["center_inside"])
        self.assertFalse(diagnostic["eligible"])

    def test_explicit_hand_pose_changes_center_test_frame(self):
        actor = _Actor("block", [0.50, 0.0, -0.04])
        env = _environment(actor)

        current = env.get_flying_hand_grasp_diagnostic(actor)
        future = env.get_flying_hand_grasp_diagnostic(
            actor,
            hand_pose=sapien.Pose([0.40, 0.0, 0.0]),
        )

        self.assertFalse(current["center_inside"])
        self.assertTrue(future["center_inside"])

    def test_explicit_actor_pose_validates_scripted_follow_target(self):
        actor = _Actor("block", [0.10, 0.0, -0.04])
        env = _environment(actor)

        valid = env.get_flying_hand_grasp_diagnostic(
            actor,
            actor_pose=sapien.Pose([0.10, 0.0, -0.04]),
        )
        invalid = env.get_flying_hand_grasp_diagnostic(
            actor,
            actor_pose=sapien.Pose([0.10, 0.10, -0.04]),
        )

        self.assertTrue(valid["eligible"])
        self.assertFalse(invalid["eligible"])

    def test_waypoints_are_tracked_at_fixed_rate_without_minco(self):
        policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
        policy.attached_actor = None
        policy.attached_pose = None
        policy.grasp_commanded = False
        policy.gripper_state = "open"
        policy.waypoint_diagnostics = []
        policy._waypoint_reference_position = None
        policy._waypoint_reference_velocity = np.zeros(3)
        policy._waypoint_reference_orientation = None
        policy._record_flight_sample = lambda task_env: None

        class Hand:
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

        class Dynamics:
            def __init__(self):
                self.v = np.zeros(3)
                self.w = np.zeros(3)
                self.references = []

            def step(self, pose, velocity, acceleration, grasped):
                self.references.append((
                    np.array(pose.p),
                    np.array(velocity),
                    np.array(acceleration),
                ))
                self.v = np.array(velocity)
                return pose, self.v

        env = type("Env", (), {})()
        env.flying_hand = Hand()
        env.flying_hand_dynamics = Dynamics()
        env.enable_dynamics = True
        env.is_grasping = False
        env.save_freq = 10
        env.sim_timestep = 0.005
        env.flying_hand_waypoint_tracking = {
            "max_velocity": 0.6,
            "max_acceleration": 0.8,
            "max_yaw_rate": 0.5,
        }
        env.initial_imu_odom_pose = sapien.Pose()
        env.flying_hand_initial_pose = sapien.Pose()

        actions = np.array([
            [0.02, 0.00, 0.00, 0.00, 0.0],
            [0.04, 0.01, 0.00, 0.05, 0.0],
        ])
        with mock.patch("envs.flying_hand.planner.step"), mock.patch(
            "envs.flying_hand.planner.plan_and_move_minco"
        ) as minco:
            policy._track_flying_hand_waypoints(env, actions)

        minco.assert_not_called()
        self.assertEqual(len(env.flying_hand_dynamics.references), 20)
        for _, velocity, acceleration in env.flying_hand_dynamics.references:
            np.testing.assert_allclose(velocity, 0.0)
            np.testing.assert_allclose(acceleration, 0.0)
        diagnostic = policy.waypoint_diagnostics[-1]
        self.assertLessEqual(diagnostic["max_reference_velocity_mps"], 0.6 + 1e-9)
        self.assertLessEqual(diagnostic["max_reference_acceleration_mps2"], 0.8 + 1e-9)
        self.assertLess(policy._waypoint_reference_position[0], actions[-1, 0])
        self.assertAlmostEqual(
            diagnostic["duration_seconds"],
            0.1,
        )

    def test_grasp_edge_keeps_full_fixed_cadence_chunk(self):
        policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
        policy.grasp_commanded = False
        policy._record_action_chunk = mock.Mock()
        policy._track_flying_hand_waypoints = mock.Mock()
        env = type("Env", (), {})()
        env.flying_hand_grasp_validation = {
            "close_threshold": 0.5,
            "open_threshold": 0.4,
        }
        env.flying_hand_ref_pose = sapien.Pose()
        actions = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 1.0],
            [0.3, 0.0, 0.0, 0.0, 1.0],
        ])

        consumed = policy._execute_flying_hand_waypoint_chunk(env, actions)

        self.assertEqual(consumed, 4)
        policy._record_action_chunk.assert_called_once_with(env, actions)
        np.testing.assert_array_equal(
            policy._track_flying_hand_waypoints.call_args.args[1],
            actions,
        )
        np.testing.assert_array_equal(
            policy._track_flying_hand_waypoints.call_args.kwargs["grasp_states"],
            np.array([False, False, True, True]),
        )


if __name__ == "__main__":
    unittest.main()
