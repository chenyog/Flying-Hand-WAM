#!/usr/bin/env python3
"""Numerical unit tests for the Flying-Hand MINCO planner."""

import unittest
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import sapien

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.flying_hand import planner


def _pose(position, yaw=0.0):
    return sapien.Pose(
        position,
        [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)],
    )


class FlyingHandMincoPlannerTest(unittest.TestCase):
    def test_release_collision_filter_preserves_scene_collisions_and_restores_groups(self):
        class Shape:
            def __init__(self, groups):
                self.groups = list(groups)

            def get_collision_groups(self):
                return list(self.groups)

            def set_collision_groups(self, groups):
                self.groups = list(groups)

        class Component:
            def __init__(self, shapes):
                self.shapes = shapes

            def get_collision_shapes(self):
                return self.shapes

        class Link:
            def __init__(self, shape):
                self.shape = shape

            def get_collision_shapes(self):
                return [self.shape]

        actor_shape = Shape([1, 1, 0, 0])
        actor = type("Actor", (), {})()
        actor.actor = type(
            "Entity",
            (),
            {"components": [Component([actor_shape])]},
        )()
        env = type("Env", (), {})()
        env._released_actor_collision_state = None
        env.flying_hand = type(
            "Hand",
            (),
            {"get_links": lambda self: [Link(Shape([1, 1, 1 << 8, 0x143]))]},
        )()

        planner._suppress_released_actor_gripper_collisions(env, actor)

        self.assertEqual(actor_shape.groups, [1, 1, 1 << 8, 0x143])
        self.assertIs(env._released_actor_collision_state["actor"], actor)
        planner.restore_released_actor_collisions(env)
        self.assertEqual(actor_shape.groups, [1, 1, 0, 0])
        self.assertIsNone(env._released_actor_collision_state)

    def test_xyz_and_yaw_minco_are_continuous_at_internal_waypoint(self):
        points = np.array([[0.0, 0.0, 0.0], [0.4, 0.2, 0.1], [0.9, 0.0, 0.2]])
        yaw_points = np.array([3.0, np.pi, 3.3])
        times = np.array([1.2, 1.6])
        xyz = planner._position_coefficients(points, times)
        yaw = planner._yaw_coefficients(yaw_points, times)

        end_pva = planner.sample(xyz[0], times[0])
        start_pva = planner.sample(xyz[1], 0.0)
        for end_value, start_value in zip(end_pva, start_pva):
            np.testing.assert_allclose(end_value, start_value, atol=1e-10)
        np.testing.assert_allclose(end_pva[0], points[1], atol=1e-10)

        end_yaw = planner.sample_yaw(yaw[0], times[0])
        start_yaw = planner.sample_yaw(yaw[1], 0.0)
        np.testing.assert_allclose(end_yaw, start_yaw, atol=1e-10)
        np.testing.assert_allclose(end_yaw[0], yaw_points[1], atol=1e-10)
        np.testing.assert_allclose(
            planner.sample_yaw(yaw[0], 0.0)[:2],
            [yaw_points[0], 0.0],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            planner.sample_yaw(yaw[-1], times[-1])[:2],
            [yaw_points[-1], 0.0],
            atol=1e-10,
        )

    def test_time_optimizer_changes_positive_durations_and_reduces_cost(self):
        poses = [
            _pose([0.0, 0.0, 0.0]),
            _pose([0.45, 0.15, 0.10]),
            _pose([0.90, 0.0, 0.20]),
        ]
        optimizer = planner.MincoTimeOptimizer(
            planner.MincoOptimizationConfig(max_iteration=80)
        )
        result = optimizer.optimize(poses)

        self.assertTrue(result.success, result.message)
        self.assertTrue(np.all(result.times > 0.0))
        self.assertFalse(np.allclose(result.times, result.initial_times))
        self.assertLessEqual(result.final_cost, result.initial_cost + 1e-8)
        self.assertEqual(result.metrics["total_time_s"], np.sum(result.times))
        self.assertEqual(result.backend, "deployment_cpp_minco_lbfgs")
        self.assertGreater(result.function_evaluations, 0)
        self.assertGreaterEqual(result.optimization_wall_time_seconds, 0.0)

    def test_cpp_analytic_time_gradient_matches_finite_difference(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.2, 0.2, 0.0],
            [0.4, 0.0, 0.0],
        ])
        yaw_points = np.array([0.0, 0.1, 0.2])
        raw_times = np.array([-0.15, 0.2])
        config = planner.MincoOptimizationConfig()

        cpp_cost, analytic_gradient = planner._minco_cpp.evaluate_raw_times(
            points,
            yaw_points,
            raw_times,
            0.0,
            asdict(config),
        )
        self.assertTrue(np.isfinite(cpp_cost))
        finite_difference = np.empty_like(raw_times)
        epsilon = 1e-6
        for index in range(len(raw_times)):
            positive = raw_times.copy()
            negative = raw_times.copy()
            positive[index] += epsilon
            negative[index] -= epsilon
            positive_cost = planner._minco_cpp.evaluate_raw_times(
                points,
                yaw_points,
                positive,
                0.0,
                asdict(config),
            )[0]
            negative_cost = planner._minco_cpp.evaluate_raw_times(
                points,
                yaw_points,
                negative,
                0.0,
                asdict(config),
            )[0]
            finite_difference[index] = (
                positive_cost - negative_cost
            ) / (2.0 * epsilon)

        np.testing.assert_allclose(
            analytic_gradient,
            finite_difference,
            rtol=2e-7,
            atol=2e-6,
        )

    def test_deployment_path_densification_preserves_endpoints_and_minimum_pieces(self):
        poses = [_pose([0.0, 0.0, 0.0]), _pose([0.2, 0.0, 0.0])]
        config = planner.MincoOptimizationConfig(
            segment_per_distance=0.5,
            min_piece_num=2,
        )
        dense = planner._densify_plan(poses, config)

        self.assertEqual(len(dense), 3)
        np.testing.assert_allclose(dense[0].p, poses[0].p)
        np.testing.assert_allclose(dense[-1].p, poses[-1].p)
        np.testing.assert_allclose(dense[1].p, [0.1, 0.0, 0.0])

    def test_densification_matches_cpp_rounding_at_positive_half_integer(self):
        poses = [_pose([0.0, 0.0, 0.0]), _pose([1.25, 0.0, 0.0])]
        config = planner.MincoOptimizationConfig(
            segment_per_distance=0.5,
            min_piece_num=1,
        )

        dense = planner._densify_plan(poses, config)

        self.assertEqual(len(dense) - 1, 3)

    def test_densification_preserves_nearby_key_waypoints_without_forced_midpoints(self):
        poses = [
            _pose([0.0, 0.0, 0.0], yaw=0.0),
            _pose([0.2, 0.0, 0.0], yaw=0.4),
            _pose([0.2, 0.2, 0.0], yaw=0.8),
        ]
        config = planner.MincoOptimizationConfig(
            segment_per_distance=0.5,
            min_piece_num=1,
        )

        dense = planner._densify_plan(poses, config)

        self.assertEqual(len(dense), 3)
        np.testing.assert_allclose(dense[1].p, poses[1].p)
        np.testing.assert_allclose(dense[-1].p, poses[-1].p)


if __name__ == "__main__":
    unittest.main()
