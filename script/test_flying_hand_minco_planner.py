#!/usr/bin/env python3
"""Numerical unit tests for the Flying-Hand MINCO planner."""

import unittest
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
    def test_positive_time_mapping_round_trip(self):
        durations = np.array([0.05, 0.5, 1.0, 2.0, 10.0])
        np.testing.assert_allclose(
            planner._forward_time(planner._backward_time(durations)),
            durations,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_xyz_and_yaw_minco_are_continuous_at_internal_waypoint(self):
        points = np.array([[0.0, 0.0, 0.0], [0.4, 0.2, 0.1], [0.9, 0.0, 0.2]])
        yaw_points = np.array([3.0, np.pi, 3.3])
        times = np.array([1.2, 1.6])
        xyz = planner.minco(points, times)
        yaw = planner.minco_yaw(yaw_points, times)

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

    def test_densification_inserts_a_midpoint_between_each_key_waypoint(self):
        poses = [
            _pose([0.0, 0.0, 0.0], yaw=0.0),
            _pose([0.2, 0.0, 0.0], yaw=0.4),
            _pose([0.2, 0.2, 0.0], yaw=0.8),
        ]
        config = planner.MincoOptimizationConfig(
            segment_per_distance=0.5,
            min_piece_num_per_key_segment=2,
            min_piece_num=1,
        )

        dense = planner._densify_plan(poses, config)

        self.assertEqual(len(dense), 5)
        np.testing.assert_allclose(dense[1].p, [0.1, 0.0, 0.0])
        np.testing.assert_allclose(dense[2].p, poses[1].p)
        np.testing.assert_allclose(dense[3].p, [0.2, 0.1, 0.0])
        np.testing.assert_allclose(dense[-1].p, poses[-1].p)

    def test_path_deviation_uses_distance_to_waypoint_chord(self):
        distance = planner._point_segment_distance(
            np.array([0.5, 0.1, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )

        self.assertAlmostEqual(distance, 0.1)


if __name__ == "__main__":
    unittest.main()
