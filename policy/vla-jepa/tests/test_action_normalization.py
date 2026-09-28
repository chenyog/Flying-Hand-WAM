import unittest

import numpy as np

from starVLA.model.action_normalization import unnormalize_actions


class ActionNormalizationTests(unittest.TestCase):
    def test_flying_hand_five_dimensions_and_no_mutation(self):
        values = np.array([[-1, 0, 1, 0.5, 0.4], [1, 0, -1, -0.5, 0.6]], dtype=float)
        original = values.copy()
        stats = dict(q01=[0, 0, 0, -2, 0], q99=[2, 4, 6, 2, 1],
                     mask=[True, True, True, True, False],
                     binary_mask=[False, False, False, False, True])
        result = unnormalize_actions(values, stats)
        np.testing.assert_allclose(result, [[0, 2, 6, 1, 0], [2, 2, 0, -1, 1]])
        np.testing.assert_array_equal(values, original)

    def test_legacy_seven_dimensions(self):
        stats = dict(q01=[-2] * 6 + [0], q99=[2] * 6 + [1], mask=[True] * 6 + [False])
        result = unnormalize_actions(np.full((2, 7), 0.5), stats)
        np.testing.assert_allclose(result, np.ones((2, 7)))

    def test_only_binary_unnormalized_dimensions_are_thresholded(self):
        stats = dict(q01=[0, -1, 0], q99=[1, 1, 1], mask=[True, False, False])
        result = unnormalize_actions(np.array([[0.6, 0.6, 0.6]]), stats)
        np.testing.assert_allclose(result, [[0.8, 0.6, 1]])

    def test_explicit_constant_gripper_and_batched_shape(self):
        stats = dict(q01=[0, 0], q99=[1, 0], mask=[True, False], binary_mask=[False, True])
        result = unnormalize_actions(np.full((2, 3, 2), 0.6), stats)
        self.assertEqual(result.shape, (2, 3, 2))
        np.testing.assert_allclose(result[..., 1], 1)

    def test_bad_dimensions_fail_clearly(self):
        with self.assertRaises(ValueError):
            unnormalize_actions(np.zeros((8, 5)), dict(q01=[0] * 7, q99=[1] * 7))


if __name__ == "__main__":
    unittest.main()
