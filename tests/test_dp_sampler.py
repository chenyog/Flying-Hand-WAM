from pathlib import Path
import sys
import unittest

import numpy as np


DP_ROOT = Path(__file__).resolve().parents[1] / "policy" / "DP"
sys.path.insert(0, str(DP_ROOT))

from diffusion_policy.common.sampler import (  # noqa: E402
    get_task_stratified_val_mask,
    get_val_mask,
)
from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset  # noqa: E402


class DPValidationMaskTest(unittest.TestCase):
    def test_respects_requested_ratio_and_seed(self):
        n_episodes = 10
        val_ratio = 0.2
        seed = 42

        mask = get_val_mask(n_episodes, val_ratio, seed)
        expected_indices = np.random.default_rng(seed).choice(
            n_episodes,
            size=2,
            replace=False,
        )

        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, (n_episodes,))
        self.assertEqual(mask.sum(), 2)
        np.testing.assert_array_equal(np.flatnonzero(mask), np.sort(expected_indices))

    def test_is_deterministic_and_not_fixed_to_last_episode(self):
        first = get_val_mask(100, 0.2, seed=7)
        second = get_val_mask(100, 0.2, seed=7)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.sum(), 20)
        self.assertFalse(np.array_equal(np.flatnonzero(first), np.array([99])))

    def test_preserves_at_least_one_training_episode(self):
        mask = get_val_mask(3, 1.0, seed=0)

        self.assertEqual(mask.sum(), 2)
        self.assertEqual((~mask).sum(), 1)

    def test_handles_disabled_validation_and_one_episode(self):
        self.assertFalse(get_val_mask(5, 0.0, seed=0).any())
        np.testing.assert_array_equal(get_val_mask(1, 0.2, seed=0), np.array([False]))


class DPTaskStratifiedValidationMaskTest(unittest.TestCase):
    def test_retains_validation_episodes_for_every_task(self):
        task_ids = np.repeat(np.arange(10), 100)

        mask = get_task_stratified_val_mask(task_ids, val_ratio=0.02, seed=42)

        self.assertEqual(mask.sum(), 20)
        for task_id in np.unique(task_ids):
            self.assertEqual(mask[task_ids == task_id].sum(), 2)

    def test_is_deterministic_and_keeps_single_episode_tasks_in_training(self):
        task_ids = np.array([0, 1, 1, 2, 2, 2])

        first = get_task_stratified_val_mask(task_ids, val_ratio=0.2, seed=7)
        second = get_task_stratified_val_mask(task_ids, val_ratio=0.2, seed=7)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(first[0])
        self.assertEqual(first[task_ids == 1].sum(), 1)
        self.assertEqual(first[task_ids == 2].sum(), 1)

    def test_rejects_non_vector_task_ids(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            get_task_stratified_val_mask(np.zeros((2, 2)), val_ratio=0.2, seed=0)


class DPRgbLayoutTest(unittest.TestCase):
    @staticmethod
    def dataset_stub():
        dataset = object.__new__(RobotImageDataset)
        dataset.image_keys = ["head_camera"]
        dataset.image_obs_keys = ["head_cam"]
        dataset.state_key = "state"
        dataset.action_key = "action"
        return dataset

    def test_current_nchw_layout_is_accepted(self):
        sample = {
            "head_camera": np.zeros((2, 3, 4, 5), dtype=np.uint8),
            "state": np.zeros((2, 5), dtype=np.float32),
            "action": np.zeros((2, 5), dtype=np.float32),
        }
        data = self.dataset_stub()._sample_to_data(sample)
        self.assertEqual(data["obs"]["head_cam"].shape, (2, 3, 4, 5))

    def test_legacy_nhwc_layout_is_rejected(self):
        sample = {
            "head_camera": np.zeros((2, 4, 5, 3), dtype=np.uint8),
            "state": np.zeros((2, 5), dtype=np.float32),
            "action": np.zeros((2, 5), dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "current NCHW RGB layout"):
            self.dataset_stub()._sample_to_data(sample)


if __name__ == "__main__":
    unittest.main()
