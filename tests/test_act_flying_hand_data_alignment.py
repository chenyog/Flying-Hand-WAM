import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from policy.ACT.utils import EpisodicDataset


class ActFlyingHandDataAlignmentTest(unittest.TestCase):
    @staticmethod
    def _stats():
        return {
            "qpos_mean": np.zeros(5, dtype=np.float32),
            "qpos_std": np.ones(5, dtype=np.float32),
            "action_mean": np.zeros(5, dtype=np.float32),
            "action_std": np.ones(5, dtype=np.float32),
        }

    def test_sim_aligned_episode_pairs_qpos_t_with_action_t(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_0.hdf5"
            with h5py.File(path, "w") as root:
                root.attrs["sim"] = True
                root.create_dataset("action", data=np.arange(20, dtype=np.float32).reshape(4, 5))
                obs = root.create_group("observations")
                obs.create_dataset("qpos", data=np.arange(20, dtype=np.float32).reshape(4, 5) + 100)
                images = obs.create_group("images")
                images.create_dataset("cam_high", data=np.zeros((4, 2, 2, 3), dtype=np.uint8))
                images.create_dataset("cam_wrist", data=np.zeros((4, 2, 2, 3), dtype=np.uint8))

            dataset = EpisodicDataset(
                [0], directory, ["cam_high", "cam_wrist"], self._stats(),
                max_action_len=4, action_horizon=2,
            )
            with mock.patch("numpy.random.choice", return_value=2):
                _, qpos, actions, is_pad = dataset[0]
            np.testing.assert_array_equal(qpos.numpy(), np.arange(10, 15, dtype=np.float32) + 100)
            np.testing.assert_array_equal(actions[0].numpy(), np.arange(10, 15, dtype=np.float32))
            self.assertFalse(bool(is_pad[0]))

    def test_legacy_episode_keeps_historical_minus_one_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_0.hdf5"
            with h5py.File(path, "w") as root:
                root.create_dataset("action", data=np.arange(20, dtype=np.float32).reshape(4, 5))
                obs = root.create_group("observations")
                obs.create_dataset("qpos", data=np.zeros((4, 5), dtype=np.float32))
                images = obs.create_group("images")
                images.create_dataset("cam_high", data=np.zeros((4, 2, 2, 3), dtype=np.uint8))
                images.create_dataset("cam_wrist", data=np.zeros((4, 2, 2, 3), dtype=np.uint8))

            dataset = EpisodicDataset(
                [0], directory, ["cam_high", "cam_wrist"], self._stats(),
                max_action_len=4, action_horizon=2,
            )
            with mock.patch("numpy.random.choice", return_value=2):
                _, _, actions, _ = dataset[0]
            np.testing.assert_array_equal(actions[0].numpy(), np.arange(5, 10, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
