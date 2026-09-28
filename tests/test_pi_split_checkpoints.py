"""Run with either openpi venv and its src on PYTHONPATH, JAX_PLATFORMS=cpu."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import orbax.checkpoint as ocp
from openpi.training import checkpoints as c


class CheckpointTest(unittest.TestCase):
    def test_legacy_full_state_is_readable(self):
        with tempfile.TemporaryDirectory() as d:
            legacy = ocp.CheckpointManager(d, item_handlers={
                "params": ocp.PyTreeCheckpointHandler(),
                "train_state": ocp.PyTreeCheckpointHandler(),
                "assets": c.CallbackHandler(),
            })
            legacy.save(5, {"params": {"params": {"x": np.array([5])}},
                            "train_state": {"step": np.array(5)}, "assets": lambda p: None})
            legacy.wait_until_finished()
            legacy.close()
            m, resuming = c.initialize_checkpoint_dir(d, keep_period=5, overwrite=False, resume=True)
            self.assertTrue(resuming)
            restored = m.restore(None, {"params": {"params": {"x": np.array([0])}},
                                        "train_state": {"step": np.array(0)}})
            self.assertEqual(int(restored["train_state"]["step"]), 5)
            m.state.close()
            m.weights.close()

    def test_retention_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            m, _ = c.initialize_checkpoint_dir(d, keep_period=2, overwrite=False, resume=True)
            for step in (1, 2, 3):
                params = {"params": {"x": np.array([step])}}
                m.weights.save(step, {"params": params, "assets": lambda path: None})
                m.state.save(step, {"params": params, "train_state": {"step": np.array(step)},
                                    "assets": lambda path: None})
                m.wait_until_finished()
            self.assertEqual(m.weights.all_steps(), [2, 3])
            self.assertEqual(m.state.all_steps(), [3])
            self.assertFalse((Path(d) / "2/train_state").exists())
            restored = m.restore(None, {"params": {"params": {"x": np.array([0])}},
                                        "train_state": {"step": np.array(0)}})
            self.assertEqual(int(restored["train_state"]["step"]), 3)
            m.state.close()
            m.weights.close()
            m, resuming = c.initialize_checkpoint_dir(d, keep_period=2, overwrite=False, resume=True)
            self.assertTrue(resuming)
            self.assertEqual(m.state.latest_step(), 3)
            m.state.close()
            m.weights.close()


if __name__ == "__main__":
    unittest.main()
