"""CPU integration test; no pretrained models or GPU required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from starVLA.training.trainer_utils.resume import ResumeMixin


class ResumeTest(unittest.TestCase):
    def test_roundtrip_and_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = ResumeMixin()
            trainer.accelerator = Accelerator(cpu=True)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
            model, optimizer = trainer.accelerator.prepare(model, optimizer)
            trainer.config = OmegaConf.create({"trainer": {"save_interval": 1091}})
            trainer.checkpoint_dir = directory
            trainer.completed_steps = 0
            trainer._init_resume()
            self.assertEqual(trainer.state_interval, 1091)
            def step():
                optimizer.zero_grad()
                trainer.accelerator.backward(model(torch.ones(1, 2)).square().sum())
                optimizer.step()
                trainer.lr_scheduler.step()
                trainer.completed_steps += 1
            step()
            trainer._save_resume_state()
            first = Path(directory) / "state/step_000001"
            step()
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            expected_lr = trainer.lr_scheduler.get_last_lr()
            with patch.object(trainer.accelerator, "save_state", side_effect=OSError("simulated write failure")):
                with self.assertRaises(OSError):
                    trainer._save_resume_state()
            self.assertTrue(first.is_dir())
            trainer._save_resume_state()
            second = Path(directory) / "state/step_000002"
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())
            step()
            trainer._load_checkpoint(second)
            self.assertEqual(trainer.completed_steps, 2)
            self.assertEqual(trainer.lr_scheduler.get_last_lr(), expected_lr)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, expected[k])
            for state in optimizer.state.values():
                self.assertEqual(state["step"].item(), 2)
            step()
            self.assertEqual(trainer.completed_steps, 3)
            with self.assertRaises(ValueError):
                trainer._load_checkpoint(first)
            metadata = second / "trainer_state.json"
            data = json.loads(metadata.read_text())
            data["world_size"] = 8
            metadata.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                trainer._load_checkpoint(second)


if __name__ == "__main__":
    unittest.main()
