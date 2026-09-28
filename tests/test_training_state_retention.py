"""CPU roundtrips for inference retention and latest-only resume state."""
import importlib.util
import ast
import copy
import os
import random
import contextlib
from types import SimpleNamespace
import numpy as np
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StateTests(unittest.TestCase):
    def test_dp_loop_resume(self):
        from omegaconf import OmegaConf
        import dill
        import tqdm
        base = load_module("dp_loop_base", "policy/DP/diffusion_policy/workspace/base_workspace.py")
        tree = ast.parse((ROOT / "policy/DP/diffusion_policy/workspace/robotworkspace.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RobotWorkspace")
        func = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        class Model(torch.nn.Linear):
            def set_normalizer(self, n): pass
            def compute_loss(self, b): return self(b["obs"]).square().mean()
            def predict_action(self, obs): return {"action_pred": self(obs)}
        class Dataset:
            def get_normalizer(self): return {}
            def get_validation_dataset(self): return self
            def postprocess(self, batch, device): return batch
        class Loader(list):
            sampler = SimpleNamespace(rng=np.random.default_rng(0))
        class Log:
            def __init__(self, *a): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def log(self, *a): pass
        namespace = dict(copy=copy, pathlib=SimpleNamespace(Path=Path), os=os, torch=torch, np=np, random=random,
                         hydra=SimpleNamespace(utils=SimpleNamespace(instantiate=lambda x: Dataset())),
                         BaseImageDataset=Dataset, tqdm=tqdm, contextlib=contextlib,
                         create_dataloader=lambda *a, **kw: Loader([{"obs": torch.ones(1, 2), "action": torch.zeros(1, 1)}]),
                         get_scheduler=lambda name, optimizer, **kw: torch.optim.lr_scheduler.StepLR(optimizer, 1, .9),
                         TopKCheckpointManager=lambda **kw: SimpleNamespace(get_ckpt_path=lambda d: None),
                         optimizer_to=lambda *a: None, JsonLogger=Log)
        exec(compile(ast.Module(body=[func], type_ignores=[]), "dp_loop", "exec"), namespace)
        class Workspace(base.BaseWorkspace):
            distributed = False
            is_main_process = True
            include_keys = ["epoch", "global_step"]
            run = namespace["run"]
            def _model_module(self): return self.model
            def _init_wandb(self, cfg): return None
            def save_checkpoint(self, path=None, *args, **kwargs):
                if path is not None and Path(path).suffix == ".incomplete":
                    self.saved_state_epochs.append(self.epoch)
                return super().save_checkpoint(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as d:
            cfg = OmegaConf.create({"training": {"resume": False, "use_ema": False,
                "num_epochs": 2, "max_train_steps": 1, "gradient_accumulate_every": 1,
                "lr_scheduler": "unused", "lr_warmup_steps": 0, "device": "cpu", "debug": False,
                "freeze_encoder": False, "tqdm_interval_sec": 100, "val_every": 1,
                "max_val_steps": 1, "sample_every": 100, "checkpoint_every": 2}, "task": {"dataset": {}},
                "dataloader": {}, "val_dataloader": {}, "checkpoint": {"topk": {"monitor_key": "val_loss"}}})
            w = Workspace(cfg, output_dir=d)
            w.model = Model(2, 1)
            w.ema_model = None
            w.optimizer = torch.optim.AdamW(w.model.parameters(), lr=.01)
            w.epoch = w.global_step = 0
            w.saved_state_epochs = []
            w.run()
            self.assertEqual(w.saved_state_epochs, [2])
            saved = torch.load(w.get_checkpoint_path(), pickle_module=dill)
            self.assertIn("optimizer", saved["state_dicts"])
            self.assertNotIn("optimizer", torch.load(Path(d) / "checkpoints/epoch=0002.ckpt", pickle_module=dill)["state_dicts"])
            self.assertEqual(w.epoch, 2)
            cfg.training.resume = True
            cfg.training.num_epochs = 3
            w.run()
            self.assertEqual(w.epoch, 3)
            self.assertEqual(w.saved_state_epochs, [2, 3])
            self.assertTrue(all(s["step"] == 3 for s in w.optimizer.state.values()))

    def test_act_training_loop_resume(self):
        # Execute the actual training loop with a tiny CPU policy, without
        # importing the simulator/ResNet deployment dependencies.
        helper = load_module("act_loop_state", "policy/ACT/training_state.py")
        saved_epochs = []
        def save(state, path):
            saved_epochs.append(state["next_epoch"])
            helper.atomic_save(state, path)
        tree = ast.parse((ROOT / "policy/ACT/imitate_episodes.py").read_text())
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_bc")
        class Loader(list):
            sampler = object()
        loader = Loader([torch.ones(1, 2)])
        namespace = dict(torch=torch, np=np, os=os, deepcopy=copy.deepcopy,
                         set_seed=torch.manual_seed, tqdm=lambda x, **kw: x,
                         make_policy=lambda *a: torch.nn.Linear(2, 1),
                         make_optimizer=lambda _, m: torch.optim.AdamW(m.parameters(), lr=.01),
                         forward_pass=lambda x, m: {"loss": m(x).square().mean()},
                         detach_dict=lambda d: {k: v.detach() for k, v in d.items()},
                         compute_dict_mean=lambda ds: {k: torch.stack([d[k] for d in ds]).mean() for k in ds[0]},
                         init_wandb=lambda _: None, plot_history=lambda *a, **kw: None,
                         atomic_save=save, capture_rng=helper.capture_rng,
                         restore_rng=helper.restore_rng)
        exec(compile(ast.Module(body=[func], type_ignores=[]), "act_loop", "exec"), namespace)
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(num_epochs=2, ckpt_dir=d, seed=0, policy_class="ACT", policy_config={},
                       device=torch.device("cpu"), save_freq=2, dataset_stats={})
            namespace["train_bc"](loader, loader, cfg)
            state_path = Path(d) / "state/latest.pt"
            saved = torch.load(state_path, weights_only=False)
            self.assertEqual(saved["next_epoch"], 2)
            self.assertEqual(saved_epochs, [2])
            self.assertNotIn("optimizer", torch.load(Path(d) / "policy_epoch_2_seed_0.ckpt", weights_only=True))
            cfg.update(num_epochs=3, loaded_resume_state=saved)
            namespace["train_bc"](loader, loader, cfg)
            resumed = torch.load(state_path, weights_only=False)
            self.assertEqual(resumed["next_epoch"], 3)
            self.assertEqual(saved_epochs, [2, 3])
            self.assertTrue(all(s["step"] == 3 for s in resumed["optimizer"]["state"].values()))

    def test_openvla_roundtrip(self):
        helper = load_module("oft_state", "policy/openvla-oft/prismatic/util/training_state.py")
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, .9)
        def update():
            opt.zero_grad()
            model(torch.ones(1, 2)).square().sum().backward()
            opt.step()
            scheduler.step()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state/latest.pt"
            update()
            helper.save_state(path, {"model": model}, opt, scheduler, 1, {"action": [1]})
            saved = {k: v.clone() for k, v in model.state_dict().items()}
            update()
            step, stats = helper.load_state(path, {"model": model}, opt, scheduler)
            self.assertEqual(step, 1)
            self.assertEqual(stats, {"action": [1]})
            self.assertEqual(scheduler.last_epoch, 1)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, saved[key])
            self.assertTrue(all(s["step"] == 1 for s in opt.state.values()))
            update()
            helper.save_state(path, {"model": model}, opt, scheduler, 2, stats)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_act_atomic_and_rng(self):
        helper = load_module("act_state", "policy/ACT/training_state.py")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state/latest.pt"
            rng = helper.capture_rng()
            expected = torch.rand(4)
            helper.atomic_save({"rng": rng, "next_epoch": 100}, path)
            payload = torch.load(path, weights_only=False)
            helper.restore_rng(payload["rng"])
            torch.testing.assert_close(torch.rand(4), expected)
            helper.atomic_save({"next_epoch": 200}, path)
            self.assertEqual(torch.load(path, weights_only=False)["next_epoch"], 200)
            self.assertEqual(list(path.parent.iterdir()), [path])
            with patch.object(helper.torch, "save", side_effect=OSError("simulated write failure")):
                with self.assertRaises(OSError):
                    helper.atomic_save({"next_epoch": 300}, path)
            self.assertEqual(torch.load(path, weights_only=False)["next_epoch"], 200)

    def test_dp_inference_excludes_optimizer(self):
        from omegaconf import OmegaConf
        helper = load_module("dp_workspace", "policy/DP/diffusion_policy/workspace/base_workspace.py")
        import dill
        with tempfile.TemporaryDirectory() as d:
            workspace = helper.BaseWorkspace(OmegaConf.create({}), output_dir=d)
            workspace.model = torch.nn.Linear(2, 1)
            workspace.optimizer = torch.optim.AdamW(workspace.model.parameters())
            workspace.model(torch.ones(1, 2)).sum().backward()
            workspace.optimizer.step()
            weights = Path(d) / "weights.ckpt"
            workspace.save_checkpoint(weights, exclude_keys=("optimizer",), use_thread=False)
            payload = torch.load(weights, pickle_module=dill)
            self.assertEqual(set(payload["state_dicts"]), {"model"})
            full = Path(d) / "latest.ckpt"
            workspace.save_checkpoint(full, use_thread=False)
            workspace.optimizer.state.clear()
            workspace.load_checkpoint(full)
            self.assertTrue(workspace.optimizer.state)


if __name__ == "__main__":
    unittest.main()
