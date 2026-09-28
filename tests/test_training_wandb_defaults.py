import ast
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path):
    return (PROJECT_ROOT / relative_path).read_text()


def _dataclass_default(relative_path, field_name):
    tree = ast.parse(_read(relative_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == field_name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"Could not find {field_name} in {relative_path}")


class TrainingWandbDefaultsTest(unittest.TestCase):
    def test_dp_and_fastwam_yaml_defaults_are_online(self):
        dp = yaml.safe_load(
            _read("policy/DP/diffusion_policy/config/robot_dp_5.yaml")
        )
        self.assertEqual(dp["logging"]["mode"], "online")

        fastwam = yaml.safe_load(_read("policy/FastWAM/configs/train.yaml"))
        self.assertTrue(fastwam["wandb"]["enabled"])
        self.assertEqual(fastwam["wandb"]["mode"], "online")

    def test_pi_train_configs_default_to_online(self):
        for path in (
            "policy/pi0/src/openpi/training/config.py",
            "policy/pi05/src/openpi/training/config.py",
        ):
            self.assertTrue(_dataclass_default(path, "wandb_enabled"), path)
            self.assertEqual(_dataclass_default(path, "wandb_mode"), "online", path)

    def test_flying_hand_wrappers_default_to_online(self):
        paths = (
            "policy/DP/train_flying_hand.sh",
            "policy/ACT/train_flying_hand.sh",
            "policy/pi0/train_flying_hand.sh",
            "policy/pi05/train_flying_hand.sh",
            "policy/FastWAM/scripts/train_zero1.sh",
            "policy/FastWAM/scripts/train_zero2.sh",
            "policy/openvla-oft/train_flying_hand.sh",
            "policy/vla-jepa/scripts/train_flying_hand.sh",
            "policy/vla-jepa/scripts/run_vlajepa_libero_ft.sh",
            "policy/vla-jepa/scripts/vlajepa_pretrain.sh",
            "policy/vla-jepa/examples/SimplerEnv/train_files/run_oxe_train.sh",
        )
        for path in paths:
            script = _read(path)
            self.assertIn("WANDB_MODE", script, path)
            self.assertIn("online", script, path)

    def test_vla_jepa_enables_wandb_tracker_online(self):
        for path in (
            "policy/vla-jepa/scripts/configs/vlajepa_flying_hand_ft.yaml",
            "policy/vla-jepa/scripts/configs/vlajepa_libero_ft.yaml",
            "policy/vla-jepa/scripts/configs/vlajepa_pretrain.yaml",
            "policy/vla-jepa/examples/SimplerEnv/train_files/vlajepa_ft.yaml",
        ):
            config = yaml.safe_load(_read(path))
            self.assertIn("wandb", config["trackers"], path)
            self.assertEqual(config["wandb_mode"], "online", path)

    def test_runtime_initializers_receive_the_selected_mode(self):
        self.assertIn("mode=config.wandb_mode", _read("policy/pi0/scripts/train.py"))
        self.assertIn("mode=config.wandb_mode", _read("policy/pi05/scripts/train.py"))
        self.assertIn("mode=config[\"wandb_mode\"]", _read("policy/ACT/imitate_episodes.py"))
        self.assertIn("**OmegaConf.to_container(cfg.logging", _read(
            "policy/DP/diffusion_policy/workspace/robotworkspace.py"
        ))
        self.assertIn("mode=os.environ.get(\"WANDB_MODE\"", _read(
            "policy/vla-jepa/starVLA/training/train_starvla.py"
        ))
        self.assertIn("mode=os.environ.get(\"WANDB_MODE\"", _read(
            "policy/vla-jepa/starVLA/training/train_jevla_cotrain.py"
        ))


if __name__ == "__main__":
    unittest.main()
