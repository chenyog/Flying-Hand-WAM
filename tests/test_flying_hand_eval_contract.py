import unittest
import tempfile
from pathlib import Path
from unittest import mock

import yaml

from script.eval_policy import (
    apply_task_overrides,
    flying_hand_simulation_contract,
    rerender_flying_hand_eval_histories,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FlyingHandEvalContractTest(unittest.TestCase):
    def test_all_task_configs_share_the_20hz_eval_contract(self):
        names = (
            "flying_hand_clean",
            "flying_hand_randomized",
            "flying_hand_direct_clean",
            "flying_hand_direct_randomized",
        )
        configs = [
            yaml.safe_load((PROJECT_ROOT / "task_config" / f"{name}.yml").read_text())
            for name in names
        ]
        for config in configs:
            self.assertAlmostEqual(1.0 / (config["timestep"] * config["save_freq"]), 20.0)
            self.assertEqual(config["render_shader"], "rt")
            self.assertEqual(config["texture_type"], "seen")
            self.assertTrue(config["eval_video_log"])
            self.assertNotIn("eval_video_mode", config)
            self.assertNotIn("eval_history_episode_limit", config)
            self.assertNotIn("replan_steps", config)

        # Direct and dynamics variants intentionally differ only in the
        # execution model.  Policy choice must not alter cameras, rendering,
        # timing, randomization, or physical controller parameters.
        for dynamics, direct in ((configs[0], configs[2]), (configs[1], configs[3])):
            dynamics = dict(dynamics)
            direct = dict(direct)
            self.assertTrue(dynamics.pop("enable_dynamics"))
            self.assertFalse(direct.pop("enable_dynamics"))
            self.assertEqual(dynamics, direct)

    def test_flying_hand_simulation_fields_cannot_be_overridden_by_policy(self):
        task = {
            "embodiment": ["flying-hand"],
            "timestep": 0.005,
            "save_freq": 10,
            "render_shader": "rt",
            "enable_dynamics": False,
            "eval_video_log": True,
        }
        with self.assertRaisesRegex(ValueError, "owned by the selected task YAML"):
            apply_task_overrides(task, {"render_shader": "default"})
        with self.assertRaisesRegex(ValueError, "owned by the selected task YAML"):
            apply_task_overrides(task, {"enable_dynamics": True})
        apply_task_overrides(task, {"eval_video_log": False})
        self.assertFalse(task["eval_video_log"])

    def test_simulation_contract_hash_excludes_policy_and_detects_env_changes(self):
        config = yaml.safe_load(
            (PROJECT_ROOT / "task_config/flying_hand_direct_clean.yml").read_text()
        )
        config["task_config"] = "flying_hand_direct_clean"
        first = flying_hand_simulation_contract(config, "seen")
        second = flying_hand_simulation_contract(dict(config), "seen")
        self.assertEqual(first["sha256"], second["sha256"])
        changed = dict(config)
        changed["timestep"] = 0.01
        self.assertNotEqual(
            first["sha256"],
            flying_hand_simulation_contract(changed, "seen")["sha256"],
        )

    def test_all_policy_deploy_configs_use_seen_instructions(self):
        paths = (
            "policy/pi0/deploy_policy.yml",
            "policy/pi05/deploy_policy.yml",
            "policy/DP/deploy_policy.yml",
            "policy/ACT/deploy_policy.yml",
        )
        for relative_path in paths:
            config = yaml.safe_load((PROJECT_ROOT / relative_path).read_text())
            self.assertEqual(config["instruction_type"], "seen", relative_path)
        fastwam = yaml.safe_load(
            (PROJECT_ROOT / "policy/FastWAM/configs/sim_flying_hand.yaml").read_text()
        )
        self.assertEqual(fastwam["EVALUATION"]["instruction_type"], "seen")
        self.assertEqual(fastwam["EVALUATION"]["replan_steps"], 24)

    def test_eval_configs_do_not_expose_noop_parameters(self):
        dp = yaml.safe_load(
            (PROJECT_ROOT / "policy/DP/deploy_policy.yml").read_text()
        )
        self.assertNotIn("head_camera_type", dp)

        act = yaml.safe_load(
            (PROJECT_ROOT / "policy/ACT/deploy_policy.yml").read_text()
        )
        for key in (
            "policy_class",
            "num_epochs",
            "kl_weight",
            "lr",
            "lr_backbone",
            "weight_decay",
        ):
            self.assertNotIn(key, act)

        for name in ("sim_flying_hand.yaml", "sim_robotwin.yaml"):
            fastwam = yaml.safe_load(
                (PROJECT_ROOT / "policy/FastWAM/configs" / name).read_text()
            )
            self.assertNotIn("skip_get_obs_within_replan", fastwam["EVALUATION"])
            self.assertNotIn("enabled", fastwam["MULTIRUN"])

    def test_eval_histories_are_rendered_in_one_full_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay_dir = root / "replay"
            replay_dir.mkdir()
            (replay_dir / "episode0_history.npz").touch()
            (replay_dir / "episode1_history.npz").touch()
            video_dir = root / "video"
            with mock.patch("script.eval_policy.subprocess.run") as run:
                count = rerender_flying_hand_eval_histories(replay_dir, video_dir)
            self.assertEqual(count, 2)
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertEqual(command[-2:], ["--output-dir", str(video_dir)])
            self.assertNotIn("--max-seconds", command)
            self.assertTrue(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
