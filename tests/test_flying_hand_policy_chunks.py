import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from policy.ACT.act_policy import ACTPolicy, temporal_aggregate_actions
from policy.ACT.deploy_policy import (
    _checkpoint_chunk_size,
    _is_flying_hand_deployment,
    get_model,
)


class FlyingHandPolicyChunkTest(unittest.TestCase):
    def test_act_deployment_builds_no_optimizer(self):
        model = torch.nn.Identity()
        with (
            mock.patch(
                "policy.ACT.act_policy.build_ACT_model_for_inference",
                return_value=model,
            ) as build_inference,
            mock.patch(
                "policy.ACT.act_policy.build_ACT_model_and_optimizer",
            ) as build_training,
        ):
            policy = ACTPolicy({}, object())

        self.assertIs(policy.model, model)
        self.assertIsNone(policy.optimizer)
        self.assertEqual(policy.kl_weight, 0.0)
        build_inference.assert_called_once()
        build_training.assert_not_called()

    def test_act_uses_explicit_robot_type_instead_of_task_name(self):
        self.assertTrue(_is_flying_hand_deployment({"robot_type": "flying_hand"}))
        self.assertTrue(_is_flying_hand_deployment({"robot_type": "flying-hand"}))
        self.assertFalse(_is_flying_hand_deployment({"task_name": "flying_hand_fake"}))
        self.assertFalse(_is_flying_hand_deployment({"task_name": "move_bottle"}))

    def test_act_flying_hand_model_contract_comes_from_explicit_robot_type(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.save(
                {"model.query_embed.weight": torch.zeros((32, 512))},
                Path(directory) / "policy_last.ckpt",
            )
            config = {
                "task_name": "move_bottle",
                "robot_type": "flying_hand",
                "ckpt_dir": directory,
                "action_dim": 14,
                "state_dim": 14,
                "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
                "chunk_size": 100,
            }
            with mock.patch("policy.ACT.deploy_policy.ACT") as act:
                get_model(config)
            effective_config = act.call_args.args[0]
            self.assertEqual(effective_config["action_dim"], 5)
            self.assertEqual(effective_config["state_dim"], 5)
            self.assertEqual(effective_config["camera_names"], ["cam_high", "cam_wrist"])
            self.assertEqual(effective_config["chunk_size"], 32)

    def test_act_flying_hand_requires_checkpoint_directory(self):
        with self.assertRaisesRegex(ValueError, "requires ckpt_dir"):
            _checkpoint_chunk_size(None)

    def test_act_chunk_size_comes_from_checkpoint_query_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy_last.ckpt"
            torch.save(
                {
                    "model.query_embed.weight": torch.zeros((32, 512)),
                    "model.input_proj.weight": torch.zeros((512, 5)),
                },
                checkpoint,
            )
            self.assertEqual(_checkpoint_chunk_size(directory), 32)

    def test_act_checkpoint_requires_unambiguous_query_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy_last.ckpt"
            torch.save({"other.weight": torch.zeros((32, 512))}, checkpoint)
            with self.assertRaisesRegex(ValueError, "Could not infer one ACT chunk_size"):
                _checkpoint_chunk_size(directory)

    def test_act_temporal_aggregation_keeps_zero_coordinates(self):
        actions = torch.zeros((3, 5, 5), dtype=torch.float32)
        valid = torch.zeros((3, 5), dtype=torch.bool)
        actions[0, 2] = torch.tensor([1.0, 0.0, 3.0, 0.0, 1.0])
        actions[1, 2] = torch.tensor([2.0, 0.0, 4.0, 0.0, 0.0])
        valid[0, 2] = True
        valid[1, 2] = True

        result = temporal_aggregate_actions(actions, valid, timestep=2)

        self.assertEqual(tuple(result.shape), (1, 5))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(float(result[0, 1]), 0.0)
        self.assertEqual(float(result[0, 3]), 0.0)

    def test_act_temporal_aggregation_requires_a_valid_prediction(self):
        actions = torch.zeros((2, 3, 5), dtype=torch.float32)
        valid = torch.zeros((2, 3), dtype=torch.bool)
        with self.assertRaisesRegex(RuntimeError, "No valid ACT prediction"):
            temporal_aggregate_actions(actions, valid, timestep=1)


if __name__ == "__main__":
    unittest.main()
