"""Small CPU contract tests, independent of the OpenVLA model dependencies."""

import ast
from dataclasses import dataclass
import importlib.util
import itertools
import os
import random
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from check_flying_hand_data import load_reader


ROOT = Path(__file__).resolve().parents[1]
reader = load_reader()


def load_file(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FlyingHandTests(unittest.TestCase):
    def test_terminal_chunk_holds_absolute_target(self):
        np.testing.assert_array_equal(reader.chunk_indices(3, 5, 8), [3, 4, 4, 4, 4, 4, 4, 4])
        np.testing.assert_array_equal(reader.chunk_indices(0, 1, 8), np.zeros(8))

    def test_normalization_and_constant_dimension(self):
        stats = {"q01": [0, 4], "q99": [10, 4], "min": [0, 4], "max": [10, 4]}
        actual = reader.normalize(np.array([[0, 4], [5, 4], [20, 4]]), stats)
        np.testing.assert_allclose(actual, [[-1, 0], [0, 0], [1, 0]], atol=1e-7)

    def test_explicit_platform_precedes_checkpoint_name(self):
        with patch.dict(os.environ, {"OPENVLA_ROBOT_PLATFORM": "FLYING_HAND"}), patch.object(sys, "argv", ["libero-model"]):
            constants = load_file("test_flying_hand_constants", "prismatic/vla/constants.py")
        self.assertEqual((constants.ACTION_DIM, constants.PROPRIO_DIM, constants.NUM_ACTIONS_CHUNK), (5, 5, 8))

    def test_collator_keeps_single_example_proprio_batch(self):
        module = load_file("test_flying_hand_collator", "prismatic/util/data_utils.py")
        collator = module.PaddedCollatorForActionPrediction(128, 0)
        item = dict(input_ids=torch.tensor([1, 2]), labels=torch.tensor([-100, 2]),
                    pixel_values=torch.zeros(6, 4, 4), pixel_values_wrist=torch.ones(6, 4, 4),
                    actions=np.zeros((8, 5), np.float32), proprio=np.zeros((1, 5), np.float32))
        for count in (1, 2):
            batch = collator([item] * count)
            self.assertEqual(tuple(batch["proprio"].shape), (count, 5))
            self.assertEqual(tuple(batch["actions"].shape), (count, 8, 5))
            self.assertEqual(tuple(batch["pixel_values"].shape), (count, 12, 4, 4))

    def test_rank_partition_and_infinite_epochs(self):
        class Source:
            episodes = list(range(8))

            def read_episode(self, episode):
                return {"action": [episode]}

            def sample(self, data, *args):
                return data["action"][0]

        def dataset(rank):
            result = object.__new__(reader.FlyingHandDataset)
            result.source = Source()
            result.rank, result.world_size, result.seed, result.chunk_size = rank, 2, 0, 8
            result.batch_transform = lambda sample: sample
            result.dataset_statistics = {reader.DATASET_NAME: {}}
            return result

        ranks = [list(itertools.islice(iter(dataset(rank)), 8)) for rank in range(2)]
        for start in (0, 4):
            left, right = set(ranks[0][start:start + 4]), set(ranks[1][start:start + 4])
            self.assertFalse(left & right)
            self.assertEqual(left | right, set(range(8)))

    def test_upstream_action_head_and_proprio_projector(self):
        # Load the real CPU modules without importing the 7B backbone/package initializer.
        with patch.dict(os.environ, {"OPENVLA_ROBOT_PLATFORM": "FLYING_HAND"}):
            constants = load_file("test_flying_hand_head_constants", "prismatic/vla/constants.py")
        with patch.dict(sys.modules, {"prismatic.vla.constants": constants}):
            heads = load_file("test_flying_hand_heads", "prismatic/models/action_heads.py")
        projectors = load_file("test_flying_hand_projectors", "prismatic/models/projectors.py")
        head = heads.L1RegressionActionHead(input_dim=16, hidden_dim=32, action_dim=5)
        projector = projectors.ProprioProjector(llm_dim=16, proprio_dim=5)
        state = torch.randn(2, 5, requires_grad=True)
        features = projector(state).unsqueeze(1).expand(-1, 40, -1)
        predicted = head.predict_action(features)
        self.assertEqual(tuple(predicted.shape), (2, 8, 5))
        predicted.abs().mean().backward()
        self.assertTrue(torch.isfinite(state.grad).all())

    def test_real_batch_transform_masks_40_tokens_not_text_characters(self):
        # Execute the production class without loading unrelated TensorFlow/backbone imports.
        path = ROOT / "prismatic/vla/datasets/datasets.py"
        tree = ast.parse(path.read_text())
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RLDSBatchTransform")
        module = ast.parse("from __future__ import annotations")
        module.body.append(node)
        namespace = dict(__name__=__name__, dataclass=dataclass, torch=torch, np=np,
                         Image=Image, random=random, IGNORE_INDEX=-100)
        exec(compile(module, str(path), "exec"), namespace)
        prompts = load_file("test_flying_hand_prompts", "prismatic/models/backbones/llm/prompting/base_prompter.py")

        def action_tokenizer(actions):
            # Each discrete token decodes to several characters, as can happen with BPE.
            if actions.ndim == 1:
                return "ACTION" * len(actions)
            return ["ACTION" * row.size for row in actions]

        transform = namespace["RLDSBatchTransform"](
            action_tokenizer=action_tokenizer,
            base_tokenizer=lambda prompt, **kwargs: SimpleNamespace(input_ids=[1, 100, 200] + [31900] * 40 + [2]),
            image_transform=lambda image: torch.zeros(6, 4, 4),
            prompt_builder_fn=prompts.PurePromptBuilder, use_wrist_image=True, use_proprio=True,
        )
        batch = transform({
            "dataset_name": reader.DATASET_NAME, "action": np.zeros((8, 5), dtype=np.float32),
            "task": {"language_instruction": "Move the bottle"},
            "observation": {"image_primary": np.zeros((1, 4, 4, 3), np.uint8),
                            "image_wrist": np.zeros((1, 4, 4, 3), np.uint8),
                            "proprio": np.zeros((1, 5), np.float32)},
        })
        self.assertEqual(batch["labels"][:3].tolist(), [-100] * 3)
        self.assertEqual(int((batch["labels"] > 31743).sum()), 40)
        self.assertEqual(tuple(batch["pixel_values_wrist"].shape), (6, 4, 4))


if __name__ == "__main__":
    unittest.main()
