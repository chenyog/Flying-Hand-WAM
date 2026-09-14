import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT / "policy/FastWAM/scripts/convert_flying_hand_to_lerobot.py"
)


def load_converter():
    spec = importlib.util.spec_from_file_location("fastwam_flying_hand_converter", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FastWAMFlyingHandConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.converter = load_converter()

    def test_episode_task_uses_saved_instruction_without_template_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instructions = root / "instructions"
            instructions.mkdir()
            (instructions / "episode7.json").write_text(
                json.dumps({"seen": ["exact per-episode instruction"], "unseen": ["other"]}),
                encoding="utf-8",
            )
            self.assertEqual(
                self.converter.episode_task(root, 7, "seen"),
                "exact per-episode instruction",
            )

    def test_episode_task_fails_if_requested_domain_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instructions = root / "instructions"
            instructions.mkdir()
            (instructions / "episode0.json").write_text(
                json.dumps({"seen": ["seen only"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unseen"):
                self.converter.episode_task(root, 0, "unseen")


if __name__ == "__main__":
    unittest.main()
