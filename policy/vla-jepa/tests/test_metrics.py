import json
import unittest
import numpy as np
import torch
from starVLA.training.trainer_utils.metrics import metric_json_default


class MetricsJSONTests(unittest.TestCase):
    def test_evaluation_metrics_remain_numeric(self):
        metrics = {"step": 22, "action_loss": 1.2, "eval": {"mse": np.float32(0.25)},
                   "count": np.int64(3), "valid": np.bool_(True),
                   "array": np.array([0.5, 0.75], dtype=np.float32),
                   "tensor": torch.tensor(0.5, requires_grad=True)}
        result = json.loads(json.dumps(metrics, default=metric_json_default))
        self.assertEqual(result["eval"]["mse"], 0.25)
        self.assertIsInstance(result["eval"]["mse"], float)
        self.assertEqual(result["array"], [0.5, 0.75])
        self.assertEqual(result["tensor"], 0.5)
        self.assertEqual(result["count"], 3)
        self.assertIs(result["valid"], True)

    def test_unsupported_metric_is_not_silently_hidden(self):
        with self.assertRaises(TypeError):
            json.dumps({"metric": object()}, default=metric_json_default)


if __name__ == "__main__":
    unittest.main()
