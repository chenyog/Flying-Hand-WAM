"""JSON encoding for numeric training and evaluation metrics."""
import numpy as np
import torch


def metric_json_default(value):
    """Preserve numeric values without silently stringifying unsupported objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
