"""Data transforms for the five-dimensional RoboTwin Flying-Hand action space."""

import dataclasses

import einops
import numpy as np

from openpi import transforms


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        # LeRobot normally returns float images in [0, 1], but tolerate already
        # byte-scaled float arrays as well.
        if image.size and np.nanmax(image) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class FlyingHandInputs(transforms.DataTransformFn):
    """Map Flying-Hand state/actions and two cameras to the OpenPI format."""

    # Pretrained pi0 checkpoints use a 32-dimensional internal action space.
    # Flying-Hand data occupies only the first five dimensions.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        head = _parse_image(data["images"]["head_camera"])
        wrist = _parse_image(data["images"]["wrist_camera"])
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != 5:
            raise ValueError(f"Flying-Hand state must have 5 values, got {state.shape}")

        inputs = {
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(head),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "state": transforms.pad_to_dim(state, self.action_dim),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 5:
                raise ValueError(f"Flying-Hand actions must have 5 values, got {actions.shape}")
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class FlyingHandOutputs(transforms.DataTransformFn):
    """Keep only the five dimensions used by Flying-Hand."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        return {"actions": actions[..., :5]}
