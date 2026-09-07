import numpy as np

from openpi.policies import flying_hand_policy


def test_flying_hand_inputs_pad_state_and_actions():
    transform = flying_hand_policy.FlyingHandInputs(action_dim=32)
    data = {
        "images": {
            "head_camera": np.ones((3, 8, 12), dtype=np.float32),
            "wrist_camera": np.full((8, 12, 3), 128.0, dtype=np.float32),
        },
        "state": np.arange(5, dtype=np.float32),
        "actions": np.ones((4, 5), dtype=np.float32),
        "prompt": "move the flying hand",
    }

    result = transform(data)

    assert result["state"].shape == (32,)
    assert result["actions"].shape == (4, 32)
    np.testing.assert_array_equal(result["state"][:5], data["state"])
    np.testing.assert_array_equal(result["state"][5:], 0)
    assert result["image"]["base_0_rgb"].shape == (8, 12, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert result["image"]["base_0_rgb"].max() == 255
    assert result["image"]["left_wrist_0_rgb"].max() == 128
    assert not result["image_mask"]["right_wrist_0_rgb"]


def test_flying_hand_outputs_keep_five_dimensions():
    actions = np.arange(3 * 32, dtype=np.float32).reshape(3, 32)
    result = flying_hand_policy.FlyingHandOutputs()({"actions": actions})
    np.testing.assert_array_equal(result["actions"], actions[:, :5])
