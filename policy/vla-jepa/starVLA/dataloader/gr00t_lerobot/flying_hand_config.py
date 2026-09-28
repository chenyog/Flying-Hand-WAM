"""Flying-Hand's absolute 5D command and two-camera contract at 20 Hz."""


class FlyingHandDataConfig:
    video_keys = ["video.head_camera", "video.wrist_camera"]
    state_keys = [f"state.{key}" for key in ("x", "y", "z", "yaw", "gripper")]
    action_keys = [f"action.{key}" for key in ("x", "y", "z", "yaw", "gripper")]
    language_keys = ["annotation.human.action.task_description"]
    image_keys = ["observation.images.head_camera", "observation.images.wrist_camera"]
    vector_names = ["x", "y", "z", "yaw", "grasp"]

    def __init__(self, observation_indices, action_indices):
        self.observation_indices = observation_indices
        self.action_indices = action_indices

    def modality_config(self):
        from .datasets import ModalityConfig

        return {
            name: ModalityConfig(delta_indices=indices, modality_keys=keys)
            for name, indices, keys in (
                ("video", self.observation_indices, self.video_keys),
                ("state", [0], self.state_keys),
                ("action", self.action_indices, self.action_keys),
                ("language", [0], self.language_keys),
            )
        }

    def transform(self):
        from .transform.base import ComposedModalityTransform
        from .transform.state_action import StateActionToTensor, StateActionTransform

        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys[:-1]},
            ),
        ])
