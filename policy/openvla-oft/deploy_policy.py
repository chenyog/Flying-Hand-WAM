import numpy as np
from dataclasses import dataclass

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
from experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
)


@dataclass
class InferenceConfig:
    pretrained_checkpoint: str
    base_model_path: str | None = None
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    use_proprio: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    num_images_in_input: int = 2
    center_crop: bool = False
    unnorm_key: str = ""
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    lora_rank: int = 32


def encode_obs(obs: dict) -> dict:
    from envs.flying_hand.eval_control import is_flying_hand_observation

    if is_flying_hand_observation(obs):
        return {
            "full_image": obs["observation"]["head_camera"]["rgb"],
            "wrist_image": obs["observation"]["wrist_camera"]["rgb"],
            "state": np.asarray(obs["flying_hand"]["actual_state"], dtype=np.float32),
            "instruction": obs["language"],
        }
    return {
        "full_image": obs["observation"]["head_camera"]["rgb"],
        "left_wrist_image": obs["observation"]["left_camera"]["rgb"],
        "right_wrist_image": obs["observation"]["right_camera"]["rgb"],
        "state": obs["joint_action"]["vector"],
        "instruction": obs["language"],
    }


class Model:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.vla = get_vla(cfg)
        self.processor = get_processor(cfg)
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(
                cfg, self.vla.llm_dim, PROPRIO_DIM
            )
        from envs.flying_hand.eval_control import reset_execution_state

        reset_execution_state(self)

    def get_action(self, observation: dict):
        obs = encode_obs(observation)
        actions = get_vla_action(
            cfg=self.cfg,
            vla=self.vla,
            processor=self.processor,
            obs=obs,
            task_label=obs["instruction"],
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )
        return actions


def get_model(usr_args: dict):
    config_args = {
        "pretrained_checkpoint": usr_args["checkpoint_path"],
        "base_model_path": usr_args.get("base_model_path"),
        "use_l1_regression": usr_args.get("use_l1_regression", True),
        "use_diffusion": usr_args.get("use_diffusion", False),
        "use_film": usr_args.get("use_film", False),
        "use_proprio": usr_args.get("use_proprio", True),
        "load_in_8bit": usr_args.get("load_in_8bit", False),
        "load_in_4bit": usr_args.get("load_in_4bit", False),
        "num_images_in_input": usr_args.get("num_images_in_input", 2),
        "center_crop": usr_args.get("center_crop", False),
        "unnorm_key": usr_args["unnorm_key"],
        "num_open_loop_steps": usr_args.get("num_open_loop_steps", NUM_ACTIONS_CHUNK),
        "lora_rank": usr_args.get("lora_rank", 32),
    }

    cfg = InferenceConfig(**config_args)
    return Model(cfg)


def reset_model(model=None):
    from envs.flying_hand.eval_control import reset_execution_state

    reset_execution_state(model)


def eval(TASK_ENV, model: Model, observation: dict):
    observation["language"] = TASK_ENV.get_instruction()

    actions = model.get_action(observation)
    from envs.flying_hand.eval_control import execute_action_chunk, is_flying_hand_observation

    if is_flying_hand_observation(observation):
        execute_action_chunk(TASK_ENV, model, actions[: model.cfg.num_open_loop_steps])
        return
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
