import numpy as np
from .dp_model import DP
from pathlib import Path
from envs.flying_hand.eval_control import (
    execute_action_chunk,
    is_flying_hand_observation,
    reset_execution_state,
)

def _encode_flying_hand_obs(observation):
    def image(name):
        value = observation["observation"][name]["rgb"]
        return np.moveaxis(np.asarray(value), -1, 0).astype(np.float32) / 255.0

    return {
        "head_cam": image("head_camera"),
        "wrist_cam": image("wrist_camera"),
        "agent_pos": np.asarray(observation["flying_hand"]["actual_state"], dtype=np.float32),
    }

def encode_obs(observation):
    if is_flying_hand_observation(observation):
        return _encode_flying_hand_obs(observation)
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):
    ckpt_override = usr_args.get("checkpoint_path")
    if ckpt_override is not None:
        ckpt_file = str(Path(ckpt_override).expanduser())
    else:
        if str(usr_args.get("task_name", "")).startswith("flying_hand"):
            raise ValueError(
                "Flying-Hand DP deployment requires `checkpoint_path` pointing to a .ckpt file."
            )
        ckpt_file = f"./policy/DP/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
    if not Path(ckpt_file).is_file():
        raise FileNotFoundError(f"Diffusion Policy checkpoint not found: {ckpt_file}")
    # Observation/action horizons are checkpoint architecture, not deployment
    # defaults. Reading them from the payload keeps older 3/6 checkpoints and
    # new OpenVLA/DROID-style 2/8 checkpoints independently deployable.
    return DP(ckpt_file, device=str(usr_args.get("device", "cuda:0")))


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)

    # ======== Get Action ========
    actions = model.get_action(obs)
    if is_flying_hand_observation(observation):
        # DPRunner has already selected exactly the checkpoint's
        # n_action_steps; the shared simulator executes that complete chunk.
        execute_action_chunk(TASK_ENV, model, actions)
        # The outer evaluator provides the next post-execution observation.
        # Fetching/updating here duplicated that frame in DP's obs deque and
        # also rendered one unnecessary RGB pair per replan window.
        return None

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        model.update_obs(encode_obs(observation))
    return observation

def reset_model(model):
    model.reset_obs()
    reset_execution_state(model)
