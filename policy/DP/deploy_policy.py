import numpy as np
from .dp_model import DP
import yaml
from pathlib import Path
from envs.flying_hand.eval_control import (
    execute_action_chunk as _execute_action_chunk,
    finite_difference_actions as _shared_finite_difference_actions,
    is_flying_hand_observation,
    reset_reference as _reset_reference,
)


def _is_flying_hand_observation(observation):
    return is_flying_hand_observation(observation)


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
    if _is_flying_hand_observation(observation):
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
    action_dim = 5 if str(usr_args.get("task_name", "")).startswith("flying_hand") else usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2
    
    load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']
    
    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
              device=str(usr_args.get("device", "cuda:0")))


def _finite_difference_flying_hand_actions(task_env, actions):
    """Compatibility alias for the shared waypoint derivative helper."""
    return _shared_finite_difference_actions(task_env, actions)


def _reset_flying_hand_reference(model):
    """Compatibility alias for the shared executor reset."""
    _reset_reference(model)


def _execute_flying_hand_action(
    task_env,
    model,
    action,
    target_velocity=None,
    target_acceleration=None,
):
    # Keep the historical helper name importable while routing execution
    # through the single shared Flying-Hand control interface.
    return _execute_action_chunk(task_env, model, np.asarray(action)[None, :])

def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)

    # ======== Get Action ========
    actions = model.get_action(obs)
    if _is_flying_hand_observation(observation):
        _execute_action_chunk(TASK_ENV, model, actions)
        observation = TASK_ENV.get_obs()
        model.update_obs(encode_obs(observation))
        return observation

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        model.update_obs(encode_obs(observation))
    return observation

def reset_model(model):
    model.reset_obs()
    model.grasp_commanded = False
    model.attached_actor = None
    model.attached_pose = None
    _reset_reference(model)
