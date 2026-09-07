import numpy as np
import torch
import dill
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *
from envs.flying_hand.eval_control import execute_action_chunk, is_flying_hand_observation, reset_reference


# Encode observation for the model
def encode_obs(observation):
    if is_flying_hand_observation(observation):
        return {
            "images": {
                "head_camera": observation["observation"]["head_camera"]["rgb"],
                "wrist_camera": observation["observation"]["wrist_camera"]["rgb"],
            },
            "state": np.asarray(observation["flying_hand"]["actual_state"], dtype=np.float32),
        }
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    return PI0(train_config_name, model_name, checkpoint_id, pi0_step)


def eval(TASK_ENV, model, observation):

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    encoded = encode_obs(observation)
    if is_flying_hand_observation(observation):
        model.update_observation_window(encoded["images"], encoded["state"])
        actions = model.get_action()
        execute_action_chunk(TASK_ENV, model, actions)
        return TASK_ENV.get_obs()

    input_rgb_arr, input_state = encoded
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action()[:model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
    reset_reference(model)
