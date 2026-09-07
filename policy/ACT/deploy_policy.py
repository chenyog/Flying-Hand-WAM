import numpy as np
import torch
import cv2
from .act_policy import ACT
from argparse import Namespace
from envs.flying_hand.eval_control import (
    execute_action_chunk,
    is_flying_hand_observation,
    reset_reference,
)


def _is_flying_hand_observation(observation):
    return is_flying_hand_observation(observation)

def encode_obs(observation):
    if _is_flying_hand_observation(observation):
        def image(name):
            value = observation["observation"][name]["rgb"]
            value = cv2.resize(value, (640, 480), interpolation=cv2.INTER_LINEAR)
            return np.moveaxis(value, -1, 0).astype(np.float32) / 255.0

        return {
            "head_cam": image("head_camera"),
            "wrist_cam": image("wrist_camera"),
            "qpos": np.asarray(observation["flying_hand"]["actual_state"], dtype=np.float32),
        }
    head_cam = cv2.resize(observation["observation"]["head_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    left_cam = cv2.resize(observation["observation"]["left_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    right_cam = cv2.resize(observation["observation"]["right_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    head_cam = np.moveaxis(head_cam, -1, 0) / 255.0
    left_cam = np.moveaxis(left_cam, -1, 0) / 255.0
    right_cam = np.moveaxis(right_cam, -1, 0) / 255.0
    qpos = (observation["joint_action"]["left_arm"] + [observation["joint_action"]["left_gripper"]] +
            observation["joint_action"]["right_arm"] + [observation["joint_action"]["right_gripper"]])
    return {
        "head_cam": head_cam,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "qpos": qpos,
    }

def get_model(usr_args):
    config = dict(usr_args)
    if str(config.get("task_name", "")).startswith("flying_hand"):
        # ACT checkpoints trained from Flying-Hand data have a 5D state/action
        # and two cameras, unlike the default bimanual arm checkpoint.
        config["action_dim"] = 5
        config["state_dim"] = 5
        config["camera_names"] = ["cam_high", "cam_wrist"]
    return ACT(config, Namespace(**config))


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    # instruction = TASK_ENV.get_instruction()

    # Get action from model
    actions = model.get_action(obs)
    if _is_flying_hand_observation(observation):
        execute_action_chunk(TASK_ENV, model, actions)
        return TASK_ENV.get_obs()
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
    return observation


def reset_model(model):
    # Reset temporal aggregation state if enabled
    if model.temporal_agg:
        model.all_time_actions = torch.zeros([
            model.max_timesteps,
            model.max_timesteps + model.num_queries,
            model.state_dim,
        ]).to(model.device)
        model.t = 0
        print("Reset temporal aggregation state")
    else:
        model.t = 0
    reset_reference(model)
    model.grasp_commanded = False
    model.attached_actor = None
    model.attached_pose = None
