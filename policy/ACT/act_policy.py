import torch.nn as nn
import os
import torch
import numpy as np
import pickle
from torch.nn import functional as F
import torchvision.transforms as transforms

try:
    from detr.main import (
        build_ACT_model_for_inference,
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
except:
    from .detr.main import (
        build_ACT_model_for_inference,
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
import IPython

e = IPython.embed


class ACTPolicy(nn.Module):

    def __init__(self, args_override, RoboTwin_Config=None):
        super().__init__()
        if RoboTwin_Config is None:
            model, optimizer = build_ACT_model_and_optimizer(args_override)
        else:
            # Deployment never updates weights. Building an AdamW optimizer here
            # made lr/weight_decay look like evaluation controls even though they
            # cannot affect inference, and allocated optimizer-only Python state.
            model = build_ACT_model_for_inference(RoboTwin_Config)
            optimizer = None
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        # kl_weight is a training loss coefficient. Keep it internal for the
        # training call path instead of exposing it in deploy_policy.yml.
        self.kl_weight = float(args_override.get("kl_weight", 0.0))
        if RoboTwin_Config is None:
            print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def temporal_aggregate_actions(all_time_actions, valid_mask, timestep, k=0.01):
    """Aggregate predictions using explicit occupancy, not action values.

    Zero is a valid Flying-Hand coordinate (notably yaw and gripper), so a
    prediction containing zero must not be mistaken for an unused slot.
    """
    actions = all_time_actions[:, timestep]
    actions = actions[valid_mask[:, timestep]]
    if len(actions) == 0:
        raise RuntimeError(f"No valid ACT prediction for timestep {timestep}")
    weights = torch.exp(
        -float(k) * torch.arange(len(actions), device=actions.device, dtype=actions.dtype)
    )
    weights = weights / weights.sum()
    return (actions * weights.unsqueeze(1)).sum(dim=0, keepdim=True)


class ACT:

    def __init__(self, args_override=None, RoboTwin_Config=None):
        if args_override is None:
            args_override = {
                "kl_weight": 0.1,  # Default value, can be overridden
                "device": "cuda:0",
            }
        self.policy = ACTPolicy(args_override, RoboTwin_Config)
        self.device = torch.device(args_override["device"])
        self.policy.to(self.device)
        self.policy.eval()

        # Temporal aggregation settings
        self.temporal_agg = args_override.get("temporal_agg", False)
        self.num_queries = args_override["chunk_size"]
        self.state_dim = int(getattr(RoboTwin_Config, "action_dim", args_override.get("action_dim", 14)))
        self.camera_names = list(
            getattr(
                RoboTwin_Config,
                "camera_names",
                args_override.get("camera_names", ["cam_high", "cam_right_wrist", "cam_left_wrist"]),
            )
        )
        self.max_timesteps = 3000  # Large enough for deployment

        # Set query frequency based on temporal_agg - matching imitate_episodes.py logic
        self.query_frequency = self.num_queries
        if self.temporal_agg:
            self.query_frequency = 1
            # Initialize with zeros matching imitate_episodes.py format
            self.all_time_actions = torch.zeros([
                self.max_timesteps,
                self.max_timesteps + self.num_queries,
                self.state_dim,
            ]).to(self.device)
            self.all_time_action_valid = torch.zeros(
                [self.max_timesteps, self.max_timesteps + self.num_queries],
                dtype=torch.bool,
                device=self.device,
            )
            print(f"Temporal aggregation enabled with {self.num_queries} queries")

        self.t = 0  # Current timestep

        # Load statistics for normalization
        ckpt_dir = args_override.get("ckpt_dir", "")
        flying_hand_checkpoint = int(args_override.get("action_dim", 14)) == 5
        if ckpt_dir:
            # Load dataset stats for normalization
            stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, "rb") as f:
                    self.stats = pickle.load(f)
                print(f"Loaded normalization stats from {stats_path}")
            else:
                if flying_hand_checkpoint:
                    raise FileNotFoundError(
                        f"Flying-Hand ACT requires checkpoint normalization stats: {stats_path}"
                    )
                print(f"Warning: Could not find stats file at {stats_path}")
                self.stats = None

            # Load policy weights
            ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
            print("current pwd:", os.getcwd())
            if os.path.exists(ckpt_path):
                # Load through CPU so a CUDA-authored checkpoint does not
                # create an extra full-sized temporary allocation on GPU.
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                loading_status = self.policy.load_state_dict(state_dict)
                print(f"Loaded policy weights from {ckpt_path}")
                print(f"Loading status: {loading_status}")
            else:
                if flying_hand_checkpoint:
                    raise FileNotFoundError(f"Flying-Hand ACT checkpoint not found: {ckpt_path}")
                print(f"Warning: Could not find policy checkpoint at {ckpt_path}")
        else:
            self.stats = None

    def pre_process(self, qpos):
        """Normalize input joint positions"""
        if self.stats is not None:
            return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return qpos

    def post_process(self, action):
        """Denormalize model outputs"""
        if self.stats is not None:
            return action * self.stats["action_std"] + self.stats["action_mean"]
        return action

    def get_action(self, obs=None):
        if obs is None:
            return None

        # Convert observations to tensors and normalize qpos - matching imitate_episodes.py
        qpos_numpy = np.array(obs["qpos"])
        qpos_normalized = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)

        # Prepare images following imitate_episodes.py pattern
        # Stack images from all cameras
        curr_images = []
        camera_key = {
            "cam_high": "head_cam",
            "cam_wrist": "wrist_cam",
            "cam_left_wrist": "left_cam",
            "cam_right_wrist": "right_cam",
        }
        for cam_name in self.camera_names:
            key = camera_key.get(cam_name, cam_name)
            if key not in obs:
                raise KeyError(f"Missing camera `{key}` required by ACT: {self.camera_names}")
            curr_images.append(obs[key])
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            # Only query the policy at specified intervals - exactly like imitate_episodes.py
            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)

            if self.temporal_agg:
                # Match temporal aggregation exactly from imitate_episodes.py
                self.all_time_actions[[self.t], self.t:self.t + self.num_queries] = (self.all_actions)
                self.all_time_action_valid[self.t, self.t:self.t + self.num_queries] = True
                raw_action = temporal_aggregate_actions(
                    self.all_time_actions,
                    self.all_time_action_valid,
                    self.t,
                )
            else:
                # Direct action selection, same as imitate_episodes.py
                raw_action = self.all_actions[:, self.t % self.query_frequency]

        # Denormalize action
        raw_action = raw_action.cpu().numpy()
        action = self.post_process(raw_action)

        self.t += 1
        return action
