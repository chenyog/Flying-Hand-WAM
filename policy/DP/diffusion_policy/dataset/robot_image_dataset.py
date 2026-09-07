from typing import Dict
try:
    import numba
except ImportError:  # Keep dataset loading usable in minimal inference environments.
    numba = None
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import pdb


class RobotImageDataset(BaseImageDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
        image_keys=None,
        image_obs_keys=None,
        state_key="state",
        action_key="action",
    ):

        super().__init__()
        self.image_keys = list(image_keys or ["head_camera"])
        self.image_obs_keys = list(image_obs_keys or ["head_cam"])
        if len(self.image_keys) != len(self.image_obs_keys):
            raise ValueError("image_keys and image_obs_keys must have the same length")
        self.state_key = state_key
        self.action_key = action_key
        # Keep the large image arrays on disk.  ``copy_from_path`` expands the
        # complete Zarr into RAM, which is unsafe when every DDP rank loads its
        # own copy.  SequenceSampler and batch_sample_sequence already support
        # Zarr-backed arrays and only read the requested sequences.
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer[self.action_key],
            "agent_pos": self.replay_buffer[self.state_key],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for key in self.image_obs_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        data = {"obs": {"agent_pos": sample[self.state_key].astype(np.float32)}}
        for image_key, obs_key in zip(self.image_keys, self.image_obs_keys):
            image = sample[image_key]
            # DP Zarr data is normally NCHW; accept legacy NHWC stores too.
            if image.ndim == 4 and image.shape[-1] == 3:
                image = np.moveaxis(image, -1, 1)
            data["obs"][obs_key] = image / 255
        data["action"] = sample[self.action_key].astype(np.float32)
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        obs = {"agent_pos": samples[self.state_key].to(device, non_blocking=True)}
        for image_key, obs_key in zip(self.image_keys, self.image_obs_keys):
            obs[obs_key] = samples[image_key].to(device, non_blocking=True) / 255.0
        return {"obs": obs, "action": samples[self.action_key].to(device, non_blocking=True)}


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    iterator = numba.prange(len(idx)) if numba is not None else range(len(idx))
    for i in iterator:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


if numba is not None:
    _batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
    _batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)
else:
    _batch_sample_sequence_sequential = _batch_sample_sequence
    _batch_sample_sequence_parallel = _batch_sample_sequence


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
