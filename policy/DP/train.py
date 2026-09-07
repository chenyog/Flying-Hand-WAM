"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra, pdb
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
import pathlib, yaml
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def init_distributed(cfg):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    cfg.training.device = f"cuda:{local_rank}"
    return True


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf):
    distributed = init_distributed(cfg)
    try:
        # Resolve immediately so all ${now:} resolvers use the same time.
        head_camera_type = cfg.head_camera_type
        head_camera_cfg = get_camera_config(head_camera_type)
        wrist_camera_type = getattr(cfg, "wrist_camera_type", None)
        wrist_camera_cfg = get_camera_config(wrist_camera_type) if wrist_camera_type else None
        cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
        cfg.task.shape_meta.obs.head_cam.shape = [
            3,
            head_camera_cfg["h"],
            head_camera_cfg["w"],
        ]
        if wrist_camera_cfg is not None and "wrist_cam" in cfg.task.shape_meta.obs:
            cfg.task.shape_meta.obs.wrist_cam.shape = [
                3,
                wrist_camera_cfg["h"],
                wrist_camera_cfg["w"],
            ]
        OmegaConf.resolve(cfg)
        cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
        cfg.task.shape_meta.obs.head_cam.shape = [
            3,
            head_camera_cfg["h"],
            head_camera_cfg["w"],
        ]
        if wrist_camera_cfg is not None and "wrist_cam" in cfg.task.shape_meta.obs:
            cfg.task.shape_meta.obs.wrist_cam.shape = [
                3,
                wrist_camera_cfg["h"],
                wrist_camera_cfg["w"],
            ]

        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        if not distributed or dist.get_rank() == 0:
            print(cfg.task.dataset.zarr_path, cfg.task_name)
        workspace.run()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
