import os

import numpy as np
import sapien.core as sapien
import torch
import yaml
from PIL import ImageColor

from envs._GLOBAL_CONFIGS import CONFIGS_PATH


class Camera:
    def __init__(self, bias=0, **kwags):
        self.pcd_crop = kwags.get("pcd_crop", False)
        self.pcd_down_sample_num = kwags.get("pcd_down_sample_num", 0)
        self.pcd_crop_bbox = kwags.get("bbox", [[-0.6, -0.35, 0.7401], [0.6, 0.35, 2]])
        self.pcd_crop_bbox[0][2] += bias
        self.cfg = kwags["left_embodiment_config"]
        self.head_camera_type = kwags["camera"].get("head_camera_type", "D435")
        self.wrist_camera_type = kwags["camera"].get("wrist_camera_type", "D435")
        self.collect_head_camera = kwags["camera"].get("collect_head_camera", True)
        self.collect_wrist_camera = kwags["camera"].get("collect_wrist_camera", True)
        self.random_head_camera_dis = kwags.get("random_head_camera_dis", 0)
        self.wrist_camera_name = self.cfg["camera"]["wrist_name"]

    def load_camera(self, scene):
        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
            camera_args = yaml.load(f.read(), Loader=yaml.FullLoader)

        def pose(info):
            p = np.array(info["position"])
            if info["name"] == "head_camera" and self.random_head_camera_dis > 0:
                random_dir = np.random.randn(3)
                random_dir = random_dir / np.linalg.norm(random_dir)
                p = p + random_dir * np.random.uniform(0, self.random_head_camera_dis)
            f = np.array(info["forward"]) / np.linalg.norm(info["forward"])
            l = np.array(info["left"]) / np.linalg.norm(info["left"])
            mat = np.eye(4)
            mat[:3, :3] = np.stack([f, l, np.cross(f, l)], axis=1)
            mat[:3, 3] = p
            return sapien.Pose(mat)

        def add(name, camera_type):
            c = camera_args[camera_type]
            return scene.add_camera(
                name=name,
                width=c["w"],
                height=c["h"],
                fovy=np.deg2rad(c["fovy"]),
                near=0.1,
                far=100,
            )

        if self.collect_head_camera:
            info = [c for c in self.cfg["static_camera_list"] if c["name"] == "head_camera"][0]
            self.head_camera = add("head_camera", self.head_camera_type)
            self.head_camera.entity.set_pose(pose(info))

        if self.collect_wrist_camera:
            self.wrist_camera = add(self.wrist_camera_name, self.wrist_camera_type)

    def update_picture(self):
        if self.collect_head_camera:
            self.head_camera.take_picture()
        if self.collect_wrist_camera:
            self.wrist_camera.take_picture()

    def update_wrist_camera(self, pose):
        if self.collect_wrist_camera:
            self.wrist_camera.entity.set_pose(pose)

    def _cameras(self):
        res = []
        if self.collect_head_camera:
            res.append(("head_camera", self.head_camera))
        if self.collect_wrist_camera:
            res.append((self.wrist_camera_name, self.wrist_camera))
        return res

    def get_config(self):
        return {
            name: {
                "intrinsic_cv": camera.get_intrinsic_matrix(),
                "extrinsic_cv": camera.get_extrinsic_matrix(),
                "cam2world_gl": camera.get_model_matrix(),
            }
            for name, camera in self._cameras()
        }

    def get_rgba(self):
        return {
            name: {"rgba": (camera.get_picture("Color") * 255).clip(0, 255).astype("uint8")}
            for name, camera in self._cameras()
        }

    def get_rgb(self):
        return {
            name: {"rgb": data["rgba"][:, :, :3]}
            for name, data in self.get_rgba().items()
        }

    def get_segmentation(self, level="mesh"):
        palette = np.array([ImageColor.getrgb(c) for c in sorted(set(ImageColor.colormap.values()))], dtype=np.uint8)
        idx = 0 if level == "mesh" else 1
        return {
            name: {f"{level}_segmentation": palette[camera.get_picture("Segmentation")[..., idx].astype(np.uint8)]}
            for name, camera in self._cameras()
        }

    def get_depth(self):
        rgba = self.get_rgba()
        return {
            name: {
                "depth": (-camera.get_picture("Position")[..., 2] * 1000.0).astype(np.float64)
                * rgba[name]["rgba"][:, :, 3]
                / 255
            }
            for name, camera in self._cameras()
        }

    def get_pcd(self, if_combine=False):
        def pcd(camera):
            rgba = camera.get_picture_cuda("Color").torch()
            position = camera.get_picture_cuda("Position").torch()
            mat = torch.tensor(camera.get_model_matrix(), dtype=torch.float32, device=position.device)
            mask = position[..., 3] < 1
            points = torch.bmm(
                position[..., :3][mask].view(1, -1, 3),
                mat[:3, :3].transpose(0, 1).view(-1, 3, 3),
            ).squeeze(1) + mat[:3, 3]
            if self.pcd_crop:
                lo = torch.tensor(self.pcd_crop_bbox[0], dtype=torch.float32, device=position.device)
                hi = torch.tensor(self.pcd_crop_bbox[1], dtype=torch.float32, device=position.device)
                keep = ((points >= lo) & (points <= hi)).all(dim=1)
                points, mask_color = points[keep], rgba[mask][keep]
            else:
                mask_color = rgba[mask]
            return np.hstack((points.cpu().numpy(), torch.clamp(mask_color[:, :3], 0, 1).cpu().numpy()))

        points = np.vstack([pcd(camera) for _, camera in self._cameras()]) if if_combine else pcd(self.head_camera)
        if points.shape[0] < self.pcd_down_sample_num:
            points = np.vstack((points, np.zeros((self.pcd_down_sample_num - points.shape[0], 6))))
        if self.pcd_down_sample_num > 0:
            points = points[np.linspace(0, len(points) - 1, self.pcd_down_sample_num).astype(int)]
        return points
