from copy import deepcopy
from pathlib import Path
import glob
import os
import pickle
import shutil

import gymnasium as gym
import numpy as np
import sapien
import transforms3d as t3d
import yaml
from sapien.render import clear_cache as sapien_clear_cache
from sapien.utils.viewer import Viewer

from envs.utils import *
from envs._GLOBAL_CONFIGS import CONFIGS_PATH

from .camera import Camera
from .dynamics import FlyingHandDynamics
from . import planner


class FlyingHandBaseTask(gym.Env):
    board_width = 0.73
    board_height = 1.65
    board_thickness = 0.02
    board_center_x = 1.0
    board_center_y = 0.0
    board_center_z = 1.275
    shelf_length = 0.30
    shelf_width = 0.50
    shelf_thickness = 0.02
    shelf_count = 4
    shelf_margin = 0.225
    shelf_object_gap = 0.02
    clutter_object_count_range = (4, 7)
    flying_hand_board_distance = 2.45
    flying_hand_initial_z_offset = 0.08
    flying_hand_asset_dir = Path("./assets/embodiments/flying-hand")
    vertical_object_qpos = [0.70710678, 0.70710678, 0.0, 0.0]
    initial_to_pre_grasp_seconds = 1.8
    pre_grasp_to_grasp_seconds = 1.3
    grasp_hold_seconds = 0.7
    grasp_to_pull_out_seconds = 1.1
    pull_out_to_place_seconds = 1.8
    release_to_retreat_seconds = pre_grasp_to_grasp_seconds
    release_hold_seconds = 0.5
    flying_hand_black_color = [0.101960784313725, 0.101960784313725, 0.101960784313725, 1.0]
    flying_hand_silver_color = [0.8, 0.8, 0.8, 1.0]
    flying_hand_black_link_names = {
        "left_up_link",
        "left_down_link",
        "right_down_link",
        "right_up_link",
        "left_tof_link",
        "right_tof_link",
    }
    flying_hand_silver_link_names = {
        "left_slide_link",
        "down_slide_link",
        "right_slide_link",
    }

    @property
    def FRAME_IDX(self):
        return self.frame_idx

    @FRAME_IDX.setter
    def FRAME_IDX(self, value):
        self.frame_idx = value

    def setup_demo(self, **kwags):
        self._init_flying_hand_task_env_(**kwags)

    def _init_flying_hand_task_env_(self, table_xy_bias=[0, 0], table_height_bias=0, **kwags):
        super().__init__()
        np.random.seed(kwags.get("seed", 0))
        self.frame_idx = 0
        self.flying_hand_save_step = 0
        self.task_name = kwags.get("task_name")
        self.save_dir = kwags.get("save_path", "data")
        self.ep_num = kwags.get("now_ep_num", 0)
        self.render_freq = kwags.get("render_freq", 10)
        self.data_type = deepcopy(kwags.get("data_type", {}))
        self.data_type["endpose"] = False
        self.data_type["qpos"] = False
        self.video_cameras = kwags.get("camera", {}).get("video_cameras", [])
        self.save_data = kwags.get("save_data", False)
        self.eval_mode = kwags.get("eval_mode", False)
        self.eval_video_path = kwags.get("eval_video_save_dir", None)
        self.save_freq = kwags.get("save_freq", 15)
        self.enable_dynamics = kwags.get("enable_dynamics", False)
        self.flying_hand_config = self._load_flying_hand_config()
        self._apply_flying_hand_config()
        self.plan_success = True
        self.step_lim = None
        self.eval_success = False
        self.need_plan = kwags.get("need_plan", True)
        self.stage_success_tag = False
        self.is_grasping = False
        self.record_flying_hand_trajectory = False
        self.flying_hand_ref_pose = None
        self.flying_hand_target_state_path = []
        self.flying_hand_actual_state_path = []
        self.left_joint_path = []
        self.right_joint_path = []
        self.task_actors = []
        self.task_failed = False

        random_setting = kwags.get("domain_randomization", {})
        self.random_background = random_setting.get("random_background", False)
        self.cluttered_board = random_setting.get("cluttered_board", False)
        self.clean_background_rate = random_setting.get("clean_background_rate", 1)
        self.random_head_camera_dis = random_setting.get("random_head_camera_dis", 0)
        self.random_board_height = random_setting.get("random_board_height", 0)
        self.random_light = random_setting.get("random_light", False)
        self.crazy_random_light_rate = random_setting.get("crazy_random_light_rate", 0)
        self.crazy_random_light = 0 if not self.random_light else np.random.rand() < self.crazy_random_light_rate
        self.random_flying_hand_init_pos = random_setting.get("random_flying_hand_init_pos", [0, 0, 0])
        self.board_z_bias = np.random.uniform(-self.random_board_height, self.random_board_height) + table_height_bias
        self.table_z_bias = self.board_z_bias
        self.clutter_object_count = int(np.random.randint(self.clutter_object_count_range[0], self.clutter_object_count_range[1] + 1))

        self.record_cluttered_objects = []
        self.now_obs = {}
        self.take_action_cnt = 0
        self.instruction = None
        self.eval_video_ffmpeg = None

        self.setup_scene(**kwags)
        self.create_table_and_wall(table_xy_bias=table_xy_bias, table_height=0.74)
        self.load_camera(**kwags)
        self.load_actors()
        self.load_flying_hand()
        if self.cluttered_board:
            self.get_cluttered_board()
        if self.eval_mode:
            with open(os.path.join(CONFIGS_PATH, "_eval_step_limit.yml"), "r", encoding="utf-8") as f:
                self.step_lim = yaml.safe_load(f).get(os.path.basename(self.task_name), 1000)

        self.info = {
            "cluttered_board_info": self.record_cluttered_objects,
            "texture_info": {"board_texture": self.board_texture},
            "info": {},
        }

    def setup_scene(self, **kwargs):
        self.engine = sapien.Engine()
        from sapien.render import set_global_config

        set_global_config(max_num_materials=50000, max_num_textures=50000)
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")
        self.scene = self.engine.create_scene(sapien.SceneConfig())
        self.sim_timestep = kwargs.get("timestep", 1 / 500)
        self.scene.set_timestep(self.sim_timestep)
        self.ground_height = kwargs.get("ground_height", 0)
        self.scene.add_ground(self.ground_height)
        self.scene.default_physical_material = self.scene.create_physical_material(
            kwargs.get("static_friction", 0.5),
            kwargs.get("dynamic_friction", 0.5),
            kwargs.get("restitution", 0),
        )
        self.scene.set_ambient_light(kwargs.get("ambient_light", [0.68, 0.68, 0.68]))
        point_lights = kwargs.get(
            "point_lights",
            [
                [[-1.2, -0.75, 2.05], [0.95, 0.95, 0.95]],
                [[-1.2, 0.75, 2.05], [0.95, 0.95, 0.95]],
                [[-1.3, 0.0, 1.15], [0.72, 0.72, 0.72]],
            ],
        )
        self.point_light_lst = []
        for point_light in point_lights:
            if self.random_light:
                point_light[1] = np.random.uniform(0.68, 1.0, 3).tolist()
            self.point_light_lst.append(self.scene.add_point_light(point_light[0], point_light[1], shadow=False))
        if self.render_freq:
            self.viewer = Viewer(self.renderer)
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(
                x=kwargs.get("camera_xyz_x", 0.85),
                y=kwargs.get("camera_xyz_y", -0.85),
                z=kwargs.get("camera_xyz_z", 1.65),
            )
            self.viewer.set_camera_rpy(
                r=kwargs.get("camera_rpy_r", 0),
                p=kwargs.get("camera_rpy_p", -0.65),
                y=kwargs.get("camera_rpy_y", 2.35),
            )

    def load_camera(self, **kwags):
        camera_kwargs = deepcopy(kwags)
        camera_kwargs["left_embodiment_config"] = self.flying_hand_config
        camera_kwargs["right_embodiment_config"] = self.flying_hand_config
        self.cameras = Camera(bias=self.board_z_bias, random_head_camera_dis=self.random_head_camera_dis, **camera_kwargs)
        self.cameras.load_camera(self.scene)
        self.scene.step()
        self.scene.update_render()

    def _load_flying_hand_config(self):
        with open(self.flying_hand_asset_dir / "config.yml", "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    def _apply_flying_hand_config(self):
        material_config = self.flying_hand_config["materials"]
        self.flying_hand_black_color = material_config["black_color"]
        self.flying_hand_silver_color = material_config["silver_color"]
        self.flying_hand_black_link_names = set(material_config["black_link_names"])
        self.flying_hand_silver_link_names = set(material_config["silver_link_names"])

    def _update_render(self):
        if self.crazy_random_light:
            for light in self.point_light_lst:
                light.set_color(np.random.uniform(0.62, 1.0, 3).tolist())
            self.scene.set_ambient_light(np.clip(np.array(self.scene.ambient_light) + np.random.rand(3) * 0.08 - 0.04, 0, 0.78))
        self.cameras.update_wrist_camera(self.flying_hand.find_link_by_name(self.flying_hand_config["camera"]["wrist_link"]).get_pose())
        self.scene.update_render()

    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        self.table_xy_bias = table_xy_bias
        self.board_texture = None
        self.wall = None
        self.table = None
        if self.random_background and np.random.rand() > self.clean_background_rate:
            texture_type = "seen" if not self.eval_mode else "unseen"
            path = f"./assets/background_texture/{texture_type}"
            self.board_texture = f"{texture_type}/{np.random.randint(0, len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]))}"
        self._create_vertical_board()

    def _get_available_model_ids(self, modelname):
        return sorted(
            int(os.path.basename(path).replace("model_data", "").replace(".json", ""))
            for path in glob.glob(os.path.join("assets/objects", modelname, "model_data*.json"))
        )

    def _sample_clutter_pose(self, p, gap):
        radius = p["radius"] + gap
        if radius > self.shelf_length / 2 or radius > self.shelf_width / 2:
            return None
        qpos = t3d.quaternions.qmult(
            t3d.euler.euler2quat(0, 0, np.random.uniform(-np.pi, np.pi)),
            self.vertical_object_qpos,
        ).tolist()
        for slot_id in np.random.permutation(len(self.board_slots)):
            xy = self._sample_shelf_xy(slot_id, [radius, radius], radius, random_x=True)
            if xy is not None:
                return qpos, int(slot_id), *xy, self.board_slots[int(slot_id)][1] - p["z_offset"]

    def _get_actor_world_bounds(self, actor, default_size=(0.1, 0.1, 0.1)):
        data = getattr(actor, "config", None)
        if data is not None:
            center = np.array(data.get("center", [0.0, 0.0, 0.0]), dtype=float)
            extents = np.array(data.get("extents", default_size), dtype=float)
            scale = np.array(data.get("scale", [1.0, 1.0, 1.0]), dtype=float)
            scale = np.array([float(scale)] * 3) if scale.ndim == 0 else scale
            center, half = center * scale, extents * scale / 2
            corners = np.array([
                [x, y, z]
                for x in [center[0] - half[0], center[0] + half[0]]
                for y in [center[1] - half[1], center[1] + half[1]]
                for z in [center[2] - half[2], center[2] + half[2]]
            ])
        else:
            corners = []
            for component in actor.actor.components:
                shapes = getattr(component, "render_shapes", None) or getattr(component, "collision_shapes", None)
                if not shapes:
                    continue
                for shape in shapes:
                    if not hasattr(shape, "half_size"):
                        continue
                    half = np.array(shape.half_size, dtype=float)
                    local = np.array([
                        [x, y, z]
                        for x in (-half[0], half[0])
                        for y in (-half[1], half[1])
                        for z in (-half[2], half[2])
                    ])
                    local_pose = shape.local_pose.to_transformation_matrix()
                    local = (local_pose[:3, :3] @ local.T).T + local_pose[:3, 3]
                    corners.append(local)
            if corners:
                corners = np.concatenate(corners, axis=0)
            else:
                half = np.array(default_size, dtype=float) / 2
                corners = np.array([
                    [x, y, z]
                    for x in [-half[0], half[0]]
                    for y in [-half[1], half[1]]
                    for z in [-half[2], half[2]]
                ])
        mat = actor.get_pose().to_transformation_matrix()
        corners = (mat[:3, :3] @ corners.T).T + mat[:3, 3]
        return np.array([corners.min(axis=0), corners.max(axis=0)])

    def _set_actor_bbox_center(self, actor, center):
        bounds = self._get_actor_world_bounds(actor)
        pose = actor.get_pose()
        actor.actor.set_pose(sapien.Pose((pose.p + np.array(center, dtype=float) - (bounds[0] + bounds[1]) / 2).tolist(), pose.q))

    def _shelf_xy_limits(self, slot_id, half_xy):
        shelf_y = self.board_slots[slot_id][0]
        return (
            self._board_front_x() - self.shelf_length + half_xy[0],
            self._board_front_x() - half_xy[0],
            shelf_y - self.shelf_width / 2 + half_xy[1],
            shelf_y + self.shelf_width / 2 - half_xy[1],
        )

    def _sample_shelf_xy(self, slot_id, half_xy, radius, x=None, y=None, random_x=False):
        x_min, x_max, y_min, y_max = self._shelf_xy_limits(slot_id, half_xy)
        if x_max < x_min or y_max < y_min:
            return None
        for _ in range(100):
            px = np.random.uniform(x_min, x_max) if random_x else x
            py = np.random.uniform(y_min, y_max) if y is None else y
            if px is None:
                return None
            if px < x_min or px > x_max or py < y_min or py > y_max:
                continue
            if all((px - ox) ** 2 + (py - oy) ** 2 > (radius + ro) ** 2 for ox, oy, ro in self.shelf_xy_areas[slot_id]):
                self.shelf_xy_areas[slot_id].append((px, py, radius))
                return px, py

    def _place_actor_on_shelf(self, actor, slot_id, x=None, y=None, padding=None, random_x=False, reserve=False):
        _, z = self.board_slots[slot_id]
        bounds = self._get_actor_world_bounds(actor)
        half_xy = (bounds[1] - bounds[0])[:2] / 2
        radius = max(half_xy) + (self.shelf_object_gap if padding is None else padding)
        x = self._object_x(actor) if x is None else x
        y = self.board_slots[slot_id][0] if y is None else y
        if reserve:
            x, y = self._sample_shelf_xy(slot_id, half_xy, radius, x, y, random_x=random_x)
            if x is None:
                raise RuntimeError(f"no free shelf space for object on shelf {slot_id}")
        self._set_actor_bbox_center(actor, [x, y, z + (bounds[1][2] - bounds[0][2]) / 2])
        return actor

    def add_task_objects(self, *actors):
        self.task_actors.extend(actors)

    def _task_objects_safe(self):
        self.task_failed = self.task_failed or any(
            self._get_actor_world_bounds(actor)[0][2] <= self.ground_height + 0.03
            for actor in self.task_actors
        )
        return not self.task_failed

    def add_prohibit_area(self, actor, padding=0.01):
        bounds = self._get_actor_world_bounds(actor)
        center = (bounds[0] + bounds[1]) / 2
        slot_id = int(np.argmin([abs(bounds[0][2] - z) for _, z in self.board_slots]))
        radius = max((bounds[1] - bounds[0])[:2]) / 2 + padding
        self.shelf_xy_areas[slot_id].append((center[0], center[1], radius))

    def _sample_board_slots(self):
        slots = []
        for z in np.linspace(
            self.board_center_z + self.board_height / 2 + self.board_z_bias - self.shelf_margin,
            self.board_center_z - self.board_height / 2 + self.board_z_bias + self.shelf_margin,
            self.shelf_count,
        ):
            y = np.random.uniform(-self.board_width / 2 + self.shelf_width / 2, self.board_width / 2 - self.shelf_width / 2)
            slots.append((y, z))
        return slots

    def _reset_board_slots(self):
        self.board_slots = self._sample_board_slots()
        self.shelf_xy_areas = {idx: [] for idx in range(len(self.board_slots))}
        self.shelves = [
            create_box(
                self.scene,
                sapien.Pose(p=[
                    self._board_front_x() - self.shelf_length / 2,
                    y,
                    z - self.shelf_thickness / 2,
                ]),
                half_size=[self.shelf_length / 2, self.shelf_width / 2, self.shelf_thickness / 2],
                color=(1, 1, 1),
                name=f"shelf_{idx + 1}",
                is_static=True,
            )
            for idx, (y, z) in enumerate(self.board_slots)
        ]

    def _board_front_x(self):
        return self.board_x - self.board_thickness / 2

    def _object_x(self, actor):
        bounds = self._get_actor_world_bounds(actor)
        return self._board_front_x() - self.shelf_length + (bounds[1][0] - bounds[0][0]) / 2

    def _board_center(self):
        return np.array([self.board_center_x, self.board_center_y, self.board_center_z + self.board_z_bias])

    def _sample_xyz_jitter(self, random_range):
        random_range = np.array(random_range, dtype=float)
        return np.random.uniform(-random_range, random_range)

    def _seconds_to_steps(self, seconds):
        return max(1, int(round(seconds / self.sim_timestep)))

    def _create_vertical_board(self):
        self.vertical_board = create_box(
            self.scene,
            sapien.Pose(p=self._board_center().tolist()),
            half_size=[self.board_thickness / 2, self.board_width / 2, self.board_height / 2],
            color=(1, 1, 1),
            name="vertical_board",
            texture_id=self.board_texture,
            is_static=True,
        )
        self.board_x = self.board_center_x

    def _create_board_actor(self, modelname, model_id, slot_id, mass=None, is_static=False, qpos=None, x=None, y=None, padding=None, random_x=False, reserve=False):
        shelf_y, z = self.board_slots[slot_id]
        actor = create_actor(
            scene=self.scene,
            pose=sapien.Pose([self._board_front_x() - self.shelf_length / 2, shelf_y, z], qpos or self.vertical_object_qpos),
            modelname=modelname,
            convex=True,
            model_id=model_id,
            is_static=is_static,
        )
        self._place_actor_on_shelf(actor, slot_id, x=x, y=y, padding=padding, random_x=random_x, reserve=reserve)
        if mass is not None:
            actor.set_mass(mass)
        self.add_task_objects(actor)
        return actor

    def _get_flying_hand_pose_from_u_center(self, u_center):
        root_config = self.flying_hand_config["root"]
        root_q = root_config["qpos"]
        root_pos = np.array(u_center, dtype=float) - t3d.quaternions.quat2mat(root_q) @ np.array(root_config["u_center_offset"], dtype=float)
        return sapien.Pose(root_pos.tolist(), root_q)

    def _get_flying_hand_initial_pose(self):
        u_center = self._board_center() + np.array([-self.flying_hand_board_distance, 0.0, self.flying_hand_initial_z_offset])
        return self._get_flying_hand_pose_from_u_center(u_center + self._sample_xyz_jitter(self.random_flying_hand_init_pos))

    def _get_flying_hand_pose(self, actor, x_offset, z_offset=0.0):
        bounds = self._get_actor_world_bounds(actor)
        center = (bounds[0] + bounds[1]) / 2
        return self._get_flying_hand_pose_from_u_center([center[0] + x_offset, center[1], center[2] + z_offset])

    def load_flying_hand(self):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        loader.load_multiple_collisions_from_file = True
        self.flying_hand = loader.load(str(self.flying_hand_asset_dir / self.flying_hand_config["urdf_path"]))
        self.flying_hand.set_name("flying_hand")
        self._set_flying_hand_materials()
        for link in self.flying_hand.get_links():
            link.set_mass(self.flying_hand_config["link_mass"])
        self.flying_hand_joints = {joint.get_name(): joint for joint in self.flying_hand.get_active_joints()}
        gripper_config = self.flying_hand_config["gripper"]
        self.flying_hand_gripper_joints = [self.flying_hand_joints[name] for name in gripper_config["joint_names"]]
        for joint in self.flying_hand_gripper_joints:
            joint.set_drive_property(gripper_config["stiffness"], gripper_config["damping"])
        self.set_flying_hand_gripper(gripper_config["open_qpos"])
        self.flying_hand_initial_pose = self._get_flying_hand_initial_pose()
        planner.set_pose(self, self.flying_hand_initial_pose)
        if self.enable_dynamics:
            self.flying_hand_dynamics = FlyingHandDynamics(self.flying_hand_config["dynamics"], self.sim_timestep)
            self.flying_hand_dynamics.reset(self.flying_hand_initial_pose)
        self.imu_odom_link = self.flying_hand.find_link_by_name("imu_odom_link")
        self.initial_imu_odom_pose = self.imu_odom_link.get_pose()
        planner.hold(self, self.flying_hand_initial_pose, 30)

    def _set_flying_hand_materials(self):
        for link in self.flying_hand.get_links():
            link_name = link.get_name()
            if link_name in self.flying_hand_black_link_names:
                color = self.flying_hand_black_color
            elif link_name in self.flying_hand_silver_link_names:
                color = self.flying_hand_silver_color
            else:
                continue
            for component in link.entity.components:
                if not isinstance(component, sapien.render.RenderBodyComponent):
                    continue
                for shape in component.render_shapes:
                    if shape.material is not None:
                        shape.material.set_base_color(color)

    def set_flying_hand_gripper(self, qpos, is_grasp=None):
        if is_grasp is not None:
            self.is_grasping = bool(is_grasp)
        for idx, joint in enumerate(self.flying_hand_gripper_joints):
            joint.set_drive_target(qpos[idx])
            joint.set_drive_velocity_target(0)

    def _get_flying_hand_xyzyaw(self, root_pose):
        pose = self.initial_imu_odom_pose.inv() * (root_pose * (self.flying_hand_initial_pose.inv() * self.initial_imu_odom_pose))
        return np.array([
            *pose.p.tolist(),
            np.arctan2(2 * (pose.q[0] * pose.q[3] + pose.q[1] * pose.q[2]), 1 - 2 * (pose.q[2] ** 2 + pose.q[3] ** 2)),
            float(self.is_grasping),
        ], dtype=np.float32)

    def _get_flying_hand_target_state(self):
        return self._get_flying_hand_xyzyaw(self.flying_hand_ref_pose or self.flying_hand_initial_pose)

    def _get_flying_hand_actual_state(self):
        return self._get_flying_hand_xyzyaw(self.flying_hand.get_root_pose())

    def _record_flying_hand_state(self):
        if self.record_flying_hand_trajectory:
            self.flying_hand_target_state_path.append(self._get_flying_hand_target_state())
            self.flying_hand_actual_state_path.append(self._get_flying_hand_actual_state())

    def reset_flying_hand_trajectory(self):
        self.flying_hand_target_state_path = []
        self.flying_hand_actual_state_path = []
        self.record_flying_hand_trajectory = True

    def get_obs(self):
        self._update_render()
        self.cameras.update_picture()
        cameras = [name for name, _ in self.cameras._cameras() if not self.video_cameras or name in self.video_cameras]
        obs = {
            "observation": {name: {} for name in cameras},
            "flying_hand": {
                "target_state": self._get_flying_hand_target_state(),
                "actual_state": self._get_flying_hand_actual_state(),
            },
        }
        if self.data_type.get("rgb", False):
            rgb = self.cameras.get_rgb()
            for name in cameras:
                obs["observation"][name].update(rgb[name])
        self.now_obs = deepcopy(obs)
        if self.eval_video_path is not None:
            self.eval_video_ffmpeg.stdin.write(obs["observation"][wrist]["rgb"].tobytes())
        return obs

    def _take_picture(self):
        if not self.save_data:
            return
        print("saving: episode = ", self.ep_num, " index = ", self.FRAME_IDX, end="\r")
        if self.FRAME_IDX == 0:
            self.folder_path = {"cache": f"{self.save_dir}/.cache/episode{self.ep_num}/"}
            os.makedirs(self.folder_path["cache"], exist_ok=True)
            for file in os.listdir(self.folder_path["cache"]):
                os.remove(self.folder_path["cache"] + file)
        save_pkl(self.folder_path["cache"] + f"{self.FRAME_IDX}.pkl", self.get_obs())
        self.FRAME_IDX += 1

    def _save_flying_hand_frame(self, save_freq, force=False):
        if save_freq is not None and save_freq > 0 and (force or self.flying_hand_save_step % save_freq == 0):
            self._update_render()
            self._record_flying_hand_state()
            self._take_picture()

    def start_flying_hand_record(self):
        self.set_flying_hand_gripper(self.flying_hand_config["gripper"]["open_qpos"], is_grasp=False)
        self.reset_flying_hand_trajectory()
        save_freq = self.save_freq if self.save_data else -1
        self._save_flying_hand_frame(save_freq, force=True)
        return save_freq

    def finish_flying_hand_record(self, save_freq):
        self._save_flying_hand_frame(save_freq, force=True)

    def merge_pkl_to_hdf5_video(self):
        if not self.save_data:
            return
        os.makedirs(f"{self.save_dir}/data", exist_ok=True)
        cameras = [name for name, _ in self.cameras._cameras() if not self.video_cameras or name in self.video_cameras]
        process_folder_to_hdf5_video(
            self.folder_path["cache"],
            f"{self.save_dir}/data/episode{self.ep_num}.hdf5",
            {name: f"{self.save_dir}/video/{name}/episode{self.ep_num}.mp4" for name in cameras},
        )

    def save_traj_data(self, idx):
        save_pkl(os.path.join(self.save_dir, "_traj_data", f"episode{idx}.pkl"), {
            "left_joint_path": deepcopy(self.left_joint_path),
            "right_joint_path": deepcopy(self.right_joint_path),
            "flying_hand_target_state": np.array(self.flying_hand_target_state_path, dtype=np.float32),
            "flying_hand_actual_state": np.array(self.flying_hand_actual_state_path, dtype=np.float32),
        })

    def load_tran_data(self, idx):
        assert self.save_dir is not None, "self.save_dir is None"
        with open(os.path.join(self.save_dir, "_traj_data", f"episode{idx}.pkl"), "rb") as f:
            return pickle.load(f)

    def remove_data_cache(self):
        if hasattr(self, "folder_path") and os.path.exists(self.folder_path["cache"]):
            shutil.rmtree(self.folder_path["cache"])

    def set_instruction(self, instruction=None):
        self.instruction = instruction

    def get_instruction(self, instruction=None):
        return self.instruction

    def set_path_lst(self, args):
        self.need_plan = args.get("need_plan", True)
        self.left_joint_path = args.get("left_joint_path", [])
        self.right_joint_path = args.get("right_joint_path", [])

    def _set_eval_video_ffmpeg(self, ffmpeg):
        self.eval_video_ffmpeg = ffmpeg

    def _del_eval_video_ffmpeg(self):
        if self.eval_video_ffmpeg:
            self.eval_video_ffmpeg.stdin.close()
            self.eval_video_ffmpeg.wait()
            self.eval_video_ffmpeg = None

    def close_env(self, clear_cache=False):
        if clear_cache:
            sapien_clear_cache()
        self.close()

    def check_actors_contact(self, actor1, actor2):
        for contact in self.scene.get_contacts():
            if {contact.bodies[0].entity.name, contact.bodies[1].entity.name} == {actor1, actor2}:
                return True
        return False

    def get_cluttered_board(self):
        if np.random.rand() < self.clean_background_rate:
            return
        clutter_gap = self.shelf_object_gap + 0.02
        task_objects = [
            actor.get_name()
            for actor in self.scene.get_all_actors()
            if actor.get_name() not in ["", "ground", "vertical_board"] and not actor.get_name().startswith("shelf")
        ]
        obj_names, info = get_available_cluttered_objects(task_objects)
        obj_names = [
            name for name in obj_names
            if info[name]["type"] == "glb"
            and os.path.isdir(os.path.join("assets/objects", name))
        ]
        candidates = [(name, model_id) for name in obj_names for model_id in info[name]["ids"]]
        for _ in range(self.clutter_object_count):
            np.random.shuffle(candidates)
            placed = False
            for name, model_id in candidates:
                pose = self._sample_clutter_pose(info[name]["params"][model_id], clutter_gap)
                if pose is None:
                    continue
                qpos, slot_id, x, y, z = pose
                create_actor(
                    scene=self.scene,
                    pose=sapien.Pose([x, y, z], qpos),
                    modelname=name,
                    convex=True,
                    model_id=model_id,
                    is_static=True,
                )
                self.record_cluttered_objects.append({"object_type": name, "object_index": int(model_id)})
                placed = True
                break
            if not placed:
                return
