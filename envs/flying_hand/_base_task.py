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
    board_side_strip_width = 0.03
    wall_distance = 3.0
    wall_width = 24.0
    wall_height = 10.0
    wall_thickness = 0.02
    ground_size = 24.0
    ground_thickness = 0.02
    fixture_color = (0.94, 0.88, 0.70)
    shelf_length = 0.30
    shelf_width = board_width
    shelf_thickness = 0.02
    shelf_count = 4
    shelf_ground_height = 0.12
    shelf_low_spacing = 0.40
    shelf_object_gap = 0.02
    clutter_object_count_range = (4, 7)
    flying_hand_board_distance = 2.45
    flying_hand_initial_z_offset = 0.08
    flying_hand_asset_dir = Path("./assets/embodiments/flying-hand")
    vertical_object_qpos = [0.70710678, 0.70710678, 0.0, 0.0]
    initial_to_pre_grasp_seconds = 2.0
    pre_grasp_to_grasp_seconds = 1.6
    grasp_hold_seconds = 0.8
    grasp_to_pull_out_seconds = 1.3
    pull_out_to_place_seconds = 2.1
    release_to_retreat_seconds = pre_grasp_to_grasp_seconds
    release_hold_seconds = 0.8
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

    def _get_isolated_carry_exclusions(self, actor):
        """Return scene entities to disable while an actor is carried in isolation."""
        return ()

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
        self.minco_optimization_config = planner.MincoOptimizationConfig.from_mapping(
            self.flying_hand_config["minco_optimization"]
        )
        self.minco_time_optimizer = planner.MincoTimeOptimizer(
            self.minco_optimization_config
        )
        self.goal_grasp_planner_config = planner.GoalGraspPlannerConfig.from_mapping(
            self.flying_hand_config["goal_grasp_planner"]
        )
        self.minco_plan_diagnostics = []
        self._flying_hand_carrying = False
        self._isolated_carried_actor_state = None
        self._released_actor_collision_state = None
        self.flying_hand_grasp_diagnostics = []
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
        self.random_light = random_setting.get("random_light", False)
        self.crazy_random_light_rate = random_setting.get("crazy_random_light_rate", 0)
        self.crazy_random_light = 0 if not self.random_light else np.random.rand() < self.crazy_random_light_rate
        self.random_flying_hand_init_pos = random_setting.get("random_flying_hand_init_pos", [0, 0, 0])
        self.board_z_bias = float(table_height_bias)
        self.table_z_bias = self.board_z_bias
        self.clutter_object_count = int(np.random.randint(self.clutter_object_count_range[0], self.clutter_object_count_range[1] + 1))
        self.ground_texture = None
        self.wall_texture = None
        if self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            with open(Path(__file__).with_name("textures.yml"), "r", encoding="utf-8") as f:
                texture_config = yaml.safe_load(f)
            if np.random.rand() > self.clean_background_rate:
                self.ground_texture = f"{texture_type}/{np.random.choice(texture_config['ground'][texture_type])}"
            if np.random.rand() > self.clean_background_rate:
                self.wall_texture = f"{texture_type}/{np.random.choice(texture_config['wall'][texture_type])}"

        self.record_cluttered_objects = []
        self.now_obs = {}
        self.take_action_cnt = 0
        self.instruction = None
        self.eval_video_ffmpeg = None
        self.eval_video_frame_limit = None
        self.eval_video_frames_written = 0

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
            "texture_info": {
                "ground_texture": self.ground_texture,
                "wall_texture": self.wall_texture,
            },
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
        self.scene.default_physical_material = self.scene.create_physical_material(
            kwargs.get("static_friction", 0.5),
            kwargs.get("dynamic_friction", 0.5),
            kwargs.get("restitution", 0),
        )
        if self.ground_texture is None:
            self.scene.add_ground(self.ground_height)
        else:
            self.ground = create_box(
                self.scene,
                sapien.Pose([0, 0, self.ground_height - self.ground_thickness / 2]),
                half_size=[self.ground_size / 2, self.ground_size / 2, self.ground_thickness / 2],
                color=(1, 1, 1),
                name="ground",
                texture_id=self.ground_texture,
                is_static=True,
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
                x=kwargs.get("camera_xyz_x", -1.15),
                y=kwargs.get("camera_xyz_y", 0.0),
                z=kwargs.get("camera_xyz_z", 1.45),
            )
            self.viewer.set_camera_rpy(
                r=kwargs.get("camera_rpy_r", 0),
                p=kwargs.get("camera_rpy_p", -0.35),
                y=kwargs.get("camera_rpy_y", 0.0),
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
            config = yaml.safe_load(f)
        with open(Path(__file__).with_name("_base_config.yml"), "r", encoding="utf-8") as f:
            control_config = yaml.safe_load(f)

        def merge(target, update):
            for key, value in update.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    merge(target[key], value)
                else:
                    target[key] = deepcopy(value)

        merge(config, control_config)
        return config

    def _apply_flying_hand_config(self):
        material_config = self.flying_hand_config["materials"]
        self.flying_hand_black_color = material_config["black_color"]
        self.flying_hand_silver_color = material_config["silver_color"]
        self.flying_hand_black_link_names = set(material_config["black_link_names"])
        self.flying_hand_silver_link_names = set(material_config["silver_link_names"])

        grasp_config = self.flying_hand_config["gripper"]["grasp_validation"]
        center_min = np.asarray(grasp_config["center_bounds_min"], dtype=float)
        center_max = np.asarray(grasp_config["center_bounds_max"], dtype=float)
        if (
            center_min.shape != (3,)
            or center_max.shape != (3,)
            or np.any(center_min >= center_max)
        ):
            raise ValueError("gripper grasp-validation center bounds must contain three increasing axes")
        close_threshold = float(grasp_config["close_threshold"])
        open_threshold = float(grasp_config["open_threshold"])
        if not 0.0 <= open_threshold < close_threshold <= 1.0:
            raise ValueError("gripper grasp thresholds must satisfy 0 <= open < close <= 1")
        grasp_hold_seconds = float(self.flying_hand_config["gripper"]["grasp_hold_seconds"])
        if grasp_hold_seconds <= 0.0:
            raise ValueError("gripper grasp_hold_seconds must be positive")
        self.grasp_hold_seconds = grasp_hold_seconds
        self.release_hold_seconds = grasp_hold_seconds
        self.flying_hand_grasp_validation = {
            **grasp_config,
            "center_bounds_min": center_min,
            "center_bounds_max": center_max,
        }
        waypoint_config = self.flying_hand_config["waypoint_tracking"]
        self.flying_hand_waypoint_tracking = {
            key: float(waypoint_config[key])
            for key in (
                "max_velocity",
                "max_acceleration",
                "max_yaw_rate",
            )
        }
        if any(value <= 0.0 for value in self.flying_hand_waypoint_tracking.values()):
            raise ValueError("waypoint-tracking limits must be positive")

    def _update_render(self):
        if self.crazy_random_light:
            for light in self.point_light_lst:
                light.set_color(np.random.uniform(0.62, 1.0, 3).tolist())
            self.scene.set_ambient_light(np.clip(np.array(self.scene.ambient_light) + np.random.rand(3) * 0.08 - 0.04, 0, 0.78))
        self.cameras.update_wrist_camera(self.flying_hand.find_link_by_name(self.flying_hand_config["camera"]["wrist_link"]).get_pose())
        self.scene.update_render()

    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        self.table_xy_bias = table_xy_bias
        self.table = None
        self.wall = create_box(
            self.scene,
            sapien.Pose([
                self.board_center_x + self.board_thickness / 2 + self.wall_distance + self.wall_thickness / 2,
                self.board_center_y,
                self.ground_height + self.wall_height / 2,
            ]),
            half_size=[self.wall_thickness / 2, self.wall_width / 2, self.wall_height / 2],
            color=(1, 1, 1),
            name="wall",
            texture_id=self.wall_texture,
            is_static=True,
        )
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

    def _get_actor_world_corners(
        self,
        actor,
        default_size=(0.1, 0.1, 0.1),
        actor_pose=None,
    ):
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
        pose = actor.get_pose() if actor_pose is None else actor_pose
        mat = pose.to_transformation_matrix()
        return (mat[:3, :3] @ corners.T).T + mat[:3, 3]

    def _get_actor_world_bounds(self, actor, default_size=(0.1, 0.1, 0.1)):
        corners = self._get_actor_world_corners(actor, default_size=default_size)
        return np.array([corners.min(axis=0), corners.max(axis=0)])

    def get_flying_hand_grasp_diagnostic(
        self,
        actor,
        hand_pose=None,
        actor_pose=None,
    ):
        """Check whether an actor box center is inside the grasp region."""
        root_pose = self.flying_hand.get_root_pose() if hand_pose is None else hand_pose
        u_center_offset = self.flying_hand_config["root"]["u_center_offset"]
        u_center_pose = root_pose * sapien.Pose(u_center_offset)
        u_center_inverse = np.linalg.inv(u_center_pose.to_transformation_matrix())
        world_corners = self._get_actor_world_corners(
            actor,
            actor_pose=actor_pose,
        )
        local_corners = (
            u_center_inverse[:3, :3] @ world_corners.T
        ).T + u_center_inverse[:3, 3]
        actor_min = local_corners.min(axis=0)
        actor_max = local_corners.max(axis=0)
        actor_center = 0.5 * (actor_min + actor_max)

        config = self.flying_hand_grasp_validation
        center_min = config["center_bounds_min"]
        center_max = config["center_bounds_max"]
        center_inside = bool(
            np.all(actor_center >= center_min)
            and np.all(actor_center <= center_max)
        )
        return {
            "actor": actor.get_name(),
            "eligible": center_inside,
            "center_inside": center_inside,
            "actor_center_u": actor_center.tolist(),
            "actor_bounds_min_u": actor_min.tolist(),
            "actor_bounds_max_u": actor_max.tolist(),
            "center_bounds_min_u": center_min.tolist(),
            "center_bounds_max_u": center_max.tolist(),
        }

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

    def get_flying_hand_grasp_candidates(self):
        """Return movable task actors that the policy is allowed to capture."""
        if hasattr(self, "graspable_actors"):
            return tuple(self.graspable_actors)
        if hasattr(self, "target_actor"):
            return (self.target_actor,)
        return tuple(self.task_actors)

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
        for z in np.r_[np.linspace(
            self.board_center_z + self.board_height / 2 + self.board_z_bias,
            self.ground_height + self.shelf_ground_height + self.shelf_low_spacing,
            self.shelf_count - 1,
        ), self.ground_height + self.shelf_ground_height]:
            y = np.random.uniform(
                self.board_center_y - self.board_width / 2 + self.shelf_width / 2,
                self.board_center_y + self.board_width / 2 - self.shelf_width / 2,
            )
            slots.append((y, z))
        return slots

    def _reset_board_slots(self):
        shelf_slots = self._sample_board_slots()
        self.board_slots = shelf_slots[:-1]
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
                color=self.fixture_color,
                name=f"shelf_{idx + 1}",
                is_static=True,
            )
            for idx, (y, z) in enumerate(shelf_slots)
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
        top = self.board_center_z + self.board_height / 2 + self.board_z_bias
        self.vertical_board = [
            create_box(
                self.scene,
                sapien.Pose(p=[
                    self.board_center_x,
                    self.board_center_y + side * (self.board_width - self.board_side_strip_width) / 2,
                    (top + self.ground_height) / 2,
                ]),
                half_size=[self.board_thickness / 2, self.board_side_strip_width / 2, (top - self.ground_height) / 2],
                color=self.fixture_color,
                name=f"vertical_board_{'right' if side > 0 else 'left'}",
                is_static=True,
            )
            for side in (-1, 1)
        ]
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
        self._configure_flying_hand_inertials()
        joints = self.flying_hand.get_active_joints()
        names = [joint.get_name() for joint in joints]
        self.flying_hand_joints = dict(zip(names, joints))
        gripper_config = self.flying_hand_config["gripper"]
        self.flying_hand_gripper_joint_indices = [names.index(name) for name in gripper_config["joint_names"]]
        self.flying_hand_gripper_joints = [self.flying_hand_joints[name] for name in gripper_config["joint_names"]]
        self._configure_flying_hand_gripper_control(gripper_config)
        if gripper_config.get("disable_self_collision", True):
            self._disable_flying_hand_self_collision()
        self.flying_hand_gripper_qpos = np.array(gripper_config["open_qpos"], dtype=float)
        self.flying_hand_gripper_start_qpos = self.flying_hand_gripper_qpos.copy()
        self.flying_hand_gripper_step = 0
        self.flying_hand_gripper_steps = 0
        self._set_flying_hand_gripper_qpos(self.flying_hand_gripper_qpos)
        self.set_flying_hand_gripper(gripper_config["open_qpos"])
        self.flying_hand_initial_pose = self._get_flying_hand_initial_pose()
        planner.set_pose(self, self.flying_hand_initial_pose)
        if self.enable_dynamics:
            self.flying_hand_dynamics = FlyingHandDynamics(self.flying_hand_config["dynamics"], self.sim_timestep)
            self.flying_hand_dynamics.reset(self.flying_hand_initial_pose)
        self.imu_odom_link = self.flying_hand.find_link_by_name("imu_odom_link")
        self.initial_imu_odom_pose = self.imu_odom_link.get_pose()
        planner.hold(self, self.flying_hand_initial_pose, 30)

    def _configure_flying_hand_inertials(self):
        config = self.flying_hand_config["inertial"]
        links = self.flying_hand.get_links()
        urdf_total_mass = sum(float(link.get_mass()) for link in links)
        target_mass = float(config["target_mass"])
        if urdf_total_mass <= 0 or target_mass <= 0:
            raise ValueError("Flying-hand URDF and target masses must be positive")

        unmodeled_config = config.get("unmodeled_links", {})
        unmodeled_names = set(unmodeled_config.get("names", []))
        link_names = {link.get_name() for link in links}
        unknown_names = unmodeled_names - link_names
        if unknown_names:
            raise ValueError(f"Unknown unmodeled flying-hand links: {sorted(unknown_names)}")
        unmodeled_mass = float(unmodeled_config.get("mass", 0.0))
        unmodeled_inertia = np.asarray(unmodeled_config.get("inertia", [0.0, 0.0, 0.0]), dtype=float)
        if unmodeled_names and (
            unmodeled_mass <= 0
            or unmodeled_inertia.shape != (3,)
            or np.any(unmodeled_inertia <= 0)
        ):
            raise ValueError("Unmodeled flying-hand links require positive mass and three positive inertias")
        modeled_links = [link for link in links if link.get_name() not in unmodeled_names]
        modeled_urdf_mass = sum(float(link.get_mass()) for link in modeled_links)
        modeled_target_mass = target_mass - unmodeled_mass * len(unmodeled_names)
        if modeled_urdf_mass <= 0 or modeled_target_mass <= 0:
            raise ValueError("Modeled flying-hand mass must remain positive")

        scale = modeled_target_mass / modeled_urdf_mass
        scale_inertia = bool(config.get("scale_inertia_with_mass", True))
        for link in links:
            if link.get_name() in unmodeled_names:
                link.set_mass(unmodeled_mass)
                link.set_inertia(unmodeled_inertia)
            else:
                link.set_mass(float(link.get_mass()) * scale)
                if scale_inertia:
                    link.set_inertia(np.asarray(link.get_inertia(), dtype=float) * scale)

        self.flying_hand_urdf_total_mass = urdf_total_mass
        self.flying_hand_modeled_urdf_mass = modeled_urdf_mass
        self.flying_hand_inertial_scale = scale
        self.flying_hand_total_mass = sum(float(link.get_mass()) for link in links)

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

    def _disable_flying_hand_self_collision(self):
        for link in self.flying_hand.get_links():
            for shape in link.get_collision_shapes():
                groups = shape.get_collision_groups()
                groups[2] |= 1 << 8
                groups[3] = (groups[3] & 0xFFFF0000) | 0x143
                shape.set_collision_groups(groups)

    def _configure_flying_hand_gripper_control(self, gripper_config):
        self.flying_hand_gripper_control_mode = gripper_config["joint_control_mode"]
        if self.flying_hand_gripper_control_mode not in {"direct_position", "drive"}:
            raise ValueError(f"Unsupported flying hand gripper joint_control_mode: {self.flying_hand_gripper_control_mode}")
        if self.flying_hand_gripper_control_mode != "drive":
            return

        drive_config = gripper_config["drive"]
        for joint in self.flying_hand_gripper_joints:
            joint_config = drive_config
            if "stiffness" not in drive_config:
                joint_type = "prismatic" if "prismatic" in joint.get_name() else "revolute"
                joint_config = drive_config[joint_type]
            joint.set_drive_property(
                stiffness=float(joint_config["stiffness"]),
                damping=float(joint_config["damping"]),
                force_limit=float(joint_config["force_limit"]),
            )
            joint.set_friction(float(joint_config["friction"]))

    def _set_flying_hand_gripper_qpos(self, qpos):
        joints_qpos = self.flying_hand.get_qpos()
        joints_qpos[self.flying_hand_gripper_joint_indices] = qpos
        self.flying_hand.set_qpos(joints_qpos)
        joints_qvel = self.flying_hand.get_qvel()
        joints_qvel[self.flying_hand_gripper_joint_indices] = 0.0
        self.flying_hand.set_qvel(joints_qvel)

    def _set_flying_hand_gripper_drive_target(self, qpos):
        self.flying_hand.set_qf(self.flying_hand.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
        for joint, target in zip(self.flying_hand_gripper_joints, qpos):
            joint.set_drive_target(float(target))
            joint.set_drive_velocity_target(0.0)

    def set_flying_hand_gripper(self, qpos, is_grasp=None):
        qpos = np.array(qpos, dtype=float)
        if is_grasp is not None:
            self.is_grasping = bool(is_grasp)
            if not self.is_grasping:
                planner.restore_isolated_carried_actor(
                    self,
                    suppress_gripper_collisions=True,
                )
            self.flying_hand_gripper_start_qpos = self.flying_hand.get_qpos()[self.flying_hand_gripper_joint_indices].copy()
            self.flying_hand_gripper_step = 0
            self.flying_hand_gripper_steps = self._seconds_to_steps(self.grasp_hold_seconds if self.is_grasping else self.release_hold_seconds)
        else:
            self.flying_hand_gripper_start_qpos = qpos.copy()
            self.flying_hand_gripper_step = 0
            self.flying_hand_gripper_steps = 0
        self.flying_hand_gripper_qpos = qpos
        if is_grasp is None:
            self.apply_flying_hand_gripper_qpos()

    def apply_flying_hand_gripper_qpos(self):
        if self.flying_hand_gripper_step < self.flying_hand_gripper_steps:
            self.flying_hand_gripper_step += 1
            a = self.flying_hand_gripper_step / self.flying_hand_gripper_steps
            qpos = (
                (1 - a) * self.flying_hand_gripper_start_qpos
                + a * self.flying_hand_gripper_qpos
            )
        else:
            qpos = self.flying_hand_gripper_qpos
        if self.flying_hand_gripper_control_mode == "drive":
            self._set_flying_hand_gripper_drive_target(qpos)
        else:
            self._set_flying_hand_gripper_qpos(qpos)

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
        self.minco_plan_diagnostics = []
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
        if self.eval_video_ffmpeg is not None:
            if isinstance(self.eval_video_ffmpeg, dict):
                for video_camera, ffmpeg in self.eval_video_ffmpeg.items():
                    if video_camera in obs["observation"] and "rgb" in obs["observation"][video_camera]:
                        ffmpeg.stdin.write(obs["observation"][video_camera]["rgb"].tobytes())
            else:
                video_camera = self.cameras.wrist_camera_name
                if video_camera not in obs["observation"]:
                    video_camera = next(iter(obs["observation"]))
                self.eval_video_ffmpeg.stdin.write(obs["observation"][video_camera]["rgb"].tobytes())
            self.eval_video_frames_written += 1
            if (
                self.eval_video_frame_limit is not None
                and self.eval_video_frames_written >= self.eval_video_frame_limit
            ):
                self._del_eval_video_ffmpeg()
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
            fps=1.0 / (float(self.sim_timestep) * float(self.save_freq)),
        )

    def save_traj_data(self, idx):
        save_pkl(os.path.join(self.save_dir, "_traj_data", f"episode{idx}.pkl"), {
            "left_joint_path": deepcopy(self.left_joint_path),
            "right_joint_path": deepcopy(self.right_joint_path),
            "flying_hand_target_state": np.array(self.flying_hand_target_state_path, dtype=np.float32),
            "flying_hand_actual_state": np.array(self.flying_hand_actual_state_path, dtype=np.float32),
            "minco_plan_diagnostics": deepcopy(self.minco_plan_diagnostics),
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
            if isinstance(self.eval_video_ffmpeg, dict):
                for ffmpeg in self.eval_video_ffmpeg.values():
                    ffmpeg.stdin.close()
                for ffmpeg in self.eval_video_ffmpeg.values():
                    ffmpeg.wait()
            else:
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
            if actor.get_name() not in ["", "ground", "wall"] and not actor.get_name().startswith(("shelf", "vertical_board"))
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
