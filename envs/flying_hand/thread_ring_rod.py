import numpy as np
import sapien

from envs.utils import *

from ._base_task import FlyingHandBaseTask
from . import planner


class thread_ring_rod(FlyingHandBaseTask):
    # The ring lies in the YZ plane, with its hole and both rods aligned to X.
    # 64 mm outer diameter and 56 mm inner diameter give a 4 mm radial band.
    ring_outer_radius = 0.032
    ring_inner_radius = 0.028
    # Axial ring length is 40 mm.  This is intentionally longer than the
    # previous 28 mm version while keeping the required 64/56 mm diameters.
    ring_half_length = 0.020
    ring_segments = 48
    rod_radius = 0.014
    rod_half_length = 0.18
    rod_y_offset = 0.18
    # Keep the ring close to the negative-X rod tip. This exposes its front
    # face to the gripper and leaves the rod body behind the grasp plane.
    initial_thread_x_offset = -0.14
    thread_start_clearance = 0.05
    thread_insert_depth = 0.02
    approach_high_x_offset = -0.70
    approach_high_z_offset = 0.28
    pre_grasp_x_offset = -0.42
    pre_grasp_z_offset = 0.16
    front_align_x_offset = -0.24
    front_align_z_offset = 0.09
    grasp_x_offset = -0.13
    pull_out_x_offset = -0.52
    # Grasp the upper band of the hanging ring without intersecting the source
    # rod.  Lower values caused a rod/link impact; >= 0.07 m missed the ring.
    grasp_z_offset = 0.06
    pull_out_z_offset = 0.24
    place_z_offset = 0.0
    place_pre_z_offset = 0.30
    grasp_to_place_seconds = 2.4
    thread_align_seconds = 1.0
    thread_insert_seconds = 1.4
    thread_exit_seconds = 1.0

    def _build_ring_render_mesh(self):
        vertices = []
        normals = []
        uvs = []
        triangles = []
        xs = [-self.ring_half_length, self.ring_half_length]
        n = self.ring_segments

        def add_vertex(position, normal, uv):
            vertices.append(position)
            normals.append(normal)
            uvs.append(uv)
            return len(vertices) - 1

        outer = []
        inner = []
        for xi, x in enumerate(xs):
            outer_row = []
            inner_row = []
            for i in range(n):
                a = 2 * np.pi * i / n
                ca, sa = np.cos(a), np.sin(a)
                outer_row.append(
                    add_vertex([x, self.ring_outer_radius * ca, self.ring_outer_radius * sa], [0, ca, sa], [i / n, xi])
                )
                inner_row.append(
                    add_vertex([x, self.ring_inner_radius * ca, self.ring_inner_radius * sa], [0, -ca, -sa], [i / n, xi])
                )
            outer.append(outer_row)
            inner.append(inner_row)

        for i in range(n):
            j = (i + 1) % n
            triangles.extend([
                [outer[0][i], outer[1][i], outer[1][j]],
                [outer[0][i], outer[1][j], outer[0][j]],
            ])
            triangles.extend([
                [inner[0][i], inner[1][j], inner[1][i]],
                [inner[0][i], inner[0][j], inner[1][j]],
            ])

        for xi, normal in [(0, [-1, 0, 0]), (1, [1, 0, 0])]:
            cap_outer = []
            cap_inner = []
            for i in range(n):
                a = 2 * np.pi * i / n
                ca, sa = np.cos(a), np.sin(a)
                cap_outer.append(
                    add_vertex([xs[xi], self.ring_outer_radius * ca, self.ring_outer_radius * sa], normal, [ca, sa])
                )
                cap_inner.append(
                    add_vertex([xs[xi], self.ring_inner_radius * ca, self.ring_inner_radius * sa], normal, [ca, sa])
                )
            for i in range(n):
                j = (i + 1) % n
                if xi == 0:
                    triangles.extend([
                        [cap_inner[i], cap_outer[j], cap_outer[i]],
                        [cap_inner[i], cap_inner[j], cap_outer[j]],
                    ])
                else:
                    triangles.extend([
                        [cap_inner[i], cap_outer[i], cap_outer[j]],
                        [cap_inner[i], cap_outer[j], cap_inner[j]],
                    ])

        return (
            np.array(vertices, dtype=np.float32),
            np.array(triangles, dtype=np.uint32),
            np.array(normals, dtype=np.float32),
            np.array(uvs, dtype=np.float32),
        )

    def _create_ring(self, pose):
        builder = self.scene.create_actor_builder()
        n = self.ring_segments
        shell_radius = (self.ring_outer_radius + self.ring_inner_radius) / 2
        collision_radius = (self.ring_outer_radius - self.ring_inner_radius) * 0.48
        for i in range(n):
            a = 2 * np.pi * i / n
            builder.add_cylinder_collision(
                pose=sapien.Pose([0, shell_radius * np.cos(a), shell_radius * np.sin(a)]),
                radius=collision_radius,
                half_length=self.ring_half_length,
                material=self.scene.default_physical_material,
                density=1000,
            )
        rigid = builder.build_physx_component()
        rigid.mass = 0.04
        rigid.linear_damping = 3.0
        rigid.angular_damping = 20.0

        entity = sapien.Entity()
        entity.set_name("ring")
        entity.set_pose(pose)
        render = sapien.render.RenderBodyComponent()
        ring_colors = [
            [0.92, 0.82, 0.32, 1.0],
            [0.30, 0.66, 0.92, 1.0],
            [0.86, 0.42, 0.34, 1.0],
        ]
        material = sapien.render.RenderMaterial(base_color=ring_colors[np.random.randint(len(ring_colors))])
        render.attach(sapien.render.RenderShapeTriangleMesh(*self._build_ring_render_mesh(), material))
        entity.add_component(rigid)
        entity.add_component(render)
        self.scene.add_entity(entity)
        return Actor(
            entity,
            {"center": [0, 0, 0], "extents": [self.ring_half_length * 2, self.ring_outer_radius * 2, self.ring_outer_radius * 2], "scale": [1, 1, 1]},
            mass=0.04,
        )

    def _create_black_rod(self, name, center):
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_collision(
            radius=self.rod_radius,
            half_length=self.rod_half_length,
            material=self.scene.default_physical_material,
        )
        builder.add_cylinder_visual(
            radius=self.rod_radius,
            half_length=self.rod_half_length,
            material=[0.03, 0.03, 0.03, 1.0],
        )
        rod = builder.build_static(name=name)
        rod.set_pose(sapien.Pose(center))
        return rod

    def load_actors(self):
        self._reset_board_slots()
        # Keep the source and target rods on the same shelf.  A vertical-shelf
        # change forces the carried ring through the board plane for some slot
        # combinations; separated Y positions still make them two distinct
        # black rods while preserving a collision-free transfer corridor.
        # Use the top shelf (index 0) so that the pick-and-transfer motion has
        # clearance above every shelf.  The lowest shelf caused the released
        # thin ring to fall to the ground for otherwise valid seed layouts.
        shelf_slot_id = 0
        x = self._board_front_x() - self.shelf_length + 0.04

        shelf_y, shelf_z = self.board_slots[shelf_slot_id]
        source_y = shelf_y - self.rod_y_offset
        self.source_rod_center = np.array([x, source_y, shelf_z + self.ring_outer_radius])
        self.source_rod = self._create_black_rod("source black rod", self.source_rod_center)
        # A ring around a horizontal rod hangs at the internal-tangent pose;
        # spawning both centers concentrically leaves a 14 mm gravitational
        # drop that is released as a sharp impact during the first contact.
        hanging_offset_z = -(self.ring_inner_radius - self.rod_radius)
        self.ring = self._create_ring(
            sapien.Pose(
                self.source_rod_center
                + np.array([self.initial_thread_x_offset, 0.0, hanging_offset_z])
            )
        )

        target_y = shelf_y + self.rod_y_offset
        self.target_rod_center = np.array([x, target_y, shelf_z + self.ring_outer_radius])
        self.target_rod = self._create_black_rod("target black rod", self.target_rod_center)
        self.target_ring_center = self.target_rod_center + np.array([
            -self.rod_half_length + self.ring_half_length + self.thread_insert_depth,
            0.0,
            0.0,
        ])
        self.add_task_objects(self.ring)
        self.add_prohibit_area(self.ring, padding=0.12)
        self.shelf_xy_areas[shelf_slot_id].append((x, source_y, 0.12))
        self.shelf_xy_areas[shelf_slot_id].append((x, target_y, 0.12))

    def _pose_for_carried_ring_center(self, center, ring_q, carried_pose):
        return sapien.Pose(np.asarray(center, dtype=float).tolist(), ring_q) * carried_pose.inv()

    def play_once(self):
        save_freq = self.start_flying_hand_record()
        motion = planner.TaskMotionPlanner(self, save_freq)
        approach_high = self._get_flying_hand_pose(
            self.ring,
            self.approach_high_x_offset,
            self.approach_high_z_offset,
        )
        pre = self._get_flying_hand_pose(self.ring, self.pre_grasp_x_offset, self.pre_grasp_z_offset)
        front_align = self._get_flying_hand_pose(
            self.ring,
            self.front_align_x_offset,
            self.front_align_z_offset,
        )
        grasp = self._get_flying_hand_pose(self.ring, self.grasp_x_offset, self.grasp_z_offset)
        source_exit = self._get_flying_hand_pose(
            self.ring,
            self.front_align_x_offset,
            self.front_align_z_offset,
        )
        pull = self._get_flying_hand_pose(self.ring, self.pull_out_x_offset, self.pull_out_z_offset)

        motion.move(
            [self.flying_hand_initial_pose, approach_high, pre, front_align, grasp],
            time_hints=[
                self.initial_to_pre_grasp_seconds,
                1.2,
                1.0,
                self.pre_grasp_to_grasp_seconds,
            ],
            phase_name="ring_approach_grasp",
            gripper_after_reach="close",
        )
        carried_pose = self.flying_hand.get_root_pose().inv() * self.ring.get_pose()
        ring_q = self.ring.get_pose().q
        thread_start_x = -self.rod_half_length - self.ring_half_length - self.thread_start_clearance
        thread_start_center = self.target_rod_center + np.array([thread_start_x, 0.0, self.place_z_offset])
        thread_pre_center = thread_start_center + np.array([0.0, 0.0, self.place_pre_z_offset])
        transfer_mid_center = np.array([
            thread_start_center[0],
            0.5 * (self.source_rod_center[1] + self.target_rod_center[1]),
            thread_pre_center[2],
        ])
        source_clear_center = np.array([
            thread_start_center[0],
            self.source_rod_center[1],
            thread_pre_center[2],
        ])
        thread_align_center = thread_start_center + np.array([0.0, 0.0, 0.12])
        thread_entry_x = -self.rod_half_length - self.ring_half_length - 0.015
        thread_entry_center = self.target_rod_center + np.array([
            thread_entry_x,
            0.0,
            self.place_z_offset,
        ])
        thread_end_center = self.target_ring_center + np.array([0.0, 0.0, self.place_z_offset])
        source_clear = self._pose_for_carried_ring_center(source_clear_center, ring_q, carried_pose)
        transfer_mid = self._pose_for_carried_ring_center(transfer_mid_center, ring_q, carried_pose)
        thread_pre = self._pose_for_carried_ring_center(thread_pre_center, ring_q, carried_pose)
        thread_align = self._pose_for_carried_ring_center(thread_align_center, ring_q, carried_pose)
        thread_start = self._pose_for_carried_ring_center(thread_start_center, ring_q, carried_pose)
        thread_entry = self._pose_for_carried_ring_center(thread_entry_center, ring_q, carried_pose)
        place = self._pose_for_carried_ring_center(thread_end_center, ring_q, carried_pose)
        motion.move(
            [
                grasp,
                source_exit,
                pull,
                source_clear,
                transfer_mid,
                thread_pre,
                thread_align,
                thread_start,
                thread_entry,
                place,
            ],
            time_hints=[
                0.8,
                self.grasp_to_pull_out_seconds,
                1.0,
                1.2,
                1.0,
                0.8,
                self.thread_align_seconds,
                0.8,
                self.thread_insert_seconds,
            ],
            phase_name="ring_thread_target_rod",
            carried_actor=self.ring,
            carried_pose=carried_pose,
        )
        motion.set_gripper(place, "open")
        exit_lift_center = thread_start_center + np.array([0.0, 0.0, self.pull_out_z_offset])
        exit_center = exit_lift_center + np.array([-0.25, 0.0, 0.0])
        exit_lift = self._pose_for_carried_ring_center(exit_lift_center, ring_q, carried_pose)
        exit_pose = self._pose_for_carried_ring_center(exit_center, ring_q, carried_pose)
        motion.move(
            [place, thread_entry, thread_start, exit_lift, exit_pose],
            time_hints=[0.8, 0.8, self.thread_exit_seconds, self.thread_exit_seconds],
            phase_name="ring_exit_target_rod",
        )
        self.finish_flying_hand_record(save_freq)
        self.info["info"] = {"{A}": "ring", "{B}": "target black rod"}
        return self.info

    def check_success(self):
        ring_center = (self._get_actor_world_bounds(self.ring)[0] + self._get_actor_world_bounds(self.ring)[1]) / 2
        return (
            self._task_objects_safe()
            and abs(ring_center[0] - self.target_ring_center[0]) < 0.06
            and np.linalg.norm(ring_center[1:] - self.target_ring_center[1:]) < 0.025
            and not self.is_grasping
        )
