import numpy as np
import sapien

from envs.utils import *

from ._base_task import FlyingHandBaseTask
from . import planner


class thread_tape_rod(FlyingHandBaseTask):
    tape_outer_radius = 0.04
    tape_inner_radius = 0.022
    tape_half_thickness = 0.014
    tape_segments = 48
    rod_radius = 0.014
    rod_half_length = 0.18
    thread_start_clearance = 0.05
    thread_insert_depth = 0.02
    pre_grasp_x_offset = -0.55
    grasp_x_offset = -0.13
    pull_out_x_offset = -0.52
    pre_grasp_z_offset = 0.10
    grasp_z_offset = 0.04
    pull_out_z_offset = 0.24
    place_z_offset = 0.0
    place_pre_z_offset = 0.30
    grasp_to_place_seconds = 2.4
    thread_align_seconds = 1.0
    thread_insert_seconds = 1.4
    thread_exit_seconds = 1.0

    def _annular_cylinder_mesh(self):
        vertices = []
        normals = []
        uvs = []
        triangles = []
        xs = [-self.tape_half_thickness, self.tape_half_thickness]
        n = self.tape_segments

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
                    add_vertex([x, self.tape_outer_radius * ca, self.tape_outer_radius * sa], [0, ca, sa], [i / n, xi])
                )
                inner_row.append(
                    add_vertex([x, self.tape_inner_radius * ca, self.tape_inner_radius * sa], [0, -ca, -sa], [i / n, xi])
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
                    add_vertex([xs[xi], self.tape_outer_radius * ca, self.tape_outer_radius * sa], normal, [ca, sa])
                )
                cap_inner.append(
                    add_vertex([xs[xi], self.tape_inner_radius * ca, self.tape_inner_radius * sa], normal, [ca, sa])
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

    def _create_tape(self, pose):
        builder = self.scene.create_actor_builder()
        n = self.tape_segments
        shell_radius = (self.tape_outer_radius + self.tape_inner_radius) / 2
        collision_radius = (self.tape_outer_radius - self.tape_inner_radius) * 0.48
        for i in range(n):
            a = 2 * np.pi * i / n
            builder.add_cylinder_collision(
                pose=sapien.Pose([0, shell_radius * np.cos(a), shell_radius * np.sin(a)]),
                radius=collision_radius,
                half_length=self.tape_half_thickness,
                material=self.scene.default_physical_material,
                density=1000,
            )
        rigid = builder.build_physx_component()
        rigid.mass = 0.04
        rigid.linear_damping = 3.0
        rigid.angular_damping = 20.0

        entity = sapien.Entity()
        entity.set_name("tape roll")
        entity.set_pose(pose)
        render = sapien.render.RenderBodyComponent()
        material = sapien.render.RenderMaterial(base_color=[0.92, 0.82, 0.32, 1.0])
        render.attach(sapien.render.RenderShapeTriangleMesh(*self._annular_cylinder_mesh(), material))
        entity.add_component(rigid)
        entity.add_component(render)
        self.scene.add_entity(entity)
        return Actor(
            entity,
            {"center": [0, 0, 0], "extents": [self.tape_half_thickness * 2, self.tape_outer_radius * 2, self.tape_outer_radius * 2], "scale": [1, 1, 1]},
            mass=0.04,
        )

    def load_actors(self):
        self._reset_board_slots()
        tape_slot, rod_slot = np.random.choice(len(self.board_slots), 2, replace=False)
        y, z = self.board_slots[int(tape_slot)]
        self.tape = self._create_tape(
            sapien.Pose([self._board_front_x() - self.shelf_length + 0.04, y, z + self.tape_outer_radius])
        )
        y, z = self.board_slots[int(rod_slot)]
        x = self._board_front_x() - self.shelf_length + 0.04
        rod_z = z + self.tape_outer_radius
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_collision(
            radius=self.rod_radius,
            half_length=self.rod_half_length,
            material=self.scene.default_physical_material,
        )
        builder.add_cylinder_visual(
            radius=self.rod_radius,
            half_length=self.rod_half_length,
            material=[0.35, 0.35, 0.35, 1.0],
        )
        self.rod = builder.build_static(name="rod")
        self.rod.set_pose(sapien.Pose([x, y, rod_z]))
        self.rod_center = np.array([x, y, rod_z])
        self.target_center = self.rod_center + np.array([
            -self.rod_half_length + self.tape_half_thickness + self.thread_insert_depth,
            0.0,
            0.0,
        ])
        self.add_task_objects(self.tape)
        self.add_prohibit_area(self.tape, padding=0.12)
        self.shelf_xy_areas[int(rod_slot)].append((x, y, 0.12))

    def _pose_from_center(self, center, x_offset, z_offset=0.0):
        return self._get_flying_hand_pose_from_u_center([center[0] + x_offset, center[1], center[2] + z_offset])

    def _pose_for_carried_tape_center(self, center, tape_q, carried_pose):
        return sapien.Pose(np.asarray(center, dtype=float).tolist(), tape_q) * carried_pose.inv()

    def play_once(self):
        save_freq = self.start_flying_hand_record()
        pre = self._get_flying_hand_pose(self.tape, self.pre_grasp_x_offset, self.pre_grasp_z_offset)
        grasp = self._get_flying_hand_pose(self.tape, self.grasp_x_offset, self.grasp_z_offset)
        pull = self._get_flying_hand_pose(self.tape, self.pull_out_x_offset, self.pull_out_z_offset)

        planner.move_minco(
            self,
            [self.flying_hand_initial_pose, pre, grasp],
            times=[self.initial_to_pre_grasp_seconds, self.pre_grasp_to_grasp_seconds],
            save_freq=save_freq,
        )
        self.set_flying_hand_gripper(self.flying_hand_config["gripper"]["close_qpos"], is_grasp=True)
        planner.hold(self, grasp, self._seconds_to_steps(self.grasp_hold_seconds), save_freq=save_freq)
        carried_pose = self.flying_hand.get_root_pose().inv() * self.tape.get_pose()
        tape_q = self.tape.get_pose().q
        thread_start_x = -self.rod_half_length - self.tape_half_thickness - self.thread_start_clearance
        thread_start_center = self.rod_center + np.array([thread_start_x, 0.0, self.place_z_offset])
        thread_pre_center = thread_start_center + np.array([0.0, 0.0, self.place_pre_z_offset])
        thread_end_center = self.target_center + np.array([0.0, 0.0, self.place_z_offset])
        thread_pre = self._pose_for_carried_tape_center(thread_pre_center, tape_q, carried_pose)
        thread_start = self._pose_for_carried_tape_center(thread_start_center, tape_q, carried_pose)
        place = self._pose_for_carried_tape_center(thread_end_center, tape_q, carried_pose)
        planner.move_minco(
            self,
            [grasp, pull, thread_pre, thread_start, place],
            times=[
                self.grasp_to_pull_out_seconds,
                self.grasp_to_place_seconds,
                self.thread_align_seconds,
                self.thread_insert_seconds,
            ],
            save_freq=save_freq,
            carried_actor=self.tape,
            carried_pose=carried_pose,
        )
        self.set_flying_hand_gripper(self.flying_hand_config["gripper"]["open_qpos"], is_grasp=False)
        planner.hold(self, place, self._seconds_to_steps(self.release_hold_seconds), save_freq=save_freq)
        exit_lift_center = thread_end_center + np.array([0.0, 0.0, self.pull_out_z_offset])
        exit_center = thread_start_center + np.array([0.0, 0.0, self.pull_out_z_offset])
        exit_lift = self._pose_for_carried_tape_center(exit_lift_center, tape_q, carried_pose)
        exit_pose = self._pose_for_carried_tape_center(exit_center, tape_q, carried_pose)
        planner.move_minco(
            self,
            [place, exit_lift, exit_pose],
            times=[self.thread_exit_seconds, self.thread_exit_seconds],
            save_freq=save_freq,
        )
        self.finish_flying_hand_record(save_freq)
        self.info["info"] = {"{A}": "tape roll", "{B}": "rod"}
        return self.info

    def check_success(self):
        tape = (self._get_actor_world_bounds(self.tape)[0] + self._get_actor_world_bounds(self.tape)[1]) / 2
        return (
            self._task_objects_safe()
            and np.linalg.norm(tape[:2] - self.target_center[:2]) < 0.07
            and abs(tape[2] - self.target_center[2]) < 0.08
            and not self.is_grasping
        )
