import numpy as np

from ._base_task import FlyingHandBaseTask
from . import planner


class place_fruit_skillet(FlyingHandBaseTask):
    pre_grasp_x_offset = -0.55
    grasp_x_offset = -0.09
    pull_out_x_offset = -0.50
    pre_grasp_z_offset = 0.08
    grasp_z_offset = 0.01
    pull_out_z_offset = 0.18
    place_z_offset = 0.13
    grasp_to_place_seconds = 2.0

    def load_actors(self):
        self._reset_board_slots()
        self.fruit_name = "035_apple"
        self.skillet_name = "106_skillet"
        self.fruit_id = int(np.random.choice(self._get_available_model_ids(self.fruit_name)))
        self.skillet_id = int(np.random.choice(self._get_available_model_ids(self.skillet_name)))
        fruit_slot, skillet_slot = np.random.choice(len(self.board_slots), 2, replace=False)
        self.fruit = self._create_board_actor(self.fruit_name, self.fruit_id, int(fruit_slot), mass=0.05)
        self.skillet = self._create_board_actor(self.skillet_name, self.skillet_id, int(skillet_slot), mass=0.4)
        self.add_prohibit_area(self.fruit, padding=0.08)
        self.add_prohibit_area(self.skillet, padding=0.08)

    def _pose_from_center(self, center, x_offset, z_offset=0.0):
        return self._get_flying_hand_pose_from_u_center([center[0] + x_offset, center[1], center[2] + z_offset])

    def play_once(self):
        save_freq = self.start_flying_hand_record()
        pre = self._get_flying_hand_pose(self.fruit, self.pre_grasp_x_offset, self.pre_grasp_z_offset)
        grasp = self._get_flying_hand_pose(self.fruit, self.grasp_x_offset, self.grasp_z_offset)
        pull = self._get_flying_hand_pose(self.fruit, self.pull_out_x_offset, self.pull_out_z_offset)
        target = self.skillet.get_functional_point(0)[:3]
        place_pre = self._pose_from_center(target, self.pull_out_x_offset, self.pull_out_z_offset + self.place_z_offset)
        place = self._pose_from_center(target, self.grasp_x_offset, self.place_z_offset)

        planner.move_minco(
            self,
            [self.flying_hand_initial_pose, pre, grasp],
            times=[self.initial_to_pre_grasp_seconds, self.pre_grasp_to_grasp_seconds],
            save_freq=save_freq,
        )
        self.set_flying_hand_gripper(self.flying_hand_config["gripper"]["close_qpos"], is_grasp=True)
        planner.hold(self, grasp, self._seconds_to_steps(self.grasp_hold_seconds), save_freq=save_freq)
        planner.move_minco(
            self,
            [grasp, pull, place_pre, place],
            times=[self.grasp_to_pull_out_seconds, self.grasp_to_place_seconds, self.pre_grasp_to_grasp_seconds],
            save_freq=save_freq,
            carried_actor=self.fruit,
            carried_pose=self.flying_hand.get_root_pose().inv() * self.fruit.get_pose(),
        )
        self.set_flying_hand_gripper(self.flying_hand_config["gripper"]["open_qpos"], is_grasp=False)
        planner.hold(self, place, self._seconds_to_steps(self.release_hold_seconds), save_freq=save_freq)
        self.finish_flying_hand_record(save_freq)
        self.info["info"] = {
            "{A}": f"{self.skillet_name}/base{self.skillet_id}",
            "{B}": f"{self.fruit_name}/base{self.fruit_id}",
        }
        return self.info

    def check_success(self):
        target = self.skillet.get_functional_point(0)[:3]
        fruit = self.fruit.get_pose().p
        return (
            self._task_objects_safe()
            and np.linalg.norm(fruit[:2] - target[:2]) < 0.08
            and target[2] - 0.02 < fruit[2] < target[2] + 0.16
            and not self.is_grasping
        )
