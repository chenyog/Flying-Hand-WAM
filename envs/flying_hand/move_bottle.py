import numpy as np

from envs.utils import *

from ._base_task import FlyingHandBaseTask
from . import planner


class move_bottle(FlyingHandBaseTask):
    pre_grasp_x_offset = -0.55
    grasp_x_offset = -0.09
    pull_out_x_offset = -0.45
    grasp_z_offset = -0.01
    pull_out_z_offset = 0.12
    success_pull_out_x = 0.10
    success_lift_z = 0.03

    def load_actors(self):
        self._reset_board_slots()
        self.target_slot_id = int(np.random.randint(0, len(self.board_slots)))
        self.bottle_model_id = int(np.random.choice(self._get_available_model_ids("001_bottle")))
        self.bottle = self._create_board_actor("001_bottle", self.bottle_model_id, self.target_slot_id)
        self.bottle_initial_x = self.bottle.get_pose().p[0]
        self.bottle_initial_z = self.bottle.get_pose().p[2]
        self.target_actor = self.bottle
        self.arm_tag = ArmTag("right" if self.bottle.get_pose().p[0] > 0 else "left")
        self.add_prohibit_area(self.bottle, padding=0.1)

    def play_once(self):
        save_freq = self.start_flying_hand_record()
        pre = self._get_flying_hand_pose(self.bottle, self.pre_grasp_x_offset)
        grasp = self._get_flying_hand_pose(self.bottle, self.grasp_x_offset, self.grasp_z_offset)
        pull = self._get_flying_hand_pose(self.bottle, self.pull_out_x_offset, self.pull_out_z_offset)

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
            [grasp, pull],
            times=[self.grasp_to_pull_out_seconds],
            save_freq=save_freq,
            carried_actor=self.bottle,
            carried_pose=self.flying_hand.get_root_pose().inv() * self.bottle.get_pose(),
        )
        self.finish_flying_hand_record(save_freq)
        self.info["info"] = {"{A}": f"001_bottle/base{self.bottle_model_id}"}
        return self.info

    def check_success(self):
        pose = self.bottle.get_pose().p
        return self._task_objects_safe() and pose[0] < self.bottle_initial_x - self.success_pull_out_x and pose[2] > self.bottle_initial_z + self.success_lift_z
