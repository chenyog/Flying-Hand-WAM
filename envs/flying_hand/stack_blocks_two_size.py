import numpy as np
import sapien

from envs.utils import *

from .stack_blocks_two import stack_blocks_two


class stack_blocks_two_size(stack_blocks_two):
    block_half_sizes = np.array([
        [0.024, 0.024, 0.024],
        [0.050, 0.050, 0.050],
    ])
    block_masses = [0.05, 0.12]
    block_names = ["small block", "large block"]

    def load_actors(self):
        self._reset_board_slots()
        self.block_colors = np.random.random((2, 3))
        self.small_slot_id = int(np.random.choice(len(self.board_slots)))
        self.large_slot_id = int(np.random.choice(len(self.board_slots)))
        half_y = self.block_half_sizes[:, 1].max() + self.shelf_object_gap
        if self.small_slot_id == self.large_slot_id:
            y, _ = self.board_slots[self.small_slot_id]
            y += np.random.uniform(
                -self.shelf_width / 2 + half_y - min(self.source_y_offsets),
                self.shelf_width / 2 - half_y - max(self.source_y_offsets),
            )
            small_y, large_y = y + self.source_y_offsets[0], y + self.source_y_offsets[1]
        else:
            small_y = np.random.uniform(
                self.board_slots[self.small_slot_id][0] - self.shelf_width / 2 + half_y,
                self.board_slots[self.small_slot_id][0] + self.shelf_width / 2 - half_y,
            )
            large_y = np.random.uniform(
                self.board_slots[self.large_slot_id][0] - self.shelf_width / 2 + half_y,
                self.board_slots[self.large_slot_id][0] + self.shelf_width / 2 - half_y,
            )
        self.small_block = self._create_block(0, self.small_slot_id, small_y)
        self.large_block = self._create_block(1, self.large_slot_id, large_y)
        self.add_prohibit_area(self.small_block, padding=0.06)
        self.add_prohibit_area(self.large_block, padding=0.06)

    def _block_x(self, half):
        return self._board_front_x() - self.shelf_length + half[0]

    def _create_block(self, block, slot_id, y):
        half = self.block_half_sizes[block]
        box = create_box(
            self.scene,
            sapien.Pose([self._block_x(half), y, self.board_slots[int(slot_id)][1] + half[2]], [1, 0, 0, 0]),
            half_size=half.tolist(),
            color=self.block_colors[block],
            name=self.block_names[block],
        )
        box.config["extents"] = (half * 2).tolist()
        box.config["scale"] = [1, 1, 1]
        box.set_mass(self.block_masses[block])
        self.add_task_objects(box)
        return self._place_actor_on_shelf(box, slot_id, y=y)

    def play_once(self):
        save_freq = self.start_flying_hand_record()
        self.target_center = self.large_block.get_pose().p + np.array([
            0.0,
            0.0,
            self.block_half_sizes[:, 2].sum() + self.stack_release_z_offset,
        ])
        self._move_block(self.small_block, self.target_center, save_freq)
        self.finish_flying_hand_record(save_freq)
        self.info["info"] = {"{A}": self.block_names[0], "{B}": self.block_names[1]}
        return self.info

    def check_success(self):
        small, large = self.small_block.get_pose().p, self.large_block.get_pose().p
        return (
            self._task_objects_safe()
            and np.linalg.norm(small - (large + np.array([0.0, 0.0, self.block_half_sizes[:, 2].sum()]))) < 0.08
            and abs(small[2] - large[2] - self.block_half_sizes[:, 2].sum()) < 0.04
            and not self.is_grasping
        )
