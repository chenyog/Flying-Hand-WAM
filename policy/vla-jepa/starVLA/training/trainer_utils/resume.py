"""Distributed training-state checkpoints, separate from inference weights."""
import json
import os
from pathlib import Path
import shutil


class ResumeMixin:
    def state_dict(self):
        return {"completed_steps": self.completed_steps}

    def load_state_dict(self, state):
        self.completed_steps = int(state["completed_steps"])

    def _init_resume(self):
        self.state_interval = int(self.config.trainer.save_interval)
        if self.state_interval < 0:
            raise ValueError("save_interval must be non-negative")
        # The scheduler is stepped manually and is not passed to prepare().
        self.accelerator.register_for_checkpointing(self.lr_scheduler, self)
        path = self.config.trainer.get("resume_from_checkpoint")
        if self.config.trainer.get("is_resume", False) and not path:
            raise ValueError("Set trainer.resume_from_checkpoint to a full state directory")
        if path:
            self._load_checkpoint(path)

    def _load_checkpoint(self, checkpoint_path):
        path = Path(checkpoint_path).resolve()
        if not path.is_dir() or not (path / "trainer_state.json").is_file():
            raise ValueError(f"Not a complete resume state directory: {path}")
        metadata = json.loads((path / "trainer_state.json").read_text())
        if metadata["world_size"] != self.accelerator.num_processes:
            raise ValueError("Resume requires the same GPU/process count")
        self.accelerator.load_state(str(path))
        self.accelerator.print(f"Resumed training at step {self.completed_steps}: {path}")

    def _save_resume_state(self):
        root = Path(self.checkpoint_dir) / "state"
        target = root / f"step_{self.completed_steps:06d}"
        staging = root / f"{target.name}.incomplete"
        # Finalization can coincide with a periodic save; do not rewrite or
        # temporarily remove an already committed checkpoint at this step.
        if (target / "trainer_state.json").is_file():
            self.accelerator.wait_for_everyone()
            return
        if self.accelerator.is_main_process:
            root.mkdir(parents=True, exist_ok=True)
            if staging.exists():
                shutil.rmtree(staging)
        self.accelerator.wait_for_everyone()
        # All ranks must participate to save DeepSpeed optimizer/model shards.
        self.accelerator.save_state(str(staging))
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            (staging / "trainer_state.json").write_text(json.dumps({
                "completed_steps": self.completed_steps,
                "world_size": self.accelerator.num_processes,
                "data_resume": "restart_iterator",
            }))
            if target.exists():
                shutil.rmtree(target)
            os.replace(staging, target)
            for old in root.iterdir():
                if (old != target and old.is_dir() and not old.is_symlink()
                        and old.name.startswith("step_")
                        and old.name[5:].isdigit()
                        and (old / "trainer_state.json").is_file()):
                    shutil.rmtree(old)
        self.accelerator.wait_for_everyone()
