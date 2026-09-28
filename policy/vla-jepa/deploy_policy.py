"""Flying-Hand simulator adapter; model dependencies stay in their own venv."""
import atexit
import pickle
from pathlib import Path
import subprocess

import numpy as np

from envs.flying_hand.eval_control import execute_action_chunk, reset_execution_state


class Policy:
    def __init__(self, checkpoint):
        root = Path(__file__).resolve().parent
        self.process = subprocess.Popen(
            [str(root / '.venv/bin/python'), '-u', str(root / 'scripts/inference_flying_hand.py'),
             str(Path(checkpoint).resolve())], cwd=root,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        atexit.register(self.close)
        ready = pickle.load(self.process.stdout)
        if ready.get('status') != 'ready':
            raise RuntimeError(ready)
        print('VLA_JEPA_READY', ready, flush=True)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def predict(self, observation, instruction):
        images = [np.asarray(observation['observation'][name]['rgb'])
                  for name in ('head_camera', 'wrist_camera')]
        for image in images:
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f'Expected HWC uint8 RGB, got {image.shape}, {image.dtype}')
        state = np.asarray(observation['flying_hand']['actual_state'], dtype=np.float32)
        if state.shape != (5,) or not np.isfinite(state).all():
            raise ValueError('Expected finite physical-unit state [5]')
        pickle.dump(dict(images=images, instruction=instruction, state=state), self.process.stdin)
        self.process.stdin.flush()
        result = pickle.load(self.process.stdout)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result['actions']


def get_model(usr_args):
    return Policy(usr_args['checkpoint_path'])


def eval(TASK_ENV, model, observation):
    actions = model.predict(observation, TASK_ENV.get_instruction())
    execute_action_chunk(TASK_ENV, model, actions)


def reset_model(model):
    reset_execution_state(model)
