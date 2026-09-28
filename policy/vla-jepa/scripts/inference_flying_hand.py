"""Private pipe-based inference child, not a network pickle service."""
import os
import pickle
import sys
from pathlib import Path

output = sys.stdout.buffer
sys.stdout = sys.stderr  # Keep third-party prints out of the private protocol.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def send(value):
    pickle.dump(value, output)
    output.flush()


def main():
    import ctypes
    import signal
    import random
    import numpy as np
    import torch
    from PIL import Image
    from starVLA.model.framework.base_framework import baseframework

    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != parent:
        return
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = baseframework.from_pretrained(sys.argv[1]).to('cuda:0').eval()
    stats = model.get_action_stats(norm_stats=model.norm_stats)
    if stats['mask'] != [True, True, True, True, False]:
        raise ValueError(f'Unexpected action mask: {stats["mask"]}')
    size = int(model.config.datasets.vla_data.get('resolution_size', 224))
    send(dict(status='ready', checkpoint=sys.argv[1], image_size=size,
              camera_order=['head_camera', 'wrist_camera'], color='RGB', state='physical_units'))
    while True:
        try:
            request = pickle.load(sys.stdin.buffer)
        except EOFError:
            break
        images = [Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)
                  for rgb in request['images']]
        prediction = model.predict_action(batch_images=[images],
            instructions=[request['instruction']], state=request['state'][None, None, :])
        actions = model.unnormalize_actions(prediction['normalized_actions'][0], stats)
        if actions.shape != (8, 5) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid action chunk: {actions.shape}')
        send(dict(actions=actions))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        send(dict(error=traceback.format_exc()))
        raise
