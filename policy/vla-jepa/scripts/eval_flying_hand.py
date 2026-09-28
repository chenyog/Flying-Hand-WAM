"""Four-GPU Flying-Hand evaluation queue (simulation runs in the root venv)."""
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
TASKS = ['move_bottle', 'place_fruit_skillet', 'shake_bottle', 'place_can_basket',
         'move_can_pot', 'stack_blocks_two', 'stack_blocks_two_size', 'thread_ring_rod',
         'blocks_ranking_rgb', 'blocks_ranking_size']


def worker():
    sys.argv.remove('--worker')
    sys.path[:0] = [str(ROOT), str(ROOT / 'policy'), str(ROOT / 'description/utils')]
    import sapien
    original = sapien.Scene.__init__
    def init(self, systems=None):
        if systems is None:
            systems = [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem('cuda:0')]
        original(self, systems)
    sapien.Scene.__init__ = init
    print('RENDER_DEVICE', sapien.Device('cuda:0').pci_string, flush=True)
    import random
    import numpy as np
    import torch
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    from script import eval_policy
    args = eval_policy.parse_args_and_config()
    sys.argv = [sys.argv[0]]
    eval_policy.main(args)


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--episodes', default=100, type=int)
    parser.add_argument('--tasks', nargs='+', default=TASKS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / 'manifest.json', dict(checkpoint=args.checkpoint, gpus=args.gpus,
         tasks=args.tasks, episodes_per_task=args.episodes, task_config='flying_hand_direct_clean',
         instruction_type='seen', video=False, seed=0))
    save(args.output / 'manager.status.json', dict(status='running', pid=os.getpid()))
    jobs = queue.Queue()
    for task in args.tasks:
        jobs.put(task)
    def run_gpu(gpu):
        rows = []
        while True:
            try:
                task = jobs.get_nowait()
            except queue.Empty:
                return rows
            folder = args.output / task
            folder.mkdir(exist_ok=False)
            cmd = [str(ROOT / '.venv/bin/python'), '-u', str(Path(__file__).resolve()),
                   '--worker', '--eval_output_dir', str(folder), '--overrides']
            overrides = dict(policy_name='vla-jepa', task_name=task,
                task_config='flying_hand_direct_clean', ckpt_setting=Path(args.checkpoint).stem,
                checkpoint_path=args.checkpoint, seed=0, eval_num_episodes=args.episodes,
                eval_video_log=False, instruction_type='seen', expert_data_num=0, device='cuda:0')
            for key, value in overrides.items():
                cmd.extend(['--' + key, str(value)])
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       PYTHONHASHSEED='0', PYTHONUNBUFFERED='1')
            with (folder / 'eval.log').open('w') as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                save(folder / 'status.json', dict(status='running', pid=proc.pid, gpu=gpu))
                rc = proc.wait()
            text = re.sub(r'\x1b\[[0-9;]*m', '', (folder / 'eval.log').read_text(errors='replace'))
            matches = re.findall(r'Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)', text)
            records, previous = [], 0
            for successes, count, seed in matches:
                records.append(dict(episode=int(count)-1, seed=int(seed), success=int(successes)>previous))
                previous = int(successes)
            save(folder / 'episodes.json', records)
            complete = rc == 0 and len(records) == args.episodes and (folder / '_result.txt').exists()
            row = dict(task=task, gpu=gpu, returncode=rc, completed_episodes=len(records),
                       successes=previous, status='complete' if complete else 'failed')
            save(folder / 'status.json', row)
            print(row, flush=True)
            rows.append(row)
            if not complete:
                return rows
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus.split(','))) as pool:
        rows = [row for group in pool.map(run_gpu, args.gpus.split(',')) for row in group]
    complete = len(rows) == len(args.tasks) and all(r['status']=='complete' for r in rows)
    save(args.output / 'summary.json', dict(complete=complete, tasks=rows,
         successes=sum(r['successes'] for r in rows), episodes=sum(r['completed_episodes'] for r in rows)))
    save(args.output / 'manager.status.json', dict(status='complete' if complete else 'failed', finished_at=time.time()))


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        main()
