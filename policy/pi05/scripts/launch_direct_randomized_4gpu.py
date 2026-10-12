"""Launch four-GPU randomized training with sample-budget-scaled parameters."""
import argparse
import ctypes
import datetime
import hashlib
import json
import netrc
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=48, help='Global batch size, divisible by four')
    parser.add_argument('--recompute-stats', action='store_true')
    parser.add_argument('--resume-run', help='Existing run directory name to resume')
    args = parser.parse_args()
    if args.batch_size <= 0 or args.batch_size % 4:
        parser.error('--batch-size must be positive and divisible by four')
    batch_size = args.batch_size
    warmup_steps = (126 * 64 + batch_size - 1) // batch_size
    # Full train_state contains optimizer/sharding state (~42 GiB here). Keep
    # the latest state, but do not synchronously write it at every warmup-sized
    # inference checkpoint.
    save_interval = 1680
    keep_period = 1680
    root = Path('/data/Flying-Hand-WAM')
    project = root / 'policy/pi05'
    dataset = root / 'data/lerobot/flying_hand_direct_randomized_all_tasks'
    info = json.loads((dataset / 'meta/info.json').read_text())
    if info['total_episodes'] != 1000 or float(info['fps']) != 20:
        raise ValueError('Expected 1000 randomized episodes at 20 Hz')
    env = os.environ.copy()
    cache = root / 'data/lerobot/.hf_datasets_cache'
    cache.mkdir(exist_ok=True)
    env.update(HF_DATASETS_CACHE=str(cache), HF_LEROBOT_HOME=str(root / 'data/lerobot'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if args.recompute_stats:
        stats_env = dict(env, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
        print('RECOMPUTE_STATS', json.dumps(dict(dataset=str(dataset), cache=str(cache), num_workers=32)), flush=True)
        subprocess.run([str(project / '.venv/bin/python'), '-u', 'scripts/compute_norm_stats.py',
                        '--config-name', 'pi05_flying_hand_full', '--dataset-dirs', str(dataset),
                        '--num-workers', '32'], cwd=project, env=stats_env, check=True)
        print('NORMALIZATION_COMPLETE', flush=True)
    asset_id = 'datasets_' + hashlib.sha256(str(dataset.resolve()).encode()).hexdigest()[:12]
    assets = project / 'assets/pi05_flying_hand_full'
    if not (assets / asset_id / 'norm_stats.json').is_file():
        raise FileNotFoundError('Randomized normalization statistics are missing')

    # Do not take GPUs that became occupied during the CPU preparation stage.
    nvml = ctypes.CDLL('libnvidia-ml.so.1')
    if nvml.nvmlInit_v2() != 0:
        raise RuntimeError('Cannot check GPU availability')
    class Memory(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]
    try:
        for index in (4, 5, 6, 7):
            handle = ctypes.c_void_p()
            memory = Memory()
            if nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                raise RuntimeError(f'Cannot inspect GPU {index}')
            if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0 or memory.used > 2 * 1024**3:
                raise RuntimeError(f'GPU {index} is occupied; refusing automatic training launch')
    finally:
        nvml.nvmlShutdown()

    if not env.get('WANDB_API_KEY'):
        credentials = netrc.netrc().authenticators('api.wandb.ai')
        if not credentials or not credentials[2]:
            raise RuntimeError('W&B credential unavailable; refusing offline fallback')
        env['WANDB_API_KEY'] = credentials[2]
    env.update(CUDA_VISIBLE_DEVICES='4,5,6,7', HF_LEROBOT_HOME=str(root / 'data/lerobot'),
               WANDB_MODE='online', WANDB_ENTITY='chenyog_rl',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
    env.pop('JAX_PLATFORMS', None)
    run = args.resume_run or ('pi05_flying_hand_direct_randomized_full_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    total_steps = 2 * (int(info['total_frames']) // batch_size)
    # Include transactional old/new state, retained weights, and headroom for
    # the already-running FastWAM job's checkpoint writes on the same disk.
    required_gib = 84 + 12 * (total_steps // keep_period + 2) + 96
    if shutil.disk_usage(project).free < required_gib * 1024**3:
        raise RuntimeError(f'Need at least {required_gib} GiB free for checkpoint budget; training not started')
    command = [str(project / '.venv/bin/python'), '-u', 'scripts/train.py', 'pi05_flying_hand_full',
               '--exp-name', run, '--checkpoint-base-dir', str(project / 'checkpoints'),
               '--fsdp-devices', '4', '--batch-size', str(batch_size), '--num-workers', '32', '--num-epochs', '2',
               '--data.dataset-dirs', str(dataset), '--data.repo-id', dataset.name,
               '--data.assets.assets-dir', str(assets), '--wandb-mode', 'online',
               '--lr-schedule.warmup-steps', str(warmup_steps), '--lr-schedule.decay-steps', str(total_steps),
               '--save-interval', str(save_interval), '--keep-period', str(keep_period)]
    if args.resume_run:
        command.append('--resume')
    print('TRAIN_LAUNCH', json.dumps(dict(run=run, global_batch_size=batch_size, fsdp_devices=4,
          epochs=2, expected_steps=total_steps, warmup_steps=warmup_steps, save_interval=save_interval,
          keep_period=keep_period, dataset=str(dataset), normalization=str(assets / asset_id),
          wandb_entity=env['WANDB_ENTITY'], wandb_project='openpi')), flush=True)
    result = subprocess.run(command, cwd=project, env=env)
    print(f'TRAIN_EXIT_CODE={result.returncode}', flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
