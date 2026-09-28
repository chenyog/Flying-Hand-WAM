# Flying-Hand direct-clean training

## Resume training

The Flying-Hand `train_starvla.py` entrypoint saves inference weights according
to `trainer.save_interval`, and saves full Accelerate/DeepSpeed state at the
same save event, without changing the original save frequency. Only the latest
complete `checkpoints/state/step_XXXXXX` directory is retained; periodic model
weights are not pruned. A final state is saved at normal training completion,
even when `save_interval` is zero (which disables periodic saves).

```bash
RESUME_FROM=/absolute/path/to/run/checkpoints/state/step_001091 \
NUM_GPUS=8 MAX_STEPS=5455 \
bash policy/vla-jepa/scripts/train_flying_hand.sh
```

`MAX_STEPS` is the total target, not additional steps. Keep the original model,
dataset, batch size, GPU count and scheduler configuration when resuming. A new
timestamped output run is created unless `RUN_ID` is specified. W&B likewise
starts a new run, with metrics continuing from the restored training step.
The state restores model, optimizer, scheduler, RNG and completed steps. The
data iterator restarts: sample order and worker-side augmentation are not
guaranteed to match an uninterrupted run bit-for-bit. Older weight-only `.pt`
files cannot be used as full states; use `PRETRAINED_CHECKPOINT` for a weight-only
warm start. `RESUME_FROM` takes precedence over pretrained weights.

CPU integration test (no model download):

```bash
PYTHONPATH=policy/vla-jepa policy/vla-jepa/.venv/bin/python -m unittest discover -s policy/vla-jepa/tests -p test_resume.py
```

## Launch

Use `scripts/train_flying_hand.sh` from any working directory. It launches eight
processes by default, creates a UTC timestamped run under `policy/vla-jepa/runs`,
and saves the effective configuration, action statistics, console log, JSON
metrics and checkpoints there. W&B defaults to **online**; log in beforehand or
set `WANDB_MODE=offline` / `WANDB_MODE=disabled`. `WANDB_PROJECT` and
`WANDB_ENTITY` override the YAML values. `--trackers '[json]'` removes W&B.
The existing LIBERO, co-pretraining and SimplerEnv launchers/configs follow the
same online-by-default rule; an explicit environment override still wins.

```bash
# Only validate actual data; no model load, download, GPU training or W&B run.
bash policy/vla-jepa/scripts/train_flying_hand.sh --dry-run

# The launcher defaults to policy/vla-jepa/.venv and these local checkpoints.
VLM_PATH=/data/Flying-Hand-WAM/policy/vla-jepa/checkpoints/Qwen3-VL-2B-Instruct \
VJEPA_PATH=/data/Flying-Hand-WAM/policy/vla-jepa/checkpoints/vjepa2-vitl-fpc64-256 \
PRETRAINED_CHECKPOINT=/data/Flying-Hand-WAM/policy/vla-jepa/checkpoints/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt \
BATCH_SIZE=2 NUM_WORKERS=4 MAX_STEPS=30000 WARMUP_STEPS=1500 \
bash policy/vla-jepa/scripts/train_flying_hand.sh
```

Other environment overrides: `DATA_ROOT` (the dataset directory containing
`meta/` and `data/`), `NUM_GPUS`, `MASTER_PORT`, `CONFIG_FILE`, `RUN_ROOT`,
`RUN_ID`, and optional `PRETRAINED_CHECKPOINT`. Extra `--key.path value` flags
are forwarded to the trainer, after the script defaults. The default global
batch is 16 = 8 GPUs × 2 examples, with gradient accumulation 1. RTX 4090 48 GB
上已用 ZeRO-2、每卡 batch 1、全局 batch 8 完成一步
forward/backward/optimizer 测试并保存最终 checkpoint；该 smoke test 只证明训练链
可运行，不代表默认 batch 2 或收敛效果。

## Data and timing

The read-only `flying_hand_datasets` adapter consumes the unified LeRobot v2.1
export directly. It handles Parquet `{bytes, path}` images, so no MP4 conversion,
`modality.json`, or GR00T index cache is required. The `flying_hand` configuration,
mixture, and embodiment tag are also registered alongside the GR00T definitions.
The existing GR00T video reader is unchanged.

The configured data has 1,000 episodes across 10 physical tasks; the 706
`tasks.jsonl` entries are distinct instruction strings, not 706 physical tasks.
Instructions come from each row's `task_index`, which the shared converter sets
to that episode's first `seen` instruction. No instruction resampling occurs.

State is `[x, y, z, yaw, grasp]` from actual state at time t. Action is the same
five fields from target state at t+1, already shifted by the shared converter.
The adapter does not shift again. Actions use exact dataset-wide q01/q99
normalization on x/y/z/yaw, clipped to [-1, 1]; grasp stays in [0, 1], and state
stays in physical units. Saved statistics contain the same quantiles and mask
used by `BaseFramework.unnormalize_actions`. Statistics read only numeric
Parquet columns; image payloads are decoded only for sampled episodes.
The decoder uses a binary-dimension mask for the last grasp field; it supports
both Flying-Hand's 5D actions and the existing 7D formats. Keeping state in
physical units follows the existing `Libero4in1DataConfig`, whose transform
normalizes only actions. The effective diffusion repeat count is explicitly
set to 8 in `trainer.repeated_diffusion_steps`, the key read by the model.

Both cameras are RGB, ordered head then wrist. Policy images use the current
frame, resized to 224×224. JEPA supervision uses 8 frames per camera, resized to
256×256, yielding `[2, 8, 256, 256, 3]`. At 20 Hz these are t through t+7, spaced
50 ms apart: 350 ms from first to last, or eight 50 ms frame intervals. An
8-command action chunk targets t+1 through t+8 (0.4 seconds of future commands).
JEPA tubelet size 2 produces four temporal tokens per spatial location. Future
frames supply the world-model training objective; the policy input uses only
the current images and state. At an episode boundary, frames and absolute
commands repeat their final values, without crossing into the next episode.

## Weights and validation limits

Qwen3-VL-2B-Instruct and vjepa2-vitl-fpc64-256 must already exist locally; the
launcher checks their paths. 本机权重分别位于 `checkpoints/Qwen3-VL-2B-Instruct`、
`checkpoints/vjepa2-vitl-fpc64-256`，官方 VLA-JEPA warm-start 位于
`checkpoints/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt`。
下载固定 revision 分别为 Qwen
`89644892e4d85e24eaac8bacfd4f463576704203`、V-JEPA2
`b3c1679b7c34d3255ef3547f27c7b226aefab26f`、VLA-JEPA
`db32b4a4457c39921869f667ac76cf8a8fc39da8`。
Without
`PRETRAINED_CHECKPOINT`, training starts from those base encoders with newly
initialized JEPA predictor and 5D action head.

For a compatible VLA-JEPA pretraining checkpoint, set `PRETRAINED_CHECKPOINT`.
The default partial load transfers `qwen_vl_interface,vj_predictor` only; the
original 7D action head is incompatible with Flying-Hand's 5D head. The YAML
keeps the original 28 added action tokens for Qwen checkpoint compatibility.
The base model version, vocabulary, predictor dimensions, and 8-frame clip
configuration must match the checkpoint. Loading remains strict; this is a
warm start, not an exact optimizer/distributed-state resume.

The model requires this repository's Transformers 4.57.0, Qwen-VL utilities,
FlashAttention, and compatible PyTorch/Accelerate/DeepSpeed dependencies. Use a
dedicated environment rather than changing other policies' shared environment.
The data dry run only requires existing NumPy, PyArrow, Pillow, PyTorch,
Accelerate and OmegaConf. It checks real start/end samples, shapes, finite
5D vectors, nonempty instruction lookup, RGB image views and terminal padding;
it does not establish model import, forward/backward correctness or training
convergence.
