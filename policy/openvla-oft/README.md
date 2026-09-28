# OpenVLA-OFT：Flying-Hand direct-clean 联合训练

本目录新增 `flying_hand_lerobot` 训练入口，直接读取项目已经转换好的
`data/lerobot/flying_hand_direct_all_tasks`（LeRobot v2.1 Parquet、内嵌 RGB 图片），
复用现有 `finetune.py`、`RLDSBatchTransform`、LoRA 和 L1 连续动作头。
不需要重新生成 RLDS，也不依赖 LeRobot Python API。原有 RLDS 数据入口仍是默认入口。

## 数据与动作约定

默认联合使用以下全部十类任务：`blocks_ranking_rgb`、`blocks_ranking_size`、
`move_bottle`、`move_can_pot`、`place_can_basket`、`place_fruit_skillet`、
`shake_bottle`、`stack_blocks_two`、`stack_blocks_two_size`、`thread_ring_rod`。
当前数据共 1,000 个 episode、394,802 帧。`meta/tasks.jsonl` 中的 706 个 task index
是自然语言指令类别，不是 706 种环境任务。

| 输入/输出 | 约定 |
|---|---|
| proprio | `observation.state = actual_state[t]`，5 维 `[x,y,z,yaw,grasp]` |
| action | 已转换为 `target_state[t+1]`，末帧重复 `target_state[-1]`，5 维绝对目标 |
| head | `observation.images.head_camera`，RGB、360×480 |
| wrist | `observation.images.wrist_camera`，RGB、480×640 |
| instruction | 转换时原始 episode JSON 的 `seen[0]`，由帧的 `task_index` 查找 |
| frequency/chunk | 20 Hz；每个输入预测连续 8×5 个目标，对应 0.4 秒命令窗口 |

读取器不会再次将动作向前移动，也不生成位移增量，不反转或二值化 grasp。
episode 尾部不足 8 步时重复最后一个绝对目标，绝不跨 episode 拼接。
两个相机按 head、wrist 顺序经过预训练 processor 的同一个图像变换。
提示词使用现有 OpenVLA 模板，现有模板会将指令转换为小写。

训练在全部 episode 的全部帧上计算 action/proprio 每一维的 1%/99% 分位数，
将五维都映射并裁剪至 `[-1,1]`；常量维映射为零。
统计保存在 run/checkpoint 的 `dataset_statistics.json`，key 为
`flying_hand_direct_all_tasks`。部署时必须使用同一统计对状态归一化、对动作反归一化，
不能直接把模型输出的归一化值作为物理目标。预训练权重原有的其他机器人统计不适用。

读取器无限重复数据；每轮先打乱 episode，然后打乱 episode 内帧，按分布式 rank/worker
切分 episode。每轮同一 episode 只属于一个 rank/worker，所有帧都会被覆盖。
这是按帧采样，较长任务会贡献更多训练帧；没有做十任务等权重重采样。
当前直接读取路径不提供图像增强或验证集划分，因此入口明确要求
`--image_aug False --use_val_set False`，不把训练样本用作验证集。

## 环境与权重前提

本仓库副本原本缺少 `prismatic/models/` 整个目录。此次已从官方 OpenVLA-OFT
commit `e4287e94541f459edc4feabc4e181f537cd569a8` 原样恢复全部 32 个文件，包含
动作头、projector、backbone 和 prompt builder，并逐文件比对一致。
来源与许可证见 [UPSTREAM_MODELS.md](UPSTREAM_MODELS.md)。包装脚本仍检查关键源码存在。

完整训练环境请遵循本目录 [original_SETUP.md](original_SETUP.md) 与
[pyproject.toml](pyproject.toml)，包括专用 OpenVLA-OFT transformers 分支、对应的
PyTorch/timm/PEFT/draccus/accelerate/flash-attn 依赖。原训练入口仍导入 TensorFlow/RLDS
工具，因此完整训练仍需原有 TensorFlow/dlimp 依赖，即使本读取器不使用它们。
额外 Parquet 依赖 `pyarrow>=14` 已列入 pyproject。
不要直接覆盖其他策略共用环境的依赖。本策略使用
`policy/openvla-oft/.venv`；`train_flying_hand.sh` 默认选择该解释器，仍可通过
`PYTHON` 显式覆盖。该环境已安装项目依赖以及与 Torch 2.2 匹配的
FlashAttention 2.5.5。为避免上游无上限依赖组合失效，项目明确固定
`tensorflow-metadata==1.15.0`、`wandb==0.19.1` 和 `diffusers==0.33.1`。

设置 `VLA_PATH` 为完整本地 OpenVLA 预训练模型目录（含 config、processor/tokenizer、权重）；
本机下载的官方 `openvla/openvla-7b` 位于 `policy/openvla-oft/checkpoints/openvla-7b`，
固定 revision 为 `47a0ec7fc4ec123775a391911046cf33cf9ed83f`。
包装脚本不接受缺失的本地目录，也不会主动下载模型。默认只训练 LoRA，checkpoint 保存
adapter、连续动作头、proprio projector 和归一化统计，默认不在线合并 7B 权重。
部署需要基模型加这些组件，或按原仓库流程离线合并。

## 先检查数据

在仓库根目录运行（不加载模型、不启动 GPU 训练、不创建 W&B run）：

```bash
bash policy/openvla-oft/train_flying_hand.sh --dry-run

policy/openvla-oft/.venv/bin/python policy/openvla-oft/scripts/check_flying_hand_data.py \
  --data-root data/lerobot/flying_hand_direct_all_tasks \
  --source-root data/flying_hand_direct

policy/openvla-oft/.venv/bin/python policy/openvla-oft/scripts/test_flying_hand_data.py
```

默认预检确认所有 1,000 个 Parquet 文件存在，并检查每个任务首个 episode 的全部
低维帧/时间戳以及首尾两帧图像。`--source-root` 进一步对十条真实原始 episode 的完整
state/action 和指令做精确比对，并逐像素比较首尾 RGB 与项目标准 JPEG 解码器的结果。
任务/episode 排序沿用现有转换器的按任务名称排序、每任务 episode 从零开始的约定。
可加 `--all-episodes` 验证全部 episode 的低维数据和每个 episode 首尾图像，成本较高。

## 八卡训练

环境和本地权重齐备后，在仓库根目录运行：

```bash
VLA_PATH=/data/Flying-Hand-WAM/policy/openvla-oft/checkpoints/openvla-7b \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WANDB_PROJECT=flying-hand-openvla-oft \
bash policy/openvla-oft/train_flying_hand.sh
```

默认 8 个进程、每卡 batch 5、梯度累积 1，因此全局 batch 40；LoRA rank 32，
学习率 5e-4、80,000 步、64,000 步衰减、每 1,000 步保存。
每卡 batch 5 是在 8 张 RTX 4090 48 GB 上按稳定样本吞吐率扫描后选择的默认值；
它验证了训练路径的短时前向、反向和优化器更新，但仍不等同于完整训练收敛验证。
可设置 `NPROC_PER_NODE`、`BATCH_SIZE`、`MAX_STEPS`、`DECAY_STEPS`、`SAVE_FREQ`、
`DATA_ROOT`、`RUN_ROOT`、`RUN_ID`；额外 finetune 参数可追加到脚本末尾。
run id 默认 `flying_hand_direct_clean_<UTC时间戳>_<进程号>`，所有 rank 共用同一 run id，
输出默认位于本策略目录 `checkpoints/<run_id>`。已有同名 run 会拒绝覆盖。

`WANDB_MODE` 默认 `online`，使用已登录账户；团队项目可额外设置 `WANDB_ENTITY`。
需要本地记录时显式设置 `WANDB_MODE=offline`。数据 dry-run 不接触 W&B。
脚本导出 `OPENVLA_ROBOT_PLATFORM=FLYING_HAND`，优先于模型路径中的 libero/aloha 字样，
保证模型、损失和数据始终使用 5 维动作、5 维状态和 8 步 chunk。

## 验证边界与部署

七项轻量测试覆盖末端绝对目标保持、归一化/常量维、显式平台优先级、双相机 collator、
batch=1 proprio 形状、多 rank 分片和多轮无限迭代、真实上游动作头和 projector 的
5D 前向/反向，以及提示词监督严格保留 40 个动作 token（不误用解码字符串字符数）。
本地还对全部 1,000 个 episode 的 394,802 帧完成了低维数值/20 Hz 时间戳校验与
全数据归一化统计计算；十条原始 episode 的 state/action/instruction 与首尾 RGB 对照通过。
通过这些测试代表数据接口和结构通过，不代表 7B 前向/反向、八卡 DDP、显存或任务成功率已验证。
本次没有启动训练或下载权重。

这次新增的是训练支持；原 `deploy_policy.py` 的 ALOHA 适配不是 Flying-Hand 推理适配。
Flying-Hand 推理需加载上述 5D 模型组件，以同样双相机/proprio/prompt 生成 8×5 绝对目标，
并由环境以 20 Hz 逐步执行。8 步预测窗口与每次执行几步后重规划是不同设置；
更改预测 chunk 长度需同步模型常量、权重和训练数据，不能只改执行频率。
