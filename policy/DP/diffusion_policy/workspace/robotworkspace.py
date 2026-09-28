if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import contextlib

import tqdm, random
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class RobotWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    @property
    def distributed(self):
        return dist.is_available() and dist.is_initialized()

    @property
    def is_main_process(self):
        return not self.distributed or dist.get_rank() == 0

    def _model_module(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def save_checkpoint(self, path=None, *args, **kwargs):
        if isinstance(self.model, DDP):
            wrapped_model = self.model
            self.model = wrapped_model.module
            try:
                kwargs.setdefault("use_thread", False)
                return super().save_checkpoint(path=path, *args, **kwargs)
            finally:
                self.model = wrapped_model
        return super().save_checkpoint(path=path, *args, **kwargs)

    def _init_wandb(self, cfg):
        """Create the sole rank-0 W&B run for this training job."""
        if not self.is_main_process or str(cfg.logging.mode) == "disabled":
            return None
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B logging is enabled; install the DP dependencies or set "
                "logging.mode=disabled."
            ) from exc
        run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **OmegaConf.to_container(cfg.logging, resolve=True),
        )
        run.config.update({"output_dir": str(self.output_dir)})
        return run

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        if int(cfg.training.checkpoint_every) < 0:
            raise ValueError("checkpoint_every must be non-negative")
        resume_payload = None

        # resume training
        if cfg.training.resume or cfg.training.get("resume_from"):
            lastest_ckpt_path = pathlib.Path(cfg.training.get("resume_from") or
                                            self.get_checkpoint_path())
            if not lastest_ckpt_path.is_file():
                raise FileNotFoundError(lastest_ckpt_path)
            print(f"Resuming from checkpoint {lastest_ckpt_path}")
            resume_payload = self.load_checkpoint(path=lastest_ckpt_path, include_keys=self.include_keys)
            if "optimizer" not in resume_payload["state_dicts"]:
                raise ValueError("Inference weights cannot resume training; use checkpoints/latest.ckpt")

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        if resume_payload is None:
            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema:
                self.ema_model.set_normalizer(normalizer)

        # Compute the scheduler horizon from the effective number of batches.
        # This also remains correct when a debug/config override caps an epoch.
        train_batches_per_epoch = len(train_dataloader)
        if cfg.training.max_train_steps is not None:
            train_batches_per_epoch = min(
                train_batches_per_epoch,
                int(cfg.training.max_train_steps),
            )
        if train_batches_per_epoch < 1:
            raise ValueError(
                "Training dataloader has no complete batches. Reduce the per-GPU "
                "batch size or provide more training sequences."
            )
        num_training_steps = (
            train_batches_per_epoch * int(cfg.training.num_epochs)
        ) // int(cfg.training.gradient_accumulate_every)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=num_training_steps,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
        resume_extra = resume_payload.get("resume_extra") if resume_payload else None
        if resume_extra:
            lr_scheduler.load_state_dict(resume_extra["scheduler"])
            for group, lr in zip(self.optimizer.param_groups, lr_scheduler.get_last_lr()):
                group["lr"] = lr
            if ema is not None:
                ema.optimization_step = resume_extra["ema_step"]
            for name, loader in (("train", train_dataloader), ("val", val_dataloader)):
                sampler = loader.sampler
                sampler = getattr(sampler, "batch_sampler", sampler)
                saved_rng = resume_extra.get("samplers", {}).get(name)
                if sampler.rng is not None and saved_rng is not None:
                    sampler.rng.bit_generator.state = saved_rng

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        env_runner = None

        # W&B owns one run per training job.  Non-main DDP ranks never import,
        # initialize, or write to W&B.
        wandb_run = self._init_wandb(cfg)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"),
                                             **cfg.checkpoint.topk)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        if self.distributed:
            if device.type != "cuda" or device.index is None:
                raise ValueError("DDP training requires training.device=cuda:<local_rank>")
            self.model = DDP(
                self.model,
                device_ids=[device.index],
                output_device=device.index,
                # ConditionalUnet1D contains optional local-conditioning
                # layers; this policy uses global conditioning, so those
                # parameters legitimately have no gradient.
                find_unused_parameters=True,
            )

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        # A shared output directory is used by all ranks.  Only rank zero may
        # append to the JSON log; concurrent appenders would corrupt JSONL.
        logger_context = JsonLogger(log_path) if self.is_main_process else contextlib.nullcontext(NullJsonLogger())
        with logger_context as json_logger:
            if resume_extra:
                states = resume_extra["rng"]
                rank = dist.get_rank() if self.distributed else 0
                if len(states) != (dist.get_world_size() if self.distributed else 1):
                    raise ValueError("Resume requires the same process count")
                rng = states[rank]
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"].cpu())
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state(rng["cuda"].cpu())
                for name, parameter in self._model_module().named_parameters():
                    grad = resume_extra.get("gradients", {}).get(name)
                    if grad is not None:
                        parameter.grad = grad.to(device)
            for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self._model_module().obs_encoder.eval()
                    self._model_module().obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # compute loss
                        raw_loss = (
                            self.model(batch)
                            if self.distributed
                            else self._model_module().compute_loss(batch)
                        )
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if (self.global_step % cfg.training.gradient_accumulate_every == 0):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self._model_module())

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (train_batches_per_epoch - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            json_logger.log(step_log)
                            if wandb_run is not None:
                                wandb_run.log(step_log, step=self.global_step)
                            self.global_step += 1

                        if (cfg.training.max_train_steps
                                is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss_tensor = torch.tensor(
                    np.mean(train_losses) if train_losses else float("nan"),
                    device=device,
                    dtype=torch.float32,
                )
                if self.distributed:
                    dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                    train_loss_tensor /= dist.get_world_size()
                train_loss = train_loss_tensor.item()
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self._model_module()
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if ((self.epoch + 1) % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                loss = (
                                    self.model(batch)
                                    if self.distributed
                                    else self._model_module().compute_loss(batch)
                                )
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps
                                        is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss_tensor = torch.stack(val_losses).mean()
                            if self.distributed:
                                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                                val_loss_tensor /= dist.get_world_size()
                            val_loss = val_loss_tensor.item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if self.is_main_process and (
                        (cfg.training.checkpoint_every > 0 and
                         (self.epoch + 1) % cfg.training.checkpoint_every == 0)
                        or self.epoch + 1 == cfg.training.num_epochs):
                    # Keep periodic checkpoints inside the timestamped Hydra
                    # output directory so independent runs cannot overwrite
                    # one another.
                    periodic_ckpt_path = pathlib.Path(self.output_dir).joinpath(
                        "checkpoints",
                        f"epoch={self.epoch + 1:04d}.ckpt",
                    )
                    self.save_checkpoint(
                        periodic_ckpt_path,
                        use_thread=False,
                        exclude_keys=("optimizer",),
                        include_keys=(),
                    )
                    if cfg.checkpoint.topk.monitor_key in step_log:
                        metric_dict = {
                            key.replace("/", "_"): value
                            for key, value in step_log.items()
                        }
                        metric_dict["epoch"] = self.epoch + 1
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path, use_thread=False,
                                                 exclude_keys=("optimizer",), include_keys=())

                if self.distributed:
                    dist.barrier(device_ids=[device.index])

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                json_logger.log(step_log)
                if wandb_run is not None:
                    wandb_run.log(step_log, step=self.global_step)
                self.global_step += 1
                self.epoch += 1
                state_every = int(cfg.training.checkpoint_every)
                if ((state_every > 0 and self.epoch % state_every == 0)
                        or self.epoch == cfg.training.num_epochs):
                    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
                           "torch": torch.get_rng_state(),
                           "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
                    rng_states = [None] * (dist.get_world_size() if self.distributed else 1)
                    if self.distributed:
                        dist.all_gather_object(rng_states, rng)
                    else:
                        rng_states[0] = rng
                    if self.is_main_process:
                        target = self.get_checkpoint_path()
                        staging = target.with_suffix(".incomplete")
                        self.save_checkpoint(staging, use_thread=False)
                        import dill
                        payload = torch.load(staging, pickle_module=dill, map_location="cpu")
                        payload["resume_extra"] = {
                            "scheduler": lr_scheduler.state_dict(),
                            "ema_step": ema.optimization_step if ema is not None else 0,
                            "rng": rng_states,
                            "gradients": {name: p.grad.detach().cpu() for name, p in
                                          self._model_module().named_parameters() if p.grad is not None},
                        }
                        payload["resume_extra"]["samplers"] = {}
                        for name, loader in (("train", train_dataloader), ("val", val_dataloader)):
                            sampler = getattr(loader.sampler, "batch_sampler", loader.sampler)
                            payload["resume_extra"]["samplers"][name] = (
                                sampler.rng.bit_generator.state if sampler.rng is not None else None)
                        torch.save(payload, staging, pickle_module=dill)
                        os.replace(staging, target)
                    if self.distributed:
                        dist.barrier()

        if wandb_run is not None:
            wandb_run.finish()


class BatchSampler:

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


class DistributedBatchSampler:
    """Shard complete batches across DDP ranks.

    ``RobotImageDataset`` accepts a numpy array as one index and constructs the
    whole batch in its preallocated buffers.  Sharding individual samples with
    ``DistributedSampler`` would therefore break the dataset contract and
    cause duplicate work.  This wrapper first creates the same global batch
    order on every rank, drops the incomplete tail, then gives each rank every
    ``world_size``-th complete batch.
    """

    def __init__(self, batch_sampler):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("DistributedBatchSampler requires an initialized process group")
        self.batch_sampler = batch_sampler
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.num_batch = len(batch_sampler) // self.world_size

    def __iter__(self):
        batches = list(iter(self.batch_sampler))
        usable = self.num_batch * self.world_size
        for batch in batches[self.rank:usable:self.world_size]:
            yield batch

    def __len__(self):
        return self.num_batch


class NullJsonLogger:
    """Drop-in logger used by non-main DDP ranks."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def log(self, data):
        pass


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
):
    batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)
    if dist.is_available() and dist.is_initialized():
        batch_sampler = DistributedBatchSampler(batch_sampler)

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # PyTorch rejects persistent_workers when num_workers == 0.  More
        # importantly, zero workers keeps the disk-backed dataset and its
        # reusable batch buffers inside the rank process only.
        persistent_workers=bool(persistent_workers and num_workers > 0),
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
