import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    overwrite: bool,
    resume: bool,
) -> tuple["SplitCheckpointManager", bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                                  "to indicate how to handle it.")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir / "state",
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=None,
            create=True,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. in this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    legacy_step = None
    if resuming and not mngr.all_steps():
        legacy_steps = [int(p.name) for p in checkpoint_dir.iterdir()
                        if p.name.isdigit() and (p / "train_state").is_dir()]
        legacy_step = max(legacy_steps, default=None)
    if resuming and not mngr.all_steps() and legacy_step is None:
        if any(p.name.isdigit() for p in checkpoint_dir.iterdir()):
            mngr.close()
            raise ValueError("Only inference checkpoints exist; no full training state to resume")
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    weights = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={"assets": CallbackHandler(), "params": ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(max_to_keep=1, keep_period=keep_period, create=False),
    )
    return SplitCheckpointManager(mngr, weights, checkpoint_dir, legacy_step), resuming


class SplitCheckpointManager:
    """Keep periodic inference artifacts and only the latest complete resume state."""

    def __init__(self, state, weights, directory, legacy_step=None):
        self.state = state
        self.weights = weights
        self.directory = directory
        self.legacy_step = legacy_step

    def restore(self, step, items):
        if self.state.all_steps():
            return self.state.restore(step, items=items)
        if self.legacy_step is None:
            raise ValueError("No full resume state found; inference weights are not resumable")
        with ocp.CheckpointManager(
            self.directory,
            item_handlers={"train_state": ocp.PyTreeCheckpointHandler(),
                           "params": ocp.PyTreeCheckpointHandler()},
            options=ocp.CheckpointManagerOptions(create=False),
        ) as legacy:
            return legacy.restore(self.legacy_step if step is None else step, items=items)

    def wait_until_finished(self):
        self.state.wait_until_finished()
        self.weights.wait_until_finished()


def save_state(
    checkpoint_manager: SplitCheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    *,
    save_weights: bool = True,
    save_resume: bool = True,
):

    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {
            "params": params
        },
    }
    if save_weights:
        checkpoint_manager.weights.save(
            step, {k: v for k, v in items.items() if k != "train_state"}
        )
    if save_resume:
        if save_weights:
            checkpoint_manager.weights.wait_until_finished()
        checkpoint_manager.state.save(step, items)


def restore_state(
    checkpoint_manager: SplitCheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {
                    "params": params
                },
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):

    def __call__(self, directory: epath.Path) -> None:
        ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def __init__(self):
        self._executor = futures.ThreadPoolExecutor(max_workers=1)

    def close(self):
        self._executor.shutdown()

    def save(self, directory: epath.Path, args: "CallbackSave"):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: "CallbackSave") -> list[futures.Future]:
        return [self._executor.submit(self.save, directory, args)]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs):
    ...


def _split_params(state: training_utils.TrainState, ) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
