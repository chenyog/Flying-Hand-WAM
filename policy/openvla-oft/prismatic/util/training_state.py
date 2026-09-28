"""Full DDP resume state, independent of inference checkpoint export."""
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def unwrap(module):
    return module.module if isinstance(module, torch.nn.parallel.DistributedDataParallel) else module


def save_state(path, modules, optimizer, scheduler, step, dataset_stats):
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
    distributed = dist.is_available() and dist.is_initialized()
    ranks = [None] * (dist.get_world_size() if distributed else 1)
    if distributed:
        dist.all_gather_object(ranks, rng)
    else:
        ranks[0] = rng
    if not distributed or dist.get_rank() == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(".incomplete")
        torch.save({
            "modules": {k: unwrap(v).state_dict() for k, v in modules.items() if v is not None},
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "completed_steps": step, "rng": ranks, "dataset_stats": dataset_stats,
        }, staging)
        os.replace(staging, path)
    if distributed:
        dist.barrier()


def load_state(path, modules, optimizer, scheduler):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "modules" not in state or "optimizer" not in state:
        raise ValueError("Expected full state/latest.pt, not inference weights")
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    if len(state["rng"]) != (dist.get_world_size() if distributed else 1):
        raise ValueError("Resume requires the same process count")
    expected = {k for k, v in modules.items() if v is not None}
    if expected != set(state["modules"]):
        raise ValueError("Resume model components differ from checkpoint")
    for key in expected:
        unwrap(modules[key]).load_state_dict(state["modules"][key])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng = state["rng"][rank]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng["cuda"] is not None:
        torch.cuda.set_rng_state(rng["cuda"].cpu())
    return state["completed_steps"], state["dataset_stats"]
