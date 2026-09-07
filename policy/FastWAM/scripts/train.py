import hydra
from omegaconf import DictConfig, open_dict

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    # Training runs intentionally retain one complete weights/state pair.
    # Keep this invariant at the shared Python entry point so direct launches
    # behave the same as the ZeRO-1 and ZeRO-2 wrappers.
    with open_dict(cfg):
        cfg.checkpoint_keep_last = 1
    run_training(cfg)


if __name__ == "__main__":
    main()
