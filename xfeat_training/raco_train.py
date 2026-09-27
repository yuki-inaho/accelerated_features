"""Hydra single-run entry point for independent XFeat ranking/covariance heads."""

from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts import reject_multirun
from xfeat_training.raco_task import run_raco_training


@hydra.main(version_base="1.3", config_path="../configs", config_name="raco")
def main(config: DictConfig) -> None:
    resolved = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    run_raco_training(resolved)


if __name__ == "__main__":
    reject_multirun()
    main()
