"""Hydra entry point: frozen official RaCo teacher, existing XFeat head student."""

from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts import reject_multirun
from xfeat_training.raco_distill_task import run_distillation


@hydra.main(version_base="1.3", config_path="../configs", config_name="raco_distill")
def main(config: DictConfig) -> None:
    resolved = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    run_distillation(resolved)


if __name__ == "__main__":
    reject_multirun()
    main()
