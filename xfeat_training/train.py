"""Hydra single-run entry point; all writes are owned by the validated trainer."""

from __future__ import annotations

from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts import reject_multirun
from xfeat_training.mining import repo_path


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(config: DictConfig) -> None:
    from xfeat_training.trainer import run_training

    resolved = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    if repo_path(resolved["run_dir"]).exists():
        raise FileExistsError(f"Run directory already exists: {resolved['run_dir']}")
    run_training(resolved)


if __name__ == "__main__":
    reject_multirun()
    main()
