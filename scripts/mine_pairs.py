"""uv run --no-sync python -m scripts.mine_pairs output_dir=temp/l76_run/pairs"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts import reject_multirun
from xfeat_training.mining import mine_pairs


@hydra.main(version_base="1.3", config_path="../configs/pair_mining", config_name="default")
def main(config: DictConfig) -> None:
    resolved = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    manifest = mine_pairs(resolved["data_root"], resolved["output_dir"], resolved)
    print(
        json.dumps(
            {
                "output_dir": manifest["output_dir"],
                "pair_counts": manifest["pair_counts"],
                "content_hash": manifest["content_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    reject_multirun()
    main()
