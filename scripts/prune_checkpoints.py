"""Apply checkpoint retention to a stopped run; never invoke beside a live trainer."""

from __future__ import annotations

import argparse
import json

from xfeat_training.mining import repo_path
from xfeat_training.retention import prune_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--keep-best", type=int, required=True)
    parser.add_argument(
        "--training-exited",
        action="store_true",
        required=True,
        help="Confirm that the owning training process has exited. A pruning lock does not stop a legacy trainer.",
    )
    args = parser.parse_args()
    result = prune_checkpoints(repo_path(args.run_dir), args.keep_best)
    print(json.dumps({key: result[key] for key in ("status", "top_k", "protected", "protected_over_budget")}, indent=2))


if __name__ == "__main__":
    main()
