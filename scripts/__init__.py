"""Command-line entry points for explicit data preparation and evaluation."""

from __future__ import annotations

import sys


def reject_multirun() -> None:
    if any(arg in {"-m", "--multirun"} for arg in sys.argv[1:]):
        raise SystemExit("Hydra multirun is not supported; use a separate explicit run directory")
