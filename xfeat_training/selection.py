"""One explicit validation ordering shared by checkpoint selection and retention."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def checkpoint_candidate(metrics: Mapping[str, Any], step: int, config: Mapping[str, Any]) -> dict[str, Any]:
    specification = config.get("selection_metric")
    if specification is None:
        f1, tp = metrics["primary_f1"], metrics["TP"]
        if not isinstance(f1, (int, float)) or isinstance(f1, bool) or not math.isfinite(f1) or not 0 <= f1 <= 1:
            raise ValueError("Invalid finite primary F1")
        if type(tp) is not int or tp < 0:
            raise ValueError("Invalid TP")
        return {"step": step, "primary_f1": f1, "TP": tp}
    if set(specification) != {"name", "mode"} or specification["mode"] not in {"min", "max"}:
        raise ValueError("Invalid selection metric specification")
    name = specification["name"]
    if not isinstance(name, str) or not name or name not in metrics:
        raise ValueError("Missing selection metric")
    value = metrics[name]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Selection metric must be finite numeric")
    return {"step": step, "metric": name, "mode": specification["mode"], "value": value}


def selection_rank(candidate: Mapping[str, Any]) -> tuple[float, int, int]:
    """Higher tuple is better; preserve historical F1/TP/earlier-step ordering."""
    if "metric" in candidate:
        sign = 1 if candidate["mode"] == "max" else -1
        return sign * candidate["value"], 0, -candidate["step"]
    return candidate["primary_f1"], candidate["TP"], -candidate["step"]
