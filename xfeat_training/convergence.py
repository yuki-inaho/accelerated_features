"""Explicit fixed-validation plateau heuristic; not a proof of convergence."""

import math
from statistics import mean


def plateau_decision(values: list[float], *, mode: str, window: int, minimum_gain: float) -> dict:
    if mode not in {"min", "max"} or type(window) is not int or window < 1:
        raise ValueError("Invalid plateau mode/window")
    if len(values) < 2 * window:
        raise ValueError("Not enough validation values for two plateau windows")
    if not all(math.isfinite(v) for v in values) or not math.isfinite(minimum_gain) or minimum_gain < 0:
        raise ValueError("Plateau inputs must be finite; minimum_gain must be nonnegative")
    sign = 1 if mode == "max" else -1
    previous, recent = values[-2 * window : -window], values[-window:]
    mean_gain = sign * (mean(recent) - mean(previous))
    peak_gain = max(sign * v for v in recent) - max(sign * v for v in previous)
    stop = mean_gain <= minimum_gain and peak_gain <= minimum_gain
    return {
        "stop": stop,
        "reason": "validation_plateau" if stop else "validation_improving",
        "previous": previous,
        "recent": recent,
        "mean_gain": mean_gain,
        "peak_gain": peak_gain,
        "minimum_gain": minimum_gain,
        "mode": mode,
        "window": window,
    }
