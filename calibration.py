"""Output scales from unperturbed validation predictions."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Mapping, Sequence

from .types import Instance


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    # Linear interpolation (Hyndman-Fan type 7).
    position = (len(sorted_values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low
    return (1 - weight) * sorted_values[low] + weight * sorted_values[high]


def estimate_output_scales(predictor, validation_inputs: Sequence[Instance], *,
                           n_outputs: int = 3) -> tuple[float, ...]:
    """Return max(IQR / 1.349, 1e-8) per output over validation windows.

    Each unperturbed window contributes once, without patient grouping.
    """
    if type(n_outputs) is not int or n_outputs <= 0 or not validation_inputs:
        raise ValueError("Calibration needs validation windows and positive output count")
    ids = set()
    components = [[] for _ in range(n_outputs)]
    for instance in validation_inputs:
        if not isinstance(instance.payload, Mapping) or instance.payload.get("split") != "validation":
            raise ValueError("Output scales must use original validation inputs")
        if instance.input_id in ids:
            raise ValueError("Calibration windows must have distinct IDs")
        ids.add(instance.input_id)
        output = predictor(deepcopy(instance.payload))
        if isinstance(output, (str, bytes)):
            raise ValueError("Calibration predictor must return a numeric vector")
        output = tuple(output)
        if len(output) != n_outputs:
            raise ValueError("Calibration output shape mismatch")
        for bucket, value in zip(components, output):
            if isinstance(value, (str, bytes, bool)) or not math.isfinite(float(value)):
                raise ValueError("Calibration outputs must be finite numbers")
            bucket.append(float(value))
    result = []
    for values in components:
        values.sort()
        scale = (_quantile(values, .75) - _quantile(values, .25)) / 1.349
        if not math.isfinite(scale):
            raise ValueError("Output-scale calculation overflow")
        result.append(max(scale, 1e-8))
    return tuple(result)
