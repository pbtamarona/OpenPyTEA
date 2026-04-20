"""Shared utilities for JSON serialization."""

import math
import numpy as np


def to_jsonable(obj):
    """Recursively convert numpy types to Python natives for JSON.

    NaN and Inf values are converted to None since JSON doesn't support them.
    """
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj
