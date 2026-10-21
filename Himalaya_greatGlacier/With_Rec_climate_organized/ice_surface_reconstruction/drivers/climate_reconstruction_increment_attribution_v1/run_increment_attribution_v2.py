from __future__ import annotations

import json

import numpy as np

import run_increment_attribution as core


_json_dumps = json.dumps


def _safe_json_dumps(value, *args, **kwargs):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    kwargs.setdefault("default", convert)
    return _json_dumps(value, *args, **kwargs)


def _finite_reference_anomaly(
    years: np.ndarray,
    values: np.ndarray,
    reference_years: tuple[int, int],
) -> np.ndarray:
    reference = (years >= reference_years[0]) & (years <= reference_years[1])
    reference_values = values[..., reference]
    finite = np.isfinite(reference_values)
    count = np.sum(finite, axis=-1, keepdims=True)
    total = np.nansum(reference_values, axis=-1, keepdims=True)
    mean = np.full(total.shape, np.nan, dtype=float)
    np.divide(total, count, out=mean, where=count > 0)
    return values - mean


core.json.dumps = _safe_json_dumps
core._anomaly = _finite_reference_anomaly


if __name__ == "__main__":
    raise SystemExit(core.main())
