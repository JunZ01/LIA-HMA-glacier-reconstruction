"""V2 entrypoint with an explicit 1941-2000 anomaly reference axis."""

from __future__ import annotations

import numpy as np

import run_precipitation_annual_oof_nested_blocked_field_validation_v1 as implementation


def field_anomaly(values: np.ndarray) -> np.ndarray:
    years = implementation.YEARS
    reference = (years >= 1961) & (years <= 1990)
    return values - np.nanmean(values[reference], axis=0, keepdims=True)


if __name__ == "__main__":
    implementation.EXPERIMENT_ID = "climate_reconstruction_precipitation_annual_oof_nested_blocked_field_validation_v2"
    implementation.anomaly = field_anomaly
    raise SystemExit(implementation.main())
