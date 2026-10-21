"""Modern nested field sensitivity forcing unstable proxy 75 out of every fold."""

from __future__ import annotations

import numpy as np
import scipy.io

import run_precipitation_annual_oof_nested_blocked_field_validation_v1 as implementation


if __name__ == "__main__":
    project = implementation.root()
    allowed_path = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/proxy6_allowed_1940_2000_official_baseline_v2.mat"
    allowed = scipy.io.loadmat(allowed_path, variable_names=["years", "observations"])
    years = np.asarray(allowed["years"]).reshape(-1).astype(int)
    rows = np.asarray([int(np.flatnonzero(years == year)[0]) for year in implementation.YEARS])
    proxy75_response = np.asarray(allowed["observations"], dtype=float)[rows, 74]
    original_classify = implementation.classify_proxy

    def classify_without_proxy75(response, bea_t, bea_p, cru_t, cru_p, outer_train):
        admissible, metrics = original_classify(response, bea_t, bea_p, cru_t, cru_p, outer_train)
        if np.allclose(response, proxy75_response, equal_nan=True):
            admissible = False
        return admissible, metrics

    def field_anomaly(values: np.ndarray) -> np.ndarray:
        reference = (implementation.YEARS >= 1961) & (implementation.YEARS <= 1990)
        return values - np.nanmean(values[reference], axis=0, keepdims=True)

    implementation.EXPERIMENT_ID = "climate_reconstruction_precipitation_annual_oof_nested_blocked_field_proxy75_exclusion_v1"
    implementation.classify_proxy = classify_without_proxy75
    implementation.anomaly = field_anomaly
    raise SystemExit(implementation.main())
