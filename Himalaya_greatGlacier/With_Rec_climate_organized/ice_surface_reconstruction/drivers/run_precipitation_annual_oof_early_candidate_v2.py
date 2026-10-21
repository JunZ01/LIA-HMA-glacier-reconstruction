"""V2 entrypoint enforcing the frozen HMA mask on early regional means."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import run_precipitation_annual_oof_early_candidate_v1 as implementation


if __name__ == "__main__":
    project = implementation.root()
    arrays = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_annual_oof_nested_blocked_field_validation_v2/blocked_field_arrays.npz"
    with np.load(arrays) as handle:
        hma_mask = np.asarray(handle["hma_mask"], dtype=bool)
    original_area_series = implementation.blocked.area_series

    def hma_area_series(field: np.ndarray, latitudes: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.shape != hma_mask.shape:
            raise RuntimeError(f"hma-mask-shape-conflict:{mask.shape}:{hma_mask.shape}")
        return original_area_series(field, latitudes, mask & hma_mask)

    implementation.EXPERIMENT_ID = "climate_reconstruction_precipitation_annual_oof_early_candidate_v2"
    implementation.blocked.area_series = hma_area_series
    raise SystemExit(implementation.main())
