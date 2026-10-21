"""Run the co-located experiment with sample-axis GeoTIFF prefix slicing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

import run_temperature_west_colocated_independent_validation_v2 as compatibility


experiment = compatibility.experiment


def read_tif_prefix(path: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    source = tifffile.memmap(path, series=0, mode="r")
    expected_source = (len(lat), len(lon), 1001)
    if source.shape != expected_source:
        raise RuntimeError(f"tif-sample-axis-shape-mismatch:{path}:{source.shape}:{expected_source}")
    sample_indices = experiment.YEARS - 1000
    data = np.asarray(source[:, :, sample_indices], dtype=float).transpose(2, 0, 1)
    expected = (len(experiment.YEARS), len(lat), len(lon))
    if data.shape != expected:
        raise RuntimeError(f"tif-prefix-shape-mismatch:{path}:{data.shape}:{expected}")
    data[data < -9000] = np.nan
    baseline = (experiment.YEARS >= 1961) & (experiment.YEARS <= 1990)
    data -= np.nanmean(data[baseline], axis=0)[None]
    return data


experiment.read_tif_prefix = read_tif_prefix


if __name__ == "__main__":
    raise SystemExit(experiment.main())
