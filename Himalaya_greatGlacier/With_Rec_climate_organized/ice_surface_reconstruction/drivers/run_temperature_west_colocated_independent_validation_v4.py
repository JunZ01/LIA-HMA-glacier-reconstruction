"""Run V3 with GeoTIFF-tag-derived source centers and reference interpolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

import run_temperature_west_colocated_independent_validation_v3 as compatibility


experiment = compatibility.experiment


def read_tif_prefix(path: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    source = tifffile.memmap(path, series=0, mode="r")
    expected_source = (43, 78, 1001)
    if source.shape != expected_source:
        raise RuntimeError(f"tif-sample-axis-shape-mismatch:{path}:{source.shape}:{expected_source}")
    sample_indices = experiment.YEARS - 1000
    data = np.asarray(source[:, :, sample_indices], dtype=float).transpose(2, 0, 1)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        scale = tuple(float(value) for value in page.tags[33550].value)
        tie = tuple(float(value) for value in page.tags[33922].value)
        nodata = float(page.tags[42113].value)
    source_lon = tie[3] + (np.arange(source.shape[1]) + 0.5) * scale[0]
    source_lat = tie[4] - (np.arange(source.shape[0]) + 0.5) * scale[1]
    data[np.isclose(data, nodata) | (data < -9000)] = np.nan
    projected, orientation = experiment.round23.interpolate_to_reference(
        data, source_lat, source_lon, lat, lon
    )
    if orientation != "bilinear_source_center_to_reference_center":
        raise RuntimeError(f"unexpected-interpolation:{orientation}")
    baseline = (experiment.YEARS >= 1961) & (experiment.YEARS <= 1990)
    projected -= np.nanmean(projected[baseline], axis=0)[None]
    return projected


experiment.read_tif_prefix = read_tif_prefix


if __name__ == "__main__":
    raise SystemExit(experiment.main())
