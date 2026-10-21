"""Run V5 without an unavailable or unnecessary modern-zero subtraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

import run_temperature_west_colocated_independent_validation_v5 as compatibility


experiment = compatibility.experiment


def read_tif_prefix(path: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    source = tifffile.memmap(path, series=0, mode="r")
    expected_source = (len(lat), len(lon), 1001)
    if source.shape != expected_source:
        raise RuntimeError(f"native-tif-shape-mismatch:{path}:{source.shape}:{expected_source}")
    data = np.asarray(source[:, :, experiment.YEARS - 1000], dtype=float).transpose(2, 0, 1)
    with tifffile.TiffFile(path) as tif:
        nodata = float(tif.pages[0].tags[42113].value)
    data[np.isclose(data, nodata) | (data < -9000)] = np.nan
    # Every downstream comparison is standardized on 1812-1860, so a modern
    # additive zero is mathematically irrelevant and would require forbidden
    # post-1900 reads in this experiment.
    return data


experiment.read_tif_prefix = read_tif_prefix


if __name__ == "__main__":
    raise SystemExit(experiment.main())
