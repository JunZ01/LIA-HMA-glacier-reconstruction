"""Run V1 with a strict tifffile prefix reader in the analysis environment."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import tifffile


# V1 and its established helper import rasterio at module load, although the
# V2 execution replaces the only reader that needs it before main is called.
sys.modules.setdefault("rasterio", types.ModuleType("rasterio"))

import run_temperature_west_colocated_independent_validation_v1 as experiment


def read_tif_prefix(path: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    page_indices = (experiment.YEARS - 1000).tolist()
    with tifffile.TiffFile(path) as source:
        if len(source.pages) != 1001:
            raise RuntimeError(f"unexpected-tif-band-count:{path}")
        data = np.stack([source.pages[index].asarray() for index in page_indices]).astype(float)
    expected = (len(experiment.YEARS), len(lat), len(lon))
    if data.shape != expected:
        raise RuntimeError(f"tif-reference-grid-shape-mismatch:{path}:{data.shape}:{expected}")
    data[data < -9000] = np.nan
    baseline = (experiment.YEARS >= 1961) & (experiment.YEARS <= 1990)
    data -= np.nanmean(data[baseline], axis=0)[None]
    return data


experiment.read_tif_prefix = read_tif_prefix


if __name__ == "__main__":
    raise SystemExit(experiment.main())
