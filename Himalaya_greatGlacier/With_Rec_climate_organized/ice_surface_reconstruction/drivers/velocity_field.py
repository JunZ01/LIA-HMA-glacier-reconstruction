"""ITS_LIVE velocity field sampling along connector paths (task #10).

Loads the ITS_LIVE static mosaics (v/vx/vy + errors, count, outlier_percent;
ESRI:102027, 120 m, nodata -32767) and samples them along a connector path.

The mosaic is a single ``RGI14A`` target-clip covering the JOINT extent of all
1,205 modern members — despite the ``RGI14A`` name it spans RGI first-order
regions 13/14/15 (the ITS_LIVE mosaic tag is not the RGI region code). Coverage
is therefore decided per junction by counting valid vector pixels along its
path and applying an SNR filter, never by the ``RGI60-1x`` region number.

Coordinate chain (reused from the velocity_gate_1205 audit, verified there with
0 extraction failures over 1,205 gates):

    connector pixel (row, col)                      # gdir gridded_data frame
      -> gdir local projection (x0 + col*dx, y0 + row*dy)
      -> velocity CRS (ESRI:102027) via pyproj      # always_xy=True
      -> mosaic row/col via the raster inverse transform, nearest pixel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_MOSAIC_NODATA = -32767.0
_LAYER_SUFFIX = {
    "vx": "_vx.tif",
    "vy": "_vy.tif",
    "v_error": "_v_error.tif",
    "count": "_count.tif",
    "outlier_percent": "_outlier_percent.tif",
}


@dataclass
class VelocitySamples:
    """Per-sample velocity in the mosaic (velocity CRS) frame.

    Arrays are aligned to the sampled points. ``valid`` is the per-point boolean
    after nodata + finite filtering on vx/vy. ``pixel_row``/``pixel_col`` are the
    mosaic (120 m) pixel indices each point falls in — used to DEDUP connector
    vertices/segments that share one velocity cell.
    """

    vx: np.ndarray
    vy: np.ndarray
    v_error: np.ndarray
    count: np.ndarray
    outlier_percent: np.ndarray
    valid: np.ndarray
    xy_velocity_crs: np.ndarray  # (n, 2) sampled coordinates, for audit
    pixel_row: np.ndarray        # (n,) mosaic row index (120 m cell)
    pixel_col: np.ndarray        # (n,) mosaic col index (120 m cell)

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid))


class VelocityField:
    """Lazily-opened ITS_LIVE mosaics with a connector-path sampler."""

    def __init__(self, mosaic_dir: Path, stem: str):
        import rasterio

        self._datasets = {}
        self._bands: dict[str, np.ndarray] = {}
        for layer, suffix in _LAYER_SUFFIX.items():
            path = mosaic_dir / f"{stem}{suffix}"
            if path.exists():
                self._datasets[layer] = rasterio.open(path)
        if "vx" not in self._datasets or "vy" not in self._datasets:
            raise FileNotFoundError(f"vx/vy mosaics not found under {mosaic_dir} / {stem}")
        self.crs = self._datasets["vx"].crs
        self._transform = self._datasets["vx"].transform

    @classmethod
    def from_dir(cls, mosaic_dir: Path) -> "VelocityField":
        """Auto-detect the mosaic stem (the shared prefix before ``_vx.tif``)."""
        vx = next(iter(sorted(Path(mosaic_dir).glob("*_vx.tif"))), None)
        if vx is None:
            raise FileNotFoundError(f"no *_vx.tif in {mosaic_dir}")
        stem = vx.name[: -len("_vx.tif")]
        return cls(Path(mosaic_dir), stem)

    def close(self) -> None:
        for dataset in self._datasets.values():
            dataset.close()

    def __enter__(self) -> "VelocityField":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _read_layer(self, layer: str, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        dataset = self._datasets.get(layer)
        if dataset is None:
            return np.full(rows.shape, np.nan)
        if layer not in self._bands:
            # Read the full band ONCE and cache it; sampling then indexes memory.
            self._bands[layer] = dataset.read(1)
        band = self._bands[layer]
        nrow, ncol = band.shape
        out = np.full(rows.shape, np.nan, dtype=float)
        inside = (rows >= 0) & (rows < nrow) & (cols >= 0) & (cols < ncol)
        out[inside] = band[rows[inside], cols[inside]]
        nodata = dataset.nodata if dataset.nodata is not None else _MOSAIC_NODATA
        out[out == nodata] = np.nan
        return out

    def sample_path_velocity_crs(self, xy: np.ndarray) -> VelocitySamples:
        """Sample all layers at points already in the velocity CRS.

        ``xy`` is (n, 2) in metres (ESRI:102027). Nearest-pixel via the vx
        raster inverse transform. Returns the mosaic pixel indices too, so
        callers can dedup points sharing one 120 m velocity cell.
        """
        vx_ds = self._datasets["vx"]
        inv = ~vx_ds.transform
        fcols, frows = inv * (xy[:, 0], xy[:, 1])
        cols = np.floor(np.asarray(fcols)).astype(int)
        rows = np.floor(np.asarray(frows)).astype(int)
        vx = self._read_layer("vx", rows, cols)
        vy = self._read_layer("vy", rows, cols)
        v_error = self._read_layer("v_error", rows, cols)
        count = self._read_layer("count", rows, cols)
        outlier = self._read_layer("outlier_percent", rows, cols)
        valid = np.isfinite(vx) & np.isfinite(vy)
        return VelocitySamples(
            vx=vx, vy=vy, v_error=v_error, count=count,
            outlier_percent=outlier, valid=valid, xy_velocity_crs=xy,
            pixel_row=rows, pixel_col=cols,
        )


def load_grid(gdir_dir: Path) -> dict:
    """Read the gdir projection anchors used to place pixels in metres."""
    return json.loads((gdir_dir / "glacier_grid.json").read_text())


def path_pixels_to_velocity_crs(
    pixels: list[tuple[int, int]], grid: dict, velocity_crs
) -> np.ndarray:
    """Project connector path pixels (row, col) into the velocity CRS.

    Mirrors the velocity_gate_1205 transform: local metres are
    ``(x0 + col*dx, y0 + row*dy)`` with ``dy`` negative, then reprojected.
    """
    from pyproj import CRS, Transformer

    x0, y0 = grid["x0y0"]
    dx, dy = grid["dxdy"]
    rows = np.array([r for r, _ in pixels], dtype=float)
    cols = np.array([c for _, c in pixels], dtype=float)
    local_x = x0 + cols * dx
    local_y = y0 + rows * dy
    transformer = Transformer.from_crs(
        CRS.from_user_input(grid["proj"]), velocity_crs, always_xy=True
    )
    x, y = transformer.transform(local_x, local_y)
    return np.column_stack([x, y])


def segment_midpoints_velocity_crs(
    pixels: list[tuple[int, int]], grid: dict, velocity_crs
) -> tuple[np.ndarray, np.ndarray]:
    """Return (midpoints_xy, vertex_xy) in the velocity CRS for a path.

    Velocity is sampled at each connector SEGMENT midpoint (n-1 points), not at
    vertices, so the sample sits where the tangent is defined. Both the midpoint
    coordinates and the full vertex coordinates (n points) are returned; the
    latter give per-segment tangents in the velocity CRS.
    """
    vertex_xy = path_pixels_to_velocity_crs(pixels, grid, velocity_crs)
    midpoints = 0.5 * (vertex_xy[:-1] + vertex_xy[1:])
    return midpoints, vertex_xy
