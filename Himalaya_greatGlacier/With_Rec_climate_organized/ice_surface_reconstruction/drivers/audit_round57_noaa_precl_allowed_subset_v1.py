"""Audit the strict allowed-period NOAA PREC/L HMA subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCE_PAGE = "https://psl.noaa.gov/data/gridded/data.precl.html"
SOURCE_FILE = "https://downloads.psl.noaa.gov/Datasets/precl/0.5deg/precip.mon.mean.0.5x0.5.nc"
NCSS_URL = (
    "https://psl.noaa.gov/thredds/ncss/grid/Datasets/precl/0.5deg/"
    "precip.mon.mean.0.5x0.5.nc?var=precip&north=50.25&west=55.0&east=116.0&"
    "south=19.5&disableProjSubset=on&horizStride=1&time_start=1948-01-01T00%3A00%3A00Z&"
    "time_end=2000-12-01T00%3A00%3A00Z&timeStride=1&accept=netcdf4"
)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_57_noaa_precl_allowed_subset_acquisition_v1"
    paths = {
        "subset": output / "precl_precip_1948_2000_hma_bbox.nc",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    with xr.open_dataset(paths["subset"], decode_times=True, chunks=None) as dataset:
        if "precip" not in dataset:
            raise RuntimeError(f"precip-variable-missing:{list(dataset.variables)}")
        time = dataset.time
        years = np.asarray(time.dt.year.values, dtype=int).ravel()
        months = np.asarray(time.dt.month.values, dtype=int).ravel()
        lat = np.asarray(dataset.lat.values, dtype=float).ravel()
        lon = np.asarray(dataset.lon.values, dtype=float).ravel()
        values = np.asarray(dataset.precip.values, dtype=float)
        attrs = {str(key): str(value) for key, value in dataset.precip.attrs.items()}
        global_attrs = {str(key): str(value) for key, value in dataset.attrs.items()}
    expected = [(year, month) for year in range(1948, 2001) for month in range(1, 13)]
    actual = list(zip(years.tolist(), months.tolist()))
    if actual != expected:
        raise RuntimeError(f"monthly-axis-failed:{actual[:2]}:{actual[-2:]}:{len(actual)}")
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("locked-period-present")
    if values.shape != (len(years), len(lat), len(lon)):
        raise RuntimeError(f"array-shape-failed:{values.shape}")
    lat_spacing = float(np.median(np.abs(np.diff(np.sort(lat)))))
    lon_spacing = float(np.median(np.abs(np.diff(np.sort(lon)))))
    finite_fraction = float(np.isfinite(values).mean())
    units = attrs.get("units", "")
    gates = {
        "monthly_axis_exact_1948_2000": actual == expected,
        "latitude_spacing_0_5_degree": abs(lat_spacing - 0.5) <= 1e-8,
        "longitude_spacing_0_5_degree": abs(lon_spacing - 0.5) <= 1e-8,
        "hma_bbox_covered": float(lat.min()) <= 20.0
        and float(lat.max()) >= 50.0
        and float(lon.min()) <= 55.5
        and float(lon.max()) >= 115.0,
        "finite_fraction_above_0_50": finite_fraction > 0.50,
        "units_declared": bool(units),
        "locked_period_absent": True,
    }
    summary = {
        "status": "complete",
        "source_page": SOURCE_PAGE,
        "source_file": SOURCE_FILE,
        "ncss_query": NCSS_URL,
        "locked_1921_1939_present_or_accessed": False,
        "time_start": f"{years[0]:04d}-{months[0]:02d}",
        "time_end": f"{years[-1]:04d}-{months[-1]:02d}",
        "month_count": int(len(years)),
        "latitude_range": [float(lat.min()), float(lat.max())],
        "longitude_range": [float(lon.min()), float(lon.max())],
        "grid_spacing_degrees": [lat_spacing, lon_spacing],
        "shape": list(values.shape),
        "finite_fraction": finite_fraction,
        "precipitation_attributes": attrs,
        "global_attributes": global_attrs,
        "subset_state": before["subset"],
        "gates": gates,
        "passes_acquisition_preflight": bool(all(gates.values())),
        "decision": (
            "advance_precl_to_modern_compatibility_and_candidate_adjudication"
            if all(gates.values())
            else "stop_precl_branch_on_acquisition_conflict"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    coverage = np.mean(np.isfinite(values), axis=0)
    figure, axis = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    image = axis.pcolormesh(lon, lat, coverage, shading="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set(
        xlabel="Longitude",
        ylabel="Latitude",
        title="NOAA PREC/L finite monthly coverage, 1948-2000",
    )
    figure.colorbar(image, ax=axis, label="Finite fraction")
    figure.savefig(output / "precl_allowed_subset_coverage.png", dpi=180)
    plt.close(figure)
    lines = [
        "# NOAA PREC/L Allowed-Period Acquisition",
        "",
        f"- Time: `{summary['time_start']}` to `{summary['time_end']}`; months `{len(years)}`.",
        f"- Shape: `{values.shape}`; grid spacing `{lat_spacing:.3f}/{lon_spacing:.3f}` degrees.",
        f"- Units: `{units}`; finite fraction `{finite_fraction:.3%}`.",
        f"- SHA-256: `{before['subset']['sha256']}`.",
        "",
        f"Preflight passes: `{str(summary['passes_acquisition_preflight']).lower()}`.",
        f"Decision: `{summary['decision']}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged},
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    if not summary["passes_acquisition_preflight"]:
        raise RuntimeError("precl-acquisition-preflight-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
