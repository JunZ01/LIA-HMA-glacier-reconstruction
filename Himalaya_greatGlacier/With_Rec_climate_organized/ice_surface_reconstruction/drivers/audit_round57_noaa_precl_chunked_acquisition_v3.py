"""Audit six complete NOAA PREC/L HMA NCSS chunks as one allowed-period source."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_RANGES = (
    (1948, 1957),
    (1958, 1967),
    (1968, 1977),
    (1978, 1987),
    (1988, 1997),
    (1998, 2000),
)
SOURCE_PAGE = "https://psl.noaa.gov/data/gridded/data.precl.html"
SOURCE_FILE = "https://downloads.psl.noaa.gov/Datasets/precl/0.5deg/precip.mon.mean.0.5x0.5.nc"


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


def text_attr(value: object) -> str:
    array = np.asarray(value).reshape(-1)
    if not len(array):
        return ""
    item = array[0]
    return item.decode() if isinstance(item, bytes) else str(item)


def decode_time(values: np.ndarray, units: str) -> list[datetime]:
    if units != "hours since 1800-1-1 00:00:0.0":
        raise RuntimeError(f"unexpected-time-units:{units}")
    origin = datetime(1800, 1, 1)
    return [origin + timedelta(hours=float(value)) for value in values]


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_57_noaa_precl_allowed_subset_acquisition_v3"
    files = [output / f"precl_{start}_{end}.nc" for start, end in EXPECTED_RANGES]
    paths = {f"chunk_{start}_{end}": path for (start, end), path in zip(EXPECTED_RANGES, files)}
    paths.update({"contract": output / "ROUND_CONTRACT.md", "driver": Path(__file__).resolve()})
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    reference_lat: np.ndarray | None = None
    reference_lon: np.ndarray | None = None
    all_dates: list[datetime] = []
    all_values: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    units_seen: set[str] = set()
    dataset_names: set[str] = set()
    for (start, end), path in zip(EXPECTED_RANGES, files):
        with h5py.File(path, "r") as handle:
            required = {"time", "lat", "lon", "precip"}
            if not required.issubset(handle.keys()):
                raise RuntimeError(f"required-variables-missing:{path.name}:{list(handle.keys())}")
            lat = np.asarray(handle["lat"], dtype=float)
            lon = np.asarray(handle["lon"], dtype=float)
            dates = decode_time(
                np.asarray(handle["time"], dtype=float),
                text_attr(handle["time"].attrs["units"]),
            )
            values = np.asarray(handle["precip"], dtype=float)
            fill = float(np.asarray(handle["precip"].attrs.get("missing_value", -9.96921e36)).reshape(-1)[0])
            values[np.isclose(values, fill) | (values < -1e20)] = np.nan
            units = text_attr(handle["precip"].attrs.get("units", ""))
            dataset_name = text_attr(handle["precip"].attrs.get("dataset", ""))
        expected = [(year, month) for year in range(start, end + 1) for month in range(1, 13)]
        actual = [(date.year, date.month) for date in dates]
        if actual != expected:
            raise RuntimeError(f"chunk-month-axis-failed:{path.name}:{actual[:2]}:{actual[-2:]}")
        if values.shape != (len(dates), len(lat), len(lon)):
            raise RuntimeError(f"chunk-shape-failed:{path.name}:{values.shape}")
        if reference_lat is None:
            reference_lat = lat
            reference_lon = lon
        elif not np.array_equal(reference_lat, lat) or not np.array_equal(reference_lon, lon):
            raise RuntimeError(f"chunk-grid-varies:{path.name}")
        units_seen.add(units)
        dataset_names.add(dataset_name)
        all_dates.extend(dates)
        all_values.append(values)
        rows.append(
            {
                "file": path.name,
                "start_month": f"{dates[0].year:04d}-{dates[0].month:02d}",
                "end_month": f"{dates[-1].year:04d}-{dates[-1].month:02d}",
                "month_count": len(dates),
                "shape": "x".join(map(str, values.shape)),
                "finite_fraction": float(np.isfinite(values).mean()),
                "size_bytes": before[f"chunk_{start}_{end}"]["size_bytes"],
                "sha256": before[f"chunk_{start}_{end}"]["sha256"],
            }
        )

    assert reference_lat is not None and reference_lon is not None
    expected_all = [(year, month) for year in range(1948, 2001) for month in range(1, 13)]
    actual_all = [(date.year, date.month) for date in all_dates]
    values = np.concatenate(all_values, axis=0)
    lat_spacing = float(np.median(np.abs(np.diff(np.sort(reference_lat)))))
    lon_spacing = float(np.median(np.abs(np.diff(np.sort(reference_lon)))))
    finite_fraction = float(np.isfinite(values).mean())
    gates = {
        "all_six_chunks_open_as_hdf5": len(rows) == 6,
        "combined_month_axis_exact_1948_2000": actual_all == expected_all,
        "combined_month_count_636": len(all_dates) == 636,
        "single_grid_across_chunks": True,
        "grid_spacing_0_5_degree": abs(lat_spacing - 0.5) <= 1e-8
        and abs(lon_spacing - 0.5) <= 1e-8,
        "units_are_mm_day": units_seen == {"mm/day"},
        "dataset_identity_is_precl_0_5": len(dataset_names) == 1
        and "PREC/L" in next(iter(dataset_names)),
        "finite_fraction_above_0_50": finite_fraction > 0.50,
        "locked_period_absent": not any(1921 <= date.year <= 1939 for date in all_dates),
    }
    with (output / "chunk_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "complete",
        "source_page": SOURCE_PAGE,
        "source_file": SOURCE_FILE,
        "locked_1921_1939_present_or_accessed": False,
        "v1_v2_truncated_transfers_excluded": True,
        "chunk_count": len(rows),
        "time_start": "1948-01",
        "time_end": "2000-12",
        "month_count": len(all_dates),
        "latitude_range": [float(reference_lat.min()), float(reference_lat.max())],
        "longitude_range": [float(reference_lon.min()), float(reference_lon.max())],
        "grid_spacing_degrees": [lat_spacing, lon_spacing],
        "combined_shape": list(values.shape),
        "finite_fraction": finite_fraction,
        "units": sorted(units_seen),
        "dataset_names": sorted(dataset_names),
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
    image = axis.pcolormesh(reference_lon, reference_lat, coverage, shading="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set(
        xlabel="Longitude",
        ylabel="Latitude",
        title="NOAA PREC/L finite monthly coverage, 1948-2000",
    )
    figure.colorbar(image, ax=axis, label="Finite fraction")
    figure.savefig(output / "precl_allowed_subset_coverage.png", dpi=180)
    plt.close(figure)
    lines = [
        "# NOAA PREC/L Chunked Acquisition",
        "",
        f"- Chunks/months: `{len(rows)}/{len(all_dates)}`.",
        f"- Combined shape: `{values.shape}`; grid `{lat_spacing:.3f}/{lon_spacing:.3f}` degrees.",
        f"- Units: `{sorted(units_seen)}`; finite fraction `{finite_fraction:.3%}`.",
        f"- All gates pass: `{str(summary['passes_acquisition_preflight']).lower()}`.",
        "",
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
