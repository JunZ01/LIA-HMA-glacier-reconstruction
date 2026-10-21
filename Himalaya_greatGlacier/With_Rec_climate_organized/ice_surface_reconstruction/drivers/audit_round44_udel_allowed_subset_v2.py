"""Verify identity and allowed-period bounds of the Round 44 UDel subset."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np


CANONICAL_URL = "https://downloads.psl.noaa.gov/Datasets/udel.airt.precip/precip.mon.total.v501.nc"
TRANSPORT_URL = "https://climatecharts.net/thredds/ncss/grid/data/UDEL/precip.mon.total.v501.nc_compression_0.nc"
REQUEST = "var=precip&north=50&west=55&east=115&south=20&horizStride=1&time_start=1940-01-01T00%3A00%3A00Z&time_end=2000-12-01T00%3A00%3A00Z&timeStride=1&accept=netcdf4"


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists(): return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    stat = path.stat(); return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def text_attr(value: object) -> str:
    array = np.asarray(value).reshape(-1); value = array[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def main() -> int:
    project = root(); loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"; output = loop / "round_44_udel_allowed_subset_acquisition_v2"
    paths = {"subset": output / "udel_precip_v501_1940_2000_hma_bbox.nc", "contract": output / "ROUND_CONTRACT.md", "driver": Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists(): raise FileNotFoundError(path)
    if (output / "summary.json").exists(): raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    with h5py.File(paths["subset"], "r") as handle:
        if set(handle.keys()) != {"lat", "lon", "precip", "time"}: raise RuntimeError("variable-set-failed")
        lat = np.asarray(handle["lat"], dtype=float); lon = np.asarray(handle["lon"], dtype=float); time = np.asarray(handle["time"], dtype=float); precip = np.asarray(handle["precip"], dtype=float)
        version = text_attr(handle.attrs["version"]); dataset_title = text_attr(handle.attrs["dataset_title"]); source_reference = text_attr(handle.attrs["References"]); units = text_attr(handle["precip"].attrs["units"]); time_units = text_attr(handle["time"].attrs["units"])
    if time_units != "days since 1900-01-01 00:00:00": raise RuntimeError(f"unexpected-time-units:{time_units}")
    dates = [datetime(1900, 1, 1) + timedelta(days=float(value)) for value in time]
    year_month = [(date.year, date.month) for date in dates]
    expected = [(year, month) for year in range(1940, 2001) for month in range(1, 13)]
    if year_month != expected or len(time) != 732: raise RuntimeError(f"monthly-axis-failed:{len(time)}:{year_month[:1]}:{year_month[-1:]}")
    if any(1921 <= year <= 1939 for year, _ in year_month) or any(year >= 2001 for year, _ in year_month): raise RuntimeError("forbidden-year-present")
    if precip.shape != (732, 61, 121): raise RuntimeError(f"precip-shape-failed:{precip.shape}")
    if not np.allclose(np.abs(np.diff(lat)), 0.5) or not np.allclose(np.diff(lon), 0.5): raise RuntimeError("grid-spacing-failed")
    if not (np.isclose(np.min(lat), 20) and np.isclose(np.max(lat), 50) and np.isclose(np.min(lon), 55) and np.isclose(np.max(lon), 115)): raise RuntimeError("coordinate-bounds-failed")
    if version != "5.01" or units != "cm" or "1900-2017" not in dataset_title or "UDel" not in source_reference: raise RuntimeError("udel-identity-metadata-failed")
    finite = np.isfinite(precip); finite_values = precip[finite]
    summary = {"status": "complete", "canonical_url": CANONICAL_URL, "transport_url": TRANSPORT_URL, "ncss_request": REQUEST, "version": version, "dataset_title": dataset_title, "source_reference": source_reference, "time_start": dates[0].strftime("%Y-%m-%d"), "time_end": dates[-1].strftime("%Y-%m-%d"), "monthly_count": len(dates), "locked_1921_1939_present_or_accessed": False, "latitude_count": len(lat), "longitude_count": len(lon), "latitude_bounds": [float(np.min(lat)), float(np.max(lat))], "longitude_bounds": [float(np.min(lon)), float(np.max(lon))], "precipitation_shape": list(precip.shape), "precipitation_units": units, "finite_fraction": float(np.mean(finite)), "finite_min_cm": float(np.min(finite_values)), "finite_max_cm": float(np.max(finite_values)), "subset_file": state(paths["subset"]), "decision": "advance_udel_as_third_allowed_period_observation_target"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "acquisition_manifest.json").write_text(json.dumps({"canonical_url": CANONICAL_URL, "transport_url": TRANSPORT_URL, "exact_ncss_request": REQUEST, "first_request_733_month_conflict_preserved_in": str(loop / "round_44_udel_allowed_subset_acquisition_v1")}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text("# UDel v5.01 Allowed-Period Subset\n\n" f"- Time: `{summary['time_start']}` to `{summary['time_end']}` (`{summary['monthly_count']}` months).\n" f"- Grid: `{len(lat)} x {len(lon)}`; precipitation shape `{precip.shape}`; units `{units}`.\n" f"- File SHA-256: `{summary['subset_file']['sha256']}`.\n\n" f"Decision: `{summary['decision']}`.\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}; unchanged = before == after; (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged: raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
