from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "round_219_ccsm4_netp_mjj_prior_preflight_v1"
CONTRACT = OUT / "ROUND_CONTRACT.md"
SOURCE = Path(r"D:\VM\VMshare\data\model\ccsm4_last_millenium\pr_sfc_Amon_CCSM4_past1000_085001-185012.nc")
DRIVER = Path(__file__).resolve()
TARGET_LAT = np.asarray([35, 37, 39, 41] * 8, dtype=float)
TARGET_LON = np.repeat(np.arange(93, 109, 2), 4).astype(float)
YEARS = np.arange(850, 1851, dtype=int)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def small_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}


def large_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def bracket(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = np.searchsorted(axis, values, side="right")
    lower = upper - 1
    if np.any(lower < 0) or np.any(upper >= len(axis)):
        raise RuntimeError("Target outside source coordinates")
    fraction = (values - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, fraction


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (SOURCE, CONTRACT, DRIVER, ROOT / "LIA_PROJECT_FACTS.md"):
        if not path.exists():
            raise FileNotFoundError(path)
    before = {
        "source": large_state(SOURCE),
        "contract": small_state(CONTRACT),
        "driver": small_state(DRIVER),
        "facts": small_state(ROOT / "LIA_PROJECT_FACTS.md"),
    }

    with netcdf_file(SOURCE, "r", mmap=True) as handle:
        if handle.dimensions != {"time": None, "lat": 192, "lon": 288, "bnds": 2}:
            raise RuntimeError(f"Unexpected dimensions: {handle.dimensions}")
        pr = handle.variables["pr"]
        time = np.asarray(handle.variables["time"][:], dtype=float).copy()
        lat = np.asarray(handle.variables["lat"][:], dtype=float).copy()
        lon = np.asarray(handle.variables["lon"][:], dtype=float).copy()
        if pr.shape != (12012, 192, 288):
            raise RuntimeError(f"Unexpected precipitation shape: {pr.shape}")
        if getattr(pr, "units", b"") != b"kg m-2 s-1":
            raise RuntimeError("Unexpected precipitation units")
        time_var = handle.variables["time"]
        if getattr(time_var, "units", b"") != b"days since 0850-01-01 00:00:00":
            raise RuntimeError("Unexpected time units")
        if getattr(time_var, "calendar", b"") != b"noleap":
            raise RuntimeError("Unexpected calendar")
        if len(time) != len(YEARS) * 12 or not np.all(np.diff(time) > 0):
            raise RuntimeError("Incomplete or unordered monthly axis")

        lat_lower, lat_upper, lat_fraction = bracket(lat, TARGET_LAT)
        lon_lower, lon_upper, lon_fraction = bracket(lon, TARGET_LON)
        lat_start, lat_stop = int(lat_lower.min()), int(lat_upper.max()) + 1
        lon_start, lon_stop = int(lon_lower.min()), int(lon_upper.max()) + 1
        local = np.asarray(pr[:, lat_start:lat_stop, lon_start:lon_stop], dtype=float).copy()

    li = lat_lower - lat_start
    ui = lat_upper - lat_start
    lj = lon_lower - lon_start
    uj = lon_upper - lon_start
    interpolated = np.empty((len(time), len(TARGET_LAT)), dtype=float)
    for column in range(len(TARGET_LAT)):
        f_lat = lat_fraction[column]
        f_lon = lon_fraction[column]
        q00 = local[:, li[column], lj[column]]
        q01 = local[:, li[column], uj[column]]
        q10 = local[:, ui[column], lj[column]]
        q11 = local[:, ui[column], uj[column]]
        interpolated[:, column] = (
            q00 * (1 - f_lat) * (1 - f_lon)
            + q01 * (1 - f_lat) * f_lon
            + q10 * f_lat * (1 - f_lon)
            + q11 * f_lat * f_lon
        )
    interpolated *= 86400.0
    monthly = interpolated.reshape(len(YEARS), 12, len(TARGET_LAT))
    day_weights = np.asarray([31.0, 30.0, 31.0])
    mjj = np.average(monthly[:, 4:7, :], axis=1, weights=day_weights)
    weights = np.cos(np.deg2rad(TARGET_LAT))
    regional = np.average(mjj, axis=1, weights=weights)
    regional_anomaly = regional - regional.mean()
    cell_anomaly = mjj - mjj.mean(axis=0, keepdims=True)
    loadings = cell_anomaly.T @ regional_anomaly / np.sum(regional_anomaly**2)
    loading_closure = float(np.average(loadings, weights=weights))

    gates = {
        "shape_1001x32": mjj.shape == (1001, 32),
        "all_finite": bool(np.isfinite(mjj).all()),
        "all_nonnegative": bool((mjj >= 0).all()),
        "all_cells_variable": bool((np.std(mjj, axis=0, ddof=1) > 0).all()),
        "regional_variance_positive": bool(np.std(regional, ddof=1) > 0),
        "loading_regional_closure": abs(loading_closure - 1.0) <= 1e-12,
    }
    if not all(gates.values()):
        raise RuntimeError(f"CCSM4 prior preflight failed: {gates}")

    np.savez_compressed(
        OUT / "ccsm4_netp_mjj_prior_850_1850.npz",
        years=YEARS,
        latitude=TARGET_LAT,
        longitude=TARGET_LON,
        mjj_mm_day=mjj,
        regional_mm_day=regional,
        prior_loading=loadings,
    )
    pd.DataFrame({
        "latitude": TARGET_LAT,
        "longitude": TARGET_LON,
        "prior_loading": loadings,
        "mjj_mean_mm_day": mjj.mean(axis=0),
        "mjj_sd_mm_day": mjj.std(axis=0, ddof=1),
    }).to_csv(OUT / "ccsm4_netp_cell_prior_diagnostics.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), gridspec_kw={"wspace": 0.27})
    axes[0].plot(YEARS, regional, color="#222222", linewidth=0.65)
    axes[0].plot(YEARS, pd.Series(regional).rolling(31, center=True, min_periods=31).mean(), color="#B33A3A", linewidth=1.5)
    axes[0].set(title="CCSM4 NETP MJJ prior", xlabel="Year", ylabel="mm/day")
    loading_map = loadings.reshape(8, 4).T
    image = axes[1].imshow(loading_map, origin="lower", aspect="auto", cmap="viridis", extent=[92, 108, 34, 42])
    axes[1].set(title="Regional-to-cell covariance loading", xlabel="Longitude", ylabel="Latitude")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    figure.savefig(OUT / "ccsm4_netp_mjj_prior_preflight.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    after = {
        "source": large_state(SOURCE),
        "contract": small_state(CONTRACT),
        "driver": small_state(DRIVER),
        "facts": small_state(ROOT / "LIA_PROJECT_FACTS.md"),
    }
    if before != after:
        raise RuntimeError("A source changed")
    summary = {
        "round": 219,
        "decision": "admit_ccsm4_as_independent_netp_mjj_prior_covariance",
        "source": before["source"],
        "years": [850, 1850],
        "cells": 32,
        "regional_mean_mm_day": float(regional.mean()),
        "regional_sd_mm_day": float(regional.std(ddof=1)),
        "loading_minimum": float(loadings.min()),
        "loading_maximum": float(loadings.max()),
        "loading_area_weighted_mean": loading_closure,
        "gates": gates,
        "source_states_unchanged": True,
        "proxy_or_instrumental_values_read": False,
        "locked_1901_1939_values_read": False,
        "jionglacuo_accessed": False,
        "enkf_run": False,
        "production_changed": False,
        "oggm_run": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "FINAL_RESULT.md").write_text(
        "# Round 219 Result\n\n"
        "`admit_ccsm4_as_independent_netp_mjj_prior_covariance`\n\n"
        f"The derived 1001x32 MJJ prior is finite and variable. Regional SD is `{regional.std(ddof=1):.4f}` mm/day; "
        f"loadings span `{loadings.min():.3f}` to `{loadings.max():.3f}` and close regionally at `{loading_closure:.15f}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
