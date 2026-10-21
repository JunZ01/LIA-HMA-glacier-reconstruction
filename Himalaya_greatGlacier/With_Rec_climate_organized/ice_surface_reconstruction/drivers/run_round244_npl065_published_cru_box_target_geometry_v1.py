from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1960, 2001, dtype=int)
LAT_RANGE = (27.5, 28.5)
LON_RANGE = (86.0, 87.0)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def area_mean(field: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None]
    valid = np.isfinite(field)
    numerator = np.nansum(field * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def read_box_djf(
    manifest_path: Path,
) -> tuple[np.ndarray, float, dict[str, Any], list[Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    monthly: dict[tuple[int, int], float] = {}
    support: dict[tuple[int, int], float] = {}
    files: list[Path] = []
    selected_lats = None
    selected_lons = None
    checks = []
    for record in manifest["files"]:
        first_year, last_year = [int(value) for value in record["allowed_year_range"].split("-")]
        if last_year < 1959:
            continue
        path = Path(record["netcdf_path"])
        if fingerprint(path)["sha256"] != record["netcdf_sha256"]:
            raise RuntimeError(f"cru-hash-changed:{path.name}")
        files.append(path)
        with netcdf_file(path, "r", mmap=True) as dataset:
            latitudes = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
            longitudes = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
            iy = np.flatnonzero((latitudes >= LAT_RANGE[0]) & (latitudes <= LAT_RANGE[1]))
            ix = np.flatnonzero((longitudes >= LON_RANGE[0]) & (longitudes <= LON_RANGE[1]))
            if len(iy) != 2 or len(ix) != 2:
                raise RuntimeError(f"published-box-cell-count-changed:{len(iy)}x{len(ix)}")
            lat_slice = slice(int(iy[0]), int(iy[-1]) + 1)
            lon_slice = slice(int(ix[0]), int(ix[-1]) + 1)
            tmn = np.asarray(dataset.variables["tmn"].data[:, lat_slice, lon_slice], dtype=float).copy()
            stn = np.asarray(dataset.variables["stn"].data[:, lat_slice, lon_slice], dtype=float).copy()
            fill = float(dataset.variables["tmn"]._attributes["_FillValue"])
            tmn[np.isclose(tmn, fill) | (tmn > 1e20)] = np.nan
            selected_lats = latitudes[iy].astype(float)
            selected_lons = longitudes[ix].astype(float)
        periods = pd.period_range(f"{first_year}-01", periods=12 * (last_year - first_year + 1), freq="M")
        values = area_mean(tmn, selected_lats)
        fractions = np.mean(stn > 0, axis=(1, 2))
        if len(periods) != len(values):
            raise RuntimeError(f"cru-month-count-changed:{path.name}")
        for period, value, fraction in zip(periods, values, fractions):
            key = (int(period.year), int(period.month))
            if key in monthly:
                raise RuntimeError(f"duplicate-cru-month:{key}")
            monthly[key] = float(value)
            support[key] = float(fraction)
        checks.append({"file": path.name, "range": [first_year, last_year]})
    result = np.full(len(YEARS), np.nan)
    support_slots = []
    for index, year in enumerate(YEARS):
        keys = ((int(year) - 1, 12), (int(year), 1), (int(year), 2))
        winter = np.asarray([monthly.get(key, np.nan) for key in keys], dtype=float)
        if np.isfinite(winter).all():
            result[index] = float(np.mean(winter))
        support_slots.extend(support.get(key, 0.0) for key in keys)
    if not np.isfinite(result).all():
        raise RuntimeError("incomplete-published-box-target")
    return result, float(np.mean(support_slots)), {
        "latitude_centers": selected_lats.tolist(),
        "longitude_centers": selected_lons.tolist(),
        "cell_count": int(len(selected_lats) * len(selected_lons)),
        "weighting": "cosine_latitude",
        "files": checks,
    }, files


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_244_npl065_published_cru_box_target_geometry_v1"
    output.mkdir(parents=True, exist_ok=False)
    round238 = loop / "round_238_npl065_documented_winter_tmin_modern_oof_v4"
    round208 = loop / "round_208_setp_cru_tmin_annual_target_audit_v1"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round244_npl065_published_cru_box_target_geometry_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "round238_summary": round238 / "summary.json",
        "round238_curves": round238 / "modern_oof_curves.csv",
        "cru_manifest": round208 / "cru_tmn_acquisition_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}
    curves = pd.read_csv(paths["round238_curves"])
    if not np.array_equal(curves.year.to_numpy(int), YEARS):
        raise RuntimeError("round238-year-axis-changed")
    proxy = curves.chronology.to_numpy(float)
    point_target = curves.cru_djf_tmin_observed.to_numpy(float)
    box_target, station_support, box_check, cru_files = read_box_djf(paths["cru_manifest"])
    paths.update({f"cru_tmin_{index:02d}": path for index, path in enumerate(cru_files, start=1)})
    before.update({name: fingerprint(path) for name, path in paths.items() if name not in before})

    r238 = load_module(
        "round244_round238",
        drivers / "run_round238_npl065_documented_winter_tmin_modern_oof_v4.py",
    )
    targets = {"nearest_point": point_target, "published_box": box_target}
    rows = []
    predictions = {}
    fold_rows = []
    for geometry, target in targets.items():
        metrics, prediction, folds = r238.blocked_oof(proxy, target)
        predictions[geometry] = prediction
        rows.append({
            "geometry": geometry,
            "raw_proxy_target_correlation": r238.correlation(proxy, target),
            **metrics,
        })
        fold_rows.extend({"geometry": geometry, **row} for row in folds)
    metrics = pd.DataFrame(rows).set_index("geometry")
    delta_oof = float(
        metrics.at["published_box", "oof_correlation"]
        - metrics.at["nearest_point", "oof_correlation"]
    )
    material = bool(delta_oof >= 0.05)
    decision = (
        "published_cru_box_geometry_materially_improves_npl065_skill"
        if material
        else "published_cru_box_geometry_not_material_for_npl065_skill_gap"
    )

    pd.DataFrame(rows).to_csv(output / "target_geometry_oof_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "target_geometry_fold_metrics.csv", index=False)
    pd.DataFrame({
        "year": YEARS,
        "chronology": proxy,
        "nearest_point_observed": point_target,
        "nearest_point_oof": predictions["nearest_point"],
        "published_box_observed": box_target,
        "published_box_oof": predictions["published_box"],
    }).to_csv(output / "target_geometry_curves.csv", index=False)

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 6.4), sharex=True, constrained_layout=True)
    colors = {"nearest_point": "#C44E52", "published_box": "#006D77"}
    titles = {"nearest_point": "Nearest CRU cell", "published_box": "Published four-cell CRU box"}
    for axis, geometry in zip(axes, targets):
        row = metrics.loc[geometry]
        axis.plot(YEARS, targets[geometry], color=colors[geometry], linewidth=1.5, label="Observed DJF Tmin")
        axis.plot(YEARS, predictions[geometry], color="#222222", linewidth=1.2, linestyle="--", label=f"OOF, r={row.oof_correlation:.3f}")
        axis.set_ylabel("degC")
        axis.set_title(titles[geometry])
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=2, fontsize=9)
    axes[-1].set_xlabel("Ring year")
    figure.savefig(output / "npl065_cru_target_geometry_oof_curves.png", dpi=220)
    plt.close(figure)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "published_box": box_check,
        "published_box_station_supported_cell_month_fraction": station_support,
        "metrics": rows,
        "box_minus_point_oof_correlation": delta_oof,
        "material_effect_threshold": 0.05,
        "material_effect": material,
        "decision": decision,
        "published_1960_2013_in_sample_r_directly_compared": False,
        "locked_1921_1939_values_accessed": False,
        "paleo_proxy_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 244 NPL065 published CRU-box target geometry\n\n"
        f"- Nearest-cell OOF r: `{metrics.at['nearest_point', 'oof_correlation']:.3f}`.\n"
        f"- Published-box OOF r: `{metrics.at['published_box', 'oof_correlation']:.3f}`.\n"
        f"- Box-minus-point delta r: `{delta_oof:.3f}`.\n"
        f"- Published-box raw chronology-target r: `{metrics.at['published_box', 'raw_proxy_target_correlation']:.3f}`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
