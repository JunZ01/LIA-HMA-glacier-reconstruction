from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1952, 2014, dtype=int)
HOLDOUT = YEARS <= 1959
CALIBRATION = YEARS >= 1960
LAT_RANGE = (27.5, 28.5)
LON_RANGE = (86.0, 87.0)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_259_npl065_cru_early_station_count_audit_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
        "pre2001_manifest": loop / "round_208_setp_cru_tmin_annual_target_audit_v1/cru_tmn_acquisition_manifest.json",
        "post2000_manifest": loop / "round_245_cru_ts401_tmin_post2000_acquisition_v1/acquisition_manifest.json",
        "round258_summary": loop / "round_258_npl065_precalibration_holdout_v2/summary.json",
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    monthly = {}
    source_files = []
    for manifest_path in (paths["pre2001_manifest"], paths["post2000_manifest"]):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["files"]:
            start, end = [int(value) for value in record["allowed_year_range"].split("-")]
            if end < 1951 or start > 2013:
                continue
            if start <= 1939:
                raise RuntimeError(f"forbidden-range:{start}-{end}")
            path = Path(record["netcdf_path"])
            if fingerprint(path)["sha256"] != record["netcdf_sha256"]:
                raise RuntimeError(f"cru-hash-changed:{path.name}")
            source_files.append(path)
            with netcdf_file(path, "r", mmap=True) as dataset:
                lats = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
                lons = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
                iy = np.flatnonzero((lats >= LAT_RANGE[0]) & (lats <= LAT_RANGE[1]))
                ix = np.flatnonzero((lons >= LON_RANGE[0]) & (lons <= LON_RANGE[1]))
                counts = np.asarray(dataset.variables["stn"].data[:, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1], dtype=float).copy()
                lat_centers = lats[iy].tolist()
                lon_centers = lons[ix].tolist()
            periods = pd.period_range(f"{start}-01", periods=12 * (end - start + 1), freq="M")
            for period, grid in zip(periods, counts):
                key = (int(period.year), int(period.month))
                if key in monthly:
                    raise RuntimeError(f"duplicate-month:{key}")
                monthly[key] = grid
    paths.update({f"cru_{index:02d}": path for index, path in enumerate(source_files, start=1)})
    before.update({name: fingerprint(path) for name, path in paths.items() if name not in before})

    yearly_rows = []
    period_values = {"1952-1959_holdout": [], "1960-2013_calibration": []}
    for year in YEARS:
        grids = np.stack([monthly[(int(year) - 1, 12)], monthly[(int(year), 1)], monthly[(int(year), 2)]])
        flat = grids.reshape(-1)
        period = "1952-1959_holdout" if year <= 1959 else "1960-2013_calibration"
        period_values[period].extend(flat.tolist())
        yearly_rows.append({
            "year": int(year),
            "period": period,
            "cell_month_count": len(flat),
            "station_count_min": float(flat.min()),
            "station_count_median": float(np.median(flat)),
            "station_count_mean": float(flat.mean()),
            "station_count_max": float(flat.max()),
            "cell_month_fraction_count_eq_1": float(np.mean(flat == 1)),
            "cell_month_fraction_count_ge_2": float(np.mean(flat >= 2)),
        })
    period_rows = []
    for period, values in period_values.items():
        array = np.asarray(values, dtype=float)
        period_rows.append({
            "period": period,
            "cell_month_count": len(array),
            "station_count_min": float(array.min()),
            "station_count_median": float(np.median(array)),
            "station_count_mean": float(array.mean()),
            "station_count_max": float(array.max()),
            "cell_month_fraction_count_eq_1": float(np.mean(array == 1)),
            "cell_month_fraction_count_ge_2": float(np.mean(array >= 2)),
            "cell_month_fraction_count_ge_3": float(np.mean(array >= 3)),
        })
    yearly = pd.DataFrame(yearly_rows)
    periods = pd.DataFrame(period_rows).set_index("period")
    yearly.to_csv(output / "cru_djf_station_count_by_year.csv", index=False)
    pd.DataFrame(period_rows).to_csv(output / "cru_djf_station_count_period_summary.csv", index=False)

    figure, axis = plt.subplots(figsize=(11.5, 4.6), constrained_layout=True)
    axis.plot(yearly.year, yearly.station_count_mean, color="#0072B2", linewidth=1.6, label="Mean station count across 12 cell-months")
    axis.fill_between(yearly.year, yearly.station_count_min, yearly.station_count_max, color="#56B4E9", alpha=0.22, label="Cell-month min-max")
    axis.axvspan(1951.5, 1959.5, color="#E9C46A", alpha=0.25, label="1952-1959 holdout")
    axis.set_xlabel("DJF ring year")
    axis.set_ylabel("CRU stn count")
    axis.set_title("CRU TS4.01 station support in the NPL065 four-cell box")
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3, fontsize=9)
    figure.savefig(output / "npl065_cru_early_station_count.png", dpi=220)
    plt.close(figure)

    holdout = periods.loc["1952-1959_holdout"]
    calibration = periods.loc["1960-2013_calibration"]
    ratio = float(holdout.station_count_mean / calibration.station_count_mean)
    if holdout.station_count_median < 2 or holdout.cell_month_fraction_count_eq_1 > 0.5:
        decision = "early_cru_holdout_is_station_sparse_and_not_decisive_for_level_transport"
    else:
        decision = "early_cru_holdout_has_multi_station_support_level_failure_remains_relevant"
    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "box": {"latitude_centers": lat_centers, "longitude_centers": lon_centers, "cell_count": 4},
        "period_metrics": period_rows,
        "holdout_to_calibration_mean_station_count_ratio": ratio,
        "decision": decision,
        "interpretation_boundary": "CRU_stn_is_support_metadata_not_direct_accuracy_validation",
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 259 NPL065 CRU early station-count audit\n\n"
        f"- Holdout station count mean/median: `{holdout.station_count_mean:.2f}/{holdout.station_count_median:.1f}`.\n"
        f"- Calibration station count mean/median: `{calibration.station_count_mean:.2f}/{calibration.station_count_median:.1f}`.\n"
        f"- Holdout cell-month fraction with exactly one station: `{holdout.cell_month_fraction_count_eq_1:.1%}`.\n"
        f"- Holdout/calibration mean-count ratio: `{ratio:.3f}`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
