from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1952, 2001, dtype=int)
BASELINE = YEARS >= 1960
HOLDOUT = YEARS <= 1959
LAT_RANGE = (27.5, 28.5)
LON_RANGE = (86.0, 87.0)
SITE_LAT = 27.89
SITE_LON = 86.37


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def area_mean(field: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None]
    valid = np.isfinite(field)
    numerator = np.nansum(field * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def read_cru_tmean(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    monthly = {}
    files = []
    for record in manifest["files"]:
        start, end = [int(value) for value in record["allowed_year_range"].split("-")]
        if end < 1951 or start > 2000:
            continue
        path = Path(record["netcdf_path"])
        if fingerprint(path)["sha256"] != record["netcdf_sha256"]:
            raise RuntimeError(f"cru-tmean-hash-changed:{path.name}")
        files.append(path)
        with netcdf_file(path, "r", mmap=True) as dataset:
            lats = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
            lons = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
            iy = np.flatnonzero((lats >= LAT_RANGE[0]) & (lats <= LAT_RANGE[1]))
            ix = np.flatnonzero((lons >= LON_RANGE[0]) & (lons <= LON_RANGE[1]))
            values = np.asarray(dataset.variables["tmp"].data[:, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1], dtype=float).copy()
            fill = float(dataset.variables["tmp"]._attributes["_FillValue"])
            values[np.isclose(values, fill) | (values > 1e20)] = np.nan
            lats_selected = lats[iy]
            lons_selected = lons[ix]
        periods = pd.period_range(f"{start}-01", periods=12 * (end - start + 1), freq="M")
        means = area_mean(values, lats_selected)
        for period, value in zip(periods, means):
            monthly[(int(period.year), int(period.month))] = float(value)
    result = np.full(len(YEARS), np.nan)
    for index, year in enumerate(YEARS):
        keys = ((int(year) - 1, 12), (int(year), 1), (int(year), 2))
        result[index] = np.mean([monthly[key] for key in keys])
    if not np.isfinite(result).all():
        raise RuntimeError("incomplete-cru-tmean")
    return result, files, {"latitude_centers": lats_selected.tolist(), "longitude_centers": lons_selected.tolist()}


def read_berkeley(path: Path):
    with h5py.File(path, "r") as handle:
        lats = np.asarray(handle["latitude"], dtype=float)
        lons = np.asarray(handle["longitude"], dtype=float)
        iy = int(np.argmin(np.abs(lats - SITE_LAT)))
        ix = int(np.argmin(np.abs(lons - SITE_LON)))
        start = (1951 - 1850) * 12
        stop = (2001 - 1850) * 12
        values = np.asarray(handle["temperature"][start:stop, iy, ix], dtype=float)
        values[(values <= -9000) | (values >= 1e20)] = np.nan
        cell = {"latitude": float(lats[iy]), "longitude": float(lons[ix])}
    periods = pd.period_range("1951-01", "2000-12", freq="M")
    monthly = {(int(period.year), int(period.month)): float(value) for period, value in zip(periods, values)}
    result = np.full(len(YEARS), np.nan)
    for index, year in enumerate(YEARS):
        keys = ((int(year) - 1, 12), (int(year), 1), (int(year), 2))
        result[index] = np.mean([monthly[key] for key in keys])
    if not np.isfinite(result).all():
        raise RuntimeError("incomplete-berkeley-tmean")
    return result, cell


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_260_npl065_1950s_temperature_product_level_audit_v1"
    output.mkdir(parents=True, exist_ok=False)
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
        "round258_curves": loop / "round_258_npl065_precalibration_holdout_v2/precalibration_holdout_curves.csv",
        "round258_summary": loop / "round_258_npl065_precalibration_holdout_v2/summary.json",
        "cru_tmean_manifest": loop / "round_240_cru_ts401_tmean_same_version_acquisition_v1/acquisition_manifest.json",
        "berkeley": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    curves = pd.read_csv(paths["round258_curves"]).set_index("year").loc[YEARS]
    cru_tmin = curves.cru_box_djf_tmin.to_numpy(float)
    sf_prediction = curves.signal_free_prediction.to_numpy(float)
    mod_prediction = curves.modnegex_prediction.to_numpy(float)
    cru_tmean, cru_files, cru_box = read_cru_tmean(paths["cru_tmean_manifest"])
    berkeley, berkeley_cell = read_berkeley(paths["berkeley"])
    paths.update({f"cru_tmean_{index:02d}": path for index, path in enumerate(cru_files, start=1)})
    before.update({name: fingerprint(path) for name, path in paths.items() if name not in before})
    series = {
        "CRU_TS401_Tmin": cru_tmin,
        "CRU_TS401_Tmean": cru_tmean,
        "Berkeley_Tmean": berkeley,
        "signal_free_prediction": sf_prediction,
        "modnegex_prediction": mod_prediction,
    }
    anomalies = {name: values - float(np.mean(values[BASELINE])) for name, values in series.items()}
    rows = []
    for name, values in series.items():
        anomaly = anomalies[name]
        rows.append({
            "series": name,
            "baseline_1960_2000_mean_degC": float(np.mean(values[BASELINE])),
            "holdout_1952_1959_mean_anomaly_degC": float(np.mean(anomaly[HOLDOUT])),
            "holdout_1952_1959_sd_degC": float(np.std(anomaly[HOLDOUT], ddof=1)),
            "holdout_to_baseline_mean_difference_degC": float(np.mean(values[HOLDOUT]) - np.mean(values[BASELINE])),
        })
    product_names = ("CRU_TS401_Tmin", "CRU_TS401_Tmean", "Berkeley_Tmean")
    correlation_rows = []
    for left_index, left in enumerate(product_names):
        for right in product_names[left_index + 1:]:
            correlation_rows.append({
                "left": left,
                "right": right,
                "annual_1952_2000_correlation": correlation(anomalies[left], anomalies[right]),
                "first_difference_correlation": correlation(np.diff(anomalies[left]), np.diff(anomalies[right])),
            })
    summary_frame = pd.DataFrame(rows).set_index("series")
    pd.DataFrame(rows).to_csv(output / "1950s_level_summary.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(output / "temperature_product_correlations.csv", index=False)
    pd.DataFrame({"year": YEARS, **{f"{name}_anomaly": values for name, values in anomalies.items()}}).to_csv(output / "1950s_temperature_product_curves.csv", index=False)

    colors = {
        "CRU_TS401_Tmin": "#0072B2",
        "CRU_TS401_Tmean": "#009E73",
        "Berkeley_Tmean": "#CC79A7",
        "signal_free_prediction": "#D55E00",
        "modnegex_prediction": "#555555",
    }
    figure, axis = plt.subplots(figsize=(11.8, 5.2), constrained_layout=True)
    for name in series:
        style = "-" if name in product_names else "--"
        width = 1.5 if name in product_names else 1.2
        axis.plot(YEARS, anomalies[name], color=colors[name], linestyle=style, linewidth=width, label=f"{name}: 1950s mean={summary_frame.at[name, 'holdout_1952_1959_mean_anomaly_degC']:+.2f}")
    axis.axhline(0, color="#888888", linewidth=0.8)
    axis.axvspan(1951.5, 1959.5, color="#E9C46A", alpha=0.22)
    axis.set_xlabel("DJF ring year")
    axis.set_ylabel("anomaly from 1960-2000 mean (degC)")
    axis.set_title("NPL065 1950s temperature-level product audit")
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, fontsize=8.5)
    figure.savefig(output / "npl065_1950s_temperature_product_level.png", dpi=220)
    plt.close(figure)

    product_shifts = summary_frame.loc[list(product_names), "holdout_1952_1959_mean_anomaly_degC"].to_numpy(float)
    same_sign = bool(np.all(np.sign(product_shifts) == np.sign(product_shifts[0])))
    spread = float(product_shifts.max() - product_shifts.min())
    if same_sign and spread <= 0.30:
        decision = "three_temperature_products_support_common_1950s_level_shift"
    elif same_sign:
        decision = "three_temperature_products_agree_on_sign_but_not_1950s_amplitude"
    else:
        decision = "temperature_products_disagree_on_1950s_level_sign"
    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "baseline_years": [1960, 2000],
        "holdout_years": [1952, 1959],
        "series_metrics": rows,
        "product_correlations": correlation_rows,
        "product_holdout_shift_same_sign": same_sign,
        "product_holdout_shift_spread_degC": spread,
        "cru_box": cru_box,
        "berkeley_cell": berkeley_cell,
        "decision": decision,
        "interpretation_boundary": "Tmin_and_Tmean_level_comparison_is_cross_variable_support_not_identity",
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")
    lines = ["# Round 260 NPL065 1950s temperature-product level audit", ""]
    for name in series:
        lines.append(f"- {name}: 1952-1959 anomaly `{summary_frame.at[name, 'holdout_1952_1959_mean_anomaly_degC']:+.3f} degC` from 1960-2000.")
    lines.extend(["", f"Decision: `{decision}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
