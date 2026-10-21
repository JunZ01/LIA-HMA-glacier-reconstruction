"""Test whether annual project P and T can explain the MADA PDSI phase conflict."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = PROJECT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUTPUT = LOOP / "round_104_mada_pt_moisture_operator_v1"
ROUND102 = LOOP / "round_102_tree_ring_hydroclimate_evidence_v4"
EXTERNAL = PROJECT / "validation_data/external_sources/hma_tree_ring_hydroclimate_audit_v1"
PUBLIC = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
MASK_PATH = PROJECT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
ACTUAL_PDSI = EXTERNAL / "jja-pdsi-act.txt"
XY = EXTERNAL / "jja-mada-xy.txt"
CAL_YEARS = np.arange(1940, 2001)
EARLY_YEARS = np.arange(1901, 1921)
LOCKED = set(range(1921, 1940))
REGIONS = {"hma": (-np.inf, np.inf), "west_lt80": (-np.inf, 80.0), "central_80_90": (80.0, 90.0), "east_ge90": (90.0, np.inf)}


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def parse_actual_pdsi(path: Path) -> tuple[np.ndarray, int]:
    rows: dict[int, np.ndarray] = {}
    locked_skipped = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = line.split()
            if not tokens: continue
            try: year = int(tokens[0])
            except ValueError: continue
            if year in LOCKED:
                locked_skipped += 1
                continue
            if year not in set(CAL_YEARS):
                continue
            values = np.asarray([float(value) for value in tokens[1:]], dtype=float)
            values[values < -90] = np.nan
            rows[year] = values
    if not np.array_equal(np.asarray(sorted(rows)), CAL_YEARS):
        raise RuntimeError("actual-pdsi-calibration-years-failed")
    widths = {len(row) for row in rows.values()}
    if len(widths) != 1:
        raise RuntimeError("actual-pdsi-column-count-varies")
    return np.stack([rows[int(year)] for year in CAL_YEARS]), locked_skipped


def point_overlap_weights(coords: np.ndarray, lat: np.ndarray, lon: np.ndarray, hma_mask: np.ndarray) -> np.ndarray:
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    fine_weights = np.cos(np.deg2rad(yy)) * hma_mask
    result = np.zeros(len(coords), dtype=float)
    for index, (longitude, latitude) in enumerate(coords):
        inside = (np.abs(yy - latitude) <= 1.25 + 1e-8) & (np.abs(xx - longitude) <= 1.25 + 1e-8)
        result[index] = float(np.sum(fine_weights[inside]))
    return result


def weighted_columns(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    denominator = np.sum(valid * weights[None], axis=1)
    return np.nansum(values * weights[None], axis=1) / np.where(denominator == 0, np.nan, denominator)


def weighted_field(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(values)
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return np.nansum(values * weights[None], axis=(1, 2)) / np.where(denominator == 0, np.nan, denominator)


def standardize(train: np.ndarray, apply: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(train)); scale = float(np.std(train, ddof=1))
    if scale <= 1e-12: raise RuntimeError("constant-calibration-predictor")
    return (apply - mean) / scale, mean, scale


def fit_predict(y: np.ndarray, p: np.ndarray, t: np.ndarray, use_temperature: bool) -> tuple[np.ndarray, np.ndarray]:
    columns = [np.ones(len(y)), p]
    if use_temperature: columns.append(t)
    x = np.column_stack(columns)
    coefficient, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coefficient, x @ coefficient


def cross_validated(y: np.ndarray, p: np.ndarray, t: np.ndarray, use_temperature: bool) -> np.ndarray:
    predicted = np.full(len(y), np.nan)
    decades = (CAL_YEARS // 10) * 10
    for block in np.unique(decades):
        test = decades == block; train = ~test
        yp, ym, ys = standardize(y[train], y)
        pp, pm, ps = standardize(p[train], p)
        tp, tm, ts = standardize(t[train], t)
        coefficient, _ = fit_predict(yp[train], pp[train], tp[train], use_temperature)
        columns = [np.ones(test.sum()), pp[test]]
        if use_temperature: columns.append(tp[test])
        predicted[test] = np.column_stack(columns) @ coefficient
    return predicted


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return float(np.corrcoef(left[valid], right[valid])[0, 1]) if valid.sum() >= 3 else np.nan


def trend(values: np.ndarray, years: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0) if valid.sum() >= 3 else np.nan


def main() -> int:
    sources = {"actual_pdsi": ACTUAL_PDSI, "xy": XY, "public_fields": PUBLIC, "hma_mask": MASK_PATH, "round102_curves": ROUND102 / "allowed_period_standardized_curves.csv", "driver": Path(__file__).resolve()}
    for path in sources.values():
        if not path.exists(): raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text("\n".join(["# Round 104 Contract", "", "- Calibrate annual project precipitation and temperature against actual JJA PDSI only over open years 1940-2000.", "- Apply a cross-validated diagnostic operator to 1901-1920 MADA reconstruction; do not access 1921-1939.", "- Interpret only regions with leave-one-decade-out correlation >=0.5 and physically signed coefficients (P positive, T negative).", "- This is a variable-definition mechanism test, not a replacement PDSI reconstruction or EnKF tuning."]) + "\n", encoding="utf-8")
    before = {name: state(path) for name, path in sources.items()}

    actual, locked_skipped = parse_actual_pdsi(ACTUAL_PDSI)
    coords = np.loadtxt(XY)
    if actual.shape[1] != len(coords):
        raise RuntimeError(f"actual-xy-column-conflict:{actual.shape[1]}:{len(coords)}")
    mask = np.load(MASK_PATH).astype(bool)
    with h5py.File(PUBLIC, "r") as handle:
        years = np.asarray(handle["time"], dtype=int).ravel()
        lat = np.asarray(handle["lat"], dtype=float).ravel(); lon = np.asarray(handle["lon"], dtype=float).ravel()
        cal_index = np.flatnonzero(np.isin(years, CAL_YEARS)); early_index = np.flatnonzero(np.isin(years, EARLY_YEARS))
        if not np.array_equal(years[cal_index], CAL_YEARS) or not np.array_equal(years[early_index], EARLY_YEARS): raise RuntimeError("project-year-axis-failed")
        if np.any(np.isin(years[np.concatenate((cal_index, early_index))], list(LOCKED))): raise RuntimeError("locked-project-index-requested")
        p_cal = np.asarray(handle["pr_ensemble_mean_rec_result"][cal_index], dtype=float); p_early = np.asarray(handle["pr_ensemble_mean_rec_result"][early_index], dtype=float)
        t_cal = np.asarray(handle["tas_ensemble_mean_rec_result"][cal_index], dtype=float); t_early = np.asarray(handle["tas_ensemble_mean_rec_result"][early_index], dtype=float)
    for values in (p_cal, p_early, t_cal, t_early): values[values < -9000] = np.nan
    if mask.shape != (len(lat), len(lon)): raise RuntimeError("mask-shape-failed")
    point_weights = point_overlap_weights(coords, lat, lon, mask)
    longitude_grid = np.broadcast_to(lon[None], mask.shape)
    mada_curves = pd.read_csv(ROUND102 / "allowed_period_standardized_curves.csv")

    rows = []; curve_rows = []
    for region, (lower, upper) in REGIONS.items():
        actual_weights = point_weights * (coords[:, 0] >= lower) * (coords[:, 0] < upper)
        target = weighted_columns(actual, actual_weights)
        region_mask = mask & (longitude_grid >= lower) & (longitude_grid < upper)
        pcal = weighted_field(p_cal, region_mask, lat); pearly = weighted_field(p_early, region_mask, lat)
        tcal = weighted_field(t_cal, region_mask, lat); tearly = weighted_field(t_early, region_mask, lat)
        yz, ymean, yscale = standardize(target, target)
        pcalz, pmean, pscale = standardize(pcal, pcal); pearlyz = (pearly - pmean) / pscale
        tcalz, tmean, tscale = standardize(tcal, tcal); tearlyz = (tearly - tmean) / tscale
        cv_p = cross_validated(target, pcal, tcal, False); cv_pt = cross_validated(target, pcal, tcal, True)
        coefficient, fitted = fit_predict(yz, pcalz, tcalz, True)
        early_prediction = coefficient[0] + coefficient[1] * pearlyz + coefficient[2] * tearlyz
        early_p_component = coefficient[1] * pearlyz; early_t_component = coefficient[2] * tearlyz
        mada = mada_curves[(mada_curves.period == "early_1901_1920") & (mada_curves.evidence == "MADA_JJA_PDSI") & (mada_curves.region == region) & (mada_curves.series == "evidence")].sort_values("year")
        mada_early = mada.raw_value.to_numpy(float); mada_z = (mada_early - ymean) / yscale
        cv_p_r = correlation(yz, cv_p); cv_pt_r = correlation(yz, cv_pt)
        physical = bool(coefficient[1] > 0 and coefficient[2] < 0)
        interpretable = bool(cv_pt_r >= 0.5 and physical)
        rows.append({"region": region, "actual_mada_cell_count": int(np.sum(actual_weights > 0)), "coefficient_precip_z": coefficient[1], "coefficient_temperature_z": coefficient[2], "physical_coefficient_signs": physical, "cv_precip_only_r": cv_p_r, "cv_precip_temperature_r": cv_pt_r, "cv_improvement": cv_pt_r - cv_p_r, "operator_interpretable": interpretable, "mada_early_trend_z_per_century": trend(mada_z, EARLY_YEARS), "operator_early_trend_z_per_century": trend(early_prediction, EARLY_YEARS), "precip_component_early_trend_z_per_century": trend(early_p_component, EARLY_YEARS), "temperature_component_early_trend_z_per_century": trend(early_t_component, EARLY_YEARS), "early_prediction_mada_r": correlation(early_prediction, mada_z), "early_trend_sign_match": bool(np.sign(trend(early_prediction, EARLY_YEARS)) == np.sign(trend(mada_z, EARLY_YEARS)))})
        for i, year in enumerate(EARLY_YEARS):
            curve_rows.append({"region": region, "year": int(year), "mada_pdsi_z": mada_z[i], "pt_operator_z": early_prediction[i], "precip_component_z": early_p_component[i], "temperature_component_z": early_t_component[i]})

    metrics = pd.DataFrame(rows); curves = pd.DataFrame(curve_rows)
    metrics.to_csv(OUTPUT / "mada_pt_operator_metrics.csv", index=False); curves.to_csv(OUTPUT / "early_operator_decomposition_curves.csv", index=False)
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    for ax, region in zip(axes, REGIONS):
        data = curves[curves.region == region]
        ax.plot(data.year, data.mada_pdsi_z, color="#111111", marker="o", linewidth=2.4, label="MADA JJA PDSI")
        ax.plot(data.year, data.pt_operator_z, color="#0072B2", marker="s", linewidth=2.0, label="P+T operator")
        ax.plot(data.year, data.precip_component_z, color="#D55E00", linewidth=1.2, label="P component")
        ax.plot(data.year, data.temperature_component_z, color="#009E73", linewidth=1.2, label="T component")
        ax.axhline(0, color="#888888", linewidth=0.7); ax.set_title(region); ax.set_ylabel("Calibration z")
    axes[0].legend(frameon=False, ncol=4); axes[-1].set_xlabel("Year")
    fig.suptitle("Can annual project P and T explain early MADA PDSI?", fontsize=15)
    fig.savefig(OUTPUT / "mada_pt_operator_decomposition.png", dpi=220); plt.close(fig)
    summary = {"status": "complete", "calibration_period": [1940, 2000], "application_period": [1901, 1920], "locked_1921_1939_accessed": False, "locked_actual_pdsi_rows_skipped": locked_skipped, "regional_metrics": rows, "interpretable_regions": metrics.loc[metrics.operator_interpretable, "region"].tolist(), "decision": "interpret_temperature_offset_only_where_cross_validation_and_physical_sign_gates_pass"}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Round 104 MADA P+T Moisture Operator", "", "The operator is diagnostic only and is interpreted only where leave-one-decade-out skill is at least 0.5 and precipitation/temperature coefficients have physical signs.", ""]
    for row in rows:
        lines.append(f"- {row['region']}: CV r(P)={row['cv_precip_only_r']:.3f}, CV r(P+T)={row['cv_precip_temperature_r']:.3f}, bP={row['coefficient_precip_z']:.3f}, bT={row['coefficient_temperature_z']:.3f}, early MADA/operator trends={row['mada_early_trend_z_per_century']:.2f}/{row['operator_early_trend_z_per_century']:.2f}, P/T contributions={row['precip_component_early_trend_z_per_century']:.2f}/{row['temperature_component_early_trend_z_per_century']:.2f}, interpretable={str(row['operator_interpretable']).lower()}.")
    lines.extend(["", f"Decision: {summary['decision']}."])
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in sources.items()}; manifest = {"sources_before": before, "sources_after": after, "source_states_unchanged": before == after, "locked_1921_1939_accessed": False}
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["source_states_unchanged"]: raise RuntimeError("source-integrity-failed")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
