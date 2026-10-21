from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "temperature_annual_maat_modern_anchored_anomalies_v1"
COVERAGE_DIR = LOOP / "temperature_annual_maat_modern_anchor_coverage_v1"
POINTS = COVERAGE_DIR / "annual_maat_safe_points_1500_1920_1940_2000.csv"
COVERAGE = COVERAGE_DIR / "modern_anchor_coverage.csv"
HAIZI = LOOP / "round_216_haizi_annual_temperature_locked_safe_comparison_v1/annual_allowed_curves.csv"
PUBLIC = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
LOCKED = (1921, 1939)

SITES = {
    "Aweng": (32.75, 81.67, "two_anchor_sensitivity_only"),
    "Lugu": (27.6833, 100.75, "broad_anchor_primary"),
    "Ngamring": (29.31, 87.18, "broad_anchor_primary"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bilinear(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, site_lat: float, site_lon: float) -> np.ndarray:
    data = np.asarray(field, dtype=float)
    if lat[1] < lat[0]:
        lat = lat[::-1]
        data = data[:, ::-1, :]
    if lon[1] < lon[0]:
        lon = lon[::-1]
        data = data[:, :, ::-1]
    fy = (site_lat - lat[0]) / (lat[1] - lat[0])
    fx = (site_lon - lon[0]) / (lon[1] - lon[0])
    iy, ix = int(np.floor(fy)), int(np.floor(fx))
    wy, wx = fy - iy, fx - ix
    if iy < 0 or ix < 0 or iy >= len(lat) - 1 or ix >= len(lon) - 1:
        raise RuntimeError("site-outside-grid")
    corners = np.stack([data[:, iy, ix], data[:, iy, ix + 1], data[:, iy + 1, ix], data[:, iy + 1, ix + 1]])
    weights = np.asarray([(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx])
    valid = np.isfinite(corners) & (corners > -9000)
    numerator = np.nansum(np.where(valid, corners * weights[:, None], 0.0), axis=0)
    denominator = np.sum(np.where(valid, weights[:, None], 0.0), axis=0)
    return np.divide(numerator, denominator, out=np.full(data.shape[0], np.nan), where=denominator > 0)


def project_at_years(dataset: h5py.File, lat: np.ndarray, lon: np.ndarray, requested: np.ndarray, site_lat: float, site_lon: float) -> np.ndarray:
    years = np.asarray(dataset["time"], dtype=int).ravel()
    if np.any((requested >= LOCKED[0]) & (requested <= LOCKED[1])):
        raise RuntimeError("locked-year-request")
    indices = np.asarray([int(np.where(years == int(round(year)))[0][0]) for year in requested], dtype=int)
    fields = np.stack([np.asarray(dataset["tas_ensemble_mean_rec_result"][index], dtype=float) for index in indices])
    return bilinear(fields, lat, lon, site_lat, site_lon)


def metrics(record: str, role: str, table: pd.DataFrame, anchor_count: int, leave_one_out: list[float]) -> dict:
    residual = table["project_anomaly_degC"] - table["external_anomaly_degC"]
    years = table["year"].to_numpy(float)
    external = table["external_anomaly_degC"].to_numpy(float)
    project = table["project_anomaly_degC"].to_numpy(float)
    return {
        "record": record,
        "role": role,
        "historical_point_count": int(len(table)),
        "modern_anchor_count": int(anchor_count),
        "mean_project_minus_external_degC": float(np.mean(residual)),
        "median_project_minus_external_degC": float(np.median(residual)),
        "rmse_degC": float(np.sqrt(np.mean(np.square(residual)))),
        "fraction_project_warmer": float(np.mean(residual > 0)),
        "external_trend_degC_per_century": float(np.polyfit(years, external, 1)[0] * 100.0),
        "project_trend_degC_per_century": float(np.polyfit(years, project, 1)[0] * 100.0),
        "anchor_leave_one_out_median_residual_min_degC": float(np.min(leave_one_out)) if leave_one_out else np.nan,
        "anchor_leave_one_out_median_residual_max_degC": float(np.max(leave_one_out)) if leave_one_out else np.nan,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [POINTS, COVERAGE, HAIZI]
    before = {str(path): sha256(path) for path in sources}
    points = pd.read_csv(POINTS)
    coverage = pd.read_csv(COVERAGE)
    curve_rows = []
    metric_rows = []

    with h5py.File(PUBLIC, "r") as dataset:
        lat = np.asarray(dataset["lat"], dtype=float).ravel()
        lon = np.asarray(dataset["lon"], dtype=float).ravel()
        for record, (site_lat, site_lon, role) in SITES.items():
            selected = points[points["record"] == record].copy()
            historical = selected[selected["year_ce"] <= 1849.999999].sort_values("year_ce")
            anchors = selected[selected["year_ce"].between(1940, 2000.999999)].sort_values("year_ce")
            required = 3 if role == "broad_anchor_primary" else 2
            if len(anchors) < required:
                raise RuntimeError(f"insufficient-anchor:{record}")
            all_years = np.concatenate([historical["year_ce"].to_numpy(float), anchors["year_ce"].to_numpy(float)])
            project = project_at_years(dataset, lat, lon, all_years, site_lat, site_lon)
            n_hist = len(historical)
            project_historical, project_anchor = project[:n_hist], project[n_hist:]
            external_anchor = anchors["temperature_c"].to_numpy(float)
            external_anomaly = historical["temperature_c"].to_numpy(float) - float(np.mean(external_anchor))
            project_anomaly = project_historical - float(np.mean(project_anchor))
            table = pd.DataFrame({
                "record": record, "role": role, "year": historical["year_ce"].to_numpy(float),
                "external_anomaly_degC": external_anomaly, "project_anomaly_degC": project_anomaly,
                "project_minus_external_degC": project_anomaly - external_anomaly,
            })
            curve_rows.append(table)
            loo = []
            if len(anchors) >= 3:
                for held in range(len(anchors)):
                    keep = np.arange(len(anchors)) != held
                    ext = historical["temperature_c"].to_numpy(float) - float(np.mean(external_anchor[keep]))
                    proj = project_historical - float(np.mean(project_anchor[keep]))
                    loo.append(float(np.median(proj - ext)))
            metric_rows.append(metrics(record, role, table, len(anchors), loo))

    haizi = pd.read_csv(HAIZI)
    haizi = haizi[haizi["year"].between(1500, 1849)].copy()
    haizi_table = pd.DataFrame({
        "record": "Haizi", "role": "strict_1961_1990_primary", "year": haizi["year"].to_numpy(int),
        "external_anomaly_degC": haizi["haizi_annual_reconstruction_anomaly_degC"].to_numpy(float),
        "project_anomaly_degC": haizi["project_local_box_annual_anomaly_degC"].to_numpy(float),
    })
    haizi_table["project_minus_external_degC"] = haizi_table["project_anomaly_degC"] - haizi_table["external_anomaly_degC"]
    curve_rows.append(haizi_table)
    metric_rows.append(metrics("Haizi", "strict_1961_1990_primary", haizi_table, 30, []))

    curves = pd.concat(curve_rows, ignore_index=True)
    summary_table = pd.DataFrame(metric_rows)
    curves.to_csv(OUT / "modern_anchored_historical_anomalies.csv", index=False)
    summary_table.to_csv(OUT / "modern_anchored_record_metrics.csv", index=False)

    primary = summary_table[summary_table["role"].str.contains("primary")]
    primary_warm = int((primary["median_project_minus_external_degC"] > 0).sum())
    primary_cold = int((primary["median_project_minus_external_degC"] < 0).sum())
    same_sign = primary_warm == len(primary) or primary_cold == len(primary)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    colors = {"external": "#1f6f8b", "project": "#d1495b"}
    for axis, record in zip(axes.flat[:3], ["Lugu", "Ngamring", "Haizi"]):
        table = curves[curves["record"] == record]
        if record == "Haizi":
            display = table.assign(
                external=table["external_anomaly_degC"].rolling(31, center=True, min_periods=16).mean(),
                project=table["project_anomaly_degC"].rolling(31, center=True, min_periods=16).mean(),
            )
        else:
            display = table.assign(external=table["external_anomaly_degC"], project=table["project_anomaly_degC"])
        axis.plot(display["year"], display["external"], color=colors["external"], marker="o" if record != "Haizi" else None, label="Independent annual-MAAT")
        axis.plot(display["year"], display["project"], color=colors["project"], marker="s" if record != "Haizi" else None, label="Project")
        axis.axhline(0, color="#777777", lw=0.8)
        axis.set_title(record)
        axis.set_ylabel("Anomaly from own modern anchor (degC)")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)

    order = summary_table["record"].tolist()
    values = summary_table["median_project_minus_external_degC"].to_numpy(float)
    bar_colors = ["#c44536" if value > 0 else "#2a9d8f" for value in values]
    axes[1, 1].barh(np.arange(len(order)), values, color=bar_colors)
    axes[1, 1].axvline(0, color="#555555", lw=0.8)
    axes[1, 1].set_yticks(np.arange(len(order)), order)
    axes[1, 1].set_xlabel("Median project minus independent record (degC)")
    axes[1, 1].set_title("Modern-anchored absolute residual")
    axes[1, 1].grid(axis="x", alpha=0.2)
    fig.suptitle("Annual-MAAT absolute-level test using record-specific modern anchors", fontsize=14.5, fontweight="bold")
    fig.savefig(OUT / "annual_maat_modern_anchored_absolute_bias.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment": OUT.name,
        "status": "complete",
        "primary_records": primary["record"].tolist(),
        "primary_record_count": int(len(primary)),
        "primary_project_warmer_count": primary_warm,
        "primary_project_colder_count": primary_cold,
        "primary_records_same_residual_sign": bool(same_sign),
        "uniform_absolute_warm_bias_supported": bool(same_sign and primary_warm == len(primary)),
        "anchor_definition": "record and project separately centered on identical safe modern sample years; Haizi uses full 1961-1990 climatology",
        "interpretation_boundary": "Lugu and Ngamring broad anchors contain only 3-4 postlock samples; leave-one-anchor-out sensitivity is reported. Aweng has two anchors and is sensitivity only.",
        "sealed_1921_1939_values_used": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {str(path): sha256(path) for path in sources}
    manifest["large_project_source"] = {"path": str(PUBLIC), "read_rule": "only explicit safe years; locked years never requested"}
    manifest["prohibited_source_accessed"] = False
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Modern-anchored annual-MAAT absolute-bias test", "",
        f"Primary records with project warmer residual: `{primary_warm}`; colder residual: `{primary_cold}`.",
        f"Uniform absolute warm bias supported: `{bool(same_sign and primary_warm == len(primary))}`.", "",
    ]
    for row in summary_table.itertuples(index=False):
        lines.append(f"- {row.record}: median project-minus-record `{row.median_project_minus_external_degC:+.3f} degC`; role `{row.role}`.")
    lines.extend([
        "", "Each record and project series was centered on identical safe modern sample years; Haizi uses 1961-1990.",
        "The sealed interval was never requested. Production was unchanged, EnKF and OGGM were not run, and 20CRv3 was not accessed.",
    ])
    (OUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {str(path): sha256(path) for path in sources}
    if before != after:
        raise RuntimeError("source-changed")


if __name__ == "__main__":
    main()
