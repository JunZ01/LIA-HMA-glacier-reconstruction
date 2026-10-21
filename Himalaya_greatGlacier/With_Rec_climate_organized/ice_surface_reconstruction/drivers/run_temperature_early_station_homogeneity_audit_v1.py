from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized"
    / "ice_surface_reconstruction/drivers"
)
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
BENCHMARK = LOOP / "temperature_station_annual_anomaly_benchmark_v4"
ROUND277 = LOOP / "round_277_direct_temperature_station_panel_1850_1950_v2"
OUTPUT = LOOP / "temperature_early_station_homogeneity_audit_v1"
CONTRACT = DRIVERS / "temperature_early_station_homogeneity_audit_v1_contract.md"
DRIVER = Path(__file__).resolve()
FLAGS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/v4/ghcnm-flags.txt"
LOCKED_START = 1921
LOCKED_END = 1939
EARLY_START = 1901
EARLY_END = 1920
MIN_MATCHED_YEARS = 5


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def trend(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(years) & np.isfinite(values)
    if int(valid.sum()) < 5:
        return math.nan
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return math.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def parse_flag_file(path: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if len(line) < 3 or line[0].isspace() or line.startswith("*"):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and len(parts[0]) == 1:
            mapping[parts[0]] = {"origin": parts[1], "description": parts[2]}
    return mapping


def one_split_diagnostic(years: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(years) & np.isfinite(values)
    years = years[valid]
    values = values[valid]
    if len(values) < 12:
        return {
            "eligible": False,
            "split_year": None,
            "mean_shift_degC": math.nan,
            "sse_reduction_fraction": math.nan,
            "strong_shift": False,
        }
    order = np.argsort(years)
    years = years[order]
    values = values[order]
    sse0 = float(np.sum((values - np.mean(values)) ** 2))
    candidates = []
    for index in range(5, len(values) - 4):
        left = values[:index]
        right = values[index:]
        sse = float(
            np.sum((left - np.mean(left)) ** 2)
            + np.sum((right - np.mean(right)) ** 2)
        )
        candidates.append((sse, index))
    best_sse, best_index = min(candidates)
    shift = float(np.mean(values[best_index:]) - np.mean(values[:best_index]))
    reduction = float((sse0 - best_sse) / max(sse0, 1e-12))
    return {
        "eligible": True,
        "split_year": int(years[best_index]),
        "mean_shift_degC": shift,
        "sse_reduction_fraction": reduction,
        "strong_shift": bool(abs(shift) >= 0.30 and reduction >= 0.50),
    }


def station_metrics(curves: pd.DataFrame, source_text: dict[str, str]) -> pd.DataFrame:
    rows = []
    for station_id, part in curves.groupby("station_id"):
        part = part.sort_values("year")
        same = part[
            part.year.between(EARLY_START, EARLY_END)
            & part.qcf_station_degC.notna()
            & part.qcu_station_degC.notna()
            & part.project_six_member_degC.notna()
        ].copy()
        if len(same) < MIN_MATCHED_YEARS:
            continue
        qcf = same.qcf_station_degC.to_numpy(float)
        qcu = same.qcu_station_degC.to_numpy(float)
        project = same.project_six_member_degC.to_numpy(float)
        years = same.year.to_numpy(float)
        qcf_trend = trend(years, qcf)
        qcu_trend = trend(years, qcu)
        project_trend = trend(years, project)
        qcf_bias = float(np.mean(project - qcf))
        qcu_bias = float(np.mean(project - qcu))
        stable = bool(
            np.sign(qcf_trend) == np.sign(qcu_trend)
            and correlation(qcf, qcu) >= 0.80
            and qcf_bias > 0
            and qcu_bias > 0
        )
        if stable and len(same) >= 10:
            grade = "core"
        elif stable:
            grade = "supporting"
        else:
            grade = "homogenization_sensitive"
        meta = part.iloc[0]
        rows.append(
            {
                "station_id": station_id,
                "station_name": meta.station_name,
                "country": meta.country,
                "latitude": meta.latitude,
                "longitude": meta.longitude,
                "elevation_m": meta.elevation_m,
                "legacy_xlt_id": "XLT" in station_id,
                "matched_years": len(same),
                "first_year": int(same.year.min()),
                "last_year": int(same.year.max()),
                "source_lineages": source_text.get(station_id, "unknown"),
                "qcf_qcu_correlation": correlation(qcf, qcu),
                "qcf_minus_qcu_mean_degC": float(np.mean(qcf - qcu)),
                "project_minus_qcf_mean_degC": qcf_bias,
                "project_minus_qcu_mean_degC": qcu_bias,
                "qcf_trend_degC_century": qcf_trend,
                "qcu_trend_degC_century": qcu_trend,
                "project_trend_degC_century": project_trend,
                "qcf_qcu_trend_sign_agreement": bool(
                    np.sign(qcf_trend) == np.sign(qcu_trend)
                ),
                "project_qcf_trend_sign_agreement": bool(
                    np.sign(project_trend) == np.sign(qcf_trend)
                ),
                "evidence_grade": grade,
            }
        )
    return pd.DataFrame(rows).sort_values(["evidence_grade", "station_id"])


def adjustment_diagnostics(curves: pd.DataFrame, station_ids: set[str]) -> pd.DataFrame:
    rows = []
    for station_id, part in curves[curves.station_id.isin(station_ids)].groupby("station_id"):
        paired = part[
            part.year.between(1880, 1920)
            & part.qcf_station_degC.notna()
            & part.qcu_station_degC.notna()
        ].copy()
        values = (
            paired.qcf_station_degC - paired.qcu_station_degC
        ).to_numpy(float)
        result = one_split_diagnostic(paired.year.to_numpy(float), values)
        rows.append(
            {
                "station_id": station_id,
                "station_name": part.station_name.iloc[0],
                "year_count": len(paired),
                "mean_qcf_minus_qcu_degC": float(np.mean(values)) if len(values) else math.nan,
                "max_abs_qcf_minus_qcu_degC": float(np.max(np.abs(values))) if len(values) else math.nan,
                **result,
            }
        )
    return pd.DataFrame(rows)


def leave_one_out(metrics: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    groups = sorted(metrics[group_column].unique())
    for excluded in ["none", *groups]:
        kept = metrics if excluded == "none" else metrics[metrics[group_column].ne(excluded)]
        rows.append(
            {
                "exclusion_type": group_column,
                "excluded": excluded,
                "station_count": len(kept),
                "positive_qcf_bias_fraction": float(
                    np.mean(kept.project_minus_qcf_mean_degC > 0)
                ),
                "positive_qcu_bias_fraction": float(
                    np.mean(kept.project_minus_qcu_mean_degC > 0)
                ),
                "median_project_minus_qcf_degC": float(
                    kept.project_minus_qcf_mean_degC.median()
                ),
                "median_project_minus_qcu_degC": float(
                    kept.project_minus_qcu_mean_degC.median()
                ),
                "project_qcf_sign_agreement_fraction": float(
                    kept.project_qcf_trend_sign_agreement.mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_curves(curves: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True, constrained_layout=True)
    for axis, row in zip(axes.flat, metrics.sort_values("longitude").itertuples(index=False)):
        part = curves[
            curves.station_id.eq(row.station_id)
            & curves.year.between(EARLY_START, EARLY_END)
        ]
        axis.plot(part.year, part.qcf_station_degC, color="#111111", marker="o", linewidth=1.5, label="qcf")
        axis.plot(part.year, part.qcu_station_degC, color="#777777", marker=".", linestyle="--", linewidth=1.0, label="qcu")
        axis.plot(part.year, part.project_six_member_degC, color="#D55E00", marker="s", linewidth=1.4, label="project")
        axis.axhline(0, color="#AAAAAA", linewidth=0.6)
        axis.set_title(f"{row.station_name} | {row.evidence_grade}")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, ncol=3)
    fig.supxlabel("Year")
    fig.supylabel("Annual anomaly relative to 1961-1990 (degC)")
    fig.savefig(path, dpi=210)
    plt.close(fig)


def plot_adjustments(curves: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for row in metrics.sort_values("longitude").itertuples(index=False):
        part = curves[
            curves.station_id.eq(row.station_id)
            & curves.year.between(1880, 1920)
        ].copy()
        part["adjustment"] = part.qcf_station_degC - part.qcu_station_degC
        axis.plot(part.year, part.adjustment, marker=".", linewidth=1.2, label=row.station_name)
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set(xlabel="Year", ylabel="qcf minus qcu anomaly (degC)", title="GHCN homogenization sensitivity at early benchmark stations")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    fig.savefig(path, dpi=210)
    plt.close(fig)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    inputs = {
        "contract": CONTRACT,
        "driver": DRIVER,
        "benchmark_curves": BENCHMARK / "station_annual_anomalies_and_products.csv",
        "benchmark_metrics": BENCHMARK / "station_product_metrics.csv",
        "round277_monthly": ROUND277 / "monthly_temperature_panel.csv",
        "round277_metadata": ROUND277 / "station_metadata.csv",
        "qcf_inventory": ROUND277 / "prepared/ghcnm_tavg_qcf.inventory",
    }
    for path in inputs.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: state(path) for name, path in inputs.items()}

    curves = pd.read_csv(inputs["benchmark_curves"])
    monthly = pd.read_csv(inputs["round277_monthly"], keep_default_na=False)
    if curves.year.between(LOCKED_START, LOCKED_END).any() or monthly.year.astype(int).between(LOCKED_START, LOCKED_END).any():
        raise RuntimeError("locked-year-present-in-derived-input")
    metric_source = pd.read_csv(inputs["benchmark_metrics"])
    selected = metric_source[
        metric_source.product.eq("project_six_member_degC")
        & metric_source.segment.eq("1901_1920")
        & metric_source.year_count.ge(MIN_MATCHED_YEARS)
    ].station_id.unique()
    if len(selected) != 6:
        raise RuntimeError(f"expected-six-early-stations:{len(selected)}")
    curves = curves[curves.station_id.isin(selected)].copy()
    metadata = pd.read_csv(inputs["round277_metadata"])
    curves = curves.drop(columns=["country"], errors="ignore").merge(
        metadata[["station_id", "country"]], on="station_id", validate="many_to_one"
    )

    inventory_lines = inputs["qcf_inventory"].read_text(encoding="ascii").splitlines()
    inventory_rows = []
    for station_id in selected:
        matches = [line for line in inventory_lines if line.startswith(station_id)]
        if len(matches) != 1:
            raise RuntimeError(f"inventory-identity-count:{station_id}:{len(matches)}")
        parts = matches[0].split()
        inventory_rows.append(
            {
                "station_id": station_id,
                "inventory_latitude": float(parts[1]),
                "inventory_longitude": float(parts[2]),
                "inventory_elevation_m": float(parts[3]),
                "inventory_name": "_".join(parts[4:]),
                "legacy_xlt_id": "XLT" in station_id,
                "raw_inventory_line": matches[0],
            }
        )
    inventory = pd.DataFrame(inventory_rows)
    official = metadata[metadata.station_id.isin(selected)].merge(inventory, on="station_id", validate="one_to_one")
    if not np.allclose(official.latitude, official.inventory_latitude) or not np.allclose(official.longitude, official.inventory_longitude):
        raise RuntimeError("station-coordinate-identity-mismatch")

    temp_dir = Path(tempfile.mkdtemp(prefix="early_station_homogeneity_", dir=LOOP))
    try:
        raw_dir = temp_dir / "raw_official"
        raw_dir.mkdir()
        flags_path = raw_dir / "ghcnm-flags.txt"
        urllib.request.urlretrieve(FLAGS_URL, flags_path)
        flag_map = parse_flag_file(flags_path)
        flag_rows = []
        source_text = {}
        selected_monthly = monthly[
            monthly.station_id.isin(selected)
            & monthly.quality_accepted.astype(str).str.lower().eq("true")
        ].copy()
        for station_id, part in selected_monthly.groupby("station_id"):
            counts = part.groupby("source_flag").size().sort_values(ascending=False)
            labels = []
            for flag, count in counts.items():
                record = flag_map.get(flag, {"origin": "unknown", "description": "unknown"})
                flag_rows.append(
                    {
                        "station_id": station_id,
                        "source_flag": flag,
                        "origin": record["origin"],
                        "description": record["description"],
                        "accepted_month_rows_qcf_plus_qcu": int(count),
                    }
                )
                labels.append(f"{flag}:{record['description']}")
            source_text[station_id] = " | ".join(labels)
        flags = pd.DataFrame(flag_rows)

        metrics = station_metrics(curves, source_text)
        adjustments = adjustment_diagnostics(curves, set(selected))
        loo_station = leave_one_out(metrics, "station_id")
        loo_country = leave_one_out(metrics, "country")
        loo = pd.concat([loo_station, loo_country], ignore_index=True)

        inventory.to_csv(temp_dir / "official_station_identity.csv", index=False)
        flags.to_csv(temp_dir / "official_source_flag_composition.csv", index=False)
        metrics.to_csv(temp_dir / "early_station_metrics.csv", index=False)
        adjustments.to_csv(temp_dir / "qcf_qcu_adjustment_diagnostics.csv", index=False)
        loo.to_csv(temp_dir / "leave_one_out_robustness.csv", index=False)
        plot_curves(curves, metrics, temp_dir / "early_station_project_qcf_qcu_curves.png")
        plot_adjustments(curves, metrics, temp_dir / "qcf_qcu_adjustment_curves.png")

        full = loo[(loo.exclusion_type.eq("station_id")) & loo.excluded.eq("none")].iloc[0]
        station_loo = loo[(loo.exclusion_type.eq("station_id")) & loo.excluded.ne("none")]
        country_loo = loo[(loo.exclusion_type.eq("country")) & loo.excluded.ne("none")]
        robust = bool(
            station_loo.median_project_minus_qcf_degC.min() > 0
            and station_loo.median_project_minus_qcu_degC.min() > 0
            and country_loo.median_project_minus_qcf_degC.min() > 0
            and country_loo.median_project_minus_qcu_degC.min() > 0
        )
        summary = {
            "experiment": OUTPUT.name,
            "status": "complete",
            "selected_station_count": len(metrics),
            "selection_rule": "at_least_5_matched_qcf_project_years_in_1901_1920",
            "legacy_xlt_station_count": int(metrics.legacy_xlt_id.sum()),
            "source_lineage_count": int(flags[["origin", "description"]].drop_duplicates().shape[0]),
            "evidence_grade_counts": metrics.evidence_grade.value_counts().to_dict(),
            "positive_project_minus_qcf_station_fraction": float(np.mean(metrics.project_minus_qcf_mean_degC > 0)),
            "positive_project_minus_qcu_station_fraction": float(np.mean(metrics.project_minus_qcu_mean_degC > 0)),
            "full_median_project_minus_qcf_degC": float(full.median_project_minus_qcf_degC),
            "full_median_project_minus_qcu_degC": float(full.median_project_minus_qcu_degC),
            "leave_one_station_min_median_qcf_bias_degC": float(station_loo.median_project_minus_qcf_degC.min()),
            "leave_one_station_min_median_qcu_bias_degC": float(station_loo.median_project_minus_qcu_degC.min()),
            "leave_one_country_min_median_qcf_bias_degC": float(country_loo.median_project_minus_qcf_degC.min()),
            "leave_one_country_min_median_qcu_bias_degC": float(country_loo.median_project_minus_qcu_degC.min()),
            "strong_adjustment_shift_station_count": int(adjustments.strong_shift.sum()),
            "common_warm_residual_robust_to_homogenization_and_leave_one_out": robust,
            "decision": "western_1901_1920_common_warm_residual_supported" if robust else "western_1901_1920_residual_not_robust",
            "locked_1921_1939_accessed": False,
        }
        (temp_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        after = {name: state(path) for name, path in inputs.items()}
        manifest = {
            "experiment": OUTPUT.name,
            "inputs_before": before,
            "inputs_after": after,
            "inputs_unchanged": before == after,
            "official_flags_url": FLAGS_URL,
            "official_flags_file": state(flags_path),
            "locked_values_accessed": False,
        }
        if not manifest["inputs_unchanged"]:
            raise RuntimeError("input-integrity-failed")
        (temp_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        report = "\n".join(
            [
                "# Early Station Homogeneity Audit v1",
                "",
                f"- Selected stations: `{len(metrics)}`; legacy XLT identities: `{int(metrics.legacy_xlt_id.sum())}`.",
                f"- Distinct official source lineages: `{summary['source_lineage_count']}`.",
                f"- Positive project-minus-qcf stations: `{summary['positive_project_minus_qcf_station_fraction']:.3f}`.",
                f"- Positive project-minus-qcu stations: `{summary['positive_project_minus_qcu_station_fraction']:.3f}`.",
                f"- Full median project-minus-qcf/qcu: `{summary['full_median_project_minus_qcf_degC']:+.3f}` / `{summary['full_median_project_minus_qcu_degC']:+.3f} degC`.",
                f"- Minimum leave-one-station median qcf/qcu residual: `{summary['leave_one_station_min_median_qcf_bias_degC']:+.3f}` / `{summary['leave_one_station_min_median_qcu_bias_degC']:+.3f} degC`.",
                f"- Minimum leave-one-country median qcf/qcu residual: `{summary['leave_one_country_min_median_qcf_bias_degC']:+.3f}` / `{summary['leave_one_country_min_median_qcu_bias_degC']:+.3f} degC`.",
                f"- Strong bounded qcf-qcu adjustment shifts: `{summary['strong_adjustment_shift_station_count']}` stations.",
                "- qcf and qcu are not independent evidence; this audit tests sensitivity only.",
                "- No 1921-1939 value, correction, production file, EnKF, or OGGM run was used.",
                "",
                f"Decision: `{summary['decision']}`.",
            ]
        ) + "\n"
        (temp_dir / "FINAL_RESULT.md").write_text(report, encoding="utf-8")
        temp_dir.rename(OUTPUT)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
