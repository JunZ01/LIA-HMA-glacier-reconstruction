"""Validate Candidate v2 trend/phase against stations and IITM WHIND diagnostic."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PERIODS = ("1880-1900", "1901-1920", "1940-1950")


def root_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def trend(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return np.nan
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100)


def main() -> int:
    root = root_path()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    candidate = loop / "temperature_station_candidate_v2_allowed_years_v1"
    iitm = loop / "temperature_1850_1950_iitm_homogeneous_india_qualification_v2"
    output = loop / "temperature_candidate_v2_trend_direction_validation_v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    paths = {
        "candidate_station": candidate / "candidate_station_annual_evaluation.csv",
        "candidate_decision": loop / "temperature_station_candidate_v2_allowed_years_v2/summary.json",
        "iitm_annual": iitm / "derived/iitm_wh_ne_allowed_annual_seasonal_temperature.csv",
        "iitm_summary": iitm / "summary.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: digest(path) for name, path in paths.items()}
    candidate_summary = json.loads(paths["candidate_decision"].read_text(encoding="utf-8"))
    iitm_summary = json.loads(paths["iitm_summary"].read_text(encoding="utf-8"))
    if candidate_summary["locked_1921_1939_included_or_accessed"] or not iitm_summary["locked_period_absent_from_outputs"]:
        raise RuntimeError("lock-boundary-failed")
    annual = pd.read_csv(paths["candidate_station"])
    if annual.year.between(1921, 1939).any():
        raise RuntimeError("locked-year-present")

    station_rows = []
    for station_id, group in annual.groupby("station_id"):
        for period in PERIODS:
            start, end = map(int, period.split("-"))
            section = group[group.year.between(start, end)]
            for column, target in (("qcf_station_degC", "qcf"), ("qcu_station_degC", "qcu")):
                valid = np.isfinite(section[column]) & np.isfinite(section.original_ensemble_degC) & np.isfinite(section.candidate_v2_ensemble_degC)
                if valid.sum() < 5:
                    continue
                years = section.loc[valid, "year"].to_numpy(float)
                observed = trend(years, section.loc[valid, column].to_numpy(float))
                original = trend(years, section.loc[valid, "original_ensemble_degC"].to_numpy(float))
                corrected = trend(years, section.loc[valid, "candidate_v2_ensemble_degC"].to_numpy(float))
                station_rows.append({
                    "station_id": station_id,
                    "station_name": group.station_name.iloc[0],
                    "period": period, "target": target, "matched_years": int(valid.sum()),
                    "observed_trend_per_century": observed,
                    "original_trend_per_century": original,
                    "candidate_trend_per_century": corrected,
                    "original_abs_trend_error": abs(original - observed),
                    "candidate_abs_trend_error": abs(corrected - observed),
                    "original_sign_agreement": bool(np.sign(original) == np.sign(observed)),
                    "candidate_sign_agreement": bool(np.sign(corrected) == np.sign(observed)),
                })
    station = pd.DataFrame(station_rows)
    station.to_csv(output / "station_trend_direction_metrics.csv", index=False)
    aggregate_rows = []
    for (period, target), group in station.groupby(["period", "target"]):
        before_error = float(group.original_abs_trend_error.median())
        after_error = float(group.candidate_abs_trend_error.median())
        aggregate_rows.append({
            "period": period, "target": target, "station_count": int(len(group)),
            "original_median_abs_trend_error": before_error,
            "candidate_median_abs_trend_error": after_error,
            "trend_error_improvement_fraction": 1 - after_error / before_error if before_error else np.nan,
            "original_sign_agreement_count": int(group.original_sign_agreement.sum()),
            "candidate_sign_agreement_count": int(group.candidate_sign_agreement.sum()),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output / "station_trend_direction_summary.csv", index=False)

    iitm_data = pd.read_csv(paths["iitm_annual"])
    whind = iitm_data[(iitm_data.region_code == "WHIND") & (iitm_data.season == "annual_Jan-Dec")][["year", "tmean_derived_c"]]
    india = annual[annual.station_id.str.startswith("IN")].groupby("year", as_index=False)[["original_ensemble_degC", "candidate_v2_ensemble_degC"]].mean()
    whind_compare = whind.merge(india, on="year", how="inner")
    iitm_rows = []
    for period in ("1901-1920", "1940-1950"):
        start, end = map(int, period.split("-"))
        values = whind_compare[whind_compare.year.between(start, end)]
        years = values.year.to_numpy(float)
        target_trend = trend(years, values.tmean_derived_c.to_numpy(float))
        original_trend = trend(years, values.original_ensemble_degC.to_numpy(float))
        candidate_trend = trend(years, values.candidate_v2_ensemble_degC.to_numpy(float))
        iitm_rows.append({
            "period": period, "year_count": int(len(values)),
            "iitm_whind_tmean_trend_per_century": target_trend,
            "original_india_station_composite_trend_per_century": original_trend,
            "candidate_india_station_composite_trend_per_century": candidate_trend,
            "original_abs_trend_error": abs(original_trend - target_trend),
            "candidate_abs_trend_error": abs(candidate_trend - target_trend),
            "original_sign_agreement": bool(np.sign(original_trend) == np.sign(target_trend)),
            "candidate_sign_agreement": bool(np.sign(candidate_trend) == np.sign(target_trend)),
        })
    iitm_table = pd.DataFrame(iitm_rows)
    iitm_table.to_csv(output / "iitm_whind_trend_sensitivity.csv", index=False)

    qcf = aggregate[aggregate.target.eq("qcf")]
    station_trend_pass = bool((qcf.trend_error_improvement_fraction >= -0.10).all() and (qcf.candidate_sign_agreement_count >= qcf.original_sign_agreement_count).all())
    iitm_not_worse = bool((iitm_table.candidate_abs_trend_error <= iitm_table.original_abs_trend_error).all())
    if not station_trend_pass:
        decision = "reject_candidate_v2_temporal_shape_direct_station_trend_gate_failed"
    elif not iitm_not_worse:
        decision = "retain_candidate_v2_level_only_iitm_trend_warning"
    else:
        decision = "candidate_v2_level_and_trend_gates_pass"
    after = {name: digest(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-changed")
    summary = {
        "experiment": "temperature_candidate_v2_trend_direction_validation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "direct_station_trend_summary": aggregate.to_dict(orient="records"),
        "iitm_whind_source_correlated_sensitivity": iitm_rows,
        "direct_station_trend_gate_passes": station_trend_pass,
        "iitm_whind_trend_not_worse": iitm_not_worse,
        "decision": decision,
        "interpretation_boundary": "IITM WHIND is source-correlated with Indian station archives and is a regional processing sensitivity, not an independent vote. Direct qcf remains primary.",
        "source_hashes_unchanged": unchanged,
        "production_modified": False,
        "enkf_rerun": False,
        "oggm_run": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before_sha256": before, "after_sha256": after, "unchanged": unchanged}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    qcf_plot = aggregate[aggregate.target.eq("qcf")].set_index("period").reindex(PERIODS)
    x = np.arange(len(PERIODS))
    axes[0].bar(x - 0.18, qcf_plot.original_median_abs_trend_error, width=0.36, color="#777777", label="original")
    axes[0].bar(x + 0.18, qcf_plot.candidate_median_abs_trend_error, width=0.36, color="#B23A48", label="Candidate v2")
    axes[0].set(xticks=x, xticklabels=PERIODS, ylabel="Median absolute trend error", title="Direct qcf trend gate")
    axes[0].legend(frameon=False)
    x2 = np.arange(len(iitm_table))
    axes[1].bar(x2 - 0.18, iitm_table.original_abs_trend_error, width=0.36, color="#777777", label="original")
    axes[1].bar(x2 + 0.18, iitm_table.candidate_abs_trend_error, width=0.36, color="#B23A48", label="Candidate v2")
    axes[1].set(xticks=x2, xticklabels=iitm_table.period, ylabel="Absolute trend error", title="IITM WHIND processing sensitivity")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "candidate_v2_trend_direction_validation.png", dpi=220)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "# Candidate v2 trend/direction validation\n\n"
        f"- Direct qcf trend gate: `{str(station_trend_pass).lower()}`.\n"
        f"- IITM WHIND trend not worsened: `{str(iitm_not_worse).lower()}`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
