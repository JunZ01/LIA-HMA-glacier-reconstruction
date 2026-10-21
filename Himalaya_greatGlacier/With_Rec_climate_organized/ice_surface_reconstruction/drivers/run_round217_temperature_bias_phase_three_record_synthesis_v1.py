from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "round_217_temperature_bias_phase_three_record_synthesis_v1"
CONTRACT = OUT / "ROUND_CONTRACT.md"
SOURCES = {
    "huang_summary": LOOP / "round_206_setp_winter_temperature_locked_safe_comparison_v1/summary.json",
    "huang_trends": LOOP / "round_206_setp_winter_temperature_locked_safe_comparison_v1/diagnostic_interval_trends.csv",
    "baiyu_summary": LOOP / "round_212_wscp_temperature_locked_safe_comparison_v1/summary.json",
    "baiyu_trends": LOOP / "round_212_wscp_temperature_locked_safe_comparison_v1/diagnostic_interval_trends.csv",
    "haizi_summary": LOOP / "round_216_haizi_annual_temperature_locked_safe_comparison_v1/summary.json",
    "haizi_trends": LOOP / "round_216_haizi_annual_temperature_locked_safe_comparison_v1/diagnostic_interval_trends.csv",
    "contract": CONTRACT,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def trend_row(path: Path, start: int, end: int) -> pd.Series:
    frame = pd.read_csv(path)
    row = frame[(frame.start_year == start) & (frame.end_year == end)]
    if len(row) != 1:
        raise RuntimeError(f"Missing unique trend row {start}-{end}: {path}")
    return row.iloc[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in SOURCES.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: state(path) for name, path in SOURCES.items()}
    huang = load_json(SOURCES["huang_summary"])
    baiyu = load_json(SOURCES["baiyu_summary"])
    haizi = load_json(SOURCES["haizi_summary"])
    if any(item.get("locked_period_values_accessed") for item in (huang, baiyu, haizi)):
        raise RuntimeError("A source round reports locked-period access")

    evidence = pd.DataFrame(
        [
            {
                "record": "Huang SETP winter Tmin",
                "target": "seasonal",
                "external_negative_windows": huang["metrics"]["external_negative_window_count"],
                "project_negative_windows": huang["metrics"]["project_setp_negative_window_count"],
                "reference_negative_windows": huang["metrics"]["reference_setp_negative_window_count"],
                "external_median_degC": huang["metrics"]["median_external_winter_anomaly_1500_1899_degC"],
                "project_median_degC": huang["metrics"]["median_project_setp_annual_anomaly_1500_1899_degC"],
                "reference_median_degC": huang["metrics"]["median_reference_setp_annual_anomaly_1500_1899_degC"],
            },
            {
                "record": "Baiyu June-July",
                "target": "seasonal",
                "external_negative_windows": baiyu["metrics"]["external_negative_window_count"],
                "project_negative_windows": baiyu["metrics"]["project_box_negative_window_count"],
                "reference_negative_windows": baiyu["metrics"]["reference_box_negative_window_count"],
                "external_median_degC": baiyu["metrics"]["median_external_anomaly_1500_1899_degC"],
                "project_median_degC": baiyu["metrics"]["median_project_box_anomaly_1500_1899_degC"],
                "reference_median_degC": baiyu["metrics"]["median_reference_box_anomaly_1500_1899_degC"],
            },
            {
                "record": "Haizi annual",
                "target": "annual",
                "external_negative_windows": haizi["metrics"]["external_negative_window_count"],
                "project_negative_windows": haizi["metrics"]["project_box_negative_window_count"],
                "reference_negative_windows": haizi["metrics"]["reference_box_negative_window_count"],
                "external_median_degC": haizi["metrics"]["median_external_anomaly_1500_1899_degC"],
                "project_median_degC": haizi["metrics"]["median_project_box_anomaly_1500_1899_degC"],
                "reference_median_degC": haizi["metrics"]["median_reference_box_anomaly_1500_1899_degC"],
            },
        ]
    )
    evidence["project_minus_external_degC"] = evidence.project_median_degC - evidence.external_median_degC
    evidence["reference_minus_external_degC"] = evidence.reference_median_degC - evidence.external_median_degC
    evidence.to_csv(OUT / "three_record_historical_bias_evidence.csv", index=False)

    trend_specs = [
        ("Huang SETP winter Tmin", "seasonal", SOURCES["huang_trends"], "external_winter_tmin_trend_degC_per_century", "project_setp_annual_trend_degC_per_century", "reference_setp_annual_trend_degC_per_century"),
        ("Baiyu June-July", "seasonal", SOURCES["baiyu_trends"], "wscp_jj_reconstruction_anomaly_degC_trend_degC_per_century", "project_local_box_annual_anomaly_degC_trend_degC_per_century", "reference_local_box_annual_anomaly_degC_trend_degC_per_century"),
        ("Haizi annual", "annual", SOURCES["haizi_trends"], "haizi_annual_reconstruction_anomaly_degC_trend_degC_per_century", "project_local_box_annual_anomaly_degC_trend_degC_per_century", "reference_local_box_annual_anomaly_degC_trend_degC_per_century"),
    ]
    trend_records = []
    for start, end in ((1850, 1900), (1901, 1920), (1940, 2000)):
        for record, target, path, external_col, project_col, reference_col in trend_specs:
            row = trend_row(path, start, end)
            external = float(row[external_col])
            project = float(row[project_col])
            reference = float(row[reference_col])
            trend_records.append(
                {
                    "record": record,
                    "target": target,
                    "start_year": start,
                    "end_year": end,
                    "external_trend_degC_per_century": external,
                    "project_trend_degC_per_century": project,
                    "reference_trend_degC_per_century": reference,
                    "external_project_sign_agree": bool(np.sign(external) == np.sign(project)),
                    "external_reference_sign_agree": bool(np.sign(external) == np.sign(reference)),
                }
            )
    trends = pd.DataFrame(trend_records)
    trends.to_csv(OUT / "three_record_interval_trends.csv", index=False)

    project_rmse = float(math.sqrt(np.mean(np.square(evidence.project_minus_external_degC))))
    reference_rmse = float(math.sqrt(np.mean(np.square(evidence.reference_minus_external_degC))))
    uniform_support_count = int(np.sum(evidence.external_negative_windows >= 6))
    annual_uniform_support = bool(
        evidence.loc[evidence.target == "annual", "external_negative_windows"].iloc[0] >= 6
    )
    early = trends[(trends.start_year == 1901) & (trends.end_year == 1920)]
    early_seasonal = early[early.target == "seasonal"]
    early_annual = early[early.target == "annual"].iloc[0]
    seasonal_negative_count = int(np.sum(early_seasonal.external_trend_degC_per_century < 0))
    annual_external_positive = bool(early_annual.external_trend_degC_per_century > 0)
    annual_project_positive = bool(early_annual.project_trend_degC_per_century > 0)
    uniform_cooling_authorized = uniform_support_count >= 2 and annual_uniform_support
    common_early_phase_correction_authorized = not (
        seasonal_negative_count == 2 and annual_external_positive
    )
    if common_early_phase_correction_authorized:
        common_early_phase_correction_authorized = bool(
            np.sign(early.external_trend_degC_per_century).nunique() == 1
            and np.sum(~early.external_project_sign_agree) >= 2
        )

    summary = {
        "round": 217,
        "decision": "uniform_historical_warm_bias_rejected_early20_conflict_is_season_dependent",
        "historical_1500_1899": {
            "external_record_count": 3,
            "records_supporting_uniform_cooling": uniform_support_count,
            "annual_record_supports_uniform_cooling": annual_uniform_support,
            "external_positive_median_count": int(np.sum(evidence.external_median_degC > 0)),
            "project_median_rmse_against_external_degC": project_rmse,
            "reference_median_rmse_against_external_degC": reference_rmse,
            "project_to_reference_rmse_ratio": project_rmse / reference_rmse,
        },
        "early_1901_1920": {
            "seasonal_external_negative_count": seasonal_negative_count,
            "annual_external_trend_degC_per_century": float(early_annual.external_trend_degC_per_century),
            "annual_project_trend_degC_per_century": float(early_annual.project_trend_degC_per_century),
            "annual_external_project_sign_agree": bool(early_annual.external_project_sign_agree),
            "interpretation": "season-dependent proxy evidence, not a common annual reversal",
        },
        "allowed_interval_sign_agreement": {
            "1850_1900_project": int(trends[(trends.start_year == 1850) & trends.external_project_sign_agree].shape[0]),
            "1940_2000_project": int(trends[(trends.start_year == 1940) & trends.external_project_sign_agree].shape[0]),
        },
        "correction_authorization": {
            "uniform_translation": uniform_cooling_authorized,
            "common_early_phase_correction": common_early_phase_correction_authorized,
            "production_rerun": False,
        },
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
    }

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), gridspec_kw={"wspace": 0.28})
    x = np.arange(len(evidence))
    width = 0.24
    axes[0].bar(x - width, evidence.external_median_degC, width, color="#222222", label="Independent record")
    axes[0].bar(x, evidence.project_median_degC, width, color="#B33A3A", label="Project local")
    axes[0].bar(x + width, evidence.reference_median_degC, width, color="#147D78", label="Four-family local")
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set_xticks(x, ["Huang\nwinter", "Baiyu\nJun-Jul", "Haizi\nannual"])
    axes[0].set_ylabel("Median 1500-1899 anomaly (degC)")
    axes[0].set_title("Historical offset evidence")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    early_x = np.arange(len(early))
    axes[1].bar(early_x - width / 2, early.external_trend_degC_per_century, width, color="#222222", label="Independent record")
    axes[1].bar(early_x + width / 2, early.project_trend_degC_per_century, width, color="#B33A3A", label="Project local")
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set_xticks(early_x, ["Huang\nwinter", "Baiyu\nJun-Jul", "Haizi\nannual"])
    axes[1].set_ylabel("1901-1920 trend (degC/century)")
    axes[1].set_title("Early-20th-century phase is seasonal")
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.16)
    figure.savefig(OUT / "temperature_three_record_bias_phase_synthesis.png", dpi=220)
    plt.close(figure)

    after = {name: state(path) for name, path in SOURCES.items()}
    if before != after:
        raise RuntimeError("A derived source changed")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    result = f"""# Round 217 Result

## Decision

`{summary['decision']}`

All three independent records have positive 1500-1899 median anomalies; none
passes the frozen uniform-cooling gate. The project local medians have RMSE
`{project_rmse:.3f}` degC against the three records, versus `{reference_rmse:.3f}`
degC for the four-family reference (ratio `{project_rmse / reference_rmse:.3f}`).

For 1901-1920, both seasonal records cool, while the independent annual record
warms at `{early_annual.external_trend_degC_per_century:+.3f}` degC/century and
the project annual curve also warms. The conflict is therefore season-dependent,
not evidence for a common annual reversal. No temperature correction or
production rerun is authorized.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
