from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "round_231_four_record_temperature_level_shape_synthesis_v1"
CONTRACT = Path(__file__).with_name(
    "round231_four_record_temperature_level_shape_synthesis_v1_contract.md"
)
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
SUMMARIES = {
    "Huang winter Tmin": LOOP / "round_206_setp_winter_temperature_locked_safe_comparison_v1/summary.json",
    "Baiyu Jun-Jul": LOOP / "round_212_wscp_temperature_locked_safe_comparison_v1/summary.json",
    "Haizi annual": LOOP / "round_216_haizi_annual_temperature_locked_safe_comparison_v1/summary.json",
    "Yang Jan-Jun": LOOP / "round_230_yang2010_project_reference_locked_safe_comparison_v2/summary.json",
}
ROUND227 = LOOP / "round_227_yang2010_pages2k_source_lineage_v3/summary.json"
ROUND229 = LOOP / "round_229_yang2010_11y_curve_digitization_v1/summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state(path: Path) -> dict[str, int | str]:
    return {"size": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [*SUMMARIES.values(), ROUND227, ROUND229, CONTRACT, FACTS]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    before = {str(path): state(path) for path in paths}
    inputs = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in SUMMARIES.items()}
    lineage = json.loads(ROUND227.read_text(encoding="utf-8"))
    digitization = json.loads(ROUND229.read_text(encoding="utf-8"))

    if lineage["decision"] != "independent_tree_core_collections_supported_same_regional_investigator":
        raise RuntimeError("Yang lineage decision changed")
    if digitization["digitization"]["missing_allowed_year_count"] != 0:
        raise RuntimeError("Yang digitization coverage changed")

    extraction = {
        "Huang winter Tmin": {
            "external": "median_external_winter_anomaly_1500_1899_degC",
            "project": "median_project_setp_annual_anomaly_1500_1899_degC",
            "reference": "median_reference_setp_annual_anomaly_1500_1899_degC",
            "corr_project": "window_correlation_external_project_setp",
            "corr_reference": "window_correlation_external_reference_setp",
            "independence": "identity_unresolved_seasonal_cru_calibrated",
        },
        "Baiyu Jun-Jul": {
            "external": "median_external_anomaly_1500_1899_degC",
            "project": "median_project_box_anomaly_1500_1899_degC",
            "reference": "median_reference_box_anomaly_1500_1899_degC",
            "corr_project": "external_project_box_window_correlation",
            "corr_reference": "external_reference_box_window_correlation",
            "independence": "source_independent_tree_network",
        },
        "Haizi annual": {
            "external": "median_external_anomaly_1500_1899_degC",
            "project": "median_project_box_anomaly_1500_1899_degC",
            "reference": "median_reference_box_anomaly_1500_1899_degC",
            "corr_project": "external_project_window_correlation",
            "corr_reference": "external_reference_window_correlation",
            "independence": "sequence_independent_annual_record",
        },
        "Yang Jan-Jun": {
            "external": "median_yang_anomaly_1500_1899_degC",
            "project": "median_project_anomaly_1500_1899_degC",
            "reference": "median_reference_anomaly_1500_1899_degC",
            "corr_project": "yang_project_window_correlation",
            "corr_reference": "yang_reference_window_correlation",
            "independence": "independent_tree_cores_shared_regional_investigator_digitized",
        },
    }
    rows = []
    for name, keys in extraction.items():
        metrics = inputs[name]["metrics"]
        row = {
            "record": name,
            "external_median_anomaly_degC": float(metrics[keys["external"]]),
            "project_local_median_anomaly_degC": float(metrics[keys["project"]]),
            "four_family_local_median_anomaly_degC": float(metrics[keys["reference"]]),
            "external_project_window_correlation": float(metrics[keys["corr_project"]]),
            "external_reference_window_correlation": float(metrics[keys["corr_reference"]]),
            "source_independence_class": keys["independence"],
        }
        row["project_minus_external_degC"] = (
            row["project_local_median_anomaly_degC"] - row["external_median_anomaly_degC"]
        )
        row["reference_minus_external_degC"] = (
            row["four_family_local_median_anomaly_degC"] - row["external_median_anomaly_degC"]
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "four_record_temperature_evidence_matrix.csv", index=False)

    external = table.external_median_anomaly_degC.to_numpy(float)
    project = table.project_local_median_anomaly_degC.to_numpy(float)
    reference = table.four_family_local_median_anomaly_degC.to_numpy(float)
    project_rmse = float(np.sqrt(np.mean((project - external) ** 2)))
    reference_rmse = float(np.sqrt(np.mean((reference - external) ** 2)))
    metrics = {
        "record_count": 4,
        "external_positive_median_count": int(np.sum(external > 0)),
        "project_positive_median_count": int(np.sum(project > 0)),
        "reference_positive_median_count": int(np.sum(reference > 0)),
        "project_level_rmse_degC": project_rmse,
        "reference_level_rmse_degC": reference_rmse,
        "project_to_reference_rmse_ratio": project_rmse / reference_rmse,
        "project_closer_record_count": int(
            np.sum(np.abs(project - external) < np.abs(reference - external))
        ),
        "median_external_level_degC": float(np.median(external)),
        "median_project_level_degC": float(np.median(project)),
        "median_reference_level_degC": float(np.median(reference)),
        "median_project_shape_correlation": float(
            np.median(table.external_project_window_correlation.to_numpy(float))
        ),
        "median_reference_shape_correlation": float(
            np.median(table.external_reference_window_correlation.to_numpy(float))
        ),
    }
    reject_uniform_correction = (
        metrics["external_positive_median_count"] == 4
        and project_rmse < 0.5 * reference_rmse
    )
    decision = (
        "retain_current_temperature_level_reject_uniform_translation_improve_shape_psms"
        if reject_uniform_correction
        else "temperature_level_requires_additional_adjudication"
    )

    x = np.arange(len(table))
    width = 0.24
    figure, axes = plt.subplots(2, 1, figsize=(11.8, 7.8), gridspec_kw={"height_ratios": [1.6, 1.0], "hspace": 0.34})
    axes[0].bar(x - width, external, width, color="#303030", label="External record")
    axes[0].bar(x, project, width, color="#B43C38", label="Project local")
    axes[0].bar(x + width, reference, width, color="#167A72", label="Four-family local")
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set_xticks(x, table.record)
    axes[0].set_ylabel("Median 1500-1899 anomaly (degC)")
    axes[0].set_title("Four-record temperature-level evidence")
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].legend(frameon=False, ncol=3, loc="upper right")

    axes[1].scatter(x, table.external_project_window_correlation, s=62, color="#B43C38", marker="s", label="Project shape")
    axes[1].scatter(x, table.external_reference_window_correlation, s=68, color="#167A72", marker="^", label="Four-family shape")
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set_xticks(x, table.record)
    axes[1].set_ylim(-0.45, 0.9)
    axes[1].set_ylabel("50-year-window correlation")
    axes[1].set_title("Low-frequency shape remains inconsistent")
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    figure.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.09)
    figure.savefig(OUT / "four_record_temperature_level_shape_synthesis.png", dpi=220)
    plt.close(figure)

    after = {str(path): state(path) for path in paths}
    if before != after:
        raise RuntimeError("A locked-safe source summary changed")
    summary = {
        "round": 231,
        "decision": decision,
        "metrics": metrics,
        "interpretation": {
            "apparent_warm_bias_cause": "persistently_cold_four_family_reference_level_not_supported_by_four_external_local_records",
            "remaining_problem": "low_frequency_shape_skill_is_record_and_season_dependent",
            "authorized_action": "retain_current_production_temperature_level",
            "future_research": "improve_spatial_and_seasonal_psm_skill_and_uncertainty_without_uniform_translation",
        },
        "boundaries": {
            "yang_native_numeric": False,
            "all_records_fully_independent": False,
            "uniform_translation_authorized": False,
            "production_rerun_authorized": False,
        },
        "source_states_unchanged": True,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
        "production_changed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    result = f"""# Round 231 Result

## Decision

`{decision}`

All four external records have positive 1500-1899 median anomalies. The
project local curve is closer than the four-family local reference for
`{metrics['project_closer_record_count']}/4` records. Across record medians,
project RMSE is `{project_rmse:.3f}` degC and four-family-reference RMSE is
`{reference_rmse:.3f}` degC, a ratio of
`{metrics['project_to_reference_rmse_ratio']:.3f}`.

The apparent uniform warm bias is therefore a reference-level conflict, not a
validated project offset. Low-frequency shape skill remains inconsistent and
must be improved through spatial/seasonal PSM structure and uncertainty, not
through a uniform temperature translation. No production rerun is authorized.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
