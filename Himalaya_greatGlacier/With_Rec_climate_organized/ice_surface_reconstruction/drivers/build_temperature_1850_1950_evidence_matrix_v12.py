from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
V11 = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v11"
ORIGINAL = BASE / "temperature_original_lake_author_reconstructions_v2"
OUTPUT = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v12"
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_row(matrix: pd.DataFrame, evidence_id: str, values: dict[str, object]) -> None:
    index = matrix.index[matrix.evidence_id == evidence_id]
    if len(index) != 1:
        raise RuntimeError(f"row-identity-failed:{evidence_id}:{len(index)}")
    for key, value in values.items():
        matrix.loc[index[0], key] = value


def add_row(matrix: pd.DataFrame, values: dict[str, object]) -> pd.DataFrame:
    unknown = set(values) - set(matrix.columns)
    if unknown:
        raise RuntimeError(f"unknown-columns:{sorted(unknown)}")
    row = {column: values.get(column, "") for column in matrix.columns}
    return pd.concat([matrix, pd.DataFrame([row])], ignore_index=True)


def metric(metrics: pd.DataFrame, record: str, window: str) -> pd.Series:
    selected = metrics[(metrics.record == record) & (metrics.window == window)]
    if len(selected) != 1:
        raise RuntimeError(f"metric-identity-failed:{record}:{window}:{len(selected)}")
    return selected.iloc[0]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    matrix = pd.read_csv(V11 / "evidence_matrix.csv")
    metrics = pd.read_csv(ORIGINAL / "original_author_window_metrics.csv")

    tcqh = metric(metrics, "Tengchongqinghai", "full_safe_prelock")
    lugu_full = metric(metrics, "Lugu", "full_safe_prelock")
    lugu_bridge = metric(metrics, "Lugu", "bridge_1850_1920")
    qinghai = metric(metrics, "Qinghai", "full_safe_prelock")

    update_row(
        matrix,
        "LIANG2024_TENGCHONGQINGHAI_BRGDGT_RF",
        {
            "admitted_role": "calibration_target_sensitivity_only_not_site_direction_evidence",
            "finding": (
                f"At the same six ages, Liang 2024 common RF MAF warms {tcqh.liang_common_rf_trend_c_per_century:+.3f} C/century, "
                f"but Zhao et al. 2021 original author MAAT cools {tcqh.author_trend_c_per_century:+.3f} C/century. "
                "The direction reversal is calibration/target dependent; this row cannot represent the original archive direction."
            ),
            "next_action": "use_only_as_calibration_sensitivity;primary_site_evidence_is_Zhao2021_author_MAAT",
        },
    )
    update_row(
        matrix,
        "LIANG2024_LUGU_BRGDGT_RF",
        {
            "admitted_role": "calibration_target_sensitivity_only_not_site_direction_evidence",
            "finding": (
                f"Full 1507-1914 trends have the same weak warming sign, but over 1850-1914 the common RF warms "
                f"{lugu_bridge.liang_common_rf_trend_c_per_century:+.3f} C/century while Zhao et al. 2021 original author MAAT cools "
                f"{lugu_bridge.author_trend_c_per_century:+.3f} C/century. The bridge-window direction is calibration dependent."
            ),
            "next_action": "use_only_as_calibration_sensitivity;primary_site_evidence_is_Zhao2021_author_MAAT",
        },
    )

    matrix = add_row(
        matrix,
        {
            "evidence_id": "ZHAO2021_TENGCHONGQINGHAI_AUTHOR_BRGDGT_MAAT",
            "region": "southwestern_China_HMA_margin",
            "longitude": 98.5667,
            "coverage": "six_safe_points_1560_1899;one_point_in_1850_1920_bridge",
            "target": "mean_annual_air_temperature",
            "measurement_family": "lake_sediment_brGDGT_author_Index1_calibration",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "official_Elsevier_author_supplement;same_archive_as_Liang2024_TCQH_do_not_double_count",
            "spatial_relevance": "southeastern_HMA_margin_high",
            "season_match": "annual_target_matches_project_annual_temperature",
            "admitted_role": "primary_independent_site_direction_evidence_before_1900",
            "finding": (
                f"Original author MAAT cools {tcqh.author_trend_c_per_century:+.3f} C/century over 1560-1899, while the project warms "
                f"{tcqh.project_median_trend_at_author_years_c_per_century:+.3f} C/century at identical ages. This is a genuine direction conflict. "
                "Only one point lies in 1850-1920, so it does not define the bridge trend."
            ),
            "next_action": "retain_as_primary_site_conflict_before_1900;do_not_infer_bridge_trend_from_one_point",
            "source_url": "https://ars.els-cdn.com/content/image/1-s2.0-S2095927320307222-mmc1.xlsx",
            "sealed_period_values_used": False,
            "production_modified": False,
        },
    )
    matrix = add_row(
        matrix,
        {
            "evidence_id": "ZHAO2021_LUGU_AUTHOR_BRGDGT_MAAT",
            "region": "southeastern_Tibetan_Plateau",
            "longitude": 100.75,
            "coverage": "twenty_safe_points_1507_1914;four_points_1850_1914",
            "target": "mean_annual_air_temperature",
            "measurement_family": "lake_sediment_brGDGT_author_Index1_calibration",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "official_Elsevier_author_supplement;same_archive_as_Liang2024_Lugu_do_not_double_count",
            "spatial_relevance": "southeastern_HMA_margin_high",
            "season_match": "annual_target_matches_project_annual_temperature",
            "admitted_role": "primary_independent_site_direction_evidence_and_1850_1920_bridge",
            "finding": (
                f"Over 1507-1914 original MAAT weakly warms {lugu_full.author_trend_c_per_century:+.3f} C/century while the project cools "
                f"{lugu_full.project_median_trend_at_author_years_c_per_century:+.3f}. Over the four-point 1850-1914 bridge, original MAAT cools "
                f"{lugu_bridge.author_trend_c_per_century:+.3f} and the project cools {lugu_bridge.project_median_trend_at_author_years_c_per_century:+.3f}; "
                "the bridge directions agree."
            ),
            "next_action": "retain_full_window_conflict_and_bridge_window_agreement_separately;do_not_average_across_windows",
            "source_url": "https://ars.els-cdn.com/content/image/1-s2.0-S2095927320307222-mmc1.xlsx",
            "sealed_period_values_used": False,
            "production_modified": False,
        },
    )
    matrix = add_row(
        matrix,
        {
            "evidence_id": "WANG2021_QINGHAI_AUTHOR_SALINITY_CORRECTED_BRGDGT_MAAT",
            "region": "northeast_Tibetan_Plateau_margin",
            "longitude": 100.137,
            "coverage": "one_safe_point_1849.065;locked_rows_not_decoded",
            "target": "salinity_corrected_mean_annual_air_temperature",
            "measurement_family": "lake_sediment_brGDGT_author_IRprime6plus7_calibration",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "official_Elsevier_author_supplement;same_Qinghai_core_as_Liang2024_QH_do_not_double_count",
            "spatial_relevance": "northeast_HMA_margin_high",
            "season_match": "annual_target_matches_project_annual_temperature",
            "admitted_role": "primary_site_calibration_identity_but_insufficient_direction_points",
            "finding": (
                f"The author-preferred salinity-corrected workbook column I provides only {int(qinghai.author_n)} safe point at 1849.065. "
                "It cannot define a trend or adjudicate the independent Qinghai UK37 summer-temperature conflict."
            ),
            "next_action": "retain_as_primary_method_identity;do_not_score_direction_or_duplicate_Qinghai_site_vote",
            "source_url": "https://ars.els-cdn.com/content/image/1-s2.0-S0016703721002672-mmc2.xlsx",
            "sealed_period_values_used": False,
            "production_modified": False,
        },
    )

    if matrix.evidence_id.duplicated().any():
        raise RuntimeError("duplicate-evidence-id")
    matrix.to_csv(OUTPUT / "evidence_matrix.csv", index=False)

    previous = json.loads((V11 / "summary.json").read_text(encoding="utf-8"))
    previous.update(
        {
            "experiment": "temperature_1850_1950_low_frequency_evidence_matrix_v12",
            "supersedes": "temperature_1850_1950_low_frequency_evidence_matrix_v11",
            "row_count": int(len(matrix)),
            "new_primary_original_records": [
                "ZHAO2021_TENGCHONGQINGHAI_AUTHOR_BRGDGT_MAAT",
                "ZHAO2021_LUGU_AUTHOR_BRGDGT_MAAT",
                "WANG2021_QINGHAI_AUTHOR_SALINITY_CORRECTED_BRGDGT_MAAT",
            ],
            "calibration_reversals": {
                "Tengchongqinghai_1560_1899": True,
                "Lugu_1850_1914": True,
                "Qinghai": "not_evaluable_one_safe_point",
            },
            "decision": "reject_uniform_HMA_correction;retain_record_window_specific_conflicts;do_not_use_common_RF_as_original_direction",
            "candidate_v2_authorized": False,
            "prohibited_source": "20CRv3",
            "prohibited_source_accessed": False,
        }
    )
    (OUTPUT / "summary.json").write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "inputs": [
            {"path": str(V11 / "evidence_matrix.csv"), "sha256": sha256(V11 / "evidence_matrix.csv")},
            {"path": str(V11 / "summary.json"), "sha256": sha256(V11 / "summary.json")},
            {"path": str(ORIGINAL / "summary.json"), "sha256": sha256(ORIGINAL / "summary.json")},
            {"path": str(ORIGINAL / "original_author_window_metrics.csv"), "sha256": sha256(ORIGINAL / "original_author_window_metrics.csv")},
        ],
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = """# HMA low-frequency temperature evidence matrix v12

Original author supplements now replace the common 2024 recalibration as the
primary site-direction evidence for Tengchongqinghai, Lugu, and Qinghai.

- Tengchongqinghai original MAAT cools over 1560-1899 while the project warms;
  this is a real pre-1900 direction conflict. The common RF had reversed it.
- Lugu original MAAT and the project both cool over the four-point 1850-1914
  bridge; the common RF had reversed that bridge direction.
- Qinghai author-preferred salinity-corrected MAAT has only one safe point and
  cannot define a trend or erase the independent UK37 conflict.

The result strengthens the rule that archive, native target, calibration, and
window must be identified before a direction vote. It does not authorize a
uniform HMA correction or Candidate v2.

The sealed period was not used. Production was not changed, EnKF and OGGM were
not run, and 20CRv3 was not accessed.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
