from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
V9 = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v9"
LAKES = BASE / "temperature_lake_brgdgt_eight_record_safe_audit_v3"
OUTPUT = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v10"
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


def add_row(matrix: pd.DataFrame, row: dict[str, object]) -> pd.DataFrame:
    unknown = set(row) - set(matrix.columns)
    if unknown:
        raise RuntimeError(f"unknown-columns:{sorted(unknown)}")
    return pd.concat(
        [matrix, pd.DataFrame([{column: row.get(column, "") for column in matrix.columns}])],
        ignore_index=True,
    )


def metric(metrics: pd.DataFrame, record: str, window: str) -> pd.Series:
    selected = metrics[(metrics.record == record) & (metrics.window == window)]
    if len(selected) != 1:
        raise RuntimeError(f"metric-identity-failed:{record}:{window}:{len(selected)}")
    return selected.iloc[0]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    matrix = pd.read_csv(V9 / "evidence_matrix.csv")
    metrics = pd.read_csv(LAKES / "eight_record_window_metrics.csv")

    tiancai_full = metric(metrics, "Tiancai", "full_safe")
    tiancai_bridge = metric(metrics, "Tiancai", "bridge_1850_1920")
    update_row(
        matrix,
        "NOAA_TIANCAI_LAKE_CHIRONOMID",
        {
            "finding": (
                "The original NOAA chironomid record cools over 1789-1918.6 and agrees with the project. "
                f"A secondary standardized brGDGT RF-MAF reanalysis cools {tiancai_full.admitted_trend_c_per_century:+.3f} C/century "
                f"over its full 1518-1909 safe span while the project warms, but over 1857-1909 both cool "
                f"({tiancai_bridge.admitted_trend_c_per_century:+.3f} vs {tiancai_bridge.project_median_trend_at_proxy_years_c_per_century:+.3f} C/century). "
                "The result is window- and target-sensitive rather than a uniform bias."
            ),
            "next_action": "retain_original_and_common_calibration_as_one_site_family;do_not_double_count",
        },
    )
    cuoqia_full = metric(metrics, "Cuoqia", "full_safe")
    cuoqia_bridge = metric(metrics, "Cuoqia", "bridge_1850_1920")
    update_row(
        matrix,
        "CUOQIA_LAKE_BRGDGT_300Y",
        {
            "finding": (
                "The original figure-derived CQ1/CQ2 series remain internally mixed. The secondary common-RF record "
                f"is nearly flat over 1532-1916 ({cuoqia_full.admitted_trend_c_per_century:+.3f} C/century) and warms "
                f"over its two-point 1866-1916 bridge ({cuoqia_bridge.admitted_trend_c_per_century:+.3f}), while the project cools. "
                "No single Cuoqia offset is supported."
            ),
            "next_action": "retain_as_mixed_warm_season_shape_gate;do_not_force_one_direction",
        },
    )
    update_row(
        matrix,
        "NOAA_QINGHAI_LAKE_ALKENONE",
        {
            "finding": (
                "The official UK37 series still cools over 1657-1804 while the project warms. The 2024 standardized "
                "brGDGT workbook has only one safe Qinghai point (1850), so it cannot adjudicate or erase this conflict."
            ),
            "next_action": "retain_as_stable_conflict;seek_higher_resolution_original_Qinghai_brgdgt_or_other_proxy",
        },
    )
    update_row(
        matrix,
        "NGRAMRING_TIANCAI_AWENG_ALREADY_ASSESSED",
        {
            "admitted_role": "identity_context_only_Tiancai_Aweng_and_Ngamring_now_have_explicit_rows",
            "finding": "Tiancai, Aweng, and Ngamring now have explicit rows and must not be counted again here.",
        },
    )

    common = {"sealed_period_values_used": False, "production_modified": False}
    definitions = [
        ("Bangong", "LIANG2024_BANGONG_BRGDGT_RF_MAF", "western_Tibetan_Plateau", 78.96, "MAF", "western_HMA", "full_safe"),
        ("Ngamring", "LIANG2024_NGAMRING_BRGDGT_RF_MAF", "central_southern_Tibetan_Plateau", 87.18, "MAF", "central_HMA", "full_safe"),
        ("Gahai", "LIANG2024_GAHAI_BRGDGT_RF", "eastern_Tibetan_Plateau", 102.325, "ice_free_season_temperature", "eastern_HMA_margin", "full_safe"),
        ("Tengchongqinghai", "LIANG2024_TENGCHONGQINGHAI_BRGDGT_RF", "southwestern_China_HMA_margin", 98.57, "MAF_or_MAAT", "southeast_HMA_margin", "full_safe"),
        ("Lugu", "LIANG2024_LUGU_BRGDGT_RF", "southeastern_Tibetan_Plateau", 100.79, "MAF_or_MAAT", "southeast_HMA", "full_safe"),
    ]
    for record_name, evidence_id, region, longitude, target, spatial, window in definitions:
        row = metric(metrics, record_name, window)
        finding = (
            f"Common-RF safe trend is {row.admitted_trend_c_per_century:+.3f} C/century over "
            f"{row.first_year:.0f}-{row.last_year:.0f}; project trend at the same proxy years is "
            f"{row.project_median_trend_at_proxy_years_c_per_century:+.3f} C/century; direction "
            f"{'agrees' if row.direction_agreement else 'conflicts'}."
        )
        if record_name in {"Bangong", "Ngamring"}:
            finding += " The 2.7 C bacterial-cluster constraint is a constant shift in this safe window and does not change the trend."
        matrix = add_row(
            matrix,
            {
                "evidence_id": evidence_id,
                "region": region,
                "longitude": longitude,
                "coverage": f"safe_points_{row.first_year:.0f}-{row.last_year:.0f}",
                "target": target,
                "measurement_family": "lake_sediment_brGDGT_common_RF_reanalysis",
                "machine_readable": True,
                "numeric_now": True,
                "source_independence": "distinct_underlying_lake_archive;secondary_common_method_reanalysis_group_with_site_family",
                "spatial_relevance": spatial,
                "season_match": "warm_or_ice_free_season_not_annual_exact",
                "admitted_role": "calibration_and_period_sensitivity_not_extra_independent_vote",
                "finding": finding,
                "next_action": "retain_as_site_level_direction_context;do_not_use_for_uniform_HMA_offset",
                "source_url": "https://doi.org/10.6084/m9.figshare.26521102.v1",
                **common,
            },
        )

    if matrix.evidence_id.duplicated().any():
        raise RuntimeError("duplicate-evidence-id")
    matrix.to_csv(OUTPUT / "evidence_matrix.csv", index=False)

    previous = json.loads((V9 / "summary.json").read_text(encoding="utf-8"))
    previous.update(
        {
            "experiment": "temperature_1850_1950_low_frequency_evidence_matrix_v10",
            "supersedes": "temperature_1850_1950_low_frequency_evidence_matrix_v9",
            "row_count": int(len(matrix)),
            "decision": "reject_uniform_HMA_correction;retain_region_season_and_window_specific_diagnosis",
            "new_numeric_site_families": ["Bangong", "Ngamring", "Gahai", "Tengchongqinghai", "Lugu"],
            "common_calibration_result": (
                "The eight-lake RF reanalysis is directionally mixed. The 2.7 C ecological constraint changes absolute "
                "levels for Bangong and Ngamring but not safe-window trends. Qinghai remains unresolved by this workbook."
            ),
            "candidate_v2_authorized": False,
            "prohibited_source": "20CRv3",
            "prohibited_source_accessed": False,
        }
    )
    (OUTPUT / "summary.json").write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "inputs": [
            {"path": str(V9 / "evidence_matrix.csv"), "sha256": sha256(V9 / "evidence_matrix.csv")},
            {"path": str(V9 / "summary.json"), "sha256": sha256(V9 / "summary.json")},
            {"path": str(LAKES / "eight_record_window_metrics.csv"), "sha256": sha256(LAKES / "eight_record_window_metrics.csv")},
            {"path": str(LAKES / "summary.json"), "sha256": sha256(LAKES / "summary.json")},
        ],
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = """# HMA low-frequency temperature evidence matrix v10

The 2024 eight-lake common-RF reanalysis adds machine-readable calibration and
period sensitivity without adding duplicate independent votes. Its directions
are mixed across sites and windows. The ecological 2.7 C constraint changes
absolute Bangong and Ngamring levels but not the safe-window trends.

This strengthens the rejection of one HMA-wide offset or background mixture.
Qinghai remains a genuine unresolved conflict because the common-RF workbook
contains only one safe point there. Candidate v2 remains unauthorized.

The sealed period was not used. Production files were not changed. EnKF and
OGGM were not run. 20CRv3 was not accessed.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
