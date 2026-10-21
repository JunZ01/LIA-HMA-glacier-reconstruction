from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
V1 = BASE / "temperature_region_season_low_frequency_preflight_v1"
EVIDENCE = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v12"
ORIGINAL = BASE / "temperature_original_lake_author_reconstructions_v2"
OUTPUT = BASE / "temperature_region_season_low_frequency_preflight_v2"
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    constraints = pd.read_csv(V1 / "constraint_matrix.csv")
    additions = pd.DataFrame(
        [
            {
                "constraint_id": "LAKE_NGAMRING_AUTHOR",
                "evidence_family": "NOAA_Ngamring_author_brGDGT_MAAT",
                "region": "central_transition_HMA",
                "native_target": "annual_MAAT_possible_winter_weight",
                "safe_interval": "1597-1778",
                "metric_role": "primary_pre1850_direction_gate_no_bridge_claim",
                "independence_rule": "same_archive_as_Liang2024_Ngamring_one_site_vote",
            },
            {
                "constraint_id": "LAKE_TCQH_AUTHOR",
                "evidence_family": "Zhao2021_Tengchongqinghai_author_brGDGT_MAAT",
                "region": "southeast_monsoon_eastern_margin",
                "native_target": "annual_MAAT",
                "safe_interval": "1560-1899;one_bridge_point_only",
                "metric_role": "primary_pre1900_direction_gate",
                "independence_rule": "same_archive_as_Liang2024_TCQH_one_site_vote",
            },
            {
                "constraint_id": "LAKE_LUGU_AUTHOR",
                "evidence_family": "Zhao2021_Lugu_author_brGDGT_MAAT",
                "region": "southeast_monsoon_eastern_margin",
                "native_target": "annual_MAAT",
                "safe_interval": "1507-1914;bridge_1850-1914",
                "metric_role": "primary_full_and_bridge_direction_gate",
                "independence_rule": "same_archive_as_Liang2024_Lugu_one_site_vote",
            },
            {
                "constraint_id": "LAKE_QINGHAI_BRGDGT_AUTHOR",
                "evidence_family": "Wang2021_Qinghai_salinity_corrected_brGDGT_MAAT",
                "region": "northern_northeastern_margin",
                "native_target": "annual_MAAT",
                "safe_interval": "single_point_1849.065",
                "metric_role": "method_identity_only_no_direction_gate",
                "independence_rule": "same_core_as_Liang2024_Qinghai;group_with_Qinghai_site_not_UK37_duplicate_spatial_vote",
            },
            {
                "constraint_id": "LIANG2024_COMMON_RF",
                "evidence_family": "eight_lake_common_RF_MAF_reanalysis",
                "region": "multi_region_HMA",
                "native_target": "common_RF_MAF",
                "safe_interval": "record_specific_prelock_only",
                "metric_role": "calibration_and_target_sensitivity_only",
                "independence_rule": "never_an_extra_vote_and_never_a_substitute_for_original_author_direction",
            },
        ]
    )
    constraints = pd.concat([constraints, additions], ignore_index=True)
    constraints.to_csv(OUTPUT / "constraint_matrix.csv", index=False)

    readiness = pd.read_csv(V1 / "input_readiness.csv")
    readiness = pd.concat(
        [
            readiness,
            pd.DataFrame(
                [
                    {
                        "input_id": "original_author_lake_temperature_supplements",
                        "period": "record_specific_1500-1920_safe",
                        "shape": "Ngamring_2;TCQH_6;Lugu_20;Qinghai_1_safe_points",
                        "status": "ready_primary_site_evidence",
                        "eligible_use": "native_target_direction_and_calibration_identity_only",
                        "remaining_gap": "sparse_or_absent_1850-1920_bridge_for_Ngamring_TCQH_Qinghai",
                    },
                    {
                        "input_id": "Liang2024_eight_lake_common_RF",
                        "period": "record_specific_1500-1920_safe",
                        "shape": "eight_records_secondary_reanalysis",
                        "status": "ready_sensitivity_only",
                        "eligible_use": "calibration_target_direction_reversal_audit",
                        "remaining_gap": "cannot_replace_original_author_temperature_or_count_as_independent_vote",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    readiness.to_csv(OUTPUT / "input_readiness.csv", index=False)

    preregistration = """# Region- and season-aware low-frequency evidence preflight v2

## Purpose

Freeze the evidence identities needed to decide whether the project has a
correctable 1850-1950 transition error. This supersedes v1's stale evidence
pointer. It does not authorize an EnKF replay, a background blend, a production
candidate, or OGGM.

## New identity rule

Original author temperature reconstructions are the primary site evidence.
The Liang et al. (2024) common RF-MAF series are calibration sensitivities only.
They cannot replace native MAAT, warm-season, or author-calibrated records and
cannot add an independent vote. This rule is required because identical
chronologies reverse direction for Ngamring, Tengchongqinghai, and the Lugu
1850-1914 bridge under different calibrations or targets.

## Frozen time blocks

- 1500-1849: regional physical and reconstruction envelope only.
- 1850-1920: bridge diagnosis using direct stations where available and native
  annual or seasonal physical records without cross-target RMSE.
- 1921-1939: sealed; exclude before value decoding, interpolation, plotting,
  fitting, or scoring.
- 1940-1950: transfer validation only.
- 1951-2000: instrumental characterization; never hand-tune a pre-1950 branch.

## Decision rules

1. Never infer an HMA-wide offset from a regionally mixed evidence matrix.
2. Keep full-window and bridge-window conclusions separate.
3. A candidate may be proposed only if one physically named mechanism improves
   direct annual stations and at least one conflicting physical family without
   reversing a currently agreeing family.
4. A secondary common calibration may diagnose calibration uncertainty but may
   not choose the correction direction.
5. Sparse one-point records identify method or level only, not a trend.

## Current consequence

Tengchongqinghai is a genuine pre-1900 conflict with the project; Lugu is a
full-window conflict but agrees with the project over its four-point
1850-1914 bridge; Qinghai brGDGT has one safe point and cannot adjudicate the
independent Qinghai UK37 conflict. The evidence therefore does not authorize a
uniform HMA correction or Candidate v2.

20CRv3 is prohibited and was not accessed.
"""
    (OUTPUT / "PRE_REGISTRATION.md").write_text(preregistration, encoding="utf-8")

    summary = {
        "experiment": "temperature_region_season_low_frequency_preflight_v2",
        "status": "evidence_identity_frozen_no_scientific_run_authorized",
        "supersedes": "temperature_region_season_low_frequency_preflight_v1",
        "latest_evidence_matrix": "temperature_1850_1950_low_frequency_evidence_matrix_v12",
        "primary_original_records_added": [
            "NOAA_Ngamring_author_brGDGT_MAAT",
            "Zhao2021_Tengchongqinghai_author_brGDGT_MAAT",
            "Zhao2021_Lugu_author_brGDGT_MAAT",
            "Wang2021_Qinghai_salinity_corrected_brGDGT_MAAT",
        ],
        "secondary_common_RF_role": "calibration_target_sensitivity_only",
        "known_direction_reversals": [
            "Ngamring_1597_1778",
            "Tengchongqinghai_1560_1899",
            "Lugu_1850_1914",
        ],
        "uniform_HMA_offset_allowed": False,
        "candidate_v2_authorized": False,
        "next_read_only_action": "close_remaining_original_numeric_source_search_then_group_primary_evidence_by_region_target_and_window",
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "sealed_period_values_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Preflight v2 result\n\n"
        "The evidence identities and time blocks are frozen. The original author lake records supersede common-RF direction claims. "
        "No uniform HMA correction or Candidate v2 is authorized. The next read-only step is to close the remaining original-source "
        "search and then aggregate primary evidence by region, native target, and window.\n\n"
        "The sealed period was not used. Production was not changed, EnKF and OGGM were not run, and 20CRv3 was not accessed.\n",
        encoding="utf-8",
    )
    manifest = {
        "inputs": [
            {"path": str(V1 / "constraint_matrix.csv"), "sha256": sha256(V1 / "constraint_matrix.csv")},
            {"path": str(V1 / "input_readiness.csv"), "sha256": sha256(V1 / "input_readiness.csv")},
            {"path": str(EVIDENCE / "summary.json"), "sha256": sha256(EVIDENCE / "summary.json")},
            {"path": str(EVIDENCE / "evidence_matrix.csv"), "sha256": sha256(EVIDENCE / "evidence_matrix.csv")},
            {"path": str(ORIGINAL / "summary.json"), "sha256": sha256(ORIGINAL / "summary.json")},
        ],
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
