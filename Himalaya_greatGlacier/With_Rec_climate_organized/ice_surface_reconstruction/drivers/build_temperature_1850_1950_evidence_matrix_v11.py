from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
V10 = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v10"
NOAA = BASE / "temperature_noaa_ngamring_original_v1"
CONFLICT = BASE / "temperature_ngamring_calibration_direction_conflict_v1"
OUTPUT = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v11"
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
    return pd.concat(
        [matrix, pd.DataFrame([{column: values.get(column, "") for column in matrix.columns}])],
        ignore_index=True,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    matrix = pd.read_csv(V10 / "evidence_matrix.csv")
    noaa = json.loads((NOAA / "summary.json").read_text(encoding="utf-8"))
    conflict = json.loads((CONFLICT / "summary.json").read_text(encoding="utf-8"))
    original = noaa["records"][0]
    secondary = conflict["metrics"]

    update_row(
        matrix,
        "LIANG2024_NGAMRING_BRGDGT_RF_MAF",
        {
            "admitted_role": "calibration_target_sensitivity_only_not_site_direction_evidence",
            "finding": (
                f"At identical years 1597 and 1778 the common RF-MAF cools "
                f"{secondary['secondary_common_RF_MAF_change_c_per_century']:+.3f} C/century, but the official author "
                f"MAAT warms {secondary['original_author_MAAT_change_c_per_century']:+.3f} C/century. This row demonstrates "
                "a calibration/target-induced direction reversal and must not be treated as the original proxy direction."
            ),
            "next_action": "use_only_as_calibration_sensitivity;primary_site_evidence_is_NOAA_author_MAAT",
        },
    )
    matrix = add_row(
        matrix,
        {
            "evidence_id": "NOAA_NGAMRING_AUTHOR_BRGDGT_MAAT",
            "region": "central_southern_Tibetan_Plateau",
            "longitude": 87.18,
            "coverage": "safe_temperature_points_1597_and_1778;no_bridge_points_1850_1920",
            "target": "mean_annual_air_temperature_with_possible_winter_season_bias",
            "measurement_family": "lake_sediment_brGDGT_author_calibration",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "official_NOAA_author_contribution;same_archive_as_Liang2024_Ngamring_do_not_double_count",
            "spatial_relevance": "central_HMA_high",
            "season_match": "annual_target_but_published_possible_winter_weight",
            "admitted_role": "primary_independent_site_direction_evidence_before_1850",
            "finding": (
                f"Official author MAAT warms {original['ngamring_maat_trend_c_per_century']:+.3f} C/century over 1597-1778; "
                f"the project warms {original['project_median_trend_at_proxy_years_c_per_century']:+.3f} C/century at the same years. "
                "Direction agrees. There are no safe 1850-1920 MAAT points."
            ),
            "next_action": "retain_as_primary_Ngamring_site_record;do_not_infer_1850_1920_bridge",
            "source_url": "https://www.ncei.noaa.gov/pub/data/paleo/paleolimnology/asia/china/sun2022/sun2022-ngamring-co.txt",
            "sealed_period_values_used": False,
            "production_modified": False,
        },
    )
    if matrix.evidence_id.duplicated().any():
        raise RuntimeError("duplicate-evidence-id")
    matrix.to_csv(OUTPUT / "evidence_matrix.csv", index=False)

    previous = json.loads((V10 / "summary.json").read_text(encoding="utf-8"))
    previous.update(
        {
            "experiment": "temperature_1850_1950_low_frequency_evidence_matrix_v11",
            "supersedes": "temperature_1850_1950_low_frequency_evidence_matrix_v10",
            "row_count": int(len(matrix)),
            "ngamring_primary_evidence": "NOAA_author_MAAT_warming_agrees_with_project_1597_1778",
            "ngamring_secondary_evidence": "Liang2024_common_RF_MAF_cooling_is_calibration_target_sensitivity_only",
            "new_mechanistic_finding": "target_and_calibration_mapping_can_reverse_low_frequency_direction_on_identical_archive_and_chronology",
            "decision": "reject_uniform_HMA_correction;require_native_target_and_calibration_identity_before_direction_votes",
            "candidate_v2_authorized": False,
            "prohibited_source": "20CRv3",
            "prohibited_source_accessed": False,
        }
    )
    (OUTPUT / "summary.json").write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "inputs": [
            {"path": str(V10 / "evidence_matrix.csv"), "sha256": sha256(V10 / "evidence_matrix.csv")},
            {"path": str(V10 / "summary.json"), "sha256": sha256(V10 / "summary.json")},
            {"path": str(NOAA / "summary.json"), "sha256": sha256(NOAA / "summary.json")},
            {"path": str(CONFLICT / "summary.json"), "sha256": sha256(CONFLICT / "summary.json")},
        ],
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = """# HMA low-frequency temperature evidence matrix v11

Ngamring is corrected at the evidence-identity level. The official NOAA author
MAAT warms over 1597-1778 and agrees with the project. The 2024 common RF-MAF
reanalysis cools over the identical years; it is retained only as evidence that
target/calibration choices can reverse low-frequency direction.

This does not validate a uniform HMA correction and does not provide a
1850-1920 Ngamring bridge. It does establish that native target and calibration
identity must be resolved before any proxy direction is counted.

The sealed period was not used. Production was not changed, EnKF and OGGM were
not run, and 20CRv3 was not accessed.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
