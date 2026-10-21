from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
V7 = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v7"
ADDITIONAL = BASE / "temperature_noaa_hma_additional_physical_records_v1"
MECHANISM = BASE / "temperature_four_prior_strategy_c_physical_evidence_extension_v1"
OUTPUT = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v8"
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_row(matrix: pd.DataFrame, values: dict[str, Any]) -> pd.DataFrame:
    unknown = set(values) - set(matrix.columns)
    if unknown:
        raise RuntimeError(f"unknown-columns:{sorted(unknown)}")
    return pd.concat(
        [matrix, pd.DataFrame([{column: values.get(column, "") for column in matrix.columns}])],
        ignore_index=True,
    )


def update_row(matrix: pd.DataFrame, evidence_id: str, values: dict[str, Any]) -> None:
    index = matrix.index[matrix.evidence_id == evidence_id]
    if len(index) != 1:
        raise RuntimeError(f"row-identity-failed:{evidence_id}:{len(index)}")
    for key, value in values.items():
        matrix.loc[index[0], key] = value


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(V7 / "evidence_matrix.csv")
    metrics = pd.read_csv(ADDITIONAL / "additional_physical_project_metrics.csv")
    mechanism_summary = json.loads((MECHANISM / "summary.json").read_text(encoding="utf-8"))
    mechanism_cross = pd.read_csv(MECHANISM / "physical_evidence_cross_prior_summary.csv")

    update_row(
        matrix,
        "NGRAMRING_TIANCAI_AWENG_ALREADY_ASSESSED",
        {
            "admitted_role": "identity_context_only_Tiancai_and_Aweng_now_have_explicit_rows",
            "finding": (
                "Tiancai and Aweng official NOAA records now have explicit numeric rows and must not be double counted here; "
                "Ngamring remains sparse prior evidence."
            ),
        },
    )

    common = {"sealed_period_values_used": False, "production_modified": False}
    aweng = metrics[metrics.site_id == "aweng_co"].iloc[0]
    guliya = metrics[metrics.site_id == "guliya_ice_core"].iloc[0]
    puruo = metrics[metrics.site_id == "puruogangri_ice_core"].iloc[0]
    rows = [
        {
            "evidence_id": "NOAA_AWENG_CO_BRGDGT_MAAT",
            "region": "western_Tibetan_Plateau",
            "longitude": 81.67,
            "coverage": "latest_safe_quantitative_pair_1758-1834",
            "target": "annual_mean_air_temperature",
            "measurement_family": "lake_sediment_brGDGT_MBT_CBT",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "independent_non_tree_lake_sediment;same_record_as_prior_Aweng_grouped_audit",
            "spatial_relevance": "western_central_transition",
            "season_match": "annual",
            "admitted_role": "quantitative_low_frequency_direction_and_rough_amplitude_gate",
            "finding": f"Aweng warms {aweng.proxy_value:+.3f} C from 1758 to 1834; project median warms {aweng.project_six_member_median:+.3f} C. Direction agrees but project amplitude is much weaker; proxy uncertainty is large.",
            "next_action": "retain_as_sparse_quantitative_western_constraint_not_a_continuous_1850_1920_gate",
            "source_url": "https://www.ncei.noaa.gov/pub/data/paleo/reconstructions/climate12k/temperature/version1.0.0/Temp12k_directory_NOAA_files/AwengCo.Li.2017.txt",
            **common,
        },
        {
            "evidence_id": "NOAA_GULIYA_ICE_CORE_D18O_174Y",
            "region": "western_Kunlun_HMA",
            "longitude": 81.46,
            "coverage": "1840-1920_safe;1940-1950_safe",
            "target": "ice_core_d18O_temperature_direction",
            "measurement_family": "ice_core_stable_isotope",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "independent_non_tree_ice_core_physical_family",
            "spatial_relevance": "high_western_HMA",
            "season_match": "precipitation_season_temperature_direction_not_annual_amplitude",
            "admitted_role": "direction_only_low_frequency_gate",
            "finding": f"Guliya d18O and project both trend toward warming over 1850-1920; isotope amplitude is not converted to degC.",
            "next_action": "retain_direction_only_and_do_not_count_ions_or_dust_as_extra_temperature_votes",
            "source_url": "https://www.ncei.noaa.gov/pub/data/paleo/icecore/trop/guliya/guliya2018core1annual.txt",
            **common,
        },
        {
            "evidence_id": "NOAA_PURUOGANGRI_ICE_CORE_D18O_10Y",
            "region": "central_Tibetan_Plateau",
            "longitude": 89.0833,
            "coverage": "safe_decades_1850-1859_to_1910-1919",
            "target": "decadal_ice_core_d18O_summer_temperature_direction",
            "measurement_family": "ice_core_stable_isotope",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "independent_non_tree_ice_core_physical_family",
            "spatial_relevance": "central_HMA_north_edge",
            "season_match": "summer_precipitation_season_direction_not_annual_amplitude",
            "admitted_role": "direction_only_decadal_gate",
            "finding": "Puruogangri d18O and project both cool from the 1850s through the 1910s; the 1920-1929 bin was excluded because it crosses the sealed period.",
            "next_action": "retain_direction_only_and_do_not_convert_per_mil_to_degC",
            "source_url": "https://www.ncei.noaa.gov/pub/data/paleo/icecore/trop/puruogangri/puruogangri2006-core2-10yr-noaa.txt",
            **common,
        },
        {
            "evidence_id": "XIADA_CO_BRGDGT_PENDING_NUMERIC",
            "region": "western_Tibetan_Plateau",
            "longitude": 79.36,
            "coverage": "approximately_0-2013_CE",
            "target": "brGDGT_quantitative_MAAT_or_mixed_season_temperature",
            "measurement_family": "lake_sediment_brGDGT",
            "machine_readable": "institutional_metadata_only_no_direct_numeric_table_confirmed",
            "numeric_now": False,
            "source_independence": "independent_non_tree_lake_sediment",
            "spatial_relevance": "very_high_western_HMA",
            "season_match": "annual_or_mixed_season_requires_calibration_audit",
            "admitted_role": "highest_priority_pending_quantitative_record",
            "finding": "Scientific coverage is ideal for 1500-1879, but no no-login machine-readable time-temperature table has been confirmed.",
            "next_action": "continue_TPDC_author_and_supplement_route_without_blocking_current_decision",
            "source_url": "https://doi.org/10.11888/Paleoenv.tpdc.271728",
            **common,
        },
        {
            "evidence_id": "BAMU_CO_BRGDGT_PENDING_NUMERIC",
            "region": "central_Tibetan_Plateau",
            "longitude": 90.5473,
            "coverage": "past_approximately_2200_years",
            "target": "brGDGT_quantitative_MAAT",
            "measurement_family": "lake_sediment_brGDGT",
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "independent_non_tree_lake_sediment",
            "spatial_relevance": "very_high_central_HMA",
            "season_match": "annual",
            "admitted_role": "priority_pending_quantitative_record",
            "finding": "Approximately 25-year central-HMA record is highly relevant, but no public numeric series was found.",
            "next_action": "author_or_repository_request",
            "source_url": "https://www.sciencedirect.com/science/article/pii/S0031018223004418",
            **common,
        },
        {
            "evidence_id": "DAGZE_CO_UK37_PENDING_NUMERIC",
            "region": "central_Tibetan_Plateau",
            "longitude": 87.53,
            "coverage": "past_approximately_2000_years",
            "target": "UK37_quantitative_summer_temperature",
            "measurement_family": "lake_sediment_alkenone",
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "independent_non_tree_lake_sediment",
            "spatial_relevance": "high_central_HMA",
            "season_match": "summer",
            "admitted_role": "priority_pending_quantitative_record",
            "finding": "Decadal quantitative temperature is scientifically relevant, but the full numeric series is not publicly downloadable.",
            "next_action": "author_or_repository_request",
            "source_url": "https://doi.org/10.1177/0959683615572852",
            **common,
        },
        {
            "evidence_id": "DASUOPU_D18O_EXCLUDED_TEMPERATURE",
            "region": "central_Himalaya",
            "longitude": 85.72,
            "coverage": "1450-1996",
            "target": "ice_core_d18O_monsoon_and_precipitation_effect",
            "measurement_family": "ice_core_stable_isotope",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "independent_ice_core_but_not_a_stable_temperature_target",
            "spatial_relevance": "high",
            "season_match": "JJAS_monsoon_not_temperature",
            "admitted_role": "excluded_from_temperature_vote_monsoon_mixing_context_only",
            "finding": "Published interpretation identifies monsoon intensity and amount effects rather than a stable local temperature relation.",
            "next_action": "do_not_use_as_temperature_validation",
            "source_url": "https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=noaa-icecore-11180",
            **common,
        },
        {
            "evidence_id": "EAST_RONGBUK_D18O_EXCLUDED_TEMPERATURE",
            "region": "central_eastern_Himalaya",
            "longitude": 86.92,
            "coverage": "approximately_1840s-1997",
            "target": "ice_core_d18O_precipitation_amount_effect",
            "measurement_family": "ice_core_stable_isotope",
            "machine_readable": True,
            "numeric_now": True,
            "source_independence": "independent_ice_core_but_not_a_stable_temperature_target",
            "spatial_relevance": "high",
            "season_match": "monsoon_precipitation_not_temperature",
            "admitted_role": "excluded_from_temperature_vote_amount_effect_context_only",
            "finding": "The isotope series is inconsistent with nearby temperature because of precipitation amount effects.",
            "next_action": "do_not_use_as_temperature_validation",
            "source_url": "https://www.ncei.noaa.gov/pub/data/paleo/icecore/trop/everest/rongbuk-noaa.txt",
            **common,
        },
    ]
    for row in rows:
        matrix = append_row(matrix, row)
    if matrix.evidence_id.duplicated().any():
        raise RuntimeError("duplicate-evidence-id")
    matrix.to_csv(OUTPUT / "evidence_matrix.csv", index=False)

    plotted = pd.DataFrame(
        [
            ("Aweng", 81.67, 32.75, True),
            ("Guliya", 81.46, 35.23, True),
            ("Puruogangri", 89.0833, 33.9167, True),
            ("Naqu", 91.88, 31.38, False),
            ("Qinghai", 100.137, 36.8112, False),
            ("PX202", 102.22, 28.27, False),
        ],
        columns=["label", "longitude", "latitude", "project_direction_supported"],
    )
    fig, ax = plt.subplots(figsize=(9.7, 5.8))
    for row in plotted.itertuples(index=False):
        color = "#2b7a4b" if row.project_direction_supported else "#bb3e3e"
        ax.scatter(row.longitude, row.latitude, s=120, color=color, edgecolor="white", linewidth=1.2)
        ax.annotate(row.label, (row.longitude, row.latitude), xytext=(5, 6), textcoords="offset points", fontsize=9)
    ax.scatter([], [], s=100, color="#2b7a4b", label="New/physical direction supports project")
    ax.scatter([], [], s=100, color="#bb3e3e", label="Stable four-prior direction conflict")
    ax.set_xlim(78, 104)
    ax.set_ylim(27, 39)
    ax.set_xlabel("Longitude (degE)")
    ax.set_ylabel("Latitude (degN)")
    ax.set_title("Resolved HMA low-frequency direction evidence")
    ax.grid(color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUT / "resolved_direction_evidence_map.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment": "temperature_1850_1950_low_frequency_evidence_matrix_v8",
        "supersedes": "temperature_1850_1950_low_frequency_evidence_matrix_v7",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "reject_uniform_HMA_warm_bias_focus_stable_conflicts_on_Naqu_PX202_Qinghai_operator_path",
        "evidence_rows": int(len(matrix)),
        "new_numeric_direction_results": metrics.to_dict(orient="records"),
        "new_direction_agreement_count": int(metrics.direction_agreement.sum()),
        "stable_four_prior_conflict_evidence_ids": mechanism_cross.loc[
            mechanism_cross.all_full_branches_conflict, "evidence_id"
        ].tolist(),
        "stable_four_prior_agreement_evidence_ids": mechanism_cross.loc[
            mechanism_cross.all_full_branches_agree, "evidence_id"
        ].tolist(),
        "major_conflicts_fixed_by_omitting_central_proxies": mechanism_summary[
            "major_conflicts_fixed_by_omitting_central_proxies"
        ],
        "mechanism_verdict": (
            "The stable Naqu, PX202, and Qinghai conflicts persist across CESM, ModE-Sim, MPI, MIROC, proxy6, proxy9, "
            "and central-proxy omission. They are not CESM-specific and are not explained by central proxy9 alone."
        ),
        "candidate_v2_authorized": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "sealed_period_values_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    inputs = [
        V7 / "evidence_matrix.csv",
        V7 / "summary.json",
        ADDITIONAL / "summary.json",
        ADDITIONAL / "additional_physical_project_metrics.csv",
        MECHANISM / "summary.json",
        MECHANISM / "physical_evidence_cross_prior_summary.csv",
    ]
    manifest = {
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in inputs],
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = """# HMA low-frequency temperature evidence matrix v8

## What is now resolved

- Aweng, Guliya, and Puruogangri all agree with the project in low-frequency direction. Aweng is quantitative and shows that the project underestimates its 1758-1834 warming amplitude.
- Naqu, PX202, and Qinghai remain direction conflicts in all eight CESM/ModE/MPI/MIROC x proxy6/proxy9 Strategy-C branches.
- Omitting central proxies does not repair those three conflicts.

The evidence rejects an HMA-wide historical warm-bias correction. The remaining problem is a localized operator/selection-path failure in the central/eastern and northeastern evidence channels, not a CESM-only drift or one central proxy record.

Xiada, Bamu, and Dagze remain the highest-value pending quantitative lake records. Dasuopu and East Rongbuk are explicitly excluded as temperature votes because isotope amount/monsoon effects dominate their interpretation.

Candidate v2 remains unauthorized. No EnKF, OGGM, production modification, sealed-period value, or 20CRv3 access occurred.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
