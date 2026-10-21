from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = (
    ROOT
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "ice_surface_reconstruction"
    / "drivers"
)
BASE_PATH = DRIVERS / "build_temperature_low_frequency_evidence_matrix_v1.py"
OUT = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "temperature_1850_1950_low_frequency_evidence_matrix_v2"
)


def load_base():
    spec = importlib.util.spec_from_file_location("lf_matrix_v1", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_rows() -> list[dict]:
    common = {
        "sealed_period_values_used": False,
        "production_modified": False,
    }
    return [
        {
            "evidence_id": "PAKISTAN_WESTERN_HIMALAYA_TRW_2026",
            "region": "Pakistan_western_Himalaya",
            "longitude": "western_Himalaya",
            "coverage": "1855-2020",
            "target": "May-June_Tmax",
            "measurement_family": "tree_ring_reconstruction",
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "new_composite_but_Rauf_family_overlap_possible",
            "spatial_relevance": "high_western_HMA",
            "season_match": "pre_monsoon_Tmax_not_annual_Tmean",
            "admitted_role": "candidate_pending_author_data_and_identity",
            "finding": "40.8 percent calibration variance reported; support data contain statistics but no annual series",
            "next_action": "request_annual_series_and_three_site_sample_crosswalk",
            "source_url": "https://doi.org/10.1088/2515-7620/ae718e",
            **common,
        },
        {
            "evidence_id": "SANO_HUMLA_TRW_MXD_2005",
            "region": "western_Nepal_Humla",
            "longitude": "western_Nepal",
            "coverage": "1752-2000",
            "target": "March-September_temperature",
            "measurement_family": "TRW_and_density_reconstruction",
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "production_identity_unresolved",
            "spatial_relevance": "high_central_west_HMA",
            "season_match": "warm_season",
            "admitted_role": "candidate_pending_numeric_and_identity",
            "finding": "published curve only; calibration skill is moderate and split validation is mixed",
            "next_action": "request_annual_series_chronology_and_sample_ids",
            "source_url": "https://doi.org/10.1016/j.dendro.2005.08.003",
            **common,
        },
        {
            "evidence_id": "RARA_CHOLANGPATI_MXD_2026",
            "region": "central_Himalaya_Nepal",
            "longitude": "central_Nepal",
            "coverage": "1775-2022",
            "target": "April-September_temperature",
            "measurement_family": "MXD_network_reconstruction",
            "machine_readable": False,
            "numeric_now": False,
            "source_independence": "new_Rara_Chola_plus_old_Jumla_Kalin_mixed",
            "spatial_relevance": "high_central_HMA",
            "season_match": "warm_season",
            "admitted_role": "candidate_pending_numeric_source_partition",
            "finding": "article and supplement expose no annual numeric table",
            "next_action": "request_series_and_sitewise_reconstruction_for_new_sites_only",
            "source_url": "publication_metadata_only",
            **common,
        },
        {
            "evidence_id": "TIEN_SHAN_MXD_2019",
            "region": "Tien_Shan_westerly_edge",
            "longitude": "external_HMA_edge",
            "coverage": "1615-2005",
            "target": "July-August_temperature",
            "measurement_family": "MXD_reconstruction",
            "machine_readable": True,
            "numeric_now": False,
            "source_independence": "three_of_five_sites_overlap_project_TRW_sites",
            "spatial_relevance": "external_westerly_edge",
            "season_match": "summer",
            "admitted_role": "source_overlap_context_only",
            "finding": "mixed source overlap; previously excluded from blind validation",
            "next_action": "do_not_count_as_independent_vote",
            "source_url": "https://www.ncei.noaa.gov/pub/data/paleo/treering/reconstructions/asia/tienshan2019temp.txt",
            **common,
        },
        {
            "evidence_id": "KATHMANDU_COOK_2003",
            "region": "central_Himalaya_Nepal",
            "longitude": "Kathmandu_network",
            "coverage": "multi_century",
            "target": "temperature_sensitive_tree_ring_network",
            "measurement_family": "tree_ring_reconstruction",
            "machine_readable": True,
            "numeric_now": False,
            "source_independence": "at_least_28_of_32_records_overlap_project",
            "spatial_relevance": "high_central_HMA",
            "season_match": "seasonal",
            "admitted_role": "same_source_context_only",
            "finding": "confirmed source-network overlap",
            "next_action": "do_not_count_as_independent_vote",
            "source_url": "https://www.ncei.noaa.gov/pub/data/paleo/treering/reconstructions/asia/nepal/",
            **common,
        },
    ]


def main() -> None:
    base = load_base()
    base.OUT = OUT
    original_rows = base.evidence_rows

    def revised_rows():
        rows = original_rows()
        for row in rows:
            if row["evidence_id"] == "UPPER_INDUS_MXD_2024":
                row["source_independence"] = "same_site_high_probability_source_overlap_exact_sample_overlap_unresolved"
                row["finding"] = "same coordinates as project PAKI025/Asia-DskJUE; MXD versus TRW; no public annual table or sample crosswalk"
                row["next_action"] = "request_annual_series_and_tree_core_crosswalk_not_an_independent_vote"
            elif row["evidence_id"] == "BANERJEE_AZAM_GLACIER_LENGTH_2016":
                row["admitted_role"] = "direction_and_weak_multidecadal_amplitude_pending_native_table"
                row["finding"] = "native output is 3-year and figures are 5-year smooth; total regional warming about 1.6 +/- 0.9 K over 1860-2010"
                row["next_action"] = "request_unsmoothed_3-year_HMA_CH_WHK_series_uncertainty_and_input_ledger"
        rows.extend(candidate_rows())
        return rows

    base.evidence_rows = revised_rows
    base.main()

    queue_path = OUT / "author_data_request_queue.csv"
    queue = pd.read_csv(queue_path)
    mask = queue["record"].str.contains("glacier-length", regex=False)
    queue.loc[mask, "contact"] = "argha@iiserpune.ac.in;farooqazam@iiti.ac.in"
    queue.loc[mask, "request"] = (
        "unsmoothed native 3-year HMA/CH/WH-K series, confidence interval, "
        "glacier count by step, 43-input and 34-selected lists, c/tau parameters, "
        "baseline, code, and plot source data; mask 1921-1939 before delivery"
    )
    queue.to_csv(queue_path, index=False)

    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = OUT.name
    summary["evidence_rows"] = len(revised_rows())
    summary["upper_indus_identity"] = "same_site_high_probability_source_overlap_exact_sample_overlap_unresolved"
    summary["dokriani_identity"] = "new_sequence_regional_and_team_partial_overlap_numeric_pending"
    summary["banerjee_azam_native_resolution"] = "3_year_output_5_year_smoothed_figures"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = OUT / "FINAL_RESULT.md"
    report = report_path.read_text(encoding="utf-8").replace(
        "evidence matrix v1", "evidence matrix v2"
    )
    report += """

## Identity closure

- Dokriani/Tela is a new Abies pindrow chronology with no exact sequence match in the project ledger. It is nevertheless regionally and institutionally related to the Gangotri evidence family, so it is not yet a blind independent vote.
- Upper Indus MXD 2024 shares the PAKI025/Asia-DskJUE site coordinates already represented by project TRW. Different measurements do not prove different trees; exact sample overlap remains unresolved.
- Banerjee-Azam is independent of the proxy/PSM/EnKF family but not independent of WGMS and historical glacier-length inputs. Its early curve can contain network-composition steps and must not define a decadal correction.

The public search therefore changes the route, not the production field: prioritize retrieval of the Dokriani annual series and use Banerjee-Azam only as a low-frequency physical gate. Upper Indus MXD is retained for within-site measurement sensitivity, not counted as a new spatial vote.

A publisher search preview exposed descriptive year labels overlapping the sealed interval, but no temperature value was extracted, stored, plotted, or used in any decision.
"""
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
