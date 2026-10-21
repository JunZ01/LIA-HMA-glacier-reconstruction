"""Freeze the temperature LIA warm-bias mainline and reopening evidence needs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_lia_warm_bias_mainline_synthesis_v1"
INPUTS = {
    "confirmation": RESULTS / "climate_reconstruction_temperature_lia_warm_bias_confirmation_v1/summary.json",
    "mechanism": RESULTS / "climate_reconstruction_temperature_lia_regional_prior_update_decomposition_v1/summary.json",
    "forced_annual": RESULTS / "climate_reconstruction_temperature_forced_prior_mean_oof_v1/summary.json",
    "forced_field_low_frequency": RESULTS / "climate_reconstruction_temperature_forced_low_frequency_oof_v1/summary.json",
    "forced_common_mode": RESULTS / "climate_reconstruction_temperature_forced_common_mode_oof_v1/summary.json",
    "reference_anchor_full": RESULTS / "climate_reconstruction_temperature_hma_reference_anchored_low_frequency_v1/summary.json",
    "reference_anchor_nested_lambda": RESULTS / "climate_reconstruction_temperature_hma_reference_anchored_nested_lambda_v1/summary.json",
    "reference_anchor_nested_timescale": RESULTS / "climate_reconstruction_temperature_hma_reference_anchored_nested_timescale_v1/summary.json",
    "proxy_low_frequency": RESULTS / "climate_reconstruction_temperature_proxy_low_frequency_information_audit_v1/summary.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path in INPUTS.values():
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True)
    before = {name: sha256(path) for name, path in INPUTS.items()}
    data = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in INPUTS.items()
    }
    if not data["confirmation"]["systematic_lia_warm_bias_confirmed"]:
        raise RuntimeError("Warm-bias confirmation prerequisite failed")
    if data["mechanism"]["hma_mechanism"] != "background_dominant":
        raise RuntimeError("Mechanism prerequisite failed")
    if data["proxy_low_frequency"]["low_frequency_reconstruction_supported"]:
        raise RuntimeError("Proxy information boundary changed")

    candidate_rows = [
        {
            "stage": "confirmation",
            "candidate": "existing six-member reconstruction",
            "status": "systematic_warm_bias_confirmed",
            "key_result": f"four-family median gap {data['confirmation']['hma_four_family_median_project_minus_reference_degC']:+.3f} degC",
        },
        {
            "stage": "mechanism",
            "candidate": "background plus EnKF increment decomposition",
            "status": "background_dominant",
            "key_result": f"background gap {data['mechanism']['hma_median_background_minus_reference_degC']:+.3f} degC; increment {data['mechanism']['hma_median_increment_degC']:+.3f} degC",
        },
        {
            "stage": "correction",
            "candidate": "additive annual ModE-Sim field",
            "status": "rejected_modern_oof",
            "key_result": f"median RMSE change {100 * data['forced_annual']['metrics']['median_rmse_improvement_over_equal_six']:+.2f}%",
        },
        {
            "stage": "correction",
            "candidate": "31-year ModE-Sim spatial low-frequency replacement",
            "status": "rejected_modern_oof",
            "key_result": f"median spatial skill {100 * data['forced_field_low_frequency']['metrics']['median_spatial_residual_skill']:+.2f}%",
        },
        {
            "stage": "correction",
            "candidate": "31-year ModE-Sim HMA common mode",
            "status": "rejected_modern_oof",
            "key_result": f"median RMSE change {100 * data['forced_common_mode']['metrics']['median_rmse_improvement_over_equal_six']:+.2f}%",
        },
        {
            "stage": "correction",
            "candidate": "full four-family LP51 anchor",
            "status": "rejected_family_transfer",
            "key_result": f"median held-out 50-year improvement {100 * data['reference_anchor_full']['median_held_out_50year_rmse_improvement']:+.2f}%",
        },
        {
            "stage": "correction",
            "candidate": "nested LP51 anchor",
            "status": "rejected_family_transfer",
            "key_result": f"median held-out 50-year improvement {100 * data['reference_anchor_nested_lambda']['median_outer_50year_rmse_improvement']:+.2f}%",
        },
        {
            "stage": "correction",
            "candidate": "nested 31/51/101-year anchor",
            "status": "rejected_family_transfer",
            "key_result": f"median held-out 50-year improvement {100 * data['reference_anchor_nested_timescale']['median_outer_50year_rmse_improvement']:+.2f}%",
        },
        {
            "stage": "evidence",
            "candidate": "existing proxy9 50-year information",
            "status": "insufficient",
            "key_result": f"{data['proxy_low_frequency']['robust_low_frequency_record_count']} robust of {data['proxy_low_frequency']['selected_temperature_record_count']} documented-temperature records",
        },
    ]
    pd.DataFrame(candidate_rows).to_csv(OUTPUT / "mainline_decision_table.csv", index=False)

    regional = data["proxy_low_frequency"]["robust_record_counts_by_region"]
    requirements = [
        {
            "evidence_need": "cross-family robust low-frequency temperature records",
            "current": data["proxy_low_frequency"]["robust_low_frequency_record_count"],
            "minimum_gate": 25,
            "minimum_deficit": 25 - data["proxy_low_frequency"]["robust_low_frequency_record_count"],
            "status": "frozen_numeric_gate",
        },
        {
            "evidence_need": "west HMA robust low-frequency records",
            "current": regional["west_lt80"],
            "minimum_gate": 5,
            "minimum_deficit": 5 - regional["west_lt80"],
            "status": "frozen_numeric_gate",
        },
        {
            "evidence_need": "central HMA robust low-frequency records",
            "current": regional["central_80_90"],
            "minimum_gate": 5,
            "minimum_deficit": 5 - regional["central_80_90"],
            "status": "frozen_numeric_gate",
        },
        {
            "evidence_need": "east HMA robust low-frequency records",
            "current": regional["east_ge90"],
            "minimum_gate": 5,
            "minimum_deficit": 5 - regional["east_ge90"],
            "status": "frozen_numeric_gate",
        },
        {
            "evidence_need": "structurally independent transient forced prior ensembles",
            "current": 1,
            "minimum_gate": 3,
            "minimum_deficit": 2,
            "status": "methodological_recommendation_not_acceptance_gate",
        },
        {
            "evidence_need": "non-tree or explicitly low-frequency-preserving temperature chronologies",
            "current": data["proxy_low_frequency"]["archive_class_counts"].get("ice", 0),
            "minimum_gate": None,
            "minimum_deficit": None,
            "status": "priority_acquisition_class",
        },
    ]
    pd.DataFrame(requirements).to_csv(OUTPUT / "minimum_new_evidence_requirements.csv", index=False)

    closest = data["reference_anchor_nested_timescale"]
    summary = {
        "status": "complete",
        "goal_state": "cause_identified_but_no_correction_passes_frozen_validation",
        "systematic_lia_warm_bias_confirmed": True,
        "primary_cause": "time_invariant_analogue_prior_background_lacks_transferable_lia_low_frequency_cold_signal",
        "secondary_cause": "proxy9_and_enkf_updates_amplify_warming_in_plateau_and_himalaya_regions",
        "confirmed_hma_four_family_median_warm_gap_degC": data["confirmation"]["hma_four_family_median_project_minus_reference_degC"],
        "background_minus_reference_median_degC": data["mechanism"]["hma_median_background_minus_reference_degC"],
        "enkf_increment_median_degC": data["mechanism"]["hma_median_increment_degC"],
        "closest_diagnostic_candidate": {
            "method": "nested_reference_anchor",
            "width_years": closest["selected_final_width_years"],
            "lambda": closest["selected_final_lambda"],
            "candidate_lia_median_gap_degC": closest["candidate_lia_median_minus_anchor_degC"],
            "modern_median_rmse_improvement": closest["modern_median_rmse_improvement"],
            "status": "diagnostic_only_rejected_by_frozen_50year_transfer_gate",
        },
        "existing_proxy_low_frequency_supported": False,
        "robust_proxy_record_count": data["proxy_low_frequency"]["robust_low_frequency_record_count"],
        "minimum_additional_robust_records_needed": 25 - data["proxy_low_frequency"]["robust_low_frequency_record_count"],
        "production_temperature_correction_authorized": False,
        "spatial_field_release_authorized": False,
        "oggm_authorized": False,
        "phase_problem_used_for_tuning": False,
        "next_action": "acquire_low_frequency_preserving_temperature_chronologies_or_multiple_transient_prior_ensembles_before_reopening_correction",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# LIA Temperature Warm-Bias Mainline Synthesis",
        "",
        "## Confirmed result",
        "",
        f"The 1500-1899 HMA reconstruction is systematically warm by `{summary['confirmed_hma_four_family_median_warm_gap_degC']:+.3f} degC` relative to the four-family central estimate. The bias is positive in every 50-year window and all 15 GTN-G regions.",
        "",
        "## Cause",
        "",
        f"The prior/background contributes `{summary['background_minus_reference_median_degC']:+.3f} degC`; the median EnKF increment contributes `{summary['enkf_increment_median_degC']:+.3f} degC`. The production analogue background does not recover the LIA low-frequency cold state. Updates are secondary and regionally mixed.",
        "",
        "## Correction status",
        "",
        "No tested correction passes every frozen validation gate. The closest diagnostic candidate uses a 31-year low-frequency four-family anchor at lambda 0.75, reduces the median warm gap to `+0.113 degC`, and slightly improves modern HMA RMSE, but its held-out-family median 50-year improvement is only `39.0%` versus the required `50%`.",
        "",
        "## Evidence boundary",
        "",
        f"Only `{summary['robust_proxy_record_count']}` of 226 documented-temperature proxy9 records carry robust 50-year information across all four reference families. At least `{summary['minimum_additional_robust_records_needed']}` additional robust records are needed to reach the frozen network gate, with explicit deficits in west, central, and east HMA.",
        "",
        "Production correction, spatial-field release, and OGGM use remain unauthorized. The 1901-1920 phase problem was not used for tuning.",
    ]
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    after = {name: sha256(path) for name, path in INPUTS.items()}
    unchanged = before == after
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("Source result changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
