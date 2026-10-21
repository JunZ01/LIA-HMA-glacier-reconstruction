"""Build an auditable decision ledger for the HMA temperature warm-bias goal."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_warm_bias_west_anomaly_decision_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    paths = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "source_decision": RESULTS / "climate_reconstruction_temperature_warm_bias_source_decision_v1/summary.json",
        "mainline": RESULTS / "climate_reconstruction_temperature_lia_warm_bias_mainline_synthesis_v1/summary.json",
        "pmip_stage": RESULTS / "climate_reconstruction_temperature_pmip_prior_stage_synthesis_v1/summary.json",
        "pmip_equal_blend": RESULTS / "climate_reconstruction_temperature_pmip_equal_blend_family_oof_v1/summary.json",
        "blind_bridge": RESULTS / "climate_reconstruction_temperature_low_frequency_blind_test_1921_1939_v1/summary.json",
        "blind_family": RESULTS / "climate_reconstruction_temperature_blind_reference_family_dependency_audit_v2/summary.json",
        "west_operator": RESULTS / "climate_reconstruction_temperature_west_bivariate_operator_decomposition_1861_1900_v1/summary.json",
        "west_stability": RESULTS / "climate_reconstruction_temperature_west_psm_stability_operator_leave_group_out_v1/summary.json",
        "west_gain": RESULTS / "climate_reconstruction_temperature_west_psm_stability_direct_vs_gain_redistribution_v1/summary.json",
        "west_alpha": RESULTS / "climate_reconstruction_temperature_west_b1_alpha0p25_kalam_blind_test_v1/summary.json",
        "west_block_r": RESULTS / "climate_reconstruction_temperature_west_colocated_block_r_modern_oof_v2/summary.json",
        "west_season": RESULTS / "climate_reconstruction_temperature_west_bivariate_documented_season_oof_v1/summary.json",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in paths.items()}
    data = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
        if path.suffix == ".json"
    }

    source = data["source_decision"]
    mainline = data["mainline"]
    pmip = data["pmip_stage"]
    equal = data["pmip_equal_blend"]
    blind = data["blind_bridge"]
    blind_family = data["blind_family"]
    west = data["west_operator"]
    stability = data["west_stability"]
    gain = data["west_gain"]
    alpha = data["west_alpha"]
    block_r = data["west_block_r"]
    season = data["west_season"]

    checks = {
        "systematic_warm_bias_confirmed": bool(source["systematic_warm_bias_confirmed"]),
        "background_share_at_least_0p90": bool(source["background_share_of_long_term_gap"] >= 0.90),
        "three_forced_models_all_cooler": bool(pmip["all_models_cooler_than_background_all_50year_windows"]),
        "pmip_equal_blend_rejected": bool(not equal["promotion_branch_passes_all_gates"]),
        "blind_bridge_rejected": bool(not blind["passes_all_frozen_gates"]),
        "blind_family_status_not_restored": bool(not blind_family["blind_status_restored"]),
        "west_temperature_pathway_dominant": bool(
            west["dominant_mechanism"].startswith("temperature_operator_covariance_pathway")
        ),
        "west_instability_not_supported": bool(not stability["unstable_temperature_operator_mechanism_supported"]),
        "west_gain_redistribution_material": bool(
            gain["maximum_absolute_gain_redistribution_residual_degC_per_century"] >= 0.5
        ),
        "west_uniform_b1_attenuation_rejected": bool(not alpha["all_preregistered_gates_pass"]),
        "west_block_r_rejected": bool(not block_r["all_modern_oof_gates_pass"]),
        "west_documented_season_rejected": bool(not season["passes_all_mechanism_support_gates"]),
        "production_correction_unauthorized": bool(
            not source["production_authorized"]
            and not mainline["production_temperature_correction_authorized"]
            and not blind["production_authorized"]
        ),
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"decision-evidence-check-failed:{failed}")

    ledger = [
        {
            "question": "Is the 1500-1899 HMA warm bias systematic?",
            "status": "verified",
            "result": f"+{source['hma_warm_gap_degC']:.3f} degC versus four-family center",
            "causal_level": "observed conflict",
            "evidence": str(paths["source_decision"]),
        },
        {
            "question": "Where does the long-term warm bias enter?",
            "status": "verified",
            "result": f"background +{source['background_gap_degC']:.3f} degC; {source['background_share_of_long_term_gap']:.1%} of gap",
            "causal_level": "algorithmic source",
            "evidence": str(paths["source_decision"]),
        },
        {
            "question": "What upstream climate mechanism explains the background bias?",
            "status": "verified",
            "result": "time-invariant analogue background lacks transferable forced LIA cold state; all three forced simulations are cooler in every frozen window",
            "causal_level": "physical forcing mechanism",
            "evidence": str(paths["pmip_stage"]),
        },
        {
            "question": "Is the correction amplitude uniquely determined?",
            "status": "unresolved",
            "result": "three-model anchor gaps span -0.170 to -0.427 degC; equal blend misses frozen 50% gate",
            "causal_level": "amplitude and spatial expression",
            "evidence": str(paths["pmip_equal_blend"]),
        },
        {
            "question": "What creates the western 1861-1900 warming increment?",
            "status": "verified",
            "result": f"temperature covariance pathway +{west['ensemble_mean_effects_relative_to_zero_degC_per_century']['temperature_operator_effect']:.3f} degC/century; precipitation offsets {west['ensemble_mean_effects_relative_to_zero_degC_per_century']['precipitation_operator_effect']:.3f}",
            "causal_level": "algorithmic mechanism",
            "evidence": str(paths["west_operator"]),
        },
        {
            "question": "Why is the western temperature pathway physically nontransferable?",
            "status": "unresolved",
            "result": "instability, uniform b1 attenuation, block-R, record deletion, and documented-season mismatch all fail independent/modern gates",
            "causal_level": "physical proxy/PSM cause",
            "evidence": str(paths["west_season"]),
        },
        {
            "question": "Is documented-season temperature better than annual temperature in western bivariate PSMs?",
            "status": "rejected",
            "result": "all four pooled blocked-OOF median RMSE changes are negative; proxy6/proxy9 both fail",
            "causal_level": "modern predictive validation",
            "evidence": str(paths["west_season"]),
        },
        {
            "question": "Does any tested correction qualify for production?",
            "status": "no",
            "result": f"PMIP equal blend median family gain {equal['branch_metrics']['equal_half_forced_mean']['median_family_improvement']:.1%} < 50%; blind bridge {blind['hma_low11_rmse_improvement_fraction']:.1%} < 30%",
            "causal_level": "frozen acceptance gates",
            "evidence": str(paths["blind_bridge"]),
        },
    ]
    write_csv(OUTPUT / "temperature_cause_and_candidate_ledger.csv", ledger)

    completion = [
        {
            "objective_requirement": "Determine whether the HMA warm bias is real",
            "evidence_status": "proved",
            "remaining_requirement": "none",
        },
        {
            "objective_requirement": "Determine a verifiable cause of the long-term warm bias",
            "evidence_status": "proved",
            "remaining_requirement": "correction amplitude and spatial expression remain unresolved, but causal sign/source is closed",
        },
        {
            "objective_requirement": "Determine a verifiable cause of the western 1881-1900 anomaly",
            "evidence_status": "partial",
            "remaining_requirement": "algorithmic temperature-covariance pathway is proved; independent physical proxy/PSM cause is not proved",
        },
        {
            "objective_requirement": "Test documented-season versus annual temperature PSM",
            "evidence_status": "proved_negative",
            "remaining_requirement": "none; seasonal route is closed",
        },
        {
            "objective_requirement": "Create a correction only after modern frozen preflight passes",
            "evidence_status": "policy_satisfied_no_candidate",
            "remaining_requirement": "no tested candidate passed, so no correction was created or promoted",
        },
    ]
    write_csv(OUTPUT / "goal_completion_audit.csv", completion)

    summary = {
        "status": "complete",
        "checks": checks,
        "long_term_warm_bias": {
            "verified": True,
            "gap_degC": source["hma_warm_gap_degC"],
            "background_share": source["background_share_of_long_term_gap"],
            "verified_primary_cause": mainline["primary_cause"],
            "amplitude_resolved": False,
        },
        "western_transition": {
            "algorithmic_mechanism_verified": True,
            "mechanism": west["dominant_mechanism"],
            "physical_nontransferability_cause_verified": False,
            "rejected_hypotheses": [
                "documented_temperature_season_mismatch",
                "uniform_network_wide_b1_attenuation",
                "psm_temporal_instability_alone",
                "colocated_block_R",
                "record_or_source_family_deletion",
            ],
        },
        "candidate_status": {
            "production_candidate_exists": False,
            "best_exploratory_family": "50:50 archived background plus three-forced-model mean low-frequency blend",
            "pmip_family_median_improvement": equal["branch_metrics"]["equal_half_forced_mean"]["median_family_improvement"],
            "pmip_frozen_gate": 0.50,
            "blind_bridge_improvement": blind["hma_low11_rmse_improvement_fraction"],
            "blind_bridge_frozen_gate": 0.30,
        },
        "goal_completion_state": "incomplete_physical_cause_for_western_transition_and_no_validated_correction",
        "next_admissible_requirement": "materially new independent low-frequency western temperature evidence or chronology with a prospectively frozen validation target",
        "production_authorized": False,
        "oggm_authorized": False,
        "raw_climate_or_proxy_arrays_read": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# HMA Temperature Cause and Candidate Decision

## What is verified

- The 1500-1899 HMA warm gap is `{source['hma_warm_gap_degC']:+.3f} degC`; the background contributes `{source['background_share_of_long_term_gap']:.1%}` of it.
- Three forced simulations independently show that the production analogue background lacks the forced LIA cold state.
- The western 1861-1900 warming is generated algorithmically by the temperature covariance pathway (`+{west['ensemble_mean_effects_relative_to_zero_degC_per_century']['temperature_operator_effect']:.3f} degC/century`); precipitation offsets it.
- Documented-season temperature is not better than annual temperature under frozen modern bivariate OOF.

## What is not verified

- The correction amplitude and regional spatial expression are not uniquely determined.
- The physical reason the western proxy-temperature covariance is nontransferable remains unresolved. Instability, uniform b1 attenuation, block-R, deletion, and season mismatch are rejected explanations.

## Candidate decision

- The 50:50 three-model blend improves all four reference families but reaches only `{equal['branch_metrics']['equal_half_forced_mean']['median_family_improvement']:.1%}` versus the frozen `50%` gate.
- The consumed 1921-1939 blind bridge improves HMA low-frequency RMSE by `{blind['hma_low11_rmse_improvement_fraction']:.1%}` versus the frozen `30%` gate.
- Therefore no independent production correction candidate exists, and no EnKF/OGGM promotion is authorized.

The next admissible evidence is a materially new western low-frequency temperature chronology or independent reconstruction with a prospectively frozen target. Existing-network tuning is closed.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")

    after = {name: sha256(path) for name, path in paths.items()}
    unchanged = before == after
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
