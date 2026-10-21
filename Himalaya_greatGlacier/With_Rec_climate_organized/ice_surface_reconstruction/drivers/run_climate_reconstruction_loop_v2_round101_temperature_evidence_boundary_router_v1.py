"""Route completed temperature evidence without reading climate arrays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXPECTED_FACTS_SHA256 = "e687443b59089decea54078ee0d07b90287d5b137774b1f98c9541c0491bfdf9"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest(path)}


def main() -> int:
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "LIA_PROJECT_FACTS.md").exists())
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_101_temperature_evidence_boundary_router_v1"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": output / "ROUND_CONTRACT.md",
        "round08": loop / "round_08_documented_temperature_seasonality_v1/summary.json",
        "round09": loop / "round_09_temperature_proxy_association_stability_v1/summary.json",
        "round13": loop / "round_13_new_lmr_temperature_network_validation_v2/summary.json",
        "round14": loop / "round_14_temperature_product_cross_validation_v1/summary.json",
        "round15": loop / "round_15_network_level_regularized_signal_proof_v1/summary.json",
        "round16": loop / "round_16_purged_low_frequency_common_mode_v1/summary.json",
        "round18": loop / "round_18_nested_year_alignment_v1/summary.json",
        "round29": loop / "round_29_monotonic_nonlinear_temperature_psm_v1/summary.json",
        "round53": loop / "round_53_dod2k_temperature_identity_chronology_v1/summary.json",
        "round54": loop / "round_54_malan_icecore_temperature_nested_validation_v1/summary.json",
        "round88": loop / "round_88_post_final_test_scientific_router_v1/summary.json",
        "round97": loop / "round_97_western_ghcnm_temperature_phase_adjudication_v1/summary.json",
        "round98": loop / "round_98_project_temperature_member_attribution_v1/summary.json",
        "round99": loop / "round_99_temperature_prior_increment_station_decomposition_v1/summary.json",
        "round100": loop / "round_100_temperature_station_proxy_trend_attribution_v2/summary.json",
        "round100_manifest": loop / "round_100_temperature_station_proxy_trend_attribution_v2/source_manifest.json",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if digest(paths["facts"]) != EXPECTED_FACTS_SHA256:
        raise RuntimeError(f"facts-hash-conflict:{digest(paths['facts'])}")
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    results = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items() if name.startswith("round") and name != "round100_manifest"}
    manifest100 = json.loads(paths["round100_manifest"].read_text(encoding="utf-8"))

    closed_repairs = {
        "documented_seasonality": results["round08"]["decision"] == "close_documented_seasonality_as_network_wide_temperature_repair",
        "existing_linear_psm_temporal_stability": results["round09"]["decision"] == "current_temperature_psm_is_weak_and_temporally_unstable",
        "expanded_lmr_linear_network": results["round13"]["decision"] == "close_new_lmr_linear_temperature_psm_promotion",
        "instrumental_product_switch": results["round14"]["decision"] == "close_local_linear_temperature_psm_family",
        "regularized_network_signal": results["round15"]["decision"] == "regional_regularized_signal_proof_failed",
        "purged_low_frequency_signal": results["round16"]["decision"] == "purged_low_frequency_common_mode_failed",
        "one_year_alignment": results["round18"]["decision"] == "close_simple_one_year_misalignment_as_network_repair",
        "monotonic_nonlinear_psm": results["round29"]["decision"] == "close_monotonic_nonlinear_temperature_psm_family",
        "new_dod2k_chronology": results["round53"]["annual_temperature_psm_candidate_count"] == 1 and results["round54"]["decision"].startswith("close_"),
        "member_weighting": results["round98"]["decision"] == "common_all_member_trend_failure_close_nonnegative_weighting_route",
        "prior_or_localization": results["round99"]["decision"] == "proxy_assimilation_increment_dominant_route_to_innovation_gain_attribution",
        "record_deletion": results["round100"]["decision"] == "close_record_deletion_route_to_network_wide_psm_chronology_repair",
    }
    early_failure_verified = bool(
        results["round97"]["network_reliability"]["passes"]
        and results["round98"]["all_six_members_have_wrong_trend_sign"]
        and results["round99"]["both_ordinary_members_increment_wrong_sign"]
        and results["round100"]["all_reproduction_gates_pass"]
    )
    all_existing_repairs_closed = bool(all(closed_repairs.values()))
    no_arrays_read = True
    locked_accessed = False
    if not manifest100["local_source_states_unchanged"]:
        raise RuntimeError("round100-source-integrity-failed")
    if early_failure_verified and all_existing_repairs_closed:
        decision = "require_explicit_exploratory_approval_or_new_external_evidence"
    else:
        decision = "retain_automatic_temperature_route_unresolved"
    summary = {
        "status": "complete",
        "climate_station_proxy_or_locked_arrays_read": False,
        "locked_1921_1939_accessed": locked_accessed,
        "early_western_temperature_failure_verified": early_failure_verified,
        "closed_repair_families": closed_repairs,
        "all_existing_data_repair_families_closed": all_existing_repairs_closed,
        "automatic_existing_data_temperature_tuning_permitted": False,
        "confirmatory_route": {
            "required_input": "materially new independent temperature evidence or proxy chronology plus a newly preregistered untouched validation target",
            "examples": ["EKF400v2 after authorized WDCC access", "genuinely new independent HMA proxy chronology"],
        },
        "exploratory_route": {
            "requires_explicit_scientific_approval": True,
            "first_bounded_candidate": "background_only_or_cross_validated_global_increment_shrinkage",
            "claim_boundary": "cannot use already inspected 1901-1920 or consumed 1921-1939 as independent validation and cannot authorize production or OGGM",
        },
        "precipitation_boundary_unchanged": "requires APHRO_MA V1101R1 0.5-degree data or authorized access, or explicit approval for a non-equivalent evidence family",
        "decision": decision,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "\n".join([
            "# Temperature Evidence-Boundary Router",
            "",
            f"- Early western failure verified: {str(early_failure_verified).lower()}.",
            f"- Existing-data repair families closed: {sum(closed_repairs.values())}/{len(closed_repairs)}.",
            "- Automatic tuning with the existing proxy network is not permitted.",
            "- Confirmatory continuation requires materially new independent evidence and a new untouched target.",
            "- Exploratory continuation requires explicit approval and cannot claim independent validation from 1901-1920 or 1921-1939.",
            "- The precipitation boundary remains APHRO_MA V1101R1 access or explicit approval of a non-equivalent evidence family.",
            "",
            f"Decision: {decision}.",
        ]) + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    manifest = {"sources_before": before, "sources_after": after, "source_states_unchanged": before == after, "arrays_read": False, "locked_1921_1939_accessed": False}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["source_states_unchanged"]:
        raise RuntimeError("source-integrity-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
