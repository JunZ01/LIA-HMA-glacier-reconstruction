from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def fingerprint(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_268_temperature_branch_closure_v1"
    if output.exists():
        raise FileExistsError(output)
    sources = {
        "round217": loop / "round_217_temperature_bias_phase_three_record_synthesis_v1/summary.json",
        "round223": loop / "round_223_reconstruction_production_eligibility_synthesis_v1/summary.json",
        "round265": loop / "round_265_npl065_regional_increment_smoother_oof_v1/summary.json",
        "round267": loop / "round_267_nepa043_mam_temperature_modern_oof_v1/summary.json",
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: fingerprint(path) for name, path in sources.items()}
    data = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in sources.items()}

    expected = {
        "round217": "uniform_historical_warm_bias_rejected_early20_conflict_is_season_dependent",
        "round223": "retain_current_production_climate_and_regional_netp_temporal_research_product_only",
        "round265": "retain_npl065_as_validation_evidence_do_not_advance_regional_smoother",
        "round267": "close_nepa043_as_temperature_fitting_evidence",
    }
    for name, decision in expected.items():
        if data[name].get("decision") != decision:
            raise RuntimeError(f"upstream-decision-conflict:{name}:{data[name].get('decision')}")

    evidence = {
        "uniform_1500_1899_warm_bias_supported": False,
        "early_twentieth_century_common_annual_phase_error_supported": False,
        "npl065_difference_signal_has_modern_skill": bool(
            data["round265"]["primary_metrics"]["median_central_heldout_difference_correlation"] >= 0.30
        ),
        "npl065_regional_level_correction_generalizes": bool(data["round265"]["passes_all_gates"]),
        "nepa043_is_sequence_independent_of_nepa029": True,
        "nepa043_mam_temperature_psm_generalizes": bool(data["round267"]["passes_all_gates"]),
    }
    if evidence["npl065_regional_level_correction_generalizes"]:
        raise RuntimeError("unexpected-npl065-generalization")
    if evidence["nepa043_mam_temperature_psm_generalizes"]:
        raise RuntimeError("unexpected-nepa043-generalization")

    decision = "close_temperature_correction_search_retain_current_production_temperature"
    next_action = "advance_precipitation_evidence_improvement_without_temperature_modification"
    result = {
        "experiment_id": "round_268_temperature_branch_closure_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "evidence": evidence,
        "causal_interpretation": {
            "apparent_warm_offset": "not_a_validated_project_warm_bias_independent_records_are_closer_to_project_than_to_cold_four_family_reference",
            "remaining_phase_disagreement": "season_and_record_dependent_not_a_common_annual_temperature_reversal",
            "new_proxy_failure": "modern_response_nonstationarity_prevents_generalizable_temperature_correction",
        },
        "authorized_temperature_correction": None,
        "decision": decision,
        "next_action": next_action,
        "production_temperature_file_changed": False,
        "enkf_run": False,
        "oggm_run": False,
        "raw_proxy_or_climate_value_read": False,
        "locked_1921_1939_value_read": False,
        "source_fingerprints_unchanged": True,
    }
    after = {name: fingerprint(path) for name, path in sources.items()}
    if before != after:
        raise RuntimeError("source-summary-changed")
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 268 temperature branch closure\n\n"
        "- A uniform 1500-1899 project warm bias remains rejected.\n"
        "- The residual early-twentieth-century disagreement is seasonal and record-dependent.\n"
        "- NPL065 has useful difference skill but no generalizable regional level correction.\n"
        "- NEPA043 is sequence-independent but fails modern MAM temperature OOF.\n"
        "- No temperature correction is authorized.\n"
        f"- Decision: `{decision}`.\n"
        f"- Next action: `{next_action}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
