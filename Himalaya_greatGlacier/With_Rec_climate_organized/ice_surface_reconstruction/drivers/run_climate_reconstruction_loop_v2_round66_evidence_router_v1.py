"""Consolidate completed climate-loop evidence into explicit next-input routes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_66_climate_reconstruction_evidence_router_v1"
    paths = {
        "temperature_spatial": loop / "round_30_temperature_reference_spatial_consensus_v1/summary.json",
        "temperature_proxy_boundary": loop / "round_55_lia_only_icecore_spatial_identity_v1/summary.json",
        "aphrodite_readiness": loop / "round_56_aphrodite_external_input_readiness_v1/READINESS_STATUS.json",
        "precipitation_regional": loop / "round_59_western_four_product_candidate_consensus_v1/summary.json",
        "observation_spatial": loop / "round_62_western_observation_spatial_residual_compatibility_v1/summary.json",
        "public_spatial_residual": loop / "round_63_public_candidate_spatial_residual_inventory_v1/summary.json",
        "hgao_shrinkage": loop / "round_64_hgao_spatial_residual_nested_shrinkage_v1/summary.json",
        "regional_fingerprint": loop / "round_65_observation_regional_spatial_fingerprint_v1/summary.json",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    data = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items() if path.suffix == ".json"}
    for name, summary in data.items():
        if summary.get("locked_1921_1939_accessed", False):
            raise RuntimeError(f"locked-period-conflict:{name}")

    aphro_dir = project / data["aphrodite_readiness"]["input_directory"]
    aphro_files = [path for path in aphro_dir.iterdir() if path.is_file() and path.name.lower() != "readme.md"]
    temperature = {
        "regional_50year_curve_supported": bool(data["temperature_spatial"]["regional_50year_curve_remains_supported"]),
        "interim_spatial_field_supported": bool(data["temperature_spatial"]["interim_hma_temperature_field_supported"]),
        "new_proxy_spatial_support_available": False,
        "route": "regional_curve_only_pending_new_independent_sites_or_chronology",
    }
    precipitation = {
        "provisional_western_regional_candidates": data["precipitation_regional"]["provisional_multi_product_consensus_candidates"],
        "strict_four_product_regional_candidate_count": len(data["precipitation_regional"]["strict_four_product_candidates"]),
        "observation_spatial_target_compatible": bool(data["observation_spatial"]["precl_is_valid_cross_product_spatial_adjudicator"]),
        "robust_existing_public_spatial_residual_count": len(data["public_spatial_residual"]["robust_spatial_residual_candidates"]),
        "hgao_shrinkage_supported": bool(data["hgao_shrinkage"]["passes_nested_spatial_residual_shrinkage"]),
        "regional_to_spatial_fingerprint_supported": bool(data["regional_fingerprint"]["passes_regional_to_spatial_fingerprint_recoverability"]),
        "aphrodite_input_present": bool(aphro_files),
        "route": "regional_curve_provisional_spatial_field_closed_pending_external_evidence",
    }
    release_gates = {
        "temperature_spatial_field_supported": temperature["interim_spatial_field_supported"],
        "precipitation_strict_regional_candidate_exists": precipitation["strict_four_product_regional_candidate_count"] > 0,
        "precipitation_spatial_component_supported": precipitation["robust_existing_public_spatial_residual_count"] > 0 or precipitation["hgao_shrinkage_supported"] or precipitation["regional_to_spatial_fingerprint_supported"],
        "aphrodite_adjudication_available": precipitation["aphrodite_input_present"],
    }
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "temperature": temperature,
        "precipitation": precipitation,
        "production_release_gates": release_gates,
        "production_release_permitted": bool(all(release_gates.values())),
        "automatic_resume_triggers": [
            {
                "trigger": "aphrodite_file_added",
                "location": str(aphro_dir),
                "action": "run_round56_preflight_then_fixed_aphrodite_regional_and_spatial_adjudication",
            },
            {
                "trigger": "new_independent_hma_proxy_sites_or_chronology",
                "action": "identity_and_allowed_period_coverage_then_nested_cross_product_psm_validation",
            },
            {
                "trigger": "new_external_gridded_paleoclimate_member",
                "action": "fixed_modern_regional_and_spatial_oof_validation_before_paleo_use",
            },
        ],
        "decision": "hold_production_release_at_external_evidence_boundary",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Climate Reconstruction Evidence Router",
        "",
        f"- Temperature: regional 50-year curve supported `{str(temperature['regional_50year_curve_supported']).lower()}`; spatial field supported `{str(temperature['interim_spatial_field_supported']).lower()}`.",
        f"- Precipitation: provisional western regional candidates `{precipitation['provisional_western_regional_candidates']}`; strict candidate count `{precipitation['strict_four_product_regional_candidate_count']}`.",
        f"- Precipitation spatial component supported: `{str(release_gates['precipitation_spatial_component_supported']).lower()}`.",
        f"- APHRODITE input present: `{str(precipitation['aphrodite_input_present']).lower()}`.",
        f"- Production release permitted: `{str(summary['production_release_permitted']).lower()}`.",
        "",
        "Next automatic trigger: place the registered APHRO_MA V1101R1 0.5-degree source file in:",
        f"`{aphro_dir}`",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
