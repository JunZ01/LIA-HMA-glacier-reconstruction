"""Route climate reconstruction after the frozen temperature final test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
    }


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_88_post_final_test_scientific_router_v1"
    paths = {
        "precipitation_router": loop / "round_84_precipitation_external_evidence_router_v1/summary.json",
        "precipitation_product_audit": loop / "round_85_precipitation_external_product_suitability_v1/summary.json",
        "temperature_final_test": loop / "round_86_frozen_temperature_final_test_v1/summary.json",
        "temperature_final_manifest": loop / "round_86_frozen_temperature_final_test_v1/source_manifest.json",
        "temperature_failure": loop / "round_87_temperature_final_test_failure_decomposition_v1/summary.json",
        "temperature_failure_manifest": loop / "round_87_temperature_final_test_failure_decomposition_v1/source_manifest.json",
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
    precipitation = json.loads(paths["precipitation_router"].read_text(encoding="utf-8"))
    products = json.loads(paths["precipitation_product_audit"].read_text(encoding="utf-8"))
    temperature = json.loads(paths["temperature_final_test"].read_text(encoding="utf-8"))
    temperature_manifest = json.loads(paths["temperature_final_manifest"].read_text(encoding="utf-8"))
    failure = json.loads(paths["temperature_failure"].read_text(encoding="utf-8"))
    failure_manifest = json.loads(paths["temperature_failure_manifest"].read_text(encoding="utf-8"))

    if temperature["passes_frozen_final_test"]:
        raise RuntimeError("temperature-final-test-unexpectedly-passed")
    if failure["dominant_failure_component"] != "mean_level_bias":
        raise RuntimeError("temperature-failure-route-conflict")
    if not temperature_manifest["local_source_states_unchanged"]:
        raise RuntimeError("round86-source-integrity-failed")
    if not failure_manifest["source_states_unchanged"] or failure_manifest["locked_source_reaccessed"]:
        raise RuntimeError("round87-source-integrity-failed")

    summary = {
        "status": "complete",
        "arrays_read": False,
        "temperature": {
            "frozen_candidate_status": "rejected_by_one_time_1921_1939_final_test",
            "dominant_failure_component": "mean_level_bias",
            "locked_interval_status": "consumed_and_not_reusable_as_blind_holdout",
            "post_hoc_tuning_against_locked_result_allowed": False,
            "confirmatory_route": (
                "require a materially new external temperature reconstruction or observation family "
                "and pre-register a new untouched validation target before constructing another candidate"
            ),
            "exploratory_route": (
                "requires explicit scientific approval to build a clearly labeled exploratory reconstruction; "
                "it cannot claim independent validation from 1921-1939"
            ),
        },
        "precipitation": {
            "status": precipitation.get("decision", "external_evidence_boundary"),
            "automatic_equivalent_product_available": False,
            "confirmatory_route": "supply APHRO_MA V1101R1 0.5-degree data or authorized access",
            "alternative_route": (
                "requires explicit scientific approval of a non-equivalent evidence family and a new frozen contract"
            ),
            "product_audit_decision": products.get("decision", "no_automatic_equivalent_product"),
        },
        "shared": {
            "oggm_forcing_authorized": False,
            "production_release_authorized": False,
            "automatic_parameter_or_member_tuning_allowed": False,
            "next_progress_requires_external_input_or_explicit_scientific_scope_change": True,
        },
        "decision": "hold_at_external_evidence_and_scientific_decision_boundary",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Post-Final-Test Scientific Router",
                "",
                "## Temperature",
                "",
                "The frozen five-family candidate is rejected. The 1921-1939 interval is consumed and cannot validate a replacement candidate. Confirmatory continuation requires materially new external temperature evidence and a newly preregistered untouched target. Building an exploratory reconstruction requires explicit approval and must be labeled non-confirmatory.",
                "",
                "## Precipitation",
                "",
                "Confirmatory continuation requires APHRO_MA V1101R1 0.5-degree data or authorized access. Using a non-equivalent product requires a separately approved scientific contract.",
                "",
                "## Boundary",
                "",
                "No automatic parameter tuning, production release, or OGGM forcing is authorized.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "sources_before": before,
                "sources_after": after,
                "source_states_unchanged": unchanged,
                "arrays_read": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("round88-source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
