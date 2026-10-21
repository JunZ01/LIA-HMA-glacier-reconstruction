"""Read-only system-2005 adjudication under the uniform revised 2% policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_branch3_coarsen2x_steady50_conditional_cached_v3/summary.json"
)
POLICY = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_tributary_separation_policy_v2/contract.json"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_tributary_separation_policy_v2/system_2005_rescore.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_hash_before = sha256(SOURCE)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    old_checks = dict(source["conditional_tributary_separation"]["checks"])
    structure = source["conditional_tributary_separation"]
    checks = dict(old_checks)
    checks["detached_volume_fraction_within_limit"] = (
        float(structure["detached_head_volume_fraction_system"])
        <= float(policy["detached_tributary_volume_fraction_max"])
    )
    checks["detached_area_fraction_within_limit"] = (
        float(structure["detached_head_area_fraction_system"])
        <= float(policy["detached_tributary_area_fraction_max"])
    )
    state_path = SOURCE.parent / "lia_equilibrium_state_50y.pkl.gz"
    state_materialized = state_path.exists()
    scientific_pass = all(checks.values())
    result = {
        "schema": "lia-191-system2005-tributary-separation-policy-v2-rescore",
        "status": (
            "steady-conditional-scientific-pass-state-materialization-pending"
            if scientific_pass and not state_materialized
            else "steady-conditional-pass" if scientific_pass else "steady-conditional-fail"
        ),
        "system_id": 2005,
        "source_summary": str(SOURCE),
        "source_summary_sha256": source_hash_before,
        "policy": policy,
        "old_policy_checks": old_checks,
        "revised_policy_checks": checks,
        "scientific_pass": scientific_pass,
        "state_materialized": state_materialized,
        "expected_state_path": str(state_path),
        "historical_authorized": bool(scientific_pass and state_materialized),
        "detached_volume_fraction": structure["detached_head_volume_fraction_system"],
        "detached_area_fraction": structure["detached_head_area_fraction_system"],
        "exact_reproduction": source["exact_reproduction"],
        "source_immutable": source_hash_before == sha256(SOURCE),
        "model_rerun": False,
        "old_result_modified": False,
        "supersedes_driver_only": "rescore_system2005_tributary_separation_policy_v2.py syntax-only pre-run failure"
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "scientific_pass", "state_materialized", "historical_authorized", "detached_volume_fraction", "detached_area_fraction")}))


if __name__ == "__main__":
    main()
