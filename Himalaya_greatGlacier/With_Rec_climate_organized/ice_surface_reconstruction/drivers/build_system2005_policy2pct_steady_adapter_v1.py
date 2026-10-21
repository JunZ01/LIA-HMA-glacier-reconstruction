"""Register the accepted 2%-policy system-2005 state for historical drivers."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_branch3_coarsen2x_steady50_policy2pct_state_materialization_v1"
)
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_system2005_policy2pct_steady_adapter_v1"
)
POLICY = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_tributary_separation_policy_v2/contract.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summary_path = SOURCE / "summary.json"
    state_path = SOURCE / "lia_equilibrium_state_50y.pkl.gz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if summary.get("status") != "steady-50y-conditional-pass-tributary-separation":
        raise RuntimeError("source-steady-not-conditional-pass")
    if not summary.get("exact_reproduction", {}).get("byte_numeric_exact"):
        raise RuntimeError("source-steady-not-exact-reproduction")
    if not all(summary["conditional_tributary_separation"]["checks"].values()):
        raise RuntimeError("source-conditional-checks-not-all-pass")
    source_state_hash = sha256(state_path)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    target_state = OUTPUT / "lia_equilibrium_state_50y.pkl.gz"
    shutil.copy2(state_path, target_state)
    if sha256(target_state) != source_state_hash:
        raise RuntimeError("adapter-state-byte-copy-mismatch")
    checks = {
        "source_steady_conditional_pass": True,
        "source_exact_reproduction": True,
        "uniform_policy_volume_limit_is_2pct": float(policy["detached_tributary_volume_fraction_max"]) == 0.02,
        "uniform_policy_area_limit_is_0p5pct": float(policy["detached_tributary_area_fraction_max"]) == 0.005,
        "root_continuous": int(summary["conditional_tributary_separation"]["root_gap_branch_count"]) == 0,
        "source_state_exists": state_path.exists(),
        "state_byte_copy_exact": True,
    }
    formal = {
        "schema": "lia-191-system2005-policy2pct-steady-adapter-v1",
        "status": "steady-50y-pass",
        "decision": "accept-uniform-2pct-small-tributary-separation-policy",
        "system_id": 2005,
        "checks": checks,
        "steady_metrics": {
            "volume_change_fraction": summary["volume_change_fraction"],
            "root_length_m": summary["root"]["length_m"],
            "root_length_relative_error": summary["root"]["length_relative_error"],
            "free_downstream_m": summary["root"]["free_downstream_m"],
            "detached_volume_fraction": summary["conditional_tributary_separation"]["detached_head_volume_fraction_system"],
            "detached_area_fraction": summary["conditional_tributary_separation"]["detached_head_area_fraction_system"],
        },
        "evidence": {
            "source_summary": str(summary_path),
            "source_summary_sha256": sha256(summary_path),
            "policy_contract": str(POLICY),
            "policy_contract_sha256": sha256(POLICY),
            "formal_state": str(target_state),
            "formal_state_sha256": source_state_hash,
        },
        "model_state_modified": False,
        "historical_run": False,
    }
    (OUTPUT / "steady_50y_summary_formal.json").write_text(
        json.dumps(formal, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": formal["status"], "checks": checks, "state_sha256": source_state_hash}))


if __name__ == "__main__":
    main()
