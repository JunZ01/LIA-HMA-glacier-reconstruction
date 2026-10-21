"""Materialize a byte-identical v4 steady state in the historical input schema."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE_ROOT = RESULTS / "lia_191_system2005_policy_v6_temperature_correction_v4"
SOURCE_SUMMARY = SOURCE_ROOT / "formal50_summary.json"
SOURCE_STATE = SOURCE_ROOT / "formal50/lia_equilibrium_state_50y.pkl.gz"
OUTPUT = RESULTS / "lia_191_system2005_policy_v6_temperature_correction_v4_steady_adapter_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def controller_gate() -> dict:
    state = json.loads((ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8"))
    checks = {
        "state_id": state.get("state_id") == "2026-08-26-v10.48-steady-pass-adapter-authorized",
        "route": state.get("authoritative_route", {}).get("version") == "v10.48",
        "launch_authorized": state.get("next_action", {}).get("launch_authorized") is True,
        "action": state.get("next_action", {}).get("type")
        == "system2005-materialize-steady-historical-adapter-v1",
        "historical_not_authorized": state.get("current_focus", {}).get("full_historical_authorized")
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"controller-gate-fail:{checks}")
    return checks


def main() -> int:
    gate = controller_gate()
    source = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    if source.get("status") != "formal-50y-steady-pass" or not all(source["checks"].values()):
        raise RuntimeError("source formal steady not pass")
    expected = str(source["state_sha256"]).lower()
    if sha256(SOURCE_STATE) != expected:
        raise RuntimeError("source state hash mismatch")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    target_state = OUTPUT / "lia_equilibrium_state_50y.pkl.gz"
    shutil.copy2(SOURCE_STATE, target_state)
    if sha256(target_state) != expected:
        raise RuntimeError("adapter state is not byte-identical")
    checks = dict(source["checks"])
    adapted = {
        "schema": "lia-191-system2005-policy-v6-steady-adapter-v1",
        "status": "steady-50y-pass",
        "system_id": 2005,
        "delta_t_c": -0.21875,
        "checks": checks,
        "volume_change_fraction": source["volume_change_fraction"],
        "final_length_m": source["final_length_m"],
        "final_area_m2": source["final_area_m2"],
        "final_free_downstream_m": source["final_free_downstream_m"],
        "evidence": {
            "formal_state_sha256": expected,
            "source_summary": str(SOURCE_SUMMARY.relative_to(ROOT)).replace("\\", "/"),
            "source_summary_sha256": sha256(SOURCE_SUMMARY),
        },
        "state_modified": False,
        "simulation_run": False,
    }
    write_json(OUTPUT / "steady_50y_summary_formal.json", adapted)
    summary = {
        "schema": "lia-191-system2005-policy-v6-steady-adapter-materialization-v1",
        "status": "byte-identical-adapter-pass",
        "controller_gate": gate,
        "source_state_sha256": expected,
        "target_state_sha256": sha256(target_state),
        "byte_identical": SOURCE_STATE.read_bytes() == target_state.read_bytes(),
        "state_modified": False,
        "simulation_run": False,
        "historical_run": False,
    }
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["byte_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

