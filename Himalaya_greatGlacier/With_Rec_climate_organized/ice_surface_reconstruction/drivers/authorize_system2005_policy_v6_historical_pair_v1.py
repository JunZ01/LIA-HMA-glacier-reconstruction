"""Record the byte-identical adapter pass and authorize two historical endpoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUT = RESULTS / "lia_191_system2005_policy_v6_historical_pair_authorization_v1"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_49"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
ADAPTER = RESULTS / "lia_191_system2005_policy_v6_temperature_correction_v4_steady_adapter_v1/summary.json"
CONFIGS = (
    ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_policy_v6_historical_endp0p28125_v1.json",
    ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/system2005_policy_v6_historical_endp0p78125_v1.json",
)
RUNNER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/run_system2005_policy_v6_historical_v1.py"
EXPECTED_FACTS_SHA = "E84AF46CB61F00640EF5AC3BAEBFE525476DF1921F627B102A2C1ACC5CCE8185"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    temp = path.with_name(path.name + ".histpair.tmp")
    if temp.exists():
        raise RuntimeError(f"stale temp:{temp}")
    with temp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    ROUTE_DIR.mkdir(parents=True, exist_ok=False)
    if sha256(FACTS) != EXPECTED_FACTS_SHA:
        raise RuntimeError(f"facts changed:{sha256(FACTS)}")
    old_state = json.loads(STATE.read_text(encoding="utf-8"))
    if old_state.get("state_id") != "2026-08-26-v10.48-steady-pass-adapter-authorized":
        raise RuntimeError("controller state changed")
    adapter = json.loads(ADAPTER.read_text(encoding="utf-8"))
    if adapter.get("status") != "byte-identical-adapter-pass" or not adapter.get("byte_identical"):
        raise RuntimeError("steady adapter not pass")
    for path in (*CONFIGS, RUNNER):
        if not path.exists():
            raise FileNotFoundError(path)
    for source, name in (
        (FACTS, "LIA_PROJECT_FACTS.pre.md"),
        (STATE, "LIA_CURRENT_STATE.pre.json"),
        (DECISIONS, "LIA_DECISION_LOG.pre.md"),
    ):
        shutil.copy2(source, OUT / name)

    section = """## 360. System-2005 Policy-V6 first historical endpoint pair (2026-08-26)

- The v4 equilibrium state was materialized in the checkpointed historical
  input schema with byte-identical source and target SHA256
  `DF85D2BB28D6C08049A630164EA56D482A2F324CF54920C1B9349051DB4AC439`.
  No model step, state change, or steady reproduction occurred.
- Two and only two first-bracket historical endpoints are released from the
  passed `-0.21875 K` state: linear ramps to `+0.28125 K` and `+0.78125 K` in
  2000. This preserves the registered +0.5/+1.0 K endpoint offsets while
  changing neither geometry nor Stage-2 physics.
- Both use the verified graph-reuse solver, 12-hour nominal timestep, atomic
  annual checkpoints, exact state restoration on failed steps, and the frozen
  KKT/convergence fallback down to the unchanged 30-second floor. No tolerance
  or 400-iteration limit is relaxed. The endpoints may run concurrently because
  they use distinct output roots and immutable shared inputs.
- Policy V6 retains 10% total and 20% separation-normalized LIA/RGI length
  gates. Root/tributary disconnections remain diagnostics, with detached
  volume/area review and severe levels at 20%/50%. Neither endpoint is accepted
  without full numerical/domain completion and later independent reproduction.
  No third endpoint or bisection is authorized by this transition.
- Evidence:
  `validation_data/results/paleo_surface_validation_v0/lia_191_system2005_policy_v6_temperature_correction_v4_steady_adapter_v1/summary.json`.
""".rstrip()
    facts_text = FACTS.read_text(encoding="utf-8")
    if "## 360." in facts_text:
        raise RuntimeError("facts section 360 already exists")
    atomic_write(FACTS, (facts_text.rstrip() + "\n\n" + section + "\n").encode("utf-8"))

    route = {
        "schema": "lia-191-current-route-v10-49",
        "status": "policy-v6-first-historical-endpoint-pair-authorized",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "active": 0,
        "queue": 2,
        "launch_authorized": True,
        "full_historical_authorized": True,
        "supersedes": "lia_191_current_route_v10_48",
        "system_id": 2005,
        "steady_state_sha256": adapter["target_state_sha256"],
        "endpoint_delta_t_c": [0.28125, 0.78125],
        "parallel_distinct_outputs_authorized": True,
        "third_endpoint_authorized": False,
        "independent_reproduction_required_for_acceptance": True,
        "configs": [str(path.relative_to(ROOT)).replace("\\", "/") for path in CONFIGS],
    }
    route_path = ROUTE_DIR / "summary.json"
    atomic_write(route_path, json_bytes(route))
    new_state = {
        "schema": "lia-191-controller-state-v1",
        "state_id": "2026-08-26-v10.49-policy-v6-historical-pair-authorized",
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "scientific_contract": {"path": "LIA_PROJECT_FACTS.md", "sha256": sha256(FACTS)},
        "execution_contract": {"path": "LIA_EXECUTION_CONTRACT.md", "version": "1.0"},
        "authoritative_route": {
            "version": "v10.49",
            "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(route_path),
            "supersedes": "v10.48",
        },
        "population": 191,
        "counts": {"accepted": 54, "remaining": 137, "active": 0, "queued": 2},
        "current_focus": {
            "system_id": 2005,
            "mainline_family": "policy-v6-checkpointed-historical-first-bracket",
            "state": "two-distinct-historical-endpoints-authorized",
            "steady_delta_t_c": -0.21875,
            "steady_state_sha256": adapter["target_state_sha256"].upper(),
            "endpoint_delta_t_c": [0.28125, 0.78125],
            "minimum_length_for_20pct_separation_gate_m": 25450.125918037134,
            "full_historical_authorized": True,
            "evidence": {
                "path": str(ADAPTER.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(ADAPTER),
            },
        },
        "research_branches": old_state.get("research_branches", {}),
        "next_action": {
            "type": "system2005-policy-v6-historical-first-endpoint-pair",
            "launch_authorized": True,
            "description": "Launch the two distinct checkpointed 1470-2000 endpoint chains concurrently; no third endpoint and no reproduction yet.",
        },
    }
    atomic_write(STATE, json_bytes(new_state))
    decisions = DECISIONS.read_text(encoding="utf-8")
    note = """## 2026-08-26 — System 2005 Policy-V6 first historical pair

- Byte-identical v4 steady adapter passes.
- Authorize concurrent +0.28125/+0.78125 K endpoints with annual checkpoints.
- Keep 20% separation gate; no third endpoint or reproduction yet.
""".strip()
    atomic_write(DECISIONS, (decisions.rstrip() + "\n\n" + note + "\n").encode("utf-8"))
    summary = {
        "schema": "lia-191-system2005-policy-v6-historical-pair-authorization-v1",
        "status": "two-endpoints-authorized",
        "facts_sha256": sha256(FACTS),
        "route_sha256": sha256(route_path),
        "adapter_sha256": sha256(ADAPTER),
        "config_sha256": {path.name: sha256(path) for path in CONFIGS},
        "runner_sha256": sha256(RUNNER),
        "endpoint_delta_t_c": [0.28125, 0.78125],
        "third_endpoint_authorized": False,
    }
    atomic_write(OUT / "summary.json", json_bytes(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

