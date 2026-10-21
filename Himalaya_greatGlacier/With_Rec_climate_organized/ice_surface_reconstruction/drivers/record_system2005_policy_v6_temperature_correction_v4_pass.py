"""Record the v4 steady pass and authorize only byte-identical state adaptation."""

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
OUT = RESULTS / "lia_191_system2005_policy_v6_temperature_correction_v4_pass_record_v1"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_48"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
RESULT = RESULTS / "lia_191_system2005_policy_v6_temperature_correction_v4/temperature_correction_summary.json"
FORMAL = RESULTS / "lia_191_system2005_policy_v6_temperature_correction_v4/formal50_summary.json"
SAVED_STATE = RESULTS / "lia_191_system2005_policy_v6_temperature_correction_v4/formal50/lia_equilibrium_state_50y.pkl.gz"
ADAPTER = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/materialize_system2005_policy_v6_steady_adapter_v1.py"
EXPECTED_FACTS_SHA = "BCF6BDAD4136B6723D899C5F24E918857F039927A05B8145CFDEAD1DBD51B5A9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    temp = path.with_name(path.name + ".v4pass.tmp")
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
    if old_state.get("state_id") != "2026-08-26-v10.47-temperature-correction-v4-authorized":
        raise RuntimeError("controller state changed")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    if result.get("status") != "formal-50y-pass-historical-pending":
        raise RuntimeError("v4 wrapper result is not pass")
    if formal.get("status") != "formal-50y-steady-pass" or not all(formal["checks"].values()):
        raise RuntimeError("v4 formal gates are not all pass")
    if sha256(SAVED_STATE) != formal["state_sha256"]:
        raise RuntimeError("saved state hash mismatch")
    if not ADAPTER.exists():
        raise FileNotFoundError(ADAPTER)
    for source, name in (
        (FACTS, "LIA_PROJECT_FACTS.pre.md"),
        (STATE, "LIA_CURRENT_STATE.pre.json"),
        (DECISIONS, "LIA_DECISION_LOG.pre.md"),
    ):
        shutil.copy2(source, OUT / name)

    section = """## 359. System-2005 Policy-V6 temperature-only steady pass (2026-08-26)

- Candidate v4 reused the frozen v2 initial state without modification and
  changed only the registered temperature offset to `-0.21875 K`. Its 10-year
  preflight passed with volume change `-0.15512%`, connected-root length
  `25.550 km`, free domain `5.900 km`, zero internal gaps, and clean numerical
  diagnostics.
- The complete 50-year run passed every frozen steady gate. Volume change is
  `+1.75407%` (limit 2%), connected-root length is `25.550 km` (Policy-V6
  minimum `25.450125918 km`), free domain is `5.900 km`, terminal thickness is
  zero, and finite/nonnegative checks pass. One final internal-gap branch is a
  diagnostic under the established disconnection policy, not a hard failure.
- The saved equilibrium state SHA256 is
  `DF85D2BB28D6C08049A630164EA56D482A2F324CF54920C1B9349051DB4AC439`.
  The next action is only a byte-identical state/summary adapter for the
  checkpointed historical runner. It is not a second steady simulation.
  Historical execution remains unauthorized until that adapter passes.
- Evidence:
  `validation_data/results/paleo_surface_validation_v0/lia_191_system2005_policy_v6_temperature_correction_v4/temperature_correction_summary.json`.
""".rstrip()
    facts_text = FACTS.read_text(encoding="utf-8")
    if "## 359." in facts_text:
        raise RuntimeError("facts section 359 already exists")
    atomic_write(FACTS, (facts_text.rstrip() + "\n\n" + section + "\n").encode("utf-8"))

    route = {
        "schema": "lia-191-current-route-v10-48",
        "status": "policy-v6-temperature-only-steady-pass-adapter-authorized",
        "population": 191,
        "accepted": 54,
        "remaining": 137,
        "active": 0,
        "queue": 0,
        "launch_authorized": True,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_47",
        "system_id": 2005,
        "steady": {
            "status": "formal-50y-steady-pass",
            "delta_t_c": -0.21875,
            "volume_change_fraction": formal["volume_change_fraction"],
            "final_length_m": formal["final_length_m"],
            "state_sha256": formal["state_sha256"],
        },
        "next_action": "materialize-byte-identical-historical-input-adapter",
        "historical_forbidden": True,
    }
    route_path = ROUTE_DIR / "summary.json"
    atomic_write(route_path, json_bytes(route))
    new_state = {
        "schema": "lia-191-controller-state-v1",
        "state_id": "2026-08-26-v10.48-steady-pass-adapter-authorized",
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "scientific_contract": {"path": "LIA_PROJECT_FACTS.md", "sha256": sha256(FACTS)},
        "execution_contract": {"path": "LIA_EXECUTION_CONTRACT.md", "version": "1.0"},
        "authoritative_route": {
            "version": "v10.48",
            "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(route_path),
            "supersedes": "v10.47",
        },
        "population": 191,
        "counts": {"accepted": 54, "remaining": 137, "active": 0, "queued": 0},
        "current_focus": {
            "system_id": 2005,
            "mainline_family": "policy-v6-temperature-only-joint-shooting",
            "state": "formal-50y-steady-pass-adapter-pending",
            "candidate_delta_t_c": -0.21875,
            "steady_state_sha256": formal["state_sha256"],
            "minimum_length_for_20pct_separation_gate_m": 25450.125918037134,
            "full_historical_authorized": False,
            "evidence": {
                "path": str(RESULT.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(RESULT),
            },
        },
        "research_branches": old_state.get("research_branches", {}),
        "next_action": {
            "type": "system2005-materialize-steady-historical-adapter-v1",
            "launch_authorized": True,
            "description": "Copy the passed v4 state byte-for-byte into the historical runner input schema; do not run dynamics.",
        },
    }
    atomic_write(STATE, json_bytes(new_state))
    decisions = DECISIONS.read_text(encoding="utf-8")
    note = """## 2026-08-26 — System 2005 Policy-V6 steady pass

- Temperature-only v4 passes all 50-year steady gates at -0.21875 K.
- No steady reproduction is required; authorize only byte-identical state adaptation.
- Historical execution remains gated on the adapter result.
""".strip()
    atomic_write(DECISIONS, (decisions.rstrip() + "\n\n" + note + "\n").encode("utf-8"))
    summary = {
        "schema": "lia-191-system2005-policy-v6-temperature-correction-v4-pass-record-v1",
        "status": "steady-pass-adapter-authorized",
        "facts_sha256": sha256(FACTS),
        "route_sha256": sha256(route_path),
        "result_sha256": sha256(RESULT),
        "state_sha256": sha256(SAVED_STATE),
        "historical_authorized": False,
    }
    atomic_write(OUT / "summary.json", json_bytes(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

