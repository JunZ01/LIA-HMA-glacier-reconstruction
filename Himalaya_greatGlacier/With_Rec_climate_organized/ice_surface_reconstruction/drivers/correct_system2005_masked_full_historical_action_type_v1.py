"""Correct the full-historical next-action label after validator rejection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
VALIDATOR = HERE.parent / "validate_lia_execution_state_v1.py"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE161 = BASE / "lia_191_current_route_v10_161/summary.json"
ROUTE162 = BASE / "lia_191_current_route_v10_162"
OUTPUT = BASE / "lia_191_system2005_masked_full_historical_action_type_correction_v1"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def append(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"marker-exists:{marker}")
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    normalized = text.replace("\r\n", "\n").replace("\n", newline.decode())
    return content.rstrip() + newline + newline + normalized.encode("utf-8").rstrip() + newline


def main() -> None:
    for path in (OUTPUT, ROUTE162):
        if path.exists():
            raise FileExistsError(path)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    route161 = json.loads(ROUTE161.read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.161-system2005-masked-full-history-pair-queued":
        raise RuntimeError("unexpected-controller-state")
    if sha256(ROUTE161) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route161-hash-mismatch")
    failed = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, capture_output=True, text=True
    )
    report = json.loads(failed.stdout)
    expected_failure = ["historical authorization has a non-historical next action"]
    if failed.returncode == 0 or report.get("failures") != expected_failure:
        raise RuntimeError(f"unexpected-validator-result:{report}")

    corrected_type = "launch-system2005-masked-full-historical-colder-v1"
    correction = {
        "schema": "lia-191-system2005-masked-full-historical-action-type-correction-v1",
        "generated_at": now(),
        "status": "technical-control-label-corrected",
        "source_route": {"path": str(ROUTE161.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(ROUTE161)},
        "validator_failure_before": expected_failure[0],
        "old_next_action_type": state["next_action"]["type"],
        "new_next_action_type": corrected_type,
        "scientific_inputs_changed": False,
        "prepared_configs_changed": False,
        "driver_changed": False,
        "thresholds_changed": False,
        "model_launched": False,
        "accepted_count_effect": 0,
    }
    summary_bytes = encode(correction)
    result_hash = hashlib.sha256(summary_bytes).hexdigest().upper()
    route = dict(route161)
    route.update(
        {
            "schema": "lia-191-current-route-v10-162",
            "status": "system2005-masked-full-historical-pair-queued-colder-launch-authorized",
            "supersedes": "lia_191_current_route_v10_161",
            "result": str((OUTPUT / "summary.json").relative_to(ROOT)).replace("\\", "/"),
            "result_sha256": result_hash,
            "technical_correction": {
                "field": "LIA_CURRENT_STATE.next_action.type",
                "old": state["next_action"]["type"],
                "new": corrected_type,
            },
        }
    )
    route_bytes = encode(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(
        facts_before,
        "## 466. System-2005 full-historical action label corrected",
        """## 466. System-2005 full-historical action label corrected (2026-08-29)

- Route v10.161's scientific and resource package passed, but the controller validator rejected its next-action type because the label used `full-history` rather than the required `historical` token.
- Route v10.162 changes only that control label to `launch-system2005-masked-full-historical-colder-v1`. The two prepared configs, endpoints, driver, forcing, initial state, physics, thresholds, sequential policy, counts, and all evidence hashes are unchanged.
- No model launched under the failed route. Full-historical launch authority becomes effective only after v10.162 passes the unchanged validator.""",
    )
    decisions_after = append(
        decisions_before,
        "## 2026-08-29 - Route v10.162: full-historical action label corrected",
        """## 2026-08-29 - Route v10.162: full-historical action label corrected

- Corrected only the validator-facing next-action token from `full-history` to `full-historical`; no scientific or execution input changed and no model launched under v10.161.
- Retain the sequential two-branch queue and authorize colder first only after validator pass.""",
    )
    next_state = dict(state)
    next_state["state_id"] = "2026-08-29-v10.162-system2005-masked-full-historical-pair-queued"
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    next_state["authoritative_route"] = {
        "version": "v10.162",
        "path": str((ROUTE162 / "summary.json").relative_to(ROOT)).replace("\\", "/"),
        "sha256": route_hash,
        "supersedes": "v10.161",
    }
    next_state["current_focus"] = dict(state["current_focus"])
    next_state["current_focus"]["evidence"] = {"path": route["result"], "sha256": result_hash}
    next_state["research_branches"]["system2005_pytorch_planform2d"]["evidence"] = {
        "path": route["result"],
        "sha256": result_hash,
    }
    next_state["next_action"] = {
        "type": corrected_type,
        "launch_authorized": True,
        "description": route["next_legal_action"],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system2005_historical_label_fix_", dir=ROOT))
    try:
        (stage / "evidence").mkdir()
        (stage / "route").mkdir()
        (stage / "evidence/summary.json").write_bytes(summary_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(encode(next_state))
        OUTPUT.mkdir(parents=False)
        ROUTE162.mkdir(parents=False)
        os.replace(stage / "evidence/summary.json", OUTPUT / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE162 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(OUTPUT, ignore_errors=True)
        shutil.rmtree(ROUTE162, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    passed = subprocess.run(
        [os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    print(json.dumps({"status": correction["status"], "route": "v10.162", "validator": json.loads(passed.stdout)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
