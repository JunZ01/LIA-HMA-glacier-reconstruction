"""Register the already-running System-2005 warmer historical preflight."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import psutil


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
VALIDATOR = HERE.parent / "validate_lia_execution_state_v1.py"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE157 = BASE / "lia_191_current_route_v10_157/summary.json"
ROUTE158 = BASE / "lia_191_current_route_v10_158"
LAUNCH = BASE / "lia_191_system2005_masked_historical_preflight_warmer_launch_v1/summary.json"
OUTPUT = ROOT / "w2005mask2dhistpre_e1p334281_v1"
EXPECTED_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction."
    "drivers.run_system2005_masked_historical_branch_v3"
)
EXPECTED_CONFIG = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/"
    "system2005_masked_historical_preflight_endp1p334281_v4_launch.json"
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def payload(value: dict) -> bytes:
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
    if ROUTE158.exists():
        raise FileExistsError(ROUTE158)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    launch = json.loads(LAUNCH.read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.157-system2005-colder-preflight-pass-warmer-queued":
        raise RuntimeError("unexpected-controller-state")
    if sha256(ROUTE157) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route157-hash-mismatch")
    if not state["next_action"]["launch_authorized"]:
        raise RuntimeError("route157-launch-not-authorized")
    if launch.get("status") != "active" or launch.get("module") != EXPECTED_MODULE:
        raise RuntimeError("unexpected-launch-evidence")
    if Path(launch["config"]).as_posix() != str(EXPECTED_CONFIG.relative_to(ROOT)).replace("\\", "/"):
        raise RuntimeError("unexpected-launch-config")
    if sha256(EXPECTED_CONFIG) != launch["config_sha256"].upper():
        raise RuntimeError("launch-config-hash-mismatch")
    if not OUTPUT.is_dir():
        raise RuntimeError("warmer-output-root-absent")

    process = psutil.Process(int(launch["pid"]))
    if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
        raise RuntimeError("registered-warmer-process-not-running")
    command = " ".join(process.cmdline())
    if EXPECTED_MODULE not in command or str(EXPECTED_CONFIG) not in command:
        raise RuntimeError("registered-pid-command-mismatch")

    validation_before = subprocess.run(
        [os.sys.executable, str(VALIDATOR)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    validated = json.loads(validation_before.stdout)
    if not validated["launch_authorized"] or validated["counts"] != {
        "accepted": 58,
        "remaining": 133,
        "active": 0,
        "queued": 1,
    }:
        raise RuntimeError("unexpected-pre-registration-controller-validation")

    launch_hash = sha256(LAUNCH)
    route = {
        "schema": "lia-191-current-route-v10-158",
        "status": "system2005-masked-historical-response-preflight-warmer-active",
        "population": 191,
        "accepted": 58,
        "remaining": 133,
        "active": 1,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_157",
        "result": str(LAUNCH.relative_to(ROOT)).replace("\\", "/"),
        "result_sha256": launch_hash,
        "pid": int(launch["pid"]),
        "active_output_root": str(OUTPUT.relative_to(ROOT)),
        "checkpoint_cadence": launch["checkpoint_cadence"],
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "production_temperature_modified": False,
        "next_legal_action": (
            "monitor only the registered warmer 1470-1480 branch and adjudicate its "
            "terminal output; do not launch a duplicate or any full historical process"
        ),
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(
        facts_before,
        "## 462. System-2005 warmer masked historical response preflight active",
        f"""## 462. System-2005 warmer masked historical response preflight active (2026-08-29)

- Controller validation passed for route v10.157. The sole registered warmer 1470-1480 response branch started exactly once as Windows PID `{launch['pid']}` at `{launch['started_at']}` in `w2005mask2dhistpre_e1p334281_v1`.
- The branch uses driver v3, the immutable 120-month visible preflight window, the reproduced initial state, a linear research overlay from `+0.3342811518826305 K` in 1470 toward `+1.3342811518826305 K` in 2000, and annual atomic checkpoints. Its config SHA256 is `{launch['config_sha256']}`.
- Route v10.158 records active 1 and queued 0 and consumes launch authority. Monitoring and terminal adjudication are the only authorized actions; no duplicate, full historical continuation, production-temperature change, initial-thickness change, model-family promotion, or accepted-count change is authorized.""",
    )
    decisions_after = append(
        decisions_before,
        "## 2026-08-29 - Route v10.158: system 2005 warmer historical preflight active",
        f"""## 2026-08-29 - Route v10.158: system 2005 warmer historical preflight active

- Registered the already-running sole warmer branch as PID `{launch['pid']}` after exact process-command, config-hash, output-root, and route validation.
- Launch authority is consumed. Monitor annual atomic checkpoints and terminal evidence only; no duplicate or full historical launch is allowed during the active run.""",
    )

    next_state = dict(state)
    next_state["state_id"] = "2026-08-29-v10.158-system2005-historical-response-preflight-warmer-active"
    next_state["updated_at"] = now()
    next_state["scientific_contract"] = {
        "path": "LIA_PROJECT_FACTS.md",
        "sha256": hashlib.sha256(facts_after).hexdigest().upper(),
    }
    next_state["authoritative_route"] = {
        "version": "v10.158",
        "path": str((ROUTE158 / "summary.json").relative_to(ROOT)).replace("\\", "/"),
        "sha256": route_hash,
        "supersedes": "v10.157",
    }
    next_state["counts"] = {"accepted": 58, "remaining": 133, "active": 1, "queued": 0}
    next_state["current_focus"] = {
        "system_id": "2005",
        "mainline_family": "source-mask-planform2d-research-steady",
        "state": "historical-response-preflight-warmer-active",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": route["result"], "sha256": launch_hash},
    }
    next_state["research_branches"]["system2005_pytorch_planform2d"].update(
        {
            "status": "masked-historical-response-preflight-warmer-active",
            "evidence": {"path": route["result"], "sha256": launch_hash},
            "next_legal_action": route["next_legal_action"],
        }
    )
    next_state["next_action"] = {
        "type": "monitor-system2005-masked-historical-response-preflight-warmer-v1",
        "launch_authorized": False,
        "description": route["next_legal_action"],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system2005_warmer_active_", dir=ROOT))
    try:
        (stage / "route").mkdir()
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after)
        (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(next_state))
        ROUTE158.mkdir(parents=False)
        os.replace(stage / "route/summary.json", ROUTE158 / "summary.json")
        os.replace(stage / "facts", FACTS)
        os.replace(stage / "decisions", DECISIONS)
        os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before)
        DECISIONS.write_bytes(decisions_before)
        STATE.write_bytes(state_before)
        shutil.rmtree(ROUTE158, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    validation_after = subprocess.run(
        [os.sys.executable, str(VALIDATOR)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    print(
        json.dumps(
            {
                "status": "active",
                "pid": int(launch["pid"]),
                "route": "v10.158",
                "validator": json.loads(validation_after.stdout),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
