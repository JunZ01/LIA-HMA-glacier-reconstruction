"""Adjudicate the passed source-mask zero-MB fixture and clear its queue."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
STATE = ROOT / "LIA_CURRENT_STATE.json"
DECISIONS = ROOT / "LIA_DECISION_LOG.md"
VALIDATOR = DRIVER.parent / "validate_lia_execution_state_v1.py"
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
ROUTE137 = BASE / "lia_191_current_route_v10_137/summary.json"
RUN = BASE / "lia_191_system2005_masked_zero_mb_fixture_v1/summary.json"
OUT = BASE / "lia_191_system2005_masked_zero_mb_fixture_adjudication_v1"
ROUTE138 = BASE / "lia_191_current_route_v10_138"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def payload(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def append(content: bytes, marker: str, text: str) -> bytes:
    if marker.encode("utf-8") in content:
        raise RuntimeError(f"marker-exists:{marker}")
    nl = b"\r\n" if b"\r\n" in content else b"\n"
    value = text.replace("\r\n", "\n").replace("\n", nl.decode()).encode("utf-8")
    return content.rstrip() + nl + nl + value.rstrip() + nl


def main() -> None:
    for path in (OUT, ROUTE138):
        if path.exists():
            raise FileExistsError(path)
    state = load(STATE)
    run = load(RUN)
    if state["state_id"] != "2026-08-29-v10.137-system2005-masked-zero-mb-fixture-queued":
        raise RuntimeError("unexpected-state")
    if sha256(ROUTE137) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route137-hash-mismatch")
    if run["status"] != "masked-zero-mb-fixture-pass" or not all(run["checks"].values()):
        raise RuntimeError("masked-fixture-not-pass")
    adjudication = {
        "schema": "lia-191-system2005-masked-zero-mb-fixture-adjudication-v1",
        "status": "masked-zero-mb-fixture-formal-pass-runtime-restoration-required",
        "recorded_at": now(),
        "system_id": 2005,
        "run_summary": str(RUN.relative_to(ROOT)),
        "run_summary_sha256": sha256(RUN),
        "results": {
            "completed_years": run["completed_seconds"] / 31_536_000.0,
            "accepted_steps": run["accepted_steps"],
            "rejected_steps": run["rejected_steps"],
            "cumulative_volume_change_m3": run["cumulative_volume_change_m3"],
            "maximum_substep_closure_m3": run["maximum_substep_closure_m3"],
            "outside_domain_maximum_thickness_m": run["outside_domain_maximum_thickness_m"],
        },
        "runtime": {
            "executable": "D:/Anaconda/envs/SS4/python.exe",
            "torch_version": "2.7.1+cu118",
            "cuda_runtime": "11.8",
            "gpu": "NVIDIA GeForce RTX 5060 Ti",
            "gpu_capability": [12, 0],
            "compatibility_warning": True,
            "role": "algebraic conservation fixture only; not production/performance equivalence",
        },
        "scientific_decision": {
            "masked_flux_operator_numerically_eligible": True,
            "climate_preflight_currently_authorized": False,
            "required_before_climate": "restore an E-drive CUDA 12.8 PyTorch runtime without C-drive package/cache growth",
            "accepted_count_effect": 0,
        },
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
    }
    adjudication_bytes = payload(adjudication)
    adjudication_hash = hashlib.sha256(adjudication_bytes).hexdigest().upper()
    route = {
        "schema": "lia-191-current-route-v10-138",
        "status": "system2005-masked-zero-mb-fixture-pass-runtime-restore-next",
        "population": 191, "accepted": 58, "remaining": 133, "active": 0, "queue": 0,
        "launch_authorized": False, "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_137",
        "result": str((OUT / "summary.json").relative_to(ROOT)),
        "result_sha256": adjudication_hash,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "climate_run_authorized": False,
        "production_temperature_modified": False,
        "next_legal_action": "restore or create an E-drive PyTorch CUDA 12.8 runtime; no C-drive cache or package installation",
    }
    route_bytes = payload(route)
    route_hash = hashlib.sha256(route_bytes).hexdigest().upper()
    facts_before = FACTS.read_bytes()
    decisions_before = DECISIONS.read_bytes()
    state_before = STATE.read_bytes()
    facts_after = append(facts_before, "## 442. System-2005 masked zero-MB fixture pass", """## 442. System-2005 masked zero-MB fixture pass (2026-08-29)

- The source-native LIA-domain face-masked diffusion fixture completed 0.1 year with 192 accepted and 50 rejected adaptive steps. Cumulative volume change is exactly `0.0 m3`; maximum substep closure is `9.0962e-6 m3`; exterior-domain thickness remains exactly zero; all finite, nonnegative, source-hash, and adaptive-error gates pass.
- The fixture used available PyTorch `2.7.1+cu118` in `SS4`, which executes on the RTX 5060 Ti but reports an `sm_120` compatibility warning. It proves the masked algebraic conservation mechanism but is not accepted as production/performance equivalence to the prior PyTorch `2.11.0+cu128` runs.
- Route v10.138 clears the queue and authorizes no climate run. The next action is to restore or create an isolated CUDA 12.8 runtime on E drive, with no C-drive package or cache growth.""")
    decisions_after = append(decisions_before, "## 2026-08-29 - Route v10.138: system 2005 masked zero-MB fixture passed", """## 2026-08-29 - Route v10.138: system 2005 masked zero-MB fixture passed

- Accepted the zero-MB fixture as an algebraic conservation pass: exact cumulative conservation and zero cross-domain leakage.
- Did not authorize climate execution because the original CUDA 12.8 runtime is missing. Restore an isolated E-drive runtime before a one-year masked climate preflight.""")
    state138 = dict(state)
    state138["state_id"] = "2026-08-29-v10.138-system2005-masked-zero-mb-fixture-pass"
    state138["updated_at"] = now()
    state138["scientific_contract"] = {"path": "LIA_PROJECT_FACTS.md", "sha256": hashlib.sha256(facts_after).hexdigest().upper()}
    state138["authoritative_route"] = {
        "version": "v10.138", "path": str((ROUTE138 / "summary.json").relative_to(ROOT)),
        "sha256": route_hash, "supersedes": "v10.137",
    }
    state138["counts"] = {"accepted": 58, "remaining": 133, "active": 0, "queued": 0}
    state138["current_focus"] = {
        "system_id": "2005", "mainline_family": "source-mask-planform2d-research-fixture",
        "state": "hold-new-mechanism", "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": adjudication_hash},
    }
    state138["research_branches"]["system2005_pytorch_planform2d"].update({
        "status": "masked-zero-mb-conservation-pass-runtime-restoration-required",
        "evidence": {"path": str((OUT / "summary.json").relative_to(ROOT)), "sha256": adjudication_hash},
        "next_legal_action": route["next_legal_action"],
    })
    state138["next_action"] = {
        "type": "restore-system2005-cuda12p8-runtime-on-e-drive-v1",
        "launch_authorized": False,
        "description": route["next_legal_action"],
    }

    stage = Path(tempfile.mkdtemp(prefix=".system2005_fixture_adjudicate_", dir=ROOT))
    try:
        (stage / "evidence").mkdir(); (stage / "route").mkdir()
        (stage / "evidence/summary.json").write_bytes(adjudication_bytes)
        (stage / "route/summary.json").write_bytes(route_bytes)
        (stage / "facts").write_bytes(facts_after); (stage / "decisions").write_bytes(decisions_after)
        (stage / "state").write_bytes(payload(state138))
        OUT.mkdir(parents=False); ROUTE138.mkdir(parents=False)
        os.replace(stage / "evidence/summary.json", OUT / "summary.json")
        os.replace(stage / "route/summary.json", ROUTE138 / "summary.json")
        os.replace(stage / "facts", FACTS); os.replace(stage / "decisions", DECISIONS); os.replace(stage / "state", STATE)
    except BaseException:
        FACTS.write_bytes(facts_before); DECISIONS.write_bytes(decisions_before); STATE.write_bytes(state_before)
        shutil.rmtree(OUT, ignore_errors=True); shutil.rmtree(ROUTE138, ignore_errors=True); raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    validation = subprocess.run([os.sys.executable, str(VALIDATOR)], cwd=ROOT, check=True, capture_output=True, text=True)
    print(json.dumps({"status": "complete", "route": "v10.138", "validator": json.loads(validation.stdout)}, indent=2))


if __name__ == "__main__":
    main()
