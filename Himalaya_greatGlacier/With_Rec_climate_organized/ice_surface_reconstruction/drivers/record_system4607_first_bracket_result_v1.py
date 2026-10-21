"""Record the terminal system-4607 first bracket and release midpoint preparation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import psutil


DRIVER = Path(__file__).resolve()
PROJECT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
EXECUTION = PROJECT / "LIA_EXECUTION_CONTRACT.md"
STATE = PROJECT / "LIA_CURRENT_STATE.json"
DECISIONS = PROJECT / "LIA_DECISION_LOG.md"
PREPARATION = RESULTS / "lia_191_system4607_first_historical_bracket_preparation_v1/summary.json"
LAUNCH = RESULTS / "lia_191_system4607_first_historical_bracket_launch_v1/summary.json"
ROOT_0375 = PROJECT / "w929hist4607rampendp0375v1"
ROOT_0875 = PROJECT / "w929hist4607rampendp0875v1"
SUMMARY_0375 = ROOT_0375 / "historical_summary_formal.json"
SUMMARY_0875 = ROOT_0875 / "historical_summary_formal.json"
CURVE_0375 = ROOT_0375 / "historical_1470_current.csv"
CURVE_0875 = ROOT_0875 / "historical_1470_current.csv"
EVIDENCE_DIR = RESULTS / "lia_191_system4607_first_historical_bracket_result_v1"
ROUTE_DIR = RESULTS / "lia_191_current_route_v10_121"
EXPECTED_STATE_ID = "2026-08-28-v10.120-system4607-endp0375-fail-endp0875-active"
MIDPOINT_C = 0.625


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def file_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def endpoint_record(endpoint: float, root: Path, summary_path: Path, curve: Path) -> dict[str, object]:
    summary = read_json(summary_path)
    if summary["system_id"] != 4607 or float(summary["forward_delta_t_end_c"]) != endpoint:
        raise RuntimeError(f"endpoint-identity-failed:{endpoint}")
    if int(summary["final"]["year"]) != 2000:
        raise RuntimeError(f"endpoint-incomplete:{endpoint}")
    if summary["runtime_error"] is not None or summary["numerical_flags"]:
        raise RuntimeError(f"endpoint-not-numerically-clean:{endpoint}")
    if int(summary["rejected_steps"]) != 0:
        raise RuntimeError(f"endpoint-rejected-steps:{endpoint}")
    return {
        "endpoint_c": endpoint,
        "root": str(root.relative_to(PROJECT)),
        "summary": str(summary_path.relative_to(PROJECT)),
        "summary_sha256": sha256(summary_path),
        "curve": str(curve.relative_to(PROJECT)),
        "curve_sha256": sha256(curve),
        "status": summary["status"],
        "peak_year": int(summary["peak"]["year"]),
        "peak_length_m": float(summary["peak"]["length_m"]),
        "final_length_m": float(summary["final"]["length_m"]),
        "minimum_free_downstream_m": min(
            float(summary["initial"]["remaining_free_downstream_m"]),
            float(summary["peak"]["remaining_free_downstream_m"]),
            float(summary["final"]["remaining_free_downstream_m"]),
        ),
        "e_lia_total": float(summary["e_lia_total"]),
        "e_lia_separation": float(summary["e_lia_separation"]),
        "e_rgi_total": float(summary["e_rgi_total"]),
        "e_rgi_separation": float(summary["e_rgi_separation"]),
        "review_flags": list(summary["review_flags"]),
        "numerical_flags": list(summary["numerical_flags"]),
        "accepted_steps": int(summary["accepted_steps"]),
        "rejected_steps": int(summary["rejected_steps"]),
        "runtime_seconds": float(summary["runtime_seconds"]),
        "config_fingerprint": str(summary["config_fingerprint"]),
    }


def main() -> int:
    if EVIDENCE_DIR.exists() or ROUTE_DIR.exists():
        raise FileExistsError("v10.121-output-collision")
    required = [
        FACTS, EXECUTION, STATE, DECISIONS, PREPARATION, LAUNCH,
        SUMMARY_0375, SUMMARY_0875, CURVE_0375, CURVE_0875, DRIVER,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    controller = read_json(STATE)
    if controller["state_id"] != EXPECTED_STATE_ID:
        raise RuntimeError(f"unexpected-controller-state:{controller['state_id']}")
    if controller["counts"] != {"accepted": 57, "remaining": 134, "active": 1, "queued": 0}:
        raise RuntimeError("unexpected-controller-counts")
    launch = read_json(LAUNCH)
    pids = [int(item["pid"]) for item in launch["processes"]]
    if any(psutil.pid_exists(pid) for pid in pids):
        raise RuntimeError("registered-process-still-active")

    source_paths = {
        "facts": FACTS,
        "execution": EXECUTION,
        "state": STATE,
        "decisions": DECISIONS,
        "preparation": PREPARATION,
        "launch": LAUNCH,
        "summary_0375": SUMMARY_0375,
        "summary_0875": SUMMARY_0875,
        "curve_0375": CURVE_0375,
        "curve_0875": CURVE_0875,
        "driver": DRIVER,
    }
    before = {name: file_state(path) for name, path in source_paths.items()}
    endpoints = [
        endpoint_record(0.375, ROOT_0375, SUMMARY_0375, CURVE_0375),
        endpoint_record(0.875, ROOT_0875, SUMMARY_0875, CURVE_0875),
    ]
    if any(item["status"] != "historical-fail" for item in endpoints):
        raise RuntimeError("expected-two-scientific-failures")
    if any(item["review_flags"] != ["rgi-2000-length-gate"] for item in endpoints):
        raise RuntimeError("unexpected-review-flags")
    target = float(read_json(PREPARATION)["registered_targets_m"]["rgi"])
    if not (endpoints[0]["final_length_m"] > target > endpoints[1]["final_length_m"]):
        raise RuntimeError("modern-target-not-bracketed")
    if any(
        item["e_lia_total"] > 0.1 or item["e_lia_separation"] > 0.2
        for item in endpoints
    ):
        raise RuntimeError("lia-peak-not-common-pass")

    EVIDENCE_DIR.mkdir(parents=True)
    ROUTE_DIR.mkdir(parents=True)
    evidence = {
        "schema": "lia-191-system4607-first-historical-bracket-result-v1",
        "status": "two-numerically-clean-scientific-failures-modern-target-bracketed",
        "system_id": 4607,
        "registered_rgi_target_m": target,
        "endpoints": endpoints,
        "common_lia_peak_pass": True,
        "modern_target_strictly_bracketed": True,
        "fixed_bisection_midpoint_c": MIDPOINT_C,
        "midpoint_selection_rule": "arithmetic midpoint of the registered +0.375/+0.875 K endpoints; no response interpolation used for acceptance",
        "related_model_process_count": 0,
        "accepted_count_effect": 0,
        "initial_thickness_modified": False,
        "physics_changed": False,
        "forcing_family_changed": False,
        "scoring_changed": False,
        "prohibited_source_accessed": False,
        "next_legal_action": "prepare exactly one immutable +0.625 K midpoint chain from the unchanged -0.125 K equilibrium; do not launch until a new route validates",
    }
    atomic_json(EVIDENCE_DIR / "summary.json", evidence)
    atomic_json(
        EVIDENCE_DIR / "source_manifest.json",
        {"before": before, "computational_sources_expected_unchanged": True},
    )
    evidence_hash = sha256(EVIDENCE_DIR / "summary.json")
    route = {
        "schema": "lia-191-current-route-v10-121",
        "status": "system4607-first-historical-bracket-terminal-midpoint-preparation-released",
        "population": 191,
        "accepted": 57,
        "remaining": 134,
        "active": 0,
        "queue": 0,
        "launch_authorized": False,
        "full_historical_authorized": False,
        "supersedes": "lia_191_current_route_v10_120",
        "system_id": 4607,
        "completed_endpoints_c": [0.375, 0.875],
        "released_midpoint_c": MIDPOINT_C,
        "result_evidence": str((EVIDENCE_DIR / "summary.json").relative_to(PROJECT)).replace("\\", "/"),
        "result_evidence_sha256": evidence_hash,
        "duplicate_launch_authorized": False,
        "third_nonmidpoint_endpoint_authorized": False,
        "independent_reproduction_authorized": False,
        "accepted_count_effect": 0,
        "initial_thickness_modification_allowed": False,
        "prohibited_source_accessed": False,
        "next_legal_action": "prepare the single fixed +0.625 K midpoint chain in a new immutable root, then create a separate queued route before launch",
    }
    atomic_json(ROUTE_DIR / "summary.json", route)
    route_hash = sha256(ROUTE_DIR / "summary.json")

    for name, path in source_paths.items():
        if file_state(path) != before[name]:
            raise RuntimeError(f"source-changed-before-canonical-commit:{name}")
    facts_text = FACTS.read_text(encoding="utf-8")
    decision_text = DECISIONS.read_text(encoding="utf-8")
    facts_heading = "## 424. System-4607 first bracket brackets the modern target (2026-08-28)"
    decision_heading = "## 2026-08-28 - Route v10.121: system 4607 first bracket terminal"
    if facts_heading in facts_text or decision_heading in decision_text:
        raise RuntimeError("canonical-entry-already-exists")
    facts_append = (
        "\n\n" + facts_heading + "\n\n"
        "- Both registered historical endpoints completed through 2000 with zero rejected steps, finite nonnegative geometry, no runtime error, and at least 5.400 km free downstream domain.\n"
        "- Both share a 16.150 km LIA peak in 1492 and pass the frozen LIA total and separation gates. The +0.375 K endpoint ends too long at 10.950 km; the +0.875 K endpoint ends too short at 7.650 km. Each fails only the modern length gates.\n"
        "- The 9.514778 km modern target is therefore strictly bracketed under the unchanged forcing family. The only released next endpoint is the fixed arithmetic midpoint +0.625 K from the same -0.125 K equilibrium. This is preparation-only; no launch or acceptance is authorized by route v10.121.\n"
        "- Counts remain 57 accepted and 134 remaining, active 0 and queued 0. Evidence and route:\n"
        "  validation_data/results/paleo_surface_validation_v0/lia_191_system4607_first_historical_bracket_result_v1/summary.json\n"
        "  and validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_121/summary.json.\n"
    )
    decision_append = (
        "\n\n" + decision_heading + "\n\n"
        "- Classified both +0.375/+0.875 K chains as numerically clean scientific failures: their common LIA peak passes, while their 2000 lengths fall on opposite sides of the modern target.\n"
        "- Released preparation only for the fixed +0.625 K arithmetic midpoint from the unchanged equilibrium. Counts remain 57 accepted and 134 remaining, active 0 and queued 0; no launch or reproduction is authorized.\n"
    )
    new_facts = (facts_text + facts_append).encode("utf-8")
    new_decisions = (decision_text + decision_append).encode("utf-8")
    atomic_bytes(FACTS, new_facts)
    atomic_bytes(DECISIONS, new_decisions)

    controller["state_id"] = "2026-08-28-v10.121-system4607-first-bracket-terminal-midpoint-preparation-released"
    controller["updated_at"] = datetime.now().astimezone().isoformat()
    controller["scientific_contract"]["sha256"] = sha256(FACTS)
    controller["authoritative_route"] = {
        "version": "v10.121",
        "path": str((ROUTE_DIR / "summary.json").relative_to(PROJECT)).replace("\\", "/"),
        "sha256": route_hash,
        "supersedes": "v10.120",
    }
    controller["counts"] = {
        "accepted": 57,
        "remaining": 134,
        "active": 0,
        "queued": 0,
    }
    controller["current_focus"] = {
        "system_id": "4607",
        "mainline_family": "checkpointed-historical-forward-ramp-screening",
        "state": "first-bracket-terminal-modern-target-bracketed",
        "initial_thickness_modification_allowed": False,
        "full_historical_authorized": False,
        "evidence": {
            "path": str((EVIDENCE_DIR / "summary.json").relative_to(PROJECT)).replace("\\", "/"),
            "sha256": evidence_hash,
        },
    }
    controller["next_action"] = {
        "type": "prepare-system4607-fixed-midpoint-bisection-v1",
        "launch_authorized": False,
        "description": "Prepare exactly one +0.625 K midpoint chain from the unchanged -0.125 K equilibrium in a new immutable root; create and validate a separate queued route before launch.",
    }
    atomic_json(STATE, controller)
    after_computational = {
        name: file_state(path)
        for name, path in source_paths.items()
        if name not in {"facts", "state", "decisions"}
    }
    if any(after_computational[name] != before[name] for name in after_computational):
        raise RuntimeError("computational-source-changed-after-commit")
    print(json.dumps({"status": "complete", "route": "v10.121", "evidence_sha256": evidence_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
