"""Register the terminal v10.246 forward curve screen without changing accepted."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
SCREEN = BASE / "lia_191_all191_oggm_flowline_1d_forward_curve_screen_v10_246_v1"
TX = BASE / "lia_191_all191_oggm_flowline_1d_forward_curve_screen_v10_246_transaction_v1/transaction.json"
EVIDENCE_DIR = BASE / "lia_191_all191_oggm_flowline_1d_forward_curve_screen_v10_247_terminal_v1"
ROUTE_DIR = BASE / "lia_191_current_route_v10_247"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def append_once(path: Path, marker: str, text: str) -> None:
    current = path.read_text(encoding="utf-8")
    if marker in current:
        raise RuntimeError(f"marker-already-present:{marker}")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        if not current.endswith("\n"):
            handle.write("\n")
        handle.write(text)


def main() -> None:
    if EVIDENCE_DIR.exists() or ROUTE_DIR.exists():
        raise FileExistsError("v10.247-output-already-exists")
    state_path = ROOT / "LIA_CURRENT_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["authoritative_route"]["version"] != "v10.246":
        raise RuntimeError("unexpected-authoritative-route")
    result = json.loads((SCREEN / "summary.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(SCREEN / "cell_scores.csv")
    failures = pd.read_csv(SCREEN / "technical_failures.csv")
    if not (len(scores) == 528 and len(failures) == 2 and result["provisional_pass_cells"] == 0):
        raise RuntimeError("screen-cardinality-failure")
    shape_without_endpoint = (
        scores["lag_gate_pass"].astype(bool)
        & scores["bidirectional_multistage_gate_pass"].astype(bool)
        & scores["artifact_gate_pass"].astype(bool)
    )
    if int(shape_without_endpoint.sum()) != 378:
        raise RuntimeError("shape-without-endpoint-count-mismatch")
    endpoint = scores["original_joint_endpoint_bounds_pass"].astype(bool)
    if int(endpoint.sum()) != 0:
        raise RuntimeError("unexpected-endpoint-pass")
    closest = scores.assign(
        worst_scaled=scores[["computed_e_lia_total", "computed_e_rgi_total"]].div(0.10).join(
            scores[["computed_e_lia_separation", "computed_e_rgi_separation"]].div(0.20)
        ).max(axis=1)
    ).sort_values("worst_scaled").head(20)
    evidence = {
        "schema": "lia-all191-forward-curve-screen-terminal-v10.247",
        "status": "terminal-zero-provisional-pass-decision-required",
        "screened_cells": 528,
        "technical_failure_cells_excluded": 2,
        "provisional_pass_cells": 0,
        "curve_screen_fail_cells": 528,
        "systems_with_provisional_pass": 0,
        "shape_gates_pass_without_endpoint": 378,
        "endpoint_joint_gate_pass": 0,
        "individual_endpoint_component_pass_counts": {
            "lia_total_10pct": int((scores["computed_e_lia_total"] <= 0.10).sum()),
            "rgi_total_10pct": int((scores["computed_e_rgi_total"] <= 0.10).sum()),
            "lia_separation_20pct": int((scores["computed_e_lia_separation"] <= 0.20).sum()),
            "rgi_separation_20pct": int((scores["computed_e_rgi_separation"] <= 0.20).sum()),
        },
        "failure_reason_counts_nonexclusive": result["failure_reason_counts_nonexclusive"],
        "independent_checks": {
            "retained32_phase_metrics_exact": True,
            "retained32_sustained_share_max_abs_difference": 1.1102230246251565e-16,
            "accepted58_target_table_common_systems": 58,
            "accepted58_lia_target_max_abs_difference_m": 1.8189894035458565e-12,
            "accepted58_rgi_target_max_abs_difference_m": 1.8189894035458565e-12,
        },
        "closest_20_member_keys": closest["member_key"].astype(str).tolist(),
        "accepted_before": 58,
        "accepted_after": 58,
        "independent_reproduction_launched": False,
        "simulation_launched": False,
        "screen_summary": str((SCREEN / "summary.json").relative_to(ROOT)).replace("\\", "/"),
        "screen_summary_sha256": sha256(SCREEN / "summary.json"),
        "cell_scores_sha256": sha256(SCREEN / "cell_scores.csv"),
        "system_summary_sha256": sha256(SCREEN / "system_summary.csv"),
        "transaction_sha256": sha256(TX),
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=False)
    evidence_path = EVIDENCE_DIR / "summary.json"
    atomic_json(evidence_path, evidence)

    facts = f"""

## 547. The 528-cell frozen-v3 curve screen is terminal with zero provisional passes (2026-09-04)

- The one authorized read-only curve screen completed across all 528 successful forward cells from 187 systems. The two technical-failure cells were excluded; no simulation or independent reproduction was launched.
- Zero cells pass all frozen v3 gates. All 528 fail the joint endpoint gate. Individually, 53 meet the 10% LIA-peak total-length component, 91 meet the 10% RGI-2000 component, six meet the 20% LIA target-separation component, and 42 meet the 20% RGI target-separation component, but none meet all four simultaneously.
- Dynamic shape is not the dominant failure: 378 cells pass lag response, bidirectional persistent phases, and artifact exclusion when the endpoint gate is omitted. Nonexclusive failures are endpoint bounds 528, lag response 112, insufficient bidirectional phases 82, terminal endpoint drop 33, and numerical high frequency 10.
- Independent read-only verification reproduces the original retained-32 v3 phase counts and pass flags exactly, with maximum sustained-share difference `1.1102230246251565e-16`. The core-191 target table agrees with all 58 common accepted-registry targets within `1.8189894035458565e-12 m`. Unit, target-table, and phase-implementation mismatches are therefore ruled out.
- The result is anomalous but contract-valid. Accepted remains 58. No threshold relaxation, candidate hand-selection, retry, or reproduction is authorized without a new user decision. Terminal evidence is `validation_data/results/paleo_surface_validation_v0/lia_191_all191_oggm_flowline_1d_forward_curve_screen_v10_247_terminal_v1/summary.json`, SHA256 `{sha256(evidence_path)}`.
"""
    append_once(ROOT / "LIA_PROJECT_FACTS.md", "## 547. The 528-cell frozen-v3 curve screen is terminal", facts)
    decision = """

## 2026-09-04 v10.247 — close the frozen-v3 screen at zero provisional passes

- Consume the v10.246 screen authority after evaluating every successful forward cell.
- Record that all 528 fail the joint endpoint gate even though 378 pass all non-endpoint shape gates.
- Keep accepted=58 and enter decision-required. Do not silently relax the 20%-of-separation endpoint rule, select a glacier by hand, launch reproduction, or retry the two technical failures.
"""
    append_once(ROOT / "LIA_DECISION_LOG.md", "## 2026-09-04 v10.247", decision)
    route = {
        "schema": "lia-191-current-route-v10-247",
        "status": "forward-curve-screen-terminal-zero-pass-decision-required",
        "population": 191, "accepted": 58, "remaining": 133,
        "active": 0, "queue": 0, "screened_cells": 528,
        "provisional_pass_cells": 0, "shape_pass_without_endpoint": 378,
        "technical_failures": 2, "scientific_curve_screen_failures": 528,
        "launch_authorized": False, "full_historical_authorized": False,
        "simulation_authorized": False,
        "supersedes": "lia_191_current_route_v10_246",
        "current_focus": "all191-forward-v3-screen-zero-pass-decision-required",
        "accepted_ledger_effect": 0,
        "evidence": str(evidence_path.relative_to(ROOT)).replace("\\", "/"),
        "evidence_sha256": sha256(evidence_path),
        "next_legal_action": "user decision on the frozen joint endpoint gate; no automatic relaxation or reproduction",
    }
    ROUTE_DIR.mkdir(parents=True, exist_ok=False)
    route_path = ROUTE_DIR / "summary.json"
    atomic_json(route_path, route)
    state["state_id"] = "2026-09-04-v10.247-forward-v3-screen-zero-pass-decision-required"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["scientific_contract"]["sha256"] = sha256(ROOT / "LIA_PROJECT_FACTS.md")
    state["authoritative_route"] = {"version": "v10.247",
        "path": str(route_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(route_path), "supersedes": "v10.246"}
    state["counts"]["active"] = 0
    state["counts"]["queued"] = 0
    state["current_focus"] = {
        "system_id": "all191-forward-v3-screen-zero-pass-decision-required",
        "mainline_family": "all191-oggm-flowline-1d-historical-forward-v1",
        "state": "decision-required", "full_historical_authorized": False,
        "screened_cells": 528, "provisional_pass_cells": 0,
        "shape_pass_without_endpoint": 378, "technical_failures": 2,
        "active": 0, "queue": 0, "accepted_ledger_effect": 0,
        "evidence": {"path": str(evidence_path.relative_to(ROOT)).replace("\\", "/"),
                     "sha256": sha256(evidence_path)},
    }
    state["next_action"] = {
        "type": "decision-required-forward-endpoint-gate",
        "transaction_id": "lia-all191-forward-screen-v10.247-terminal",
        "launch_authorized": False,
        "description": "User decision is required; do not relax the frozen endpoint gate or launch reproduction automatically.",
    }
    atomic_json(state_path, state)
    print(json.dumps({"route": route, "evidence_sha256": sha256(evidence_path)}, indent=2))


if __name__ == "__main__":
    main()
