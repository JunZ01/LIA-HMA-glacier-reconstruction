"""Close downstream loop gates when Round 03 yields no promotable candidate."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state(path: Path) -> dict[str, object]:
    item = path.stat()
    return {"path": str(path), "size_bytes": int(item.st_size), "mtime_ns": int(item.st_mtime_ns)}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_round(path: Path, contract: str, config: dict, scores: list[dict], summary: dict, final: str, sources: dict[str, Path]) -> None:
    if path.exists():
        raise FileExistsError(path)
    before = {name: state(source) for name, source in sources.items()}
    path.mkdir(parents=True)
    (path / "ROUND_CONTRACT.md").write_text(contract, encoding="utf-8")
    (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_csv(path / "scores.csv", scores)
    (path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (path / "FINAL_RESULT.md").write_text(final, encoding="utf-8")
    after = {name: state(source) for name, source in sources.items()}
    if before != after:
        raise RuntimeError(f"source-changed:{path.name}")
    manifest = {
        "before": before,
        "after": after,
        "sha256": {name: digest(source) for name, source in sources.items()},
        "source_hashes_unchanged": True,
    }
    (path / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    project = root()
    loop_root = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1"
    round03_path = loop_root / "round_03_nested_ensemble_selection/operator_oof_summary_v1.json"
    contract_path = project / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/climate_reconstruction_improvement_loop_v1.md"
    lock_path = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_instrumental_baseline_v1/LOCKED_FINAL_PERIOD.json"
    facts_path = project / "LIA_PROJECT_FACTS.md"
    for path in (round03_path, contract_path, lock_path, facts_path):
        if not path.exists():
            raise FileNotFoundError(path)
    round03 = json.loads(round03_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if round03.get("decision") != "reject_ao9_operator_repair":
        raise RuntimeError("round03-decision-does-not-authorize-closure")
    if lock.get("status") != "locked_not_evaluated":
        raise RuntimeError("locked-period-state-changed")
    gate = round03["selected_oof_gate"]
    created = datetime.now(timezone.utc).isoformat()
    sources = {"round03": round03_path, "loop_contract": contract_path, "lock_contract": lock_path, "facts": facts_path}

    round04 = loop_root / "round_04_frozen_full_reconstruction"
    write_round(
        round04,
        "# Round 04 Contract\n\nDecision: run a frozen full reconstruction only if Round 03 promotes a candidate. The predeclared stop condition is closure without a production run when Gate 3 fails.\n",
        {"created_utc": created, "required_round03_state": "promoted", "observed_round03_decision": round03["decision"], "locked_period_access": "forbidden"},
        [{"candidate": "ao9_blocked_operator_oof_exp1", "relative_improvement": gate["relative_improvement"], "maximum_pair_relative_regression": gate["maximum_pair_relative_regression"], "passes_gate": gate["passes_gate"]}],
        {"status": "closed_no_validated_candidate", "decision": "do_not_run_full_reconstruction", "reason": "Round 03 failed both the primary improvement and regional no-regression gates.", "locked_1921_1939_accessed": False, "source_hashes_unchanged": True},
        "# Round 04 Result\n\nNo full reconstruction was run. AO_9 worsened the OOF median RMSE by `10.591%` and the worst region/reference pair by `16.327%`, so Gate 3 forbids production promotion.\n",
        sources,
    )

    round05 = loop_root / "round_05_validation"
    write_round(
        round05,
        "# Round 05 Contract\n\nDecision: open the final test exactly once only for a frozen candidate that passed development gates. Without such a candidate, preserve the lock and record non-entry.\n",
        {"created_utc": created, "gate_5a_requires_frozen_candidate": True, "frozen_candidate_available": False, "user_prior_authorization_recorded": True},
        [{"gate": "5A", "eligible": False, "executed": True, "result": "not_eligible_no_frozen_candidate"}, {"gate": "5B", "eligible": False, "executed": False, "result": "locked_period_preserved"}],
        {"status": "not_entered_no_frozen_candidate", "gate_5a_passed": False, "gate_5b_executed": False, "locked_1921_1939_accessed": False, "reason": "Prior user authorization does not override the frozen-candidate prerequisite."},
        "# Round 05 Result\n\nGate 5A is not eligible because no candidate passed Gate 3. The `1921-1939` final period remains sealed and was not read.\n",
        {**sources, "round04": round04 / "summary.json"},
    )

    round06 = loop_root / "round_06_release"
    write_round(
        round06,
        "# Round 06 Contract\n\nDecision: publish HMA_reconstruction_v2 only after Gates 4, 5A, and 5B pass. Otherwise issue a traceable no-release decision while retaining the legacy product.\n",
        {"created_utc": created, "required_gates": [4, "5A", "5B"], "legacy_product_mutation_allowed": False},
        [{"release_candidate": "HMA_reconstruction_v2", "gate_4": False, "gate_5a": False, "gate_5b": False, "released": False}],
        {"status": "no_release", "release_id": None, "legacy_equal_six_status": "retained_control", "locked_1921_1939_accessed": False, "reason": "No new ensemble passed operator-level blocked validation."},
        "# Round 06 Result\n\n`HMA_reconstruction_v2` is not released. The legacy equal-six reconstruction remains the control; AO_9 is closed as a production candidate.\n",
        {**sources, "round04": round04 / "summary.json", "round05": round05 / "summary.json"},
    )

    registry = loop_root / "candidate_registry.csv"
    if registry.exists():
        raise FileExistsError(registry)
    registry_rows = [
        {"candidate_id": "legacy_equal_six", "parent": "", "variable": "tas_pr", "mechanism": "legacy_equal_weight", "state": "retained_control", "result_directory": "round_01_blocked_cv_baseline", "decision": "retain", "evidence": "No promoted replacement"},
        {"candidate_id": "pr_ao9_operator_oof_exp1", "parent": "legacy_equal_six", "variable": "pr", "mechanism": "AO_9 blocked PSM and analogue selection", "state": "closed", "result_directory": "round_03_nested_ensemble_selection", "decision": "reject", "evidence": "Median RMSE -10.591% improvement; max pair regression 16.327%"},
        {"candidate_id": "HMA_reconstruction_v2", "parent": "legacy_equal_six", "variable": "tas_pr", "mechanism": "new production release", "state": "closed", "result_directory": "round_06_release", "decision": "no_release", "evidence": "No Gate 3 candidate"},
    ]
    write_csv(registry, registry_rows)
    print(json.dumps({"status": "loop_closed_without_release", "round04": str(round04), "round05": str(round05), "round06": str(round06), "locked_1921_1939_accessed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
