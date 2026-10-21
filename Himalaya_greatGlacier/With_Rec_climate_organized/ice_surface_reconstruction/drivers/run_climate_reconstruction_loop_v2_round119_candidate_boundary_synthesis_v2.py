"""Synthesize the Round 111-118 boundary with the corrected Round 113 key."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


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


def main() -> int:
    project = root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_119_candidate_boundary_synthesis_v2"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    paths = {
        "facts": project / "LIA_PROJECT_FACTS.md",
        "round111": loop / "round_111_validation_evidence_independence_ledger_v1/summary.json",
        "round113": loop / "round_113_frozen_multiscale_validation_gates_v1/summary.json",
        "round114": loop / "round_114_external_evidence_role_closure_v1/summary.json",
        "round115": loop / "round_115_temperature_low_rank_oof_v2/summary.json",
        "round116": loop / "round_116_temperature_low_rank_residual_oof_v1/summary.json",
        "round117": loop / "round_117_precipitation_documented_moisture_residual_oof_v1/summary.json",
        "round118": loop / "round_118_documented_moisture_seasonal_incremental_skill_v1/summary.json",
        "old_boundary": loop / "round_106_climate_candidate_boundary_synthesis_v1/summary.json",
        "old_precipitation": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_annual_oof_early_candidate_v2/summary.json",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: {"path": str(path), "sha256": digest(path), "size_bytes": path.stat().st_size} for name, path in paths.items()}
    data = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items() if path.suffix == ".json"}
    if data["round113"].get("consumed_1921_1939_may_be_reopened") is not False:
        raise RuntimeError("round113-lockbox-contract-failed")
    if data["round113"].get("locked_period_values_accessed_by_round113") is not False:
        raise RuntimeError("round113-locked-values-contract-failed")
    for name in ("round115", "round116", "round117", "round118"):
        if data[name].get("locked_1921_1939_accessed") is not False:
            raise RuntimeError(f"locked-period-boundary-failed:{name}")
        if data[name].get("source_hashes_unchanged") is not True:
            raise RuntimeError(f"source-integrity-failed:{name}")
    if data["round117"].get("early_1901_1920_accessed") is not False or data["round118"].get("early_1901_1920_accessed") is not False:
        raise RuntimeError("early-period-boundary-failed")
    if data["round115"]["passes_all_required_modern_gates"] or data["round116"]["passes_all_required_modern_gates"]:
        raise RuntimeError("temperature-candidate-unexpectedly-passed")
    if data["round117"]["passes_all_required_modern_gates"] or data["round118"]["passes_all_seasonal_promotion_gates"]:
        raise RuntimeError("precipitation-candidate-unexpectedly-passed")

    rows = [
        {"component": "temperature", "candidate": "proxy-only annual low-rank", "status": "rejected", "evidence": "RMSE change -23.25%; annual correlation -0.152; spatial skill -2.53%"},
        {"component": "temperature", "candidate": "equal-six plus proxy residual", "status": "rejected_after_regional_improvement", "evidence": "median improvement +6.68%, but worst pair -10.78%, path fraction 66%, spatial skill -7.60%"},
        {"component": "precipitation", "candidate": "equal-six plus documented-moisture annual residual", "status": "rejected", "evidence": "RMSE change -7.57%; absolute skill -57.08%; spatial skill -23.74%"},
        {"component": "precipitation", "candidate": "documented-season incremental PSM", "status": "rejected", "evidence": "all four median precipitation gains negative; 12% of records improve on all paths"},
        {"component": "joint", "candidate": "temperature plus precipitation release", "status": "not_available", "evidence": "neither component passes frozen modern promotion"},
    ]
    pd.DataFrame(rows).to_csv(output / "candidate_boundary_matrix.csv", index=False)
    reopen = [
        "additional low-frequency-preserving HMA temperature chronologies available for fitting and distinct from the frozen validation reserve",
        "a moisture network that first passes the frozen cross-product seasonal incremental-skill gate, or independent amplitude-capable precipitation evidence",
        "new plateau-interior instrumental station or gauge observations that materially enlarge validated spatial support",
        "an explicitly narrower publication target limited to validated regional phase, without unsupported amplitude or spatial-field claims",
    ]
    after = {name: {"path": str(path), "sha256": digest(path), "size_bytes": path.stat().st_size} for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-changed-during-round119-v2")
    summary = {
        "experiment_id": "round_119_candidate_boundary_synthesis_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "temperature_publishable_candidate": False,
        "precipitation_publishable_candidate": False,
        "joint_publishable_candidate": False,
        "sufficient_evidence_for_current_data_method_boundary": True,
        "decision": "current_data_method_boundary_no_joint_release",
        "locked_1921_1939_reopened": False,
        "round111_round114_reserve_values_opened": False,
        "source_hashes_unchanged": unchanged,
        "reopen_only_with": reopen,
        "oggm_authorized": False,
        "boundary_matrix": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2), encoding="utf-8")
    temperature = data["round116"]["metrics"]
    seasonal = data["round118"]
    report = [
        "# Round 119 Climate Candidate Boundary Synthesis V2",
        "",
        "No temperature, precipitation, or joint HMA field passes the frozen release gates under the currently available fitting data and tested method families.",
        "",
        "## Temperature",
        "",
        f"The proxy-only field failed. Equal-six plus proxy residual achieved regional median improvement `{temperature['median_rmse_improvement_over_equal_six']:.2%}` and annual correlation `{temperature['median_annual_regional_correlation']:.3f}`, but failed worst-pair, path-fraction, and absolute spatial gates. Eastern HMA and spatial amplitude remain unvalidated.",
        "",
        "## Precipitation",
        "",
        f"The documented-moisture annual candidate failed. Matching records to documented summer seasons also failed: four median incremental gains range from `{min(seasonal['median_incremental_improvements'].values()):.2%}` to `{max(seasonal['median_incremental_improvements'].values()):.2%}`, and only `{seasonal['fraction_records_all_four_paths_improve']:.1%}` of records improve on all paths.",
        "",
        "## Decision",
        "",
        "Decision: `current_data_method_boundary_no_joint_release`.",
        "",
        "The old outcome-filtered precipitation branch remains a regional phase diagnostic. The Round 111/114 reserve values remain sealed, the consumed 1921-1939 interval was not reopened, and OGGM remains blocked.",
        "",
        "## Reopen Conditions",
        "",
        *[f"- {item}." for item in reopen],
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    facts = """## 190. Climate-reconstruction long-horizon validation contract (2026-08-17)\n\n- The climate objective was expanded to a publishable 1000-2000 HMA temperature and precipitation product. Round 111 froze proxy/evidence independence, Round 113 froze five blocked 1940-2000 folds and 19 multiscale gates, and Round 114 froze external evidence roles. The consumed 1921-1939 lockbox was not reopened and external reserve values remained excluded from fitting.\n\n## 191. Temperature low-rank modern-validation closure (2026-08-17)\n\n- Round 115 used 226 documented-temperature groups. Its proxy-only field fails: median RMSE change -23.25%, annual correlation -0.152, and spatial skill -2.53%.\n- Round 116 retained equal-six and predicted its residual. Regional median RMSE improves 6.68%, absolute skill is 16.47%, and annual correlation is 0.635, but worst-pair regression is 10.78%, only 66% of paths improve, and spatial skill is -7.60%. The annual linear low-rank temperature correction family is closed.\n\n## 192. Documented-moisture precipitation closure (2026-08-17)\n\n- Round 117 used 25 documented-moisture groups and fails: median RMSE change -7.57%, absolute skill -57.08%, and spatial skill -23.74%.\n- Round 118 matched every group to documented JJA or June-September seasons. All four self/cross-product precipitation gains are negative (-0.69% to -1.42%) and only 12% of records improve on all paths. Sign agreement and migration pass, showing product consistency without independent precipitation information. The current proxy9 moisture PSM route is closed.\n\n## 193. Current climate data-method boundary (2026-08-17)\n\n- Round 119 finds no publishable temperature, precipitation, or joint field under current fitting data and tested methods. The old outcome-filtered precipitation branch remains a regional phase diagnostic only. Paleo reserve values remain sealed because no candidate passes modern promotion.\n- Reopening requires new low-frequency temperature chronologies distinct from final reserve, a moisture network that first passes seasonal incremental skill or amplitude-capable precipitation evidence, materially expanded plateau-interior stations/gauges, or a narrower phase-only publication scope. OGGM remains blocked. Evidence is in Round 111 through Round 119 v2.\n"""
    (output / "LIA_PROJECT_FACTS_APPEND_190_193.md").write_text(facts, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
