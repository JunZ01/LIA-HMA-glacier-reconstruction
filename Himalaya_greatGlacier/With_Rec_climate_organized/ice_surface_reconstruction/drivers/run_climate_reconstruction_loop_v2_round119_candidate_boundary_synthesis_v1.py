"""Synthesize the frozen Round 111-118 climate reconstruction boundary."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_119_candidate_boundary_synthesis_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "round111": loop / "round_111_validation_evidence_independence_ledger_v1/summary.json",
        "round113": loop / "round_113_frozen_multiscale_validation_gates_v1/summary.json",
        "round114": loop / "round_114_external_evidence_role_closure_v1/summary.json",
        "round115": loop / "round_115_temperature_low_rank_oof_v2/summary.json",
        "round116": loop / "round_116_temperature_low_rank_residual_oof_v1/summary.json",
        "round117": loop / "round_117_precipitation_documented_moisture_residual_oof_v1/summary.json",
        "round118": loop / "round_118_documented_moisture_seasonal_incremental_skill_v1/summary.json",
        "old_boundary": loop / "round_106_climate_candidate_boundary_synthesis_v1/summary.json",
        "old_precipitation_candidate": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_annual_oof_early_candidate_v2/summary.json",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size} for name, path in paths.items()}
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items() if path.suffix == ".json"}

    if not loaded["round113"].get("lockbox_reopen") is False:
        raise RuntimeError("round113-lockbox-contract-failed")
    for name in ("round115", "round116", "round117", "round118"):
        if loaded[name].get("locked_1921_1939_accessed") is not False:
            raise RuntimeError(f"locked-period-boundary-failed:{name}")
        if loaded[name].get("source_hashes_unchanged") is not True:
            raise RuntimeError(f"source-integrity-failed:{name}")
    if loaded["round117"].get("early_1901_1920_accessed") is not False:
        raise RuntimeError("round117-early-period-boundary-failed")
    if loaded["round118"].get("early_1901_1920_accessed") is not False:
        raise RuntimeError("round118-early-period-boundary-failed")

    temperature_pass = bool(
        loaded["round115"]["passes_all_required_modern_gates"]
        or loaded["round116"]["passes_all_required_modern_gates"]
    )
    precipitation_pass = bool(
        loaded["round117"]["passes_all_required_modern_gates"]
        or loaded["round118"]["passes_all_seasonal_promotion_gates"]
    )
    if temperature_pass or precipitation_pass:
        raise RuntimeError("boundary-synthesis-called-with-passing-new-candidate")

    rows = [
        {
            "component": "temperature",
            "tested_candidate": "proxy-only low-rank annual field",
            "result": "rejected",
            "controlling_evidence": "median RMSE change versus equal-six -23.25%; annual correlation -0.152; spatial skill -2.53%",
            "boundary": "proxy-only annual modes do not transfer modern trend or spatial pattern",
        },
        {
            "component": "temperature",
            "tested_candidate": "equal-six plus low-rank proxy residual",
            "result": "rejected_after_material_regional_improvement",
            "controlling_evidence": "regional median improvement +6.68% and correlation 0.635, but worst pair -10.78%, path fraction 66%, spatial skill -7.60%",
            "boundary": "current network carries some regional signal but cannot validate eastern/spatial amplitude",
        },
        {
            "component": "precipitation",
            "tested_candidate": "equal-six plus documented-moisture annual residual",
            "result": "rejected",
            "controlling_evidence": "median RMSE change -7.57%; absolute skill -57.08%; spatial skill -23.74%",
            "boundary": "25 metadata-pure moisture groups do not support annual residual correction",
        },
        {
            "component": "precipitation",
            "tested_candidate": "documented-season incremental PSM",
            "result": "rejected",
            "controlling_evidence": "all four median precipitation gains are negative; only 12% of records improve on all paths",
            "boundary": "stable coefficient signs do not translate into independent predictive precipitation information",
        },
        {
            "component": "joint",
            "tested_candidate": "temperature plus precipitation release",
            "result": "not_available",
            "controlling_evidence": "neither component passes its frozen modern promotion gates",
            "boundary": "Round 111/114 reserve values remain sealed because no modern-passing candidate exists",
        },
    ]
    pd.DataFrame(rows).to_csv(output / "candidate_boundary_matrix.csv", index=False)

    reopen = [
        "additional low-frequency-preserving HMA temperature chronologies that are available for fitting and are distinct from the frozen final-validation reserve",
        "a moisture-proxy network that first demonstrates at least 1% cross-product blocked seasonal incremental skill, or independent amplitude-capable precipitation evidence",
        "new plateau-interior instrumental station or gauge observations that materially enlarge the validated spatial support",
        "an explicitly narrower publication target limited to validated regional phase, with amplitude and spatial-field claims removed",
    ]
    after = {name: {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size} for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-changed-during-round119")
    summary = {
        "experiment_id": "round_119_candidate_boundary_synthesis_v1",
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
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Round 119 Climate Candidate Boundary Synthesis",
        "",
        "No temperature, precipitation, or joint HMA field passes the frozen release gates under the currently available fitting data and tested method families.",
        "",
        "## Temperature",
        "",
        f"The proxy-only low-rank field failed decisively. Keeping equal-six as a background recovered useful regional skill (median RMSE improvement `{loaded['round116']['metrics']['median_rmse_improvement_over_equal_six']:.2%}`, annual correlation `{loaded['round116']['metrics']['median_annual_regional_correlation']:.3f}`), but failed the worst-pair, path-fraction, and absolute spatial gates. The eastern HMA and spatial amplitude remain unvalidated.",
        "",
        "## Precipitation",
        "",
        f"The 25 documented-moisture groups worsened annual residual skill. Matching every record to its documented summer season did not rescue precipitation information: four median incremental improvements range from `{min(loaded['round118']['median_incremental_improvements'].values()):.2%}` to `{max(loaded['round118']['median_incremental_improvements'].values()):.2%}`, and only `{loaded['round118']['fraction_records_all_four_paths_improve']:.1%}` of records improve on all paths.",
        "",
        "## Decision",
        "",
        "Decision: `current_data_method_boundary_no_joint_release`.",
        "",
        "The old outcome-filtered precipitation branch remains a regional phase diagnostic, not a replacement product. The Round 111/114 paleo reserves remain sealed because no candidate passed modern promotion. The consumed 1921-1939 period was not reopened. OGGM remains blocked.",
        "",
        "## Reopen Conditions",
        "",
        *[f"- {item}." for item in reopen],
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    facts_text = """## 190. Climate-reconstruction long-horizon validation contract (2026-08-17)\n\n- The active climate objective was expanded from the 1901-1920 diagnostic to a 1000-2000 publishable HMA temperature and precipitation product. Proxy identity and validation evidence were frozen before new candidates. The consumed 1921-1939 temperature lockbox was not reopened; modern promotion uses five blocked 1940-2000 folds and independent paleo reserves remain excluded from fitting.\n- Round 111 found 297 reconstruction evidence groups and froze 12 source-and-method-independent hydroclimate site clusters. Round 113 froze 19 multiscale gates, including relative, absolute, annual-correlation, spatial-field, and external-paleo requirements. Round 114 classified eight hydroclimate clusters as directional phase evidence, three as mechanism-only, one as excluded, and froze new temperature evidence roles.\n\n## 191. Temperature low-rank modern-validation closure (2026-08-17)\n\n- Round 115 used 226 metadata-ready documented-temperature proxy9 evidence groups with training-only area-weighted EOFs and nested ridge selection. The proxy-only candidate fails: median RMSE change versus equal-six is -23.25%, annual regional correlation -0.152, and spatial residual skill -2.53%.\n- Round 116 retained equal-six as background and predicted only its five-product residual. This materially improves regional metrics: median RMSE improvement +6.68%, absolute skill +16.47%, and annual correlation 0.635. It still fails the frozen release gates because the worst pair regresses 10.78%, only 66% of block-region-product paths improve, median spatial residual skill is -7.60%, and only one of five products has positive spatial skill. The annual linear low-rank proxy correction family is closed.\n- Both experiments left Round 114 external values sealed, did not access 1921-1939, preserved source hashes, and did not run OGGM.\n\n## 192. Documented-moisture precipitation closure (2026-08-17)\n\n- Round 117 used exactly 25 metadata-ready documented-moisture proxy9 evidence groups, CRU/GPCC common support, and nested residual correction around equal-six. It fails decisively: median RMSE change -7.57%, median absolute skill -57.08%, and median spatial residual skill -23.74%.\n- Round 118 then matched every group to its documented JJA or June-September season and tested temperature-only versus bivariate temperature-plus-precipitation PSMs on four self/cross-product paths. All four median precipitation gains are negative (-0.69% to -1.42%); only 12% of records improve on all paths. Coefficient-sign agreement (84.87%) and migration gates pass, proving product consistency but not independent precipitation information. The current documented-moisture PSM route is closed.\n- Neither experiment accessed 1901-1920, 1921-1939, or paleo reserve values; source hashes are unchanged and OGGM was not run.\n\n## 193. Current climate data-method boundary (2026-08-17)\n\n- Round 119 finds no publishable temperature, precipitation, or joint HMA field under the currently available fitting data and tested method families. The old outcome-filtered precipitation branch remains a regional-phase diagnostic only and is not promoted under the newer metadata and absolute/spatial gates. The Round 111/114 reserves remain sealed because no modern-passing candidate exists.\n- This is a sufficient evidence-supported current data-method boundary. Reopening requires at least one of: additional low-frequency-preserving temperature chronologies distinct from the final-validation reserve; a moisture network that first passes the frozen cross-product seasonal incremental-skill gate or independent amplitude-capable precipitation evidence; materially expanded plateau-interior station/gauge support; or a narrower publication target that removes unsupported amplitude and spatial-field claims. OGGM remains blocked on this climate branch.\n- Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_111_validation_evidence_independence_ledger_v1/` through `round_119_candidate_boundary_synthesis_v1/`.\n"""
    (output / "LIA_PROJECT_FACTS_APPEND_190_193.md").write_text(facts_text, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
