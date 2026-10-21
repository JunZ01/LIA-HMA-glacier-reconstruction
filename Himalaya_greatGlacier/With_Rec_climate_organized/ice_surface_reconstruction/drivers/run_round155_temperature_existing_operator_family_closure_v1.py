"""Synthesize verified evidence and close or retain the existing temperature family."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    output = validation / "climate_reconstruction_temperature_existing_operator_family_closure_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round155_temperature_existing_operator_family_closure_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "round25": validation / "climate_reconstruction_improvement_loop_v2/round_25_temperature_prior_update_decomposition_v1/summary.json",
        "round151": validation / "climate_reconstruction_temperature_early20_prior_update_export_attribution_1901_1920_v1/summary.json",
        "round152": validation / "climate_reconstruction_temperature_early20_adaptive_background_factorial_1901_1920_v2/summary.json",
        "round153": validation / "climate_reconstruction_temperature_early20_adaptive_background_group_loo_1901_1920_v1/summary.json",
        "round154": validation / "climate_reconstruction_temperature_adaptive_background_spatial_balance_oof_v1/summary.json",
        "round154_pairs": validation / "climate_reconstruction_temperature_adaptive_background_spatial_balance_oof_v1/region_product_oof_metrics.csv",
        "fold_oenkf": validation / "temperature_fold_native_oenkf_increment_cv_v5_regem_R/summary.json",
        "fold_oenkf_pairs": validation / "temperature_fold_native_oenkf_increment_cv_v5_regem_R/heldout_region_product_scores.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: sha256(path) for name, path in paths.items()}
    payload = {
        name: json.loads(paths[name].read_text(encoding="utf-8"))
        for name in ("round25", "round151", "round152", "round153", "round154", "fold_oenkf")
    }
    round154_pairs = pd.read_csv(paths["round154_pairs"])
    oenkf_pairs = pd.read_csv(paths["fold_oenkf_pairs"])

    locked_flags = {
        "round25": bool(payload["round25"]["locked_1921_1939_values_read"]),
        "round151": bool(payload["round151"]["locked_1921_1939_accessed"]),
        "round152": bool(payload["round152"]["locked_1921_1939_accessed"]),
        "round153": bool(payload["round153"]["locked_1921_1939_accessed"]),
        "round154": bool(payload["round154"]["locked_1901_1939_accessed"]),
        "fold_oenkf": bool(payload["fold_oenkf"]["locked_1921_1939_accessed"]),
    }
    if any(locked_flags.values()):
        raise RuntimeError(f"locked-source-flag:{locked_flags}")

    global_skill = float(round154_pairs.global_skill_vs_zero.median())
    global_correlation = float(round154_pairs.global_annual_correlation.median())
    balanced_skill = float(round154_pairs.balanced_skill_vs_zero.median())
    balanced_correlation = float(round154_pairs.balanced_annual_correlation.median())

    def branch_median(branch: str, column: str) -> float:
        selected = oenkf_pairs[oenkf_pairs.branch.eq(branch)]
        if selected.empty:
            raise RuntimeError(f"missing-oenkf-branch:{branch}")
        return float(selected[column].median())

    full_skill = branch_median("full_increment_alpha1", "skill_vs_zero")
    full_correlation = branch_median("full_increment_alpha1", "annual_correlation")
    shrunk_skill = branch_median("candidate_nested_alpha", "skill_vs_zero")
    shrunk_correlation = branch_median("candidate_nested_alpha", "annual_correlation")

    rows = [
        {
            "route": "archived_global_adaptive_background",
            "modern_oof_skill": global_skill,
            "modern_oof_annual_correlation": global_correlation,
            "allowed_period_evidence": "dominates_east_1901_1920_warming",
            "decision": "close_as_validated_background_signal",
            "reason": "modern_oof_skill_nonpositive_and_east_early_warming_conflicts_with_external_evidence",
        },
        {
            "route": "equal_longitude_band_adaptive_background",
            "modern_oof_skill": balanced_skill,
            "modern_oof_annual_correlation": balanced_correlation,
            "allowed_period_evidence": "trajectory_not_opened_after_modern_failure",
            "decision": "close",
            "reason": "round154_failed_relative_absolute_network_and_split_period_gates",
        },
        {
            "route": "fixed_background_full_oenkf_increment",
            "modern_oof_skill": full_skill,
            "modern_oof_annual_correlation": full_correlation,
            "allowed_period_evidence": "increment_only_warm_in_all_ten_1400_1899_windows",
            "decision": "close",
            "reason": "phase_present_but_amplitude_skill_negative_and_lia_direction_unrepaired",
        },
        {
            "route": "fixed_background_nested_global_increment_shrinkage",
            "modern_oof_skill": shrunk_skill,
            "modern_oof_annual_correlation": shrunk_correlation,
            "allowed_period_evidence": "not_promoted_after_modern_gate_failure",
            "decision": "close",
            "reason": "modern_skill_below_5pct_and_fold_native_gate_set_failed",
        },
    ]
    decisions = pd.DataFrame(rows)
    decisions.to_csv(output / "temperature_route_decision_matrix.csv", index=False)

    early = payload["round151"]["east_contribution_trends_degC_per_century"]
    long_term = payload["round25"]
    all_closed = bool((decisions.decision == "close").sum() == 3 and (decisions.decision == "close_as_validated_background_signal").sum() == 1)
    if not all_closed:
        raise RuntimeError("route-closure-table-failed")
    after = {name: sha256(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("synthesis-input-changed")
    summary = {
        "experiment_id": "round155_temperature_existing_operator_family_closure_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "evidence_only_synthesis": True,
        "early_east_1901_1920_trends_degC_per_century": early,
        "long_term_1400_1899": {
            "increment_only_warm_window_count": int(long_term["increment_only_warmer_than_reference_window_count"]),
            "window_count": int(long_term["lia_window_count"]),
            "adaptive_background_fraction_of_warm_gap": float(long_term["median_background_fraction_of_warm_gap"]),
        },
        "modern_oof": {
            "global_adaptive_background": {"skill": global_skill, "annual_correlation": global_correlation},
            "balanced_adaptive_background": {"skill": balanced_skill, "annual_correlation": balanced_correlation},
            "fixed_background_full_oenkf_increment": {"skill": full_skill, "annual_correlation": full_correlation},
            "fixed_background_nested_increment_shrinkage": {"skill": shrunk_skill, "annual_correlation": shrunk_correlation},
        },
        "all_existing_routes_closed_for_production_promotion": all_closed,
        "decision": "close_current_archived_temperature_prior_psm_operator_repair_family",
        "current_field_status": "legacy_experimental_forcing_not_validated_hma_temperature_reconstruction",
        "next_evidence_requirement": (
            "a separately contracted new seasonal_or_low_frequency regional temperature operator "
            "or genuinely new proxy evidence must first pass blocked modern OOF and split-period stability; "
            "existing weighting banding subset scalar and gain-amplitude tuning remain closed"
        ),
        "locked_values_opened": False,
        "jionglacuo_opened": False,
        "climate_field_opened": False,
        "proxy_or_instrumental_source_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "input_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 155 Existing Temperature-Operator Family Closure\n\n"
        "- Decision: `close_current_archived_temperature_prior_psm_operator_repair_family`.\n"
        f"- Early eastern trend components (background/update/export): `{early['background']:+.3f}` / "
        f"`{early['enkf_increment']:+.3f}` / `{early['export_residual']:+.3f} degC/century`.\n"
        f"- Global adaptive-background modern OOF skill/correlation: `{global_skill:+.3f}` / `{global_correlation:.3f}`.\n"
        f"- Balanced adaptive-background modern OOF skill/correlation: `{balanced_skill:+.3f}` / `{balanced_correlation:.3f}`.\n"
        f"- Fixed-background full OEnKF skill/correlation: `{full_skill:+.3f}` / `{full_correlation:.3f}`.\n"
        f"- Fixed-background nested-shrinkage skill/correlation: `{shrunk_skill:+.3f}` / `{shrunk_correlation:.3f}`.\n"
        f"- Increment-only counterfactual remains warm in `{long_term['increment_only_warmer_than_reference_window_count']}/{long_term['lia_window_count']}` LIA windows; adaptive background explains only `{long_term['median_background_fraction_of_warm_gap']:.1%}` of the median long-term gap.\n"
        "- Current field status: legacy experimental forcing, not a validated HMA temperature reconstruction.\n"
        "- No locked value, proxy source, climate field, EnKF, or OGGM input was opened in this synthesis.\n\n"
        "Further weighting, longitude banding, proxy deletion, scalar scaling, and gain-amplitude tuning are closed. "
        "A new seasonal/low-frequency regional operator or genuinely new proxy evidence must independently pass modern OOF before any paleo trajectory is inspected.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
