"""Summarize the full turnover-equivalence scan without selecting a posterior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "019f730c-ee32-74d0-8b77-fdb7832f6a06"
)
DEFAULT_SCAN_ROOT = RUN_ROOT / "s1_turnover_f19_full_v0"
MANIFEST_PATH = (
    RUN_ROOT / "system_manifest_work" / "system_manifest_245_v0.csv"
)


def quantiles(series: pd.Series) -> dict[str, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    return {
        str(q): float(clean.quantile(q))
        for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    }


def classify_reachability(row: pd.Series) -> str:
    if row.evaluable_strong_gate_count == 0:
        return "no-evaluable-strong-gate"
    if row.best_inside_gate_count == row.evaluable_strong_gate_count:
        return "all-gates-inside"
    if row.best_inside_gate_count > 0:
        return "partial-gates-inside"
    return "no-gates-inside"


def classify_edge(row: pd.Series) -> str:
    if pd.isna(row.best_precip_prior_multiplier):
        return "no-velocity-selection"
    at_lower = np.isclose(
        row.best_precip_prior_multiplier,
        row.candidate_precip_multiplier_min,
        rtol=0,
        atol=1e-8,
    )
    at_upper = np.isclose(
        row.best_precip_prior_multiplier,
        row.candidate_precip_multiplier_max,
        rtol=0,
        atol=1e-8,
    )
    if at_lower and at_upper:
        return "degenerate-candidate-interval"
    if at_lower:
        return "low-turnover-edge"
    if at_upper:
        return "high-turnover-edge"
    return "candidate-interior"


def classify_diagnostic(row: pd.Series) -> str:
    if row.reachability_class == "no-evaluable-strong-gate":
        return "weak-identifiability-no-velocity"
    if row.reachability_class == "all-gates-inside":
        return "velocity-compatible"
    if row.reachability_class == "partial-gates-inside":
        return "partially-velocity-compatible"
    if row.candidate_edge_class == "low-turnover-edge" and row.best_median_observed_to_modeled_ratio < 1:
        return "modeled-too-fast-at-low-turnover-edge"
    if row.candidate_edge_class == "high-turnover-edge" and row.best_median_observed_to_modeled_ratio > 1:
        return "modeled-too-slow-at-high-turnover-edge"
    return "unresolved-mixed-or-interior-residual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_SCAN_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scan_root = args.scan_root
    if not scan_root.is_absolute():
        scan_root = REPO_ROOT / scan_root
    systems = pd.read_csv(scan_root / "system_summary.csv")
    candidates = pd.read_csv(scan_root / "candidate_summary.csv")
    gates = pd.read_csv(scan_root / "gate_residuals.csv")
    failures = pd.read_csv(scan_root / "failures.csv")
    manifest = pd.read_csv(MANIFEST_PATH)
    run_summary = json.loads((scan_root / "run_summary.json").read_text())
    run_manifest = json.loads((scan_root / "run_manifest.json").read_text())
    thickness_product = run_manifest.get(
        "fixed_thickness_product", "farinotti2019"
    )

    candidate_ranges = candidates.groupby("lia_unique_id").agg(
        candidate_precip_multiplier_min=("precip_prior_multiplier", "min"),
        candidate_precip_multiplier_max=("precip_prior_multiplier", "max"),
        candidate_melt_f_min=("melt_f", "min"),
        candidate_melt_f_max=("melt_f", "max"),
    )
    baselines = candidates[candidates["candidate_id"] == "stage0-baseline"]
    baselines = baselines.set_index("lia_unique_id")[["weighted_interval_loss"]]
    baselines = baselines.rename(
        columns={"weighted_interval_loss": "baseline_weighted_interval_loss"}
    )

    selected_ids = systems[["lia_unique_id", "best_candidate_id"]].dropna()
    selected_ids = selected_ids.rename(columns={"best_candidate_id": "candidate_id"})
    selected_gates = gates.merge(
        selected_ids, on=["lia_unique_id", "candidate_id"], validate="many_to_one"
    )
    gate_summary = selected_gates.groupby("lia_unique_id").agg(
        best_gate_count=("rgi_id", "size"),
        best_inside_gate_count=("inside_observation_interval", "sum"),
        best_median_observed_to_modeled_ratio=(
            "observed_to_modeled_surface_ratio",
            "median",
        ),
        best_min_observed_to_modeled_ratio=(
            "observed_to_modeled_surface_ratio",
            "min",
        ),
        best_max_observed_to_modeled_ratio=(
            "observed_to_modeled_surface_ratio",
            "max",
        ),
    )

    manifest_fields = [
        "lia_unique_id",
        "region_code",
        "region_label",
        "precip_prior_class",
        "precip_prior_multiplier_p025",
        "precip_prior_multiplier_p975",
        "debris_ratio_class",
        "debris_ratio_fraction",
    ]
    diagnostics = (
        systems.merge(candidate_ranges, on="lia_unique_id", validate="one_to_one")
        .merge(baselines, on="lia_unique_id", validate="one_to_one")
        .merge(gate_summary, on="lia_unique_id", how="left", validate="one_to_one")
        .merge(manifest[manifest_fields], on="lia_unique_id", validate="one_to_one")
    )
    diagnostics["best_inside_gate_count"] = diagnostics[
        "best_inside_gate_count"
    ].fillna(0).astype(int)
    diagnostics["best_gate_count"] = diagnostics["best_gate_count"].fillna(0).astype(int)
    diagnostics["reachability_class"] = diagnostics.apply(
        classify_reachability, axis=1
    )
    diagnostics["candidate_edge_class"] = diagnostics.apply(
        classify_edge, axis=1
    )
    diagnostics["registered_precip_lower_bound"] = np.isclose(
        diagnostics["best_precip_prior_multiplier"],
        diagnostics["precip_prior_multiplier_p025"],
        rtol=0,
        atol=1e-8,
    )
    diagnostics["registered_precip_upper_bound"] = np.isclose(
        diagnostics["best_precip_prior_multiplier"],
        diagnostics["precip_prior_multiplier_p975"],
        rtol=0,
        atol=1e-8,
    )
    diagnostics["physical_melt_lower_bound"] = np.isclose(
        diagnostics["best_melt_f"], 1.5, rtol=0, atol=1e-8
    )
    diagnostics["loss_improvement_from_stage0"] = (
        diagnostics["baseline_weighted_interval_loss"]
        - diagnostics["best_weighted_interval_loss"]
    )
    diagnostics["turnover_diagnostic"] = diagnostics.apply(
        classify_diagnostic, axis=1
    )
    diagnostics.to_csv(scan_root / "system_diagnostics.csv", index=False)

    by_region = pd.crosstab(
        diagnostics["region_label"], diagnostics["turnover_diagnostic"]
    )
    by_region.insert(0, "system_count", by_region.sum(axis=1))
    by_region.to_csv(scan_root / "diagnostics_by_region.csv")

    by_prior = pd.crosstab(
        diagnostics["precip_prior_class"], diagnostics["turnover_diagnostic"]
    )
    by_prior.insert(0, "system_count", by_prior.sum(axis=1))
    by_prior.to_csv(scan_root / "diagnostics_by_precip_prior.csv")

    evaluable = diagnostics[diagnostics["evaluable_strong_gate_count"] > 0]
    summary = {
        "run_id": run_summary["pilot_id"],
        "fixed_thickness_product": thickness_product,
        "artifact_status": "diagnostic-scan-not-frozen-parameter-posterior",
        "system_count": int(len(diagnostics)),
        "evaluable_system_count": int(len(evaluable)),
        "no_evaluable_strong_gate_system_count": int(
            (diagnostics["evaluable_strong_gate_count"] == 0).sum()
        ),
        "reachability_counts": diagnostics[
            "reachability_class"
        ].value_counts().to_dict(),
        "candidate_edge_counts": diagnostics[
            "candidate_edge_class"
        ].value_counts().to_dict(),
        "turnover_diagnostic_counts": diagnostics[
            "turnover_diagnostic"
        ].value_counts().to_dict(),
        "selected_at_registered_precip_lower_bound": int(
            diagnostics["registered_precip_lower_bound"].sum()
        ),
        "selected_at_registered_precip_upper_bound": int(
            diagnostics["registered_precip_upper_bound"].sum()
        ),
        "selected_at_physical_melt_lower_bound": int(
            diagnostics["physical_melt_lower_bound"].sum()
        ),
        "selected_gate_count": int(len(selected_gates)),
        "selected_gates_inside_observation_interval": int(
            selected_gates["inside_observation_interval"].sum()
        ),
        "best_system_median_observed_to_modeled_ratio_quantiles": quantiles(
            evaluable["best_median_observed_to_modeled_ratio"]
        ),
        "best_weighted_interval_loss_quantiles": quantiles(
            evaluable["best_weighted_interval_loss"]
        ),
        "loss_improvement_from_stage0_quantiles": quantiles(
            evaluable["loss_improvement_from_stage0"]
        ),
        "operator_failure_count": int(len(failures)),
        "operator_failure_types": failures["error_type"].value_counts().to_dict(),
        "stage0_source_immutable": bool(
            run_summary["source_stage0_key_files_immutable"]
        ),
        "semantic_validation_pass": bool(run_summary["semantic_validation_pass"]),
        "interpretation": (
            "Velocity compatibility is assessed only inside the registered "
            "modern-MB equivalence surface with fixed "
            f"{thickness_product} thickness. "
            "Best candidates are diagnostics, not published point posteriors."
        ),
    }
    (scan_root / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
