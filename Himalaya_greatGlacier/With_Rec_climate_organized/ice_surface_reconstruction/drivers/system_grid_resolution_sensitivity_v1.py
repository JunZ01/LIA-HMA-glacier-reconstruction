"""Diagnose rasterized cross-branch conflicts on nested system grids.

This driver reruns selected systems at explicitly nested grid resolutions. It
does not change duplicate-value tolerances or aggregate conflicting branch
constraints. Each factor writes to an isolated directory under the existing
run root.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import resolve
from .lia_3d_system_grid_pilot_v1 import run as run_system_grid


DEFAULT_FACTORS = (1, 2, 4, 8)
CONVERGENCE_THRESHOLDS = {
    "volume_relative_fraction": 0.005,
    "mean_thickness_relative_fraction": 0.005,
    "maximum_thickness_relative_fraction": 0.02,
}


def _is_success(status: str) -> bool:
    return str(status) in {"success", "success-with-documented-exclusion"}


def _relative_change(fine: float, coarse: float) -> float:
    if not np.isfinite(fine) or not np.isfinite(coarse) or fine == 0.0:
        return np.nan
    return float((fine - coarse) / fine)


def _comparison_table(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["rgi_id", "lia_unique_id", "bed_scenario"]
    output: list[dict] = []
    for key, group in rows.groupby(keys, dropna=False):
        by_factor = {
            int(row.resolution_factor): row
            for row in group.itertuples(index=False)
        }
        record = dict(zip(keys, key))
        for factor in DEFAULT_FACTORS:
            row = by_factor.get(factor)
            record[f"factor{factor}_status"] = (
                str(row.status) if row is not None else "not-run"
            )
            record[f"factor{factor}_failure_reason"] = (
                str(row.failure_reason) if row is not None else ""
            )
            record[f"factor{factor}_dx_m"] = (
                float(getattr(row, "system_grid_dx_m", np.nan))
                if row is not None else np.nan
            )
            for label, field in {
                "volume_km3": "volume_km3_finite_components",
                "mean_thickness_m": "mean_thickness_m_finite_components",
                "maximum_thickness_m": "maximum_thickness_m_finite_components",
                "branch_count": "branch_count",
                "connector_count": "connector_count",
            }.items():
                record[f"factor{factor}_{label}"] = (
                    float(getattr(row, field, np.nan))
                    if row is not None else np.nan
                )
        successful = [
            factor for factor, row in by_factor.items()
            if _is_success(row.status)
        ]
        record["first_success_factor"] = min(successful) if successful else np.nan
        comparisons = {
            "volume": "volume_km3_finite_components",
            "mean_thickness": "mean_thickness_m_finite_components",
            "maximum_thickness": "maximum_thickness_m_finite_components",
        }
        for coarse_factor, fine_factor in ((2, 4), (4, 8)):
            coarse_row = by_factor.get(coarse_factor)
            fine_row = by_factor.get(fine_factor)
            for label, field in comparisons.items():
                coarse = (
                    float(getattr(coarse_row, field, np.nan))
                    if coarse_row is not None else np.nan
                )
                fine = (
                    float(getattr(fine_row, field, np.nan))
                    if fine_row is not None else np.nan
                )
                record[f"factor{fine_factor}_minus_factor{coarse_factor}_{label}"] = (
                    fine - coarse
                    if np.isfinite(fine) and np.isfinite(coarse) else np.nan
                )
                record[
                    f"factor{fine_factor}_vs_factor{coarse_factor}_{label}_relative_fraction"
                ] = _relative_change(fine, coarse)
        f4 = by_factor.get(4)
        f8 = by_factor.get(8)
        both_success = (
            f4 is not None and f8 is not None
            and _is_success(f4.status) and _is_success(f8.status)
        )
        if not successful:
            verdict = "conflict-unresolved-through-factor8"
        elif not both_success:
            verdict = "conflict-resolved-but-convergence-not-demonstrated"
        else:
            volume_change = abs(record[
                "factor8_vs_factor4_volume_relative_fraction"
            ])
            mean_change = abs(record[
                "factor8_vs_factor4_mean_thickness_relative_fraction"
            ])
            maximum_change = abs(record[
                "factor8_vs_factor4_maximum_thickness_relative_fraction"
            ])
            converged = (
                volume_change <= CONVERGENCE_THRESHOLDS["volume_relative_fraction"]
                and mean_change <= CONVERGENCE_THRESHOLDS[
                    "mean_thickness_relative_fraction"
                ]
                and maximum_change <= CONVERGENCE_THRESHOLDS[
                    "maximum_thickness_relative_fraction"
                ]
            )
            verdict = (
                "factor4-converged-against-factor8"
                if converged else "factor4-factor8-not-converged"
            )
        record["resolution_verdict"] = verdict
        output.append(record)
    return pd.DataFrame(output)


def run(
    config: dict,
    selected_rgi_ids: set[str],
    factors: tuple[int, ...] = DEFAULT_FACTORS,
    *,
    reuse_existing_factors: bool = False,
) -> dict:
    if tuple(factors) != DEFAULT_FACTORS:
        raise ValueError("resolution-sensitivity-v1-requires-factors-1-2-4-8")
    if not selected_rgi_ids:
        raise ValueError("resolution-sensitivity-requires-selected-rgi-ids")
    root = resolve(config["output_dir"])
    diagnostic_subdir = str(
        config.get("system_grid_resolution_sensitivity_v1", {}).get(
            "output_subdir", "system_grid_resolution_sensitivity_v1"
        )
    )
    diagnostic_dir = root / diagnostic_subdir
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    factor_summaries: dict[str, dict] = {}
    for factor in factors:
        candidate = copy.deepcopy(config)
        grid_config = candidate.setdefault("lia_system_grid", {})
        grid_config["resolution_factor"] = int(factor)
        grid_config["grid_output_subdir"] = (
            f"{diagnostic_subdir}/factor_{factor}/lia_system_grids_v1"
        )
        candidate.setdefault("lia_3d_system_grid_pilot_v1", {})[
            "output_subdir"
        ] = f"{diagnostic_subdir}/factor_{factor}/results"
        result_path = (
            diagnostic_dir / f"factor_{factor}" / "results"
            / "lia_system_grid_results_v1.csv"
        )
        summary_path = (
            diagnostic_dir / f"factor_{factor}" / "results"
            / "lia_system_grid_summary_v1.json"
        )
        can_reuse = False
        if reuse_existing_factors and result_path.exists() and summary_path.exists():
            existing = pd.read_csv(result_path)
            can_reuse = set(existing["rgi_id"].astype(str)) == set(selected_rgi_ids)
        if can_reuse:
            factor_summaries[str(factor)] = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
        else:
            factor_summaries[str(factor)] = run_system_grid(
                candidate, selected_rgi_ids
            )
        frame = pd.read_csv(result_path)
        frame["resolution_factor"] = int(factor)
        frame["factor_output_csv"] = str(result_path.relative_to(root))
        component_path = (
            diagnostic_dir / f"factor_{factor}" / "results"
            / "lia_system_component_results_v1.csv"
        )
        if component_path.exists() and component_path.stat().st_size > 2:
            components = pd.read_csv(component_path)
            keys = ["rgi_id", "lia_unique_id", "bed_scenario"]
            topology_fields = [
                field for field in ("branch_count", "connector_count")
                if field in components
            ]
            if topology_fields:
                topology = components.groupby(keys, dropna=False)[
                    topology_fields
                ].sum(min_count=1).reset_index()
                frame = frame.merge(topology, on=keys, how="left")
            if "failure_reason" in components:
                failures = components.groupby(
                    keys, dropna=False
                )["failure_reason"].apply(
                    lambda values: ";".join(
                        str(value) for value in values
                        if pd.notna(value) and str(value)
                    )
                ).rename("component_failure_reason")
                frame = frame.merge(
                    failures.reset_index(),
                    on=["rgi_id", "lia_unique_id", "bed_scenario"],
                    how="left",
                )
        if "failure_reason" not in frame:
            frame["failure_reason"] = ""
        frame["failure_reason"] = frame["failure_reason"].fillna("")
        if "component_failure_reason" in frame:
            frame["failure_reason"] = frame["failure_reason"].where(
                frame["failure_reason"].astype(str).str.len() > 0,
                frame["component_failure_reason"].fillna(""),
            )
        rows.append(frame)
    long = pd.concat(rows, ignore_index=True, sort=False)
    comparison = _comparison_table(long)
    long_path = diagnostic_dir / "system_grid_resolution_results_long_v1.csv"
    comparison_path = diagnostic_dir / "system_grid_resolution_comparison_v1.csv"
    long.to_csv(long_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    verdict_counts = comparison["resolution_verdict"].value_counts().to_dict()
    payload = {
        "status": "completed-system-grid-resolution-sensitivity-v1",
        "selected_rgi_ids": sorted(selected_rgi_ids),
        "resolution_factors": list(factors),
        "grid_alignment": "nested-original-gdir-pixel-edges",
        "constraint_policy": "unchanged-no-average-no-tolerance-relaxation",
        "convergence_thresholds": CONVERGENCE_THRESHOLDS,
        "member_verdict_counts": verdict_counts,
        "factor_summaries": factor_summaries,
        "long_results_csv": long_path.name,
        "comparison_csv": comparison_path.name,
    }
    (diagnostic_dir / "system_grid_resolution_summary_v1.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rgi-id", action="append", required=True)
    parser.add_argument("--reuse-existing-factors", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run(
        config,
        set(args.rgi_id),
        reuse_existing_factors=args.reuse_existing_factors,
    ), indent=2))


if __name__ == "__main__":
    main()
