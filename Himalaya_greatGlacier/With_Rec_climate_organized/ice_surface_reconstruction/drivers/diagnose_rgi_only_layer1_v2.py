"""Corrected signed-mechanism diagnosis for the 48 Layer-1 RGI-only systems.

Version 1 correctly extracted the saved curves but incorrectly assumed that a
missing LIA match always meant a short peak.  This wrapper replaces that label
with a signed comparison to the LIA target, reruns the read-only extraction and
plots, and then classifies the complete stable Layer-1 response as entirely
short, entirely long, or straddling the LIA target through a branch gap.

No OGGM model is initialized or run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import diagnose_rgi_only_layer1_v1 as base
from .build_lia_191_iterative_loop import load_scores


def signed_trajectory_class(metrics: dict) -> str:
    signed_deficit = float(metrics["peak_lia_deficit_m"])
    lia_m = float(metrics["forward_peak_m"]) + signed_deficit
    start_m = float(metrics["forward_start_m"])
    if signed_deficit > 0:
        if float(metrics["peak_gain_from_start_m"]) <= base.NODE_M:
            return "rgi_match_peak_below_lia_no_growth"
        recovery = float(metrics["initial_deficit_recovery_fraction"])
        if np.isfinite(recovery) and recovery < 0.25:
            return "rgi_match_peak_below_lia_limited_growth"
        return "rgi_match_peak_below_lia_partial_recovery"
    if signed_deficit < 0:
        if start_m > 1.10 * lia_m:
            return "rgi_match_starts_above_lia"
        return "rgi_match_grows_past_lia_and_overshoots"
    return "unexpected_exact_lia_peak"


def needed_side_class(layer1: pd.DataFrame, peak_anchor: pd.Series) -> str:
    lia_m = float(peak_anchor["lia_target_m"])
    peak_m = float(peak_anchor["forward_peak_m"])
    anchor_dt = float(peak_anchor["delta_t"])
    if peak_m < lia_m:
        direction = "colder"
        directional = layer1[layer1["delta_t"].lt(anchor_dt - 1e-9)]
    elif peak_m > lia_m:
        direction = "warmer"
        directional = layer1[layer1["delta_t"].gt(anchor_dt + 1e-9)]
    else:
        return "unexpected_exact_lia_peak"
    if directional.empty:
        return f"needed_{direction}_tested_search_edge"
    if not directional["stable_chain"].any():
        return f"needed_{direction}_stability_boundary"
    return f"needed_{direction}_nonmonotonic_or_branch_response"


def stable_branch_class(stable: pd.DataFrame, lia_m: float) -> str:
    peaks = pd.to_numeric(stable["forward_peak_m"], errors="coerce").dropna()
    below = bool((peaks < 0.90 * lia_m).any())
    inside = bool(((peaks >= 0.90 * lia_m) & (peaks <= 1.10 * lia_m)).any())
    above = bool((peaks > 1.10 * lia_m).any())
    if inside:
        return "conflict_stable_lia_gate_exists"
    if below and above:
        return "stable_branches_straddle_lia_without_gate"
    if below:
        return "all_stable_peaks_below_lia"
    if above:
        return "all_stable_peaks_above_lia"
    return "unclassified_stable_peak_range"


def append_branch_diagnosis(output_dir: Path) -> dict:
    path = output_dir / "rgi_only_mechanism_diagnosis.csv"
    diagnosis = pd.read_csv(path)
    scores = load_scores()
    branch_values: list[str] = []
    stable_peak_min: list[float] = []
    stable_peak_max: list[float] = []
    stable_peak_gap: list[float] = []
    for row in diagnosis.itertuples(index=False):
        uid = int(row.lia_unique_id)
        stable = scores[
            scores["lia_unique_id"].eq(uid)
            & scores["score_layer"].eq("layer1_delta")
            & scores["stable_chain"]
        ].copy()
        lia_m = float(row.lia_target_m)
        branch_values.append(stable_branch_class(stable, lia_m))
        peaks = pd.to_numeric(stable["forward_peak_m"], errors="coerce").dropna()
        stable_peak_min.append(float(peaks.min()))
        stable_peak_max.append(float(peaks.max()))
        below = peaks[peaks < 0.90 * lia_m]
        above = peaks[peaks > 1.10 * lia_m]
        if not below.empty and not above.empty:
            stable_peak_gap.append(float(above.min() - below.max()))
        else:
            stable_peak_gap.append(np.nan)
    diagnosis["stable_peak_branch_class"] = branch_values
    diagnosis["stable_peak_min_m"] = stable_peak_min
    diagnosis["stable_peak_max_m"] = stable_peak_max
    diagnosis["stable_peak_branch_gap_m"] = stable_peak_gap
    diagnosis["search_topology"] = diagnosis["cold_side_class"]
    diagnosis.to_csv(path, index=False)

    conflict = diagnosis[
        diagnosis["stable_peak_branch_class"].eq("conflict_stable_lia_gate_exists")
    ]
    if not conflict.empty:
        raise RuntimeError(
            "router-conflict-stable-lia-gate:" + ",".join(map(str, conflict["lia_unique_id"]))
        )
    branch_counts = diagnosis["stable_peak_branch_class"].value_counts().sort_index()
    trajectory_counts = diagnosis["trajectory_class"].value_counts().sort_index()
    topology_counts = diagnosis["search_topology"].value_counts().sort_index()
    return {
        "stable_peak_branch_class_counts": {
            str(key): int(value) for key, value in branch_counts.items()
        },
        "trajectory_class_counts_corrected": {
            str(key): int(value) for key, value in trajectory_counts.items()
        },
        "search_topology_counts": {
            str(key): int(value) for key, value in topology_counts.items()
        },
        "median_stable_branch_gap_m": float(
            diagnosis["stable_peak_branch_gap_m"].dropna().median()
        )
        if diagnosis["stable_peak_branch_gap_m"].notna().any()
        else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unresolved", type=Path, default=base.default_input())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            base.project_root()
            / "validation_data"
            / "results"
            / "paleo_surface_validation_v0"
            / "lia_191_controller_v1"
            / "diagnostics"
            / "rgi_only_layer1_mechanisms_v2"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base._trajectory_class = signed_trajectory_class
    base._cold_side_class = needed_side_class
    manifest = base.run(args.unresolved, args.output_dir)
    manifest.update(append_branch_diagnosis(args.output_dir))
    manifest["supersedes"] = (
        "rgi_only_layer1_mechanisms_v1 mechanism labels; V1 numerical curves "
        "remain valid but its short-peak assumption is withdrawn"
    )
    manifest["interpretation_limit"] = (
        "signed read-only routing diagnosis; no parameter family or new OGGM batch is authorized"
    )
    (args.output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
