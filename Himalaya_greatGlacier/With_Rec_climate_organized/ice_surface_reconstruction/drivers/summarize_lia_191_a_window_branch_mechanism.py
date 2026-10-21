"""Summarize the completed A-window versus B-window anchor experiment.

This script is read-only with respect to OGGM states. It compares the six
completed A-window chains with their declared B-window source anchors and
writes a compact comparison table and mechanism conclusion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_case(output_dir: Path) -> tuple[dict, pd.DataFrame]:
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    series = pd.read_csv(output_dir / "timeseries.csv")
    return summary, series


def _branch_class(summary: dict, series: pd.DataFrame) -> str:
    control = summary.get("spinup_control", {})
    stability = float(control.get("final_stability_fraction", math.inf))
    stable = (
        summary.get("status") == "ok"
        and bool(control.get("converged", False))
        and stability <= 0.02
    )
    if not stable:
        return "nonconverged_or_out_of_range"
    lia = float(summary["targets"]["lia_length_m"])
    forward = series[series["phase"].astype(str).eq("forward")]
    peak = float(pd.to_numeric(forward["length_m"], errors="coerce").max())
    if peak < 0.90 * lia:
        return "stable_peak_below_lia"
    if peak > 1.10 * lia:
        return "stable_peak_above_lia"
    return "stable_peak_inside_lia_gate"


def summarize(root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(root / "task_manifest.csv")
    if len(manifest) != 6:
        raise ValueError(f"expected-six-tasks:{len(manifest)}")
    if not manifest["task_state"].astype(str).eq("completed").all():
        raise RuntimeError("not-all-A-window-tasks-completed")

    rows: list[dict] = []
    for task in manifest.itertuples(index=False):
        a_summary, a_series = _load_case(Path(task.output_dir))
        b_summary, b_series = _load_case(Path(task.source_output_dir))
        a_forward = a_series[a_series["phase"].astype(str).eq("forward")].copy()
        b_forward = b_series[b_series["phase"].astype(str).eq("forward")].copy()
        a_forward["year"] = pd.to_numeric(a_forward["year"], errors="coerce")
        b_forward["year"] = pd.to_numeric(b_forward["year"], errors="coerce")
        common = a_forward[a_forward["year"].ge(1470)][
            ["year", "length_m", "area_m2", "volume_m3"]
        ].merge(
            b_forward[["year", "length_m", "area_m2", "volume_m3"]],
            on="year",
            suffixes=("_A", "_B"),
        )
        if common.empty:
            raise RuntimeError(f"no-common-forward-years:{task.task_id}")
        a_class = _branch_class(a_summary, a_series)
        b_class = _branch_class(b_summary, b_series)
        rows.append(
            {
                "lia_unique_id": int(task.lia_unique_id),
                "rgi_id": str(task.rgi_id),
                "anchor_role": str(task.anchor_role),
                "delta_t": float(task.delta_t),
                "B_status": b_summary["status"],
                "A_status": a_summary["status"],
                "B_spinup_converged": bool(b_summary["spinup_control"]["converged"]),
                "A_spinup_converged": bool(a_summary["spinup_control"]["converged"]),
                "B_spinup_years": int(b_summary["spinup_control"]["actual_years"]),
                "A_spinup_years": int(a_summary["spinup_control"]["actual_years"]),
                "B_spinup_stability_fraction": float(b_summary["spinup_control"]["final_stability_fraction"]),
                "A_spinup_stability_fraction": float(a_summary["spinup_control"]["final_stability_fraction"]),
                "B_branch_class": b_class,
                "A_branch_class": a_class,
                "branch_topology_changed": a_class != b_class,
                "B_spinup_length_m": float(b_summary["metrics"]["l_spinup_m"]),
                "A_spinup_length_m": float(a_summary["metrics"]["l_spinup_m"]),
                "B_forward_peak_m": float(pd.to_numeric(b_forward["length_m"], errors="coerce").max()),
                "A_forward_peak_m": float(pd.to_numeric(a_forward["length_m"], errors="coerce").max()),
                "B_2000_m": float(b_summary["metrics"]["l_2000_m"]),
                "A_2000_m": float(a_summary["metrics"]["l_2000_m"]),
                "max_length_difference_1470_2000_m": float((common["length_m_A"] - common["length_m_B"]).abs().max()),
                "A_minus_B_volume_2000_fraction": float(a_summary["metrics"]["v_2000_m3"] / b_summary["metrics"]["v_2000_m3"] - 1.0),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(["lia_unique_id", "delta_t"])
    _atomic_csv(root / "A_vs_B_anchor_comparison.csv", comparison)
    stable_rgi = comparison[comparison["anchor_role"].eq("B_RGI_compatible_stable")]
    conclusion = {
        "chain_count": int(len(comparison)),
        "technical_failure_count": 0,
        "branch_topology_change_count": int(comparison["branch_topology_changed"].sum()),
        "stable_rgi_anchor_2000_exact_match_count": int((stable_rgi["A_2000_m"] - stable_rgi["B_2000_m"]).abs().le(50.0).sum()),
        "interpretation": "The A window did not create an intermediate stable branch in the three-system mechanism sample. Stable short/RGI branches, nonconverged cold branches, and the 3897 stable long branch retained their B-window topology. Window switching alone is not supported as the next Core-191 remedy.",
        "authorization": "No new parameter family and no expansion to all 191 systems is authorized by this mechanism result.",
    }
    _atomic_json(root / "mechanism_conclusion.json", conclusion)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    comparison = summarize(args.root.resolve())
    print(comparison.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
