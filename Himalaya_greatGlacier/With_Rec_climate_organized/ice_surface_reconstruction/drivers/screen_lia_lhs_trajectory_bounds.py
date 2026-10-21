"""Screen D-layer LHS trajectories against the LIA-RGI length envelope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _screen_trial(summary_path: Path, tolerance: float) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    targets = summary["targets"]
    lia = float(targets["lia_length_m"])
    rgi = float(targets["rgi_length_m"])
    amplitude = abs(lia - rgi)
    lower = min(lia, rgi) - tolerance * amplitude
    upper = max(lia, rgi) + tolerance * amplitude
    timeseries = pd.read_csv(summary_path.parent / "timeseries.csv")
    forward = timeseries[timeseries["phase"].eq("forward")].copy()
    below = forward["length_m"] < lower
    above = forward["length_m"] > upper
    violation = below | above
    if below.any() and above.any():
        direction = "both"
    elif below.any():
        direction = "below"
    elif above.any():
        direction = "above"
    else:
        direction = "none"
    first_year = int(forward.loc[violation, "year"].iloc[0]) if violation.any() else np.nan
    maximum_below = max(0.0, lower - float(forward["length_m"].min()))
    maximum_above = max(0.0, float(forward["length_m"].max()) - upper)
    curve_pass = not violation.any()
    original_status = str(summary.get("status", "crash"))
    return {
        "system_id": int(summary["lia_unique_id"]),
        "point": int(summary_path.parent.name.split("_")[1]),
        "original_status": original_status,
        "curve_within_band": curve_pass,
        "new_accept": original_status == "ok" and curve_pass,
        "violation_direction": direction,
        "first_violation_checkpoint_year": first_year,
        "violation_checkpoint_count": int(violation.sum()),
        "forward_checkpoint_count": int(len(forward)),
        "maximum_below_m": maximum_below,
        "maximum_above_m": maximum_above,
        "forward_min_length_m": float(forward["length_m"].min()),
        "forward_max_length_m": float(forward["length_m"].max()),
        "band_lower_m": lower,
        "band_upper_m": upper,
        "band_amplitude_m": amplitude,
        "e_lia": float(metrics["e_lia"]),
        "e_2000": float(metrics["e_2000"]),
        "e_total": float(metrics["e_total"]),
        "l_spinup_m": float(metrics["l_spinup_m"]),
        "l_2000_m": float(metrics["l_2000_m"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=0.20)
    args = parser.parse_args()
    root = args.batch_root.resolve()
    rows = [_screen_trial(path, args.tolerance) for path in sorted(root.glob("system_*/point_*/summary.json"))]
    screened = pd.DataFrame(rows).sort_values(["system_id", "point"])
    screened.to_csv(root / "trajectory_bound_screen.csv", index=False)
    accepted = screened[screened["new_accept"]].copy()
    top = (
        accepted.sort_values(["system_id", "e_total"])
        .groupby("system_id", as_index=False, group_keys=False)
        .head(10)
    )
    top.to_csv(root / "trajectory_bound_top10.csv", index=False)
    diagnostics = screened.groupby("system_id", as_index=False).agg(
        total=("point", "size"),
        original_ok=("original_status", lambda s: int((s == "ok").sum())),
        curve_within_band=("curve_within_band", "sum"),
        newly_accepted=("new_accept", "sum"),
        below_count=("violation_direction", lambda s: int(s.isin(["below", "both"]).sum())),
        above_count=("violation_direction", lambda s: int(s.isin(["above", "both"]).sum())),
        min_e_total_accepted=("e_total", lambda s: float(s[screened.loc[s.index, "new_accept"]].min()) if screened.loc[s.index, "new_accept"].any() else np.nan),
        max_below_m=("maximum_below_m", "max"),
        max_above_m=("maximum_above_m", "max"),
    )
    diagnostics["new_accept_rate"] = diagnostics["newly_accepted"] / diagnostics["total"]
    diagnostics.to_csv(root / "trajectory_bound_diagnostics.csv", index=False)
    print(diagnostics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
