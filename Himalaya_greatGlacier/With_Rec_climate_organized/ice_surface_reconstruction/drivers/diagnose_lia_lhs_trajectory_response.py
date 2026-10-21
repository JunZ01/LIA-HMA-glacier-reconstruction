"""Quantify dynamic response of all D-layer LHS trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(path: Path) -> dict:
    ts = pd.read_csv(path)
    ts = ts[ts["phase"].eq("forward")].sort_values("year")
    result = {"checkpoint_count": len(ts)}
    for name in ("length_m", "area_m2", "volume_m3"):
        values = ts[name].to_numpy(float)
        result[f"{name}_range"] = float(values.max() - values.min())
        result[f"{name}_net_change"] = float(values[-1] - values[0])
        result[f"{name}_relative_net_change"] = float((values[-1] - values[0]) / values[0]) if values[0] else np.nan
        result[f"{name}_unique_count"] = int(np.unique(values).size)
    result["length_slope_m_per_100y"] = float(np.polyfit(ts["year"], ts["length_m"], 1)[0] * 100.0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.batch_root.resolve()
    base = pd.read_csv(root / "results.csv")
    screen = pd.read_csv(root / "trajectory_bound_screen.csv")
    rows = []
    for row in base.itertuples(index=False):
        path = root / f"system_{int(row.system_id)}" / f"point_{int(row.point):04d}" / "timeseries.csv"
        response = _metrics(path)
        bound = screen[(screen.system_id == row.system_id) & (screen.point == row.point)].iloc[0]
        rows.append({
            "system_id": int(row.system_id),
            "point": int(row.point),
            "original_status": row.status,
            "new_accept": bool(bound.new_accept),
            "e_total": row.e_total,
            **response,
        })
    output = pd.DataFrame(rows)
    output.to_csv(root / "trajectory_response_diagnostics.csv", index=False)
    summary = output.groupby(["system_id", "new_accept"], as_index=False).agg(
        n=("point", "size"),
        length_range_median=("length_m_range", "median"),
        length_net_change_median=("length_m_net_change", "median"),
        area_relative_change_median=("area_m2_relative_net_change", "median"),
        volume_relative_change_median=("volume_m3_relative_net_change", "median"),
        volume_range_median=("volume_m3_range", "median"),
        length_unique_count_median=("length_m_unique_count", "median"),
    )
    summary.to_csv(root / "trajectory_response_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
