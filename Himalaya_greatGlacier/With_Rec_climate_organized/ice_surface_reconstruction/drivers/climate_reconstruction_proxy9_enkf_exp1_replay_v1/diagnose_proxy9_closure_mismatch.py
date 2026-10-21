"""Localize, without altering sources, the remaining proxy9 posterior mismatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def _period_row(frame: pd.DataFrame, start: int, end: int) -> dict[str, float | int]:
    selected = frame.loc[frame.year.between(start, end)]
    return {
        "start_year": start,
        "end_year": end,
        "year_count": int(len(selected)),
        "mean_max_abs_difference_c": float(selected.mean_max_abs_difference_c.mean()),
        "p95_max_abs_difference_c": float(selected.mean_max_abs_difference_c.quantile(0.95)),
        "maximum_max_abs_difference_c": float(selected.mean_max_abs_difference_c.max()),
        "percentile_maximum_c": float(selected.percentile_max_abs_difference_c.max()),
        "minimum_active_proxy_count": int(selected.active_proxy_count.min()),
        "maximum_active_proxy_count": int(selected.active_proxy_count.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=None)
    args = parser.parse_args()
    root = _project_root()
    output = args.result_root or root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_enkf_exp1_replay_v1"
    data = scipy.io.loadmat(output / "replay_full.mat")
    years = np.asarray(data["run_years"], dtype=int).reshape(-1)
    frame = pd.DataFrame({
        "year": years,
        "active_proxy_count": np.asarray(data["active_proxy_count"], dtype=int).reshape(-1),
        "mean_max_abs_difference_c": np.asarray(data["mean_max_abs_diff_by_year"], dtype=float).reshape(-1),
        "sigma_max_abs_difference_c": np.asarray(data["sigma_max_abs_diff_by_year"], dtype=float).reshape(-1),
        "percentile_max_abs_difference_c": np.asarray(data["prct_max_abs_diff_by_year"], dtype=float).reshape(-1),
    })
    frame.to_csv(output / "closure_by_year.csv", index=False)
    periods = pd.DataFrame([
        _period_row(frame, 1000, 1449),
        _period_row(frame, 1450, 1900),
        _period_row(frame, 1901, 1950),
        _period_row(frame, 1951, 2000),
    ])
    periods.to_csv(output / "closure_by_period.csv", index=False)
    top = frame.nlargest(15, "mean_max_abs_difference_c")
    report = {
        "status": "diagnostic-only; causal proxy attribution remains blocked",
        "top_mean_difference_years": top.to_dict(orient="records"),
        "periods": periods.to_dict(orient="records"),
        "interpretation_boundary": "This table localizes mismatch in the recovered replay relative to the archived posterior. It does not identify a physical proxy error, and it does not authorize K x innovation attribution.",
    }
    (output / "closure_mismatch_diagnosis.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
