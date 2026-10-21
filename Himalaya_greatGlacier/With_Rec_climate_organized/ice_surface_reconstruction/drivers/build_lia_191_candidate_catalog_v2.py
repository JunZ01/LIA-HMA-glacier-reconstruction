"""Compatibility entry point for the reusable saved-chain candidate catalog.

Some early controller score tables record only ``prcp_fac_multiplier`` rather
than the final numeric ``prcp_fac``.  The candidate-catalog logic does not
score that field, but it records parameters for traceability.  This wrapper
normalizes the merged score frame before calling the read-only V1 builder and
then fills actual parameter values from each selected saved time series.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import build_lia_191_candidate_catalog as base


def _load_scores_compatible() -> pd.DataFrame:
    scores = base.load_scores()
    if "prcp_fac" not in scores.columns:
        scores["prcp_fac"] = np.nan
    return scores


def _fill_saved_parameters(output_dir: Path) -> None:
    path = output_dir / "candidate_selection_verified.csv"
    frame = pd.read_csv(path)
    actual: list[dict] = []
    for row in frame.itertuples(index=False):
        series = pd.read_csv(Path(row.timeseries_path), nrows=1)
        actual.append(
            {
                "lia_unique_id": int(row.lia_unique_id),
                "delta_t": pd.to_numeric(series.get("delta_t_c", pd.Series([np.nan])), errors="coerce").iloc[0],
                "melt_f": pd.to_numeric(series.get("melt_f", pd.Series([np.nan])), errors="coerce").iloc[0],
                "prcp_fac": pd.to_numeric(series.get("prcp_fac", pd.Series([np.nan])), errors="coerce").iloc[0],
                "glen_a": pd.to_numeric(series.get("glen_a_pa_3_s_1", pd.Series([np.nan])), errors="coerce").iloc[0],
                "temp_melt": pd.to_numeric(series.get("temp_melt_c", pd.Series([np.nan])), errors="coerce").iloc[0],
            }
        )
    values = pd.DataFrame(actual).set_index("lia_unique_id")
    for column in ("delta_t", "melt_f", "prcp_fac", "glen_a", "temp_melt"):
        frame[column] = frame["lia_unique_id"].map(values[column])
    frame.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing", type=Path, default=base.default_routing_path())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base.controller_root() / "diagnostics" / "candidate_curve_catalog_v2",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base.load_scores = _load_scores_compatible
    manifest = base.build_catalog(args.routing, args.output_dir)
    _fill_saved_parameters(args.output_dir)
    manifest["compatibility_wrapper"] = "normalized missing prcp_fac score columns and filled saved parameter values from source time series"
    (args.output_dir / "catalog_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
