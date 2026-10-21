"""Build the C1-C3 pilots from the full history-aware classification table.

V2 retains the V1 experiment design and repairs the representative queue by
including the exact Stage-2-relative anchor multipliers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import build_lia_191_class_representative_pilots_v1 as _v1


CLASSIFICATION = (
    _v1.PROJECT_ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
    / "total_length_only_batch_router_v1"
    / "system_classification.csv"
)
QUEUE_V2 = CLASSIFICATION.parent / "representative_pilot_queue_v2.csv"


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def main() -> None:
    table = pd.read_csv(CLASSIFICATION)
    selected = table.loc[
        _bool(table["is_representative"])
        & table["batch_state"].eq("eligible_class_pilot")
    ].copy()
    if len(selected) != 18:
        raise RuntimeError(f"Expected 18 representatives, found {len(selected)}")
    selected.to_csv(QUEUE_V2, index=False)
    old = _v1.QUEUE
    try:
        _v1.QUEUE = Path(QUEUE_V2)
        _v1.main()
    finally:
        _v1.QUEUE = old


if __name__ == "__main__":
    main()
