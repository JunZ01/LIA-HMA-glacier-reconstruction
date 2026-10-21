"""Analytically test formal endpoint intervals against the 50 m length lattice."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DIAGNOSTICS = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / "lia_191_controller_v1" / "diagnostics"
AUDIT = DIAGNOSTICS / "unresolved_reachability_audit_v1" / "unresolved_reachability.csv"
MASTER = DIAGNOSTICS / "unresolved_master_router_v1" / "unresolved_master_queue.csv"
OUTPUT = DIAGNOSTICS / "length_grid_reachability_v1"
DX = 50.0


def lattice_interval(target: float, tolerance: float) -> tuple[int, int, int]:
    first = math.ceil((target - tolerance) / DX)
    last = math.floor((target + tolerance) / DX)
    return first, last, max(0, last - first + 1)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(AUDIT)
    flagged = audit[audit["tolerance_below_50m_grid"].eq(True)].copy()
    master = pd.read_csv(MASTER)
    joined = flagged.merge(
        master[["lia_unique_id", "lia_target_m", "rgi_target_m"]],
        on="lia_unique_id",
        how="left",
        validate="one_to_one",
    )
    if len(joined) != 16 or joined[["lia_target_m", "rgi_target_m"]].isna().any().any():
        raise RuntimeError("expected 16 fully referenced sub-grid-tolerance systems")
    rows: list[dict] = []
    for row in joined.itertuples(index=False):
        tolerance = float(row.formal_tolerance_m)
        lia_first, lia_last, lia_count = lattice_interval(float(row.lia_target_m), tolerance)
        rgi_first, rgi_last, rgi_count = lattice_interval(float(row.rgi_target_m), tolerance)
        rows.append({
            "lia_unique_id": int(row.lia_unique_id),
            "separation_m": float(row.separation_m),
            "formal_tolerance_m": tolerance,
            "lia_target_m": float(row.lia_target_m),
            "lia_first_accepted_grid_m": lia_first * DX if lia_count else math.nan,
            "lia_last_accepted_grid_m": lia_last * DX if lia_count else math.nan,
            "lia_accepted_grid_count": lia_count,
            "rgi_target_m": float(row.rgi_target_m),
            "rgi_first_accepted_grid_m": rgi_first * DX if rgi_count else math.nan,
            "rgi_last_accepted_grid_m": rgi_last * DX if rgi_count else math.nan,
            "rgi_accepted_grid_count": rgi_count,
            "both_endpoints_discretely_representable": lia_count > 0 and rgi_count > 0,
        })
    result = pd.DataFrame(rows).sort_values("lia_unique_id")
    result.to_csv(OUTPUT / "length_grid_reachability.csv", index=False)
    both = int(result["both_endpoints_discretely_representable"].sum())
    summary = {
        "systems": 16,
        "grid_spacing_m": DX,
        "both_endpoints_discretely_representable": both,
        "at_least_one_endpoint_not_representable": 16 - both,
        "decision": (
            "Systems lacking a 50 m lattice point inside either formal endpoint interval "
            "are formally unresolvable at current length discretization; do not scalar-microsearch them."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
