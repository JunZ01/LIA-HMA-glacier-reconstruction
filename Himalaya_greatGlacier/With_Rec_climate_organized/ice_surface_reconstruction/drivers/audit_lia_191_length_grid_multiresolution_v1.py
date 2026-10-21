"""Audit formal endpoint reachability on several uniform length lattices.

This is a pure analytical audit.  It does not import or run OGGM and it does
not alter the formal endpoint tolerance.  The six inputs are the systems that
the existing 50 m audit proved to have no lattice value inside at least one
formal endpoint interval.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DIAGNOSTICS = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
)
SOURCE = DIAGNOSTICS / "length_grid_reachability_v1" / "length_grid_reachability.csv"
OUTPUT = DIAGNOSTICS / "length_grid_multiresolution_v1"

SYSTEM_IDS = (1014, 1038, 2185, 2925, 3075, 3185)
SPACINGS_M = (50.0, 25.0, 20.0, 12.5)
FORMAL_TOLERANCE_FRACTION_OF_SEPARATION = 0.20


def lattice_interval(
    target_m: float, tolerance_m: float, spacing_m: float
) -> dict[str, float | int | bool]:
    """Return the exact intersection of a closed interval and a lattice."""

    first_index = math.ceil((target_m - tolerance_m) / spacing_m)
    last_index = math.floor((target_m + tolerance_m) / spacing_m)
    count = max(0, last_index - first_index + 1)
    if count:
        first_m = first_index * spacing_m
        last_m = last_index * spacing_m
        candidates = {first_m, last_m}
        nearest_m = min(candidates, key=lambda value: (abs(value - target_m), value))
        nearest_error_m = abs(nearest_m - target_m)
    else:
        first_m = math.nan
        last_m = math.nan
        nearest_index = math.floor(target_m / spacing_m)
        candidates = {
            nearest_index * spacing_m,
            (nearest_index + 1) * spacing_m,
        }
        nearest_m = min(candidates, key=lambda value: (abs(value - target_m), value))
        nearest_error_m = abs(nearest_m - target_m)
    return {
        "representable": bool(count),
        "accepted_grid_count": int(count),
        "first_accepted_grid_m": float(first_m),
        "last_accepted_grid_m": float(last_m),
        "nearest_grid_m": float(nearest_m),
        "nearest_grid_error_m": float(nearest_error_m),
    }


def main() -> None:
    source = pd.read_csv(SOURCE)
    selected = source[source["lia_unique_id"].astype(int).isin(SYSTEM_IDS)].copy()
    if set(selected["lia_unique_id"].astype(int)) != set(SYSTEM_IDS):
        missing = sorted(set(SYSTEM_IDS) - set(selected["lia_unique_id"].astype(int)))
        raise RuntimeError(f"missing-declared-50m-obstruction-systems:{missing}")
    if len(selected) != len(SYSTEM_IDS):
        raise RuntimeError("duplicate-declared-50m-obstruction-system")
    if selected["both_endpoints_discretely_representable"].astype(bool).any():
        raise RuntimeError("declared-obstruction-is-representable-at-50m")

    rows: list[dict] = []
    for item in selected.sort_values("lia_unique_id").itertuples(index=False):
        separation_m = abs(float(item.lia_target_m) - float(item.rgi_target_m))
        tolerance_m = FORMAL_TOLERANCE_FRACTION_OF_SEPARATION * separation_m
        if not math.isclose(
            tolerance_m,
            float(item.formal_tolerance_m),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"formal-tolerance-drift:{int(item.lia_unique_id)}")
        for spacing_m in SPACINGS_M:
            lia = lattice_interval(float(item.lia_target_m), tolerance_m, spacing_m)
            rgi = lattice_interval(float(item.rgi_target_m), tolerance_m, spacing_m)
            rows.append(
                {
                    "lia_unique_id": int(item.lia_unique_id),
                    "grid_spacing_m": spacing_m,
                    "lia_target_m": float(item.lia_target_m),
                    "rgi_target_m": float(item.rgi_target_m),
                    "separation_m": separation_m,
                    "formal_tolerance_fraction_of_separation": (
                        FORMAL_TOLERANCE_FRACTION_OF_SEPARATION
                    ),
                    "formal_tolerance_m": tolerance_m,
                    **{f"lia_{key}": value for key, value in lia.items()},
                    **{f"rgi_{key}": value for key, value in rgi.items()},
                    "both_endpoints_representable": bool(
                        lia["representable"] and rgi["representable"]
                    ),
                }
            )

    long = pd.DataFrame(rows).sort_values(["grid_spacing_m", "lia_unique_id"], ascending=[False, True])
    resolution_rows: list[dict] = []
    for spacing_m in SPACINGS_M:
        group = long[long["grid_spacing_m"].eq(spacing_m)]
        pass_ids = sorted(
            group.loc[group["both_endpoints_representable"], "lia_unique_id"]
            .astype(int)
            .tolist()
        )
        blocked_ids = sorted(set(SYSTEM_IDS) - set(pass_ids))
        resolution_rows.append(
            {
                "grid_spacing_m": spacing_m,
                "system_count": len(SYSTEM_IDS),
                "both_endpoints_representable_count": len(pass_ids),
                "both_endpoints_representable_ids": ";".join(map(str, pass_ids)),
                "blocked_count": len(blocked_ids),
                "blocked_ids": ";".join(map(str, blocked_ids)),
                "uniformly_feasible": len(blocked_ids) == 0,
            }
        )
    by_resolution = pd.DataFrame(resolution_rows)
    feasible = by_resolution[by_resolution["uniformly_feasible"]]
    if feasible.empty:
        coarsest_uniform_spacing_m = math.nan
    else:
        # Prefer the coarsest tested spacing that resolves all six; this is the
        # minimum-cost common representation, not a relaxation of tolerance.
        coarsest_uniform_spacing_m = float(feasible["grid_spacing_m"].max())

    OUTPUT.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUTPUT / "endpoint_lattice_reachability_long.csv", index=False)
    by_resolution.to_csv(OUTPUT / "resolution_summary.csv", index=False)

    summary = {
        "schema_version": 1,
        "status": "completed-pure-analytical-audit",
        "source": str(SOURCE.resolve()),
        "system_ids": list(SYSTEM_IDS),
        "tested_grid_spacings_m": list(SPACINGS_M),
        "formal_tolerance_fraction_of_separation": (
            FORMAL_TOLERANCE_FRACTION_OF_SEPARATION
        ),
        "formal_score_changed": False,
        "oggm_run": False,
        "coarsest_tested_uniformly_feasible_spacing_m": (
            coarsest_uniform_spacing_m
        ),
        "resolution_results": resolution_rows,
        "representative_preflight": {
            "lia_unique_id": 1038,
            "grid_spacing_m": coarsest_uniform_spacing_m,
            "reason": (
                "System 1038 is the stricter common-resolution discriminator: "
                "it remains blocked at both 25 m and 20 m, whereas system 2925 "
                "is blocked at 25 m but representable at 20 m. Run 1038 static "
                "geometry and zero-MB preflight first; retain 2925 as the second "
                "confirmation before any historical chain."
            ),
            "historical_chain_released": False,
        },
        "decision": (
            "Do not release the declared 2925 25 m historical chain. Among the "
            "tested spacings, 12.5 m is the coarsest common lattice that contains "
            "accepted values for both endpoints of all six systems."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
