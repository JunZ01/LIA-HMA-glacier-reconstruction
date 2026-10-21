"""Diagnose Layer-1 endpoint brackets for systems routed to Layer 2.

This is read-only. It launches no OGGM task and changes no parameter decision.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .build_lia_191_iterative_loop import load_scores, project_root


ROUTING = (
    project_root()
    / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1"
    / "diagnostics/iterative_loop_v2_after_delta_extension"
    / "core191_iterative_loop_table.csv"
)
OUTPUT = (
    project_root()
    / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1"
    / "diagnostics/iterative_loop_v2_after_delta_extension"
    / "layer2_endpoint_brackets_v1.csv"
)


def signed(model: float, target: float) -> float:
    return (float(model) - float(target)) / float(target)


def row_record(prefix: str, row: pd.Series) -> dict:
    return {
        f"{prefix}_task_id": row["task_id"],
        f"{prefix}_source": row["score_source"],
        f"{prefix}_delta_t": float(row["delta_t"]),
        f"{prefix}_e_lia_total": float(row["e_lia_total"]),
        f"{prefix}_e_rgi_total": float(row["e_rgi_total"]),
        f"{prefix}_peak_signed_total": signed(row["forward_peak_m"], row["lia_target_m"]),
        f"{prefix}_end_signed_total": signed(row["forward_2000_m"], row["rgi_target_m"]),
        f"{prefix}_amplitude_ratio": float(row["amplitude_ratio_to_target"]),
    }


def main() -> None:
    routing = pd.read_csv(ROUTING)
    ids = set(
        routing.loc[
            routing["next_route"].eq("layer2_melt_f_delta_t"), "lia_unique_id"
        ].astype(int)
    )
    scores = load_scores()
    layer1 = scores[
        scores["lia_unique_id"].isin(ids)
        & scores["score_layer"].eq("layer1_delta")
        & scores["stable_chain"]
    ].copy()
    rows: list[dict] = []
    for system_id in sorted(ids):
        group = layer1[layer1["lia_unique_id"].eq(system_id)].copy()
        if group.empty:
            raise RuntimeError(f"system {system_id} has no stable Layer-1 chain")
        best_lia = group.sort_values(
            ["e_lia_total", "e_rgi_total", "score_priority"]
        ).iloc[0]
        best_rgi = group.sort_values(
            ["e_rgi_total", "e_lia_total", "score_priority"]
        ).iloc[0]
        rows.append(
            {
                "lia_unique_id": system_id,
                "rgi_id": best_lia["rgi_id"],
                "stable_layer1_count": int(len(group)),
                "tested_delta_t_min": float(group["delta_t"].min()),
                "tested_delta_t_max": float(group["delta_t"].max()),
                "endpoint_delta_t_low": float(min(best_lia["delta_t"], best_rgi["delta_t"])),
                "endpoint_delta_t_high": float(max(best_lia["delta_t"], best_rgi["delta_t"])),
                "endpoint_delta_t_span": float(abs(best_lia["delta_t"] - best_rgi["delta_t"])),
                **row_record("lia_best", best_lia),
                **row_record("rgi_best", best_rgi),
            }
        )
    frame = pd.DataFrame(rows).sort_values("lia_unique_id")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(frame.to_string(index=False))
    print(f"Wrote: {OUTPUT}")


if __name__ == "__main__":
    main()
