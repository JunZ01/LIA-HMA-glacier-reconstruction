"""Read-only branch-position diagnosis for N4 neither-endpoint systems."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
RESULTS = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
INVENTORY = (
    RESULTS
    / "lia_191_controller_v1"
    / "diagnostics"
    / "dual_gate_inventory_after_no_formal_N3_v1"
)
OUTPUT = (
    RESULTS
    / "lia_191_controller_v1"
    / "diagnostics"
    / "no_formal_N4_branch_diagnosis_v1"
)


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _position(minimum: float, maximum: float) -> str:
    if maximum < -0.10:
        return "always_short"
    if minimum > 0.10:
        return "always_long"
    if minimum < -0.10 and maximum > 0.10:
        return "straddles_without_near_chain"
    return "touches_10pct_band"


def main() -> None:
    systems = pd.read_csv(INVENTORY / "system_dual_gate_inventory.csv")
    chains = pd.read_csv(INVENTORY / "chain_dual_gate_inventory.csv")
    no_formal = set(
        systems.loc[
            systems["formal_inventory_route"].eq("no_formal_pass"), "lia_unique_id"
        ].astype(int)
    )
    chains["lia_unique_id"] = chains["lia_unique_id"].astype(int)
    chains["stable_chain"] = _bool(chains["stable_chain"])
    valid = chains[
        chains["lia_unique_id"].isin(no_formal) & chains["stable_chain"]
    ].copy()
    valid["lia_signed_total"] = (
        valid["forward_peak_m"] - valid["lia_target_m"]
    ) / valid["lia_target_m"]
    valid["rgi_signed_total"] = (
        valid["forward_2000_m"] - valid["rgi_target_m"]
    ) / valid["rgi_target_m"]
    reach = valid.groupby("lia_unique_id").agg(
        lia_reachable=("e_lia_total", lambda s: bool((pd.to_numeric(s, errors="coerce") <= 0.10).any())),
        rgi_reachable=("e_rgi_total", lambda s: bool((pd.to_numeric(s, errors="coerce") <= 0.10).any())),
    )
    n4_ids = set(reach.index[~reach["lia_reachable"] & ~reach["rgi_reachable"]])
    rows = []
    for system_id, group in valid[valid["lia_unique_id"].isin(n4_ids)].groupby(
        "lia_unique_id"
    ):
        lia_min = float(group["lia_signed_total"].min())
        lia_max = float(group["lia_signed_total"].max())
        rgi_min = float(group["rgi_signed_total"].min())
        rgi_max = float(group["rgi_signed_total"].max())
        lia_position = _position(lia_min, lia_max)
        rgi_position = _position(rgi_min, rgi_max)
        if lia_position == rgi_position:
            joint = f"both_{lia_position}"
        else:
            joint = f"lia_{lia_position}__rgi_{rgi_position}"
        best = group.sort_values("objective_sep_formal").iloc[0]
        history = "|".join(
            sorted(
                set(group["score_source"].dropna().astype(str))
                | set(group["score_layer"].dropna().astype(str))
            )
        ).lower()
        rows.append(
            {
                "lia_unique_id": int(system_id),
                "rgi_id": best["rgi_id"],
                "joint_position_class": joint,
                "lia_position_class": lia_position,
                "rgi_position_class": rgi_position,
                "lia_signed_total_min": lia_min,
                "lia_signed_total_max": lia_max,
                "rgi_signed_total_min": rgi_min,
                "rgi_signed_total_max": rgi_max,
                "stable_chain_count": int(len(group)),
                "best_objective_sep": float(best["objective_sep_formal"]),
                "best_delta_t": float(best["delta_t"]),
                "best_amplitude_ratio": float(best["amplitude_ratio_to_target"]),
                "tested_melt_f": "melt" in history,
                "tested_glen_a": "glen" in history,
                "tested_prcp_fac": "prcp" in history,
                "tested_temp_melt": "temp_melt" in history,
                "history_sources": history,
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["joint_position_class", "best_objective_sep", "lia_unique_id"]
    )
    if len(table) != 35:
        raise RuntimeError(f"expected 35 N4 systems, found {len(table)}")
    class_summary = (
        table.groupby("joint_position_class", as_index=False)
        .agg(
            system_count=("lia_unique_id", "size"),
            median_best_objective=("best_objective_sep", "median"),
            median_amplitude_ratio=("best_amplitude_ratio", "median"),
            tested_melt_f_count=("tested_melt_f", "sum"),
            tested_glen_a_count=("tested_glen_a", "sum"),
        )
        .sort_values("system_count", ascending=False)
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "system_branch_position.csv", index=False)
    class_summary.to_csv(OUTPUT / "class_summary.csv", index=False)
    summary = {
        "schema_version": 1,
        "launched_oggm": False,
        "N4_system_count": int(len(table)),
        "class_counts": {
            row.joint_position_class: int(row.system_count)
            for row in class_summary.itertuples(index=False)
        },
        "decision_scope": (
            "Read-only routing only; no parameter family is authorized by this output."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(class_summary.to_string(index=False))


if __name__ == "__main__":
    main()
