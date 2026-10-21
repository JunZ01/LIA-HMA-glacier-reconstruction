"""Build a read-only reachability router for the 99 no-formal-pass systems."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
INPUT = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_191_controller_v1"
    / "diagnostics"
    / "dual_gate_inventory_after_class_round2_v1"
)
OUTPUT = INPUT.parent / "no_formal_pass_router_v1"


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _tested(text: str, token: str) -> bool:
    return token in text.lower()


def _class_name(lia_reachable: bool, rgi_reachable: bool) -> str:
    if lia_reachable and rgi_reachable:
        return "N1_endpoints_separately_reachable"
    if rgi_reachable:
        return "N2_rgi_only_reachable"
    if lia_reachable:
        return "N3_lia_only_reachable"
    return "N4_neither_endpoint_reachable"


def _representatives(group: pd.DataFrame) -> pd.DataFrame:
    n = min(5, max(1, int(np.ceil(len(group) / 10))))
    if len(group) <= n:
        out = group.copy()
        out["representative_rank"] = np.arange(1, len(out) + 1)
        return out
    features = group[
        [
            "best_signed_lia_sep",
            "best_signed_rgi_sep",
            "best_amplitude_ratio",
            "log_separation_m",
        ]
    ].astype(float).to_numpy()
    center = np.nanmedian(features, axis=0)
    scale = np.nanmedian(np.abs(features - center), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    z = (features - center) / scale
    selected = [int(np.argmin(np.linalg.norm(z, axis=1)))]
    while len(selected) < n:
        candidates = [idx for idx in range(len(group)) if idx not in selected]
        distances = [
            min(float(np.linalg.norm(z[idx] - z[chosen])) for chosen in selected)
            for idx in candidates
        ]
        selected.append(candidates[int(np.argmax(distances))])
    out = group.iloc[selected].copy()
    out["representative_rank"] = np.arange(1, len(out) + 1)
    return out


def main() -> None:
    systems = pd.read_csv(INPUT / "system_dual_gate_inventory.csv")
    chains = pd.read_csv(INPUT / "chain_dual_gate_inventory.csv")
    target_ids = set(
        systems.loc[
            systems["formal_inventory_route"].eq("no_formal_pass"),
            "lia_unique_id",
        ].astype(int)
    )
    if len(target_ids) != 99:
        raise RuntimeError(f"Expected 99 no_formal_pass systems, found {len(target_ids)}")

    chains["lia_unique_id"] = chains["lia_unique_id"].astype(int)
    chains["stable_chain"] = _bool(chains["stable_chain"])
    chains["observation_domain_overflow"] = _bool(
        chains["observation_domain_overflow"]
    )
    subset = chains.loc[chains["lia_unique_id"].isin(target_ids)].copy()
    rows = []
    for system_id, group in subset.groupby("lia_unique_id"):
        stable = group.loc[group["stable_chain"]].copy()
        if stable.empty:
            raise RuntimeError(f"no_formal_pass system {system_id} has no stable chain")
        stable["e_lia_total"] = pd.to_numeric(stable["e_lia_total"], errors="coerce")
        stable["e_rgi_total"] = pd.to_numeric(stable["e_rgi_total"], errors="coerce")
        stable["objective_sep_formal"] = pd.to_numeric(
            stable["objective_sep_formal"], errors="coerce"
        )
        stable = stable.sort_values(
            ["objective_sep_formal", "score_priority"], ascending=[True, False]
        )
        best = stable.iloc[0]
        lia_reachable = bool((stable["e_lia_total"] <= 0.10).any())
        rgi_reachable = bool((stable["e_rgi_total"] <= 0.10).any())
        history = "|".join(
            sorted(
                set(group["score_source"].dropna().astype(str))
                | set(group["score_layer"].dropna().astype(str))
            )
        ).lower()
        separation = float(best["lia_rgi_separation_m_formal"])
        overflow_rate = float(group["observation_domain_overflow"].mean())
        response_class = _class_name(lia_reachable, rgi_reachable)
        rows.append(
            {
                "lia_unique_id": int(system_id),
                "rgi_id": best["rgi_id"],
                "reachability_class": response_class,
                "lia_total_endpoint_reachable": lia_reachable,
                "rgi_total_endpoint_reachable": rgi_reachable,
                "stable_chain_count": int(len(stable)),
                "saved_chain_count": int(len(group)),
                "overflow_chain_fraction": overflow_rate,
                "best_task_id": best["task_id"],
                "best_output_dir": best["output_dir"],
                "best_delta_t": best["delta_t"],
                "best_melt_f_stage2_multiplier": best.get(
                    "melt_f_stage2_multiplier", np.nan
                ),
                "best_glen_a_stage2_multiplier": best.get(
                    "glen_a_stage2_multiplier", np.nan
                ),
                "best_prcp_fac_multiplier": best.get(
                    "prcp_fac_multiplier", np.nan
                ),
                "best_temp_melt": best.get("temp_melt", np.nan),
                "best_signed_lia_sep": (
                    float(best["forward_peak_m"]) - float(best["lia_target_m"])
                )
                / separation,
                "best_signed_rgi_sep": (
                    float(best["forward_2000_m"]) - float(best["rgi_target_m"])
                )
                / separation,
                "best_e_lia_sep": best["e_lia_sep_formal"],
                "best_e_rgi_sep": best["e_rgi_sep_formal"],
                "best_objective_sep": best["objective_sep_formal"],
                "best_amplitude_ratio": best["amplitude_ratio_to_target"],
                "target_separation_m": separation,
                "log_separation_m": np.log10(max(separation, 1.0)),
                "geometry_variant": best.get("geometry_variant", ""),
                "tested_melt_f": _tested(history, "melt"),
                "tested_glen_a": _tested(history, "glen"),
                "tested_prcp_fac": _tested(history, "prcp"),
                "tested_temp_melt": _tested(history, "temp_melt"),
                "history_sources": history,
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["reachability_class", "best_objective_sep", "lia_unique_id"]
    )
    reps = []
    for _, group in table.groupby("reachability_class", sort=True):
        reps.append(_representatives(group.reset_index(drop=True)))
    representatives = pd.concat(reps, ignore_index=True)
    table["is_representative"] = table["lia_unique_id"].isin(
        set(representatives["lia_unique_id"])
    )

    summary_rows = []
    for class_name, group in table.groupby("reachability_class", sort=True):
        rep_ids = representatives.loc[
            representatives["reachability_class"].eq(class_name), "lia_unique_id"
        ].astype(int).tolist()
        summary_rows.append(
            {
                "reachability_class": class_name,
                "system_count": int(len(group)),
                "median_best_objective_sep": float(group["best_objective_sep"].median()),
                "median_overflow_chain_fraction": float(
                    group["overflow_chain_fraction"].median()
                ),
                "tested_melt_f_count": int(group["tested_melt_f"].sum()),
                "tested_glen_a_count": int(group["tested_glen_a"].sum()),
                "tested_prcp_fac_count": int(group["tested_prcp_fac"].sum()),
                "tested_temp_melt_count": int(group["tested_temp_melt"].sum()),
                "representative_ids": "|".join(map(str, rep_ids)),
            }
        )
    class_summary = pd.DataFrame(summary_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "system_reachability_classification.csv", index=False)
    representatives.to_csv(OUTPUT / "representative_queue.csv", index=False)
    class_summary.to_csv(OUTPUT / "class_summary.csv", index=False)
    summary = {
        "schema_version": 1,
        "source_inventory": str(INPUT),
        "no_formal_pass_system_count": int(len(table)),
        "class_counts": {
            row.reachability_class: int(row.system_count)
            for row in class_summary.itertuples(index=False)
        },
        "representative_count": int(len(representatives)),
        "representatives": {
            row.reachability_class: [
                int(value)
                for value in str(row.representative_ids).split("|")
                if value
            ]
            for row in class_summary.itertuples(index=False)
        },
        "launched_oggm": False,
    }
    (OUTPUT / "router_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(class_summary.to_string(index=False))


if __name__ == "__main__":
    main()
