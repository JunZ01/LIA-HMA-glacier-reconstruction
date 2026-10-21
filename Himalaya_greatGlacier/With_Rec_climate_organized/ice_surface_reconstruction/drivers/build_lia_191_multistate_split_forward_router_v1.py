"""Route unresolved systems for an alternate-state split-forward pilot.

This is a read-only router.  It asks whether already-saved, stable spin-up
states with different ice-flow softness (``glen_a``) provide a materially
different initial geometry for the independent forward-forcing experiment.
It never advances OGGM and it excludes every frozen system.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
PALEO = ROOT / "validation_data/results/paleo_surface_validation_v0"
INVENTORY = (
    PALEO
    / "lia_191_controller_v1/diagnostics/dual_gate_inventory_after_no_formal_N3_v1"
    / "chain_dual_gate_inventory.csv"
)
FROZEN = PALEO / "lia_191_frozen_curve_registry_v1/frozen_candidates.csv"
OUTPUT = (
    PALEO
    / "lia_191_controller_v1/diagnostics/multistate_split_forward_router_v1"
)


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _number(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _split_tested_sources() -> set[str]:
    sources: set[str] = set()
    for path in PALEO.glob("lia_191_split_forward_delta_*/**/summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = payload.get("source_dir")
        if source:
            sources.add(str(Path(source).resolve()).lower())
    return sources


def _endpoint_class(signed_lia: float, signed_rgi: float) -> str:
    lia = "long" if signed_lia > 0.2 else "short" if signed_lia < -0.2 else "near"
    rgi = "long" if signed_rgi > 0.2 else "short" if signed_rgi < -0.2 else "near"
    return f"peak_{lia}__rgi_{rgi}"


def _read_state(row: pd.Series, split_sources: set[str]) -> dict | None:
    output_dir = Path(str(row["output_dir"]))
    summary_path = output_dir / "summary.json"
    state_path = output_dir / "spinup_state.pkl.gz"
    if not summary_path.exists() or not state_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if summary.get("status") != "ok":
        return None
    control = summary.get("spinup_control") or {}
    parameters = summary.get("parameters") or {}
    metrics = summary.get("metrics") or {}
    targets = summary.get("targets") or {}
    glen_a = _number(parameters.get("glen_a_pa_3_s_1"))
    melt_f = _number(parameters.get("melt_f"))
    prcp_fac = _number(parameters.get("prcp_fac"))
    delta_t = _number(parameters.get("delta_t_c"), _number(row.get("delta_t")))
    temp_melt = _number(parameters.get("temp_melt_c"), _number(row.get("temp_melt")))
    separation = _number(row.get("lia_rgi_separation_m_formal"))
    peak = _number(row.get("forward_peak_m"))
    endpoint = _number(row.get("forward_2000_m"))
    lia_target = _number(row.get("lia_target_m"), _number(targets.get("lia_length_m")))
    rgi_target = _number(row.get("rgi_target_m"), _number(targets.get("rgi_length_m")))
    if not all(math.isfinite(v) for v in (glen_a, melt_f, prcp_fac, separation, peak, endpoint, lia_target, rgi_target)):
        return None
    signed_lia = (peak - lia_target) / separation
    signed_rgi = (endpoint - rgi_target) / separation
    return {
        "lia_unique_id": int(row["lia_unique_id"]),
        "rgi_id": str(row["rgi_id"]),
        "task_id": str(row["task_id"]),
        "output_dir": str(output_dir.resolve()),
        "score_source": str(row.get("score_source", "")),
        "geometry_variant": str(row.get("geometry_variant", "")),
        "delta_t_c": delta_t,
        "melt_f": melt_f,
        "melt_f_stage2_multiplier": _number(row.get("melt_f_stage2_multiplier"), 1.0),
        "prcp_fac": prcp_fac,
        "prcp_fac_multiplier": _number(row.get("prcp_fac_multiplier"), 1.0),
        "glen_a_pa_3_s_1": glen_a,
        "glen_a_stage2_multiplier": _number(row.get("glen_a_stage2_multiplier"), 1.0),
        "temp_melt_c": temp_melt,
        "spinup_years": int(control.get("actual_years", summary.get("spinup_years", 0))),
        "spinup_stability_fraction": _number(control.get("final_stability_fraction")),
        "spinup_length_m": _number(metrics.get("l_spinup_m")),
        "spinup_area_m2": _number(metrics.get("a_spinup_m2")),
        "spinup_volume_m3": _number(metrics.get("v_spinup_m3")),
        "lia_target_m": lia_target,
        "rgi_target_m": rgi_target,
        "separation_m": separation,
        "forward_peak_m": peak,
        "forward_2000_m": endpoint,
        "signed_lia_sep": signed_lia,
        "signed_rgi_sep": signed_rgi,
        "e_lia_sep": abs(signed_lia),
        "e_rgi_sep": abs(signed_rgi),
        "objective_sep": abs(signed_lia) + abs(signed_rgi),
        "endpoint_class": _endpoint_class(signed_lia, signed_rgi),
        "already_split_forward_tested": str(output_dir.resolve()).lower() in split_sources,
    }


def _distinct_glen_states(group: pd.DataFrame) -> pd.DataFrame:
    work = group.copy()
    work["glen_key"] = work["glen_a_pa_3_s_1"].map(lambda value: f"{value:.12e}")
    return (
        work.sort_values(["objective_sep", "spinup_stability_fraction", "task_id"])
        .groupby("glen_key", as_index=False, sort=True)
        .first()
    )


def main() -> None:
    chains = pd.read_csv(INVENTORY)
    frozen_ids = set(pd.read_csv(FROZEN)["lia_unique_id"].astype(int))
    if len(frozen_ids) != 20:
        raise RuntimeError(f"Expected 20 frozen systems, found {len(frozen_ids)}")
    chains["lia_unique_id"] = chains["lia_unique_id"].astype(int)
    stable = _bool(chains["stable_chain"])
    valid = _bool(chains["audit_length_observation_valid"])
    technical = _bool(chains["technical_failure_bool"])
    subset = chains.loc[
        stable & valid & ~technical & ~chains["lia_unique_id"].isin(frozen_ids)
    ].copy()

    split_sources = _split_tested_sources()
    state_rows = []
    for _, row in subset.iterrows():
        parsed = _read_state(row, split_sources)
        if parsed is not None:
            state_rows.append(parsed)
    states = pd.DataFrame(state_rows)
    if states.empty:
        raise RuntimeError("No reusable stable states found")

    # Exact reruns of the same saved state are reduced to the best formal row.
    state_signature = [
        "lia_unique_id",
        "delta_t_c",
        "melt_f",
        "prcp_fac",
        "glen_a_pa_3_s_1",
        "temp_melt_c",
        "geometry_variant",
    ]
    states = (
        states.sort_values(["objective_sep", "spinup_stability_fraction", "task_id"])
        .drop_duplicates(state_signature, keep="first")
        .reset_index(drop=True)
    )

    routed_rows: list[dict] = []
    queue_rows: list[dict] = []
    for system_id, group in states.groupby("lia_unique_id", sort=True):
        glen_states = _distinct_glen_states(group)
        if len(glen_states) < 2:
            continue
        glen_states = glen_states.sort_values(["objective_sep", "glen_a_pa_3_s_1"])
        best = glen_states.iloc[0]
        alternatives = glen_states.iloc[1:].copy()
        untested = alternatives.loc[~alternatives["already_split_forward_tested"].astype(bool)]
        if untested.empty:
            continue
        alternative = untested.sort_values("objective_sep").iloc[0]
        volumes = pd.to_numeric(glen_states["spinup_volume_m3"], errors="coerce")
        areas = pd.to_numeric(glen_states["spinup_area_m2"], errors="coerce")
        lengths = pd.to_numeric(glen_states["spinup_length_m"], errors="coerce")
        volume_spread = (volumes.max() - volumes.min()) / volumes.median()
        area_spread = (areas.max() - areas.min()) / areas.median()
        length_spread = (lengths.max() - lengths.min()) / best["separation_m"]
        routed_rows.append(
            {
                "lia_unique_id": int(system_id),
                "rgi_id": best["rgi_id"],
                "distinct_glen_state_count": int(len(glen_states)),
                "reusable_state_count": int(len(group)),
                "best_state_task_id": best["task_id"],
                "best_state_output_dir": best["output_dir"],
                "best_state_glen_a": best["glen_a_pa_3_s_1"],
                "best_state_glen_multiplier": best["glen_a_stage2_multiplier"],
                "best_state_delta_t": best["delta_t_c"],
                "best_state_objective": best["objective_sep"],
                "best_state_endpoint_class": best["endpoint_class"],
                "alternate_task_id": alternative["task_id"],
                "alternate_output_dir": alternative["output_dir"],
                "alternate_glen_a": alternative["glen_a_pa_3_s_1"],
                "alternate_glen_multiplier": alternative["glen_a_stage2_multiplier"],
                "alternate_delta_t": alternative["delta_t_c"],
                "alternate_objective": alternative["objective_sep"],
                "alternate_endpoint_class": alternative["endpoint_class"],
                "spinup_volume_spread_fraction": volume_spread,
                "spinup_area_spread_fraction": area_spread,
                "spinup_length_spread_per_target_separation": length_spread,
                "best_state_already_split_tested": bool(best["already_split_forward_tested"]),
                "alternate_state_already_split_tested": bool(alternative["already_split_forward_tested"]),
            }
        )
        for rank, row in enumerate(glen_states.itertuples(index=False), start=1):
            queue_rows.append(
                {
                    "lia_unique_id": int(system_id),
                    "state_rank": rank,
                    "task_id": row.task_id,
                    "output_dir": row.output_dir,
                    "glen_a_pa_3_s_1": row.glen_a_pa_3_s_1,
                    "glen_a_stage2_multiplier": row.glen_a_stage2_multiplier,
                    "delta_t_c": row.delta_t_c,
                    "melt_f_stage2_multiplier": row.melt_f_stage2_multiplier,
                    "prcp_fac_multiplier": row.prcp_fac_multiplier,
                    "temp_melt_c": row.temp_melt_c,
                    "spinup_stability_fraction": row.spinup_stability_fraction,
                    "spinup_length_m": row.spinup_length_m,
                    "spinup_area_m2": row.spinup_area_m2,
                    "spinup_volume_m3": row.spinup_volume_m3,
                    "signed_lia_sep": row.signed_lia_sep,
                    "signed_rgi_sep": row.signed_rgi_sep,
                    "objective_sep": row.objective_sep,
                    "endpoint_class": row.endpoint_class,
                    "already_split_forward_tested": row.already_split_forward_tested,
                }
            )

    router = pd.DataFrame(routed_rows)
    if router.empty:
        raise RuntimeError("No unresolved system has reusable distinct glen-a states")
    router["near_gate"] = router["best_state_objective"] <= 1.0
    router = router.sort_values(
        ["near_gate", "best_state_objective", "spinup_volume_spread_fraction"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    router["router_rank"] = np.arange(1, len(router) + 1)

    # Three representatives, preferring different signed-endpoint classes.
    selected: list[int] = []
    used_classes: set[str] = set()
    for idx, row in router.iterrows():
        endpoint_class = str(row["best_state_endpoint_class"])
        if endpoint_class not in used_classes:
            selected.append(idx)
            used_classes.add(endpoint_class)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        selected.extend(idx for idx in router.index if idx not in selected)
        selected = selected[:3]
    router["selected_representative"] = router.index.isin(selected)

    queue = pd.DataFrame(queue_rows)
    selected_ids = set(router.loc[router["selected_representative"], "lia_unique_id"].astype(int))
    queue["selected_system"] = queue["lia_unique_id"].astype(int).isin(selected_ids)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    states.to_csv(OUTPUT / "reusable_saved_states.csv", index=False)
    router.to_csv(OUTPUT / "system_router.csv", index=False)
    queue.to_csv(OUTPUT / "glen_state_queue.csv", index=False)
    summary = {
        "schema_version": 1,
        "question": (
            "Can an alternate already-converged glen-a spin-up geometry, followed by "
            "independent forward delta-T, resolve endpoint tradeoffs closed under one state?"
        ),
        "source_inventory": str(INVENTORY),
        "frozen_system_count_excluded": len(frozen_ids),
        "reusable_state_count": int(len(states)),
        "eligible_multistate_system_count": int(len(router)),
        "near_gate_system_count": int(router["near_gate"].sum()),
        "selected_representatives": router.loc[
            router["selected_representative"], "lia_unique_id"
        ].astype(int).tolist(),
        "selected_classes": router.loc[
            router["selected_representative"], "best_state_endpoint_class"
        ].astype(str).tolist(),
        "launched_oggm": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        router.loc[
            router["selected_representative"],
            [
                "lia_unique_id",
                "best_state_endpoint_class",
                "best_state_objective",
                "alternate_glen_multiplier",
                "alternate_objective",
                "spinup_volume_spread_fraction",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
