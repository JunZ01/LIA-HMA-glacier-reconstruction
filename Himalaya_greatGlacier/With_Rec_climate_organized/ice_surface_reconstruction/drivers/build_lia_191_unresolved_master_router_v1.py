"""Build the current 170-system master queue and taper-mechanism route.

The router is read-only.  It combines the frozen registry, the last complete
controller inventory, and every saved split-forward result.  A 1470-1500
spin-to-forward taper is eligible only when its initial offset direction would
move the LIA peak toward the target while the year-2000 endpoint is already
near tolerance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
PALEO = ROOT / "validation_data/results/paleo_surface_validation_v0"
INVENTORY = (
    PALEO
    / "lia_191_controller_v1/diagnostics/dual_gate_inventory_after_no_formal_N3_v1"
)
FROZEN = PALEO / "lia_191_frozen_curve_registry_v1/frozen_candidates.csv"
OUTPUT = (
    PALEO
    / "lia_191_controller_v1/diagnostics/unresolved_master_router_v1"
)


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _finite(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _class(signed_lia: float, signed_rgi: float) -> str:
    lia = "long" if signed_lia > 0.2 else "short" if signed_lia < -0.2 else "near"
    rgi = "long" if signed_rgi > 0.2 else "short" if signed_rgi < -0.2 else "near"
    return f"peak_{lia}__rgi_{rgi}"


def _split_rows() -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for path in PALEO.glob("lia_191_*split_forward*/**/summary.json"):
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("experiment") != "saved_spinup_independent_forward_delta_t":
            continue
        if payload.get("status") != "ok":
            continue
        domain = payload.get("observation_domain") or {}
        if bool(domain.get("overflow", True)):
            continue
        source_dir = Path(str(payload.get("source_dir", "")))
        source_summary_path = source_dir / "summary.json"
        if not source_summary_path.exists():
            continue
        source = json.loads(source_summary_path.read_text(encoding="utf-8"))
        control = source.get("spinup_control") or {}
        if not bool(control.get("converged")):
            continue
        stability = _finite(control.get("final_stability_fraction"))
        if not math.isfinite(stability) or stability > 0.02:
            continue
        metrics = payload.get("metrics") or {}
        targets = payload.get("targets") or {}
        parameters = payload.get("parameters") or {}
        lia = _finite(targets.get("lia_length_m"))
        rgi = _finite(targets.get("rgi_length_m"))
        separation = _finite(targets.get("separation_m"), abs(lia - rgi))
        peak = _finite(metrics.get("forward_peak_m"))
        endpoint = _finite(metrics.get("forward_2000_m"))
        spin_dt = _finite(parameters.get("spinup_delta_t_c"))
        forward_dt = _finite(parameters.get("forward_delta_t_c"))
        if not all(
            math.isfinite(value)
            for value in (lia, rgi, separation, peak, endpoint, spin_dt, forward_dt)
        ):
            continue
        if separation <= 0:
            continue
        signed_lia = (peak - lia) / separation
        signed_rgi = (endpoint - rgi) / separation
        rows.append(
            {
                "lia_unique_id": int(payload["lia_unique_id"]),
                "rgi_id": str(payload["rgi_id"]),
                "source_kind": "saved_split_forward",
                "experiment": path.parts[path.parts.index("paleo_surface_validation_v0") + 1],
                "output_dir": str(path.parent.resolve()),
                "source_dir": str(source_dir.resolve()),
                "spinup_delta_t_c": spin_dt,
                "forward_delta_t_c": forward_dt,
                "delta_gap_spin_minus_forward_c": spin_dt - forward_dt,
                "spinup_stability_fraction": stability,
                "lia_target_m": lia,
                "rgi_target_m": rgi,
                "separation_m": separation,
                "forward_peak_m": peak,
                "forward_2000_m": endpoint,
                "signed_lia_sep": signed_lia,
                "signed_rgi_sep": signed_rgi,
                "e_lia_sep": abs(signed_lia),
                "e_rgi_sep": abs(signed_rgi),
                "objective_sep": abs(signed_lia) + abs(signed_rgi),
                "e_lia_total": _finite(metrics.get("e_lia_total")),
                "e_rgi_total": _finite(metrics.get("e_rgi_total")),
                "endpoint_class": _class(signed_lia, signed_rgi),
            }
        )
    return rows


def _base_best(chains: pd.DataFrame, frozen_ids: set[int]) -> pd.DataFrame:
    chains = chains.copy()
    chains["lia_unique_id"] = chains["lia_unique_id"].astype(int)
    valid = (
        _bool(chains["stable_chain"])
        & _bool(chains["audit_length_observation_valid"])
        & ~_bool(chains["technical_failure_bool"])
        & ~chains["lia_unique_id"].isin(frozen_ids)
    )
    frame = chains.loc[valid].copy()
    for column in (
        "objective_sep_formal",
        "e_lia_sep_formal",
        "e_rgi_sep_formal",
        "forward_peak_m",
        "forward_2000_m",
        "lia_target_m",
        "rgi_target_m",
        "lia_rgi_separation_m_formal",
        "e_lia_total",
        "e_rgi_total",
        "spinup_stability_fraction",
        "delta_t",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["lia_unique_id", "objective_sep_formal", "task_id"])
    frame = frame.groupby("lia_unique_id", as_index=False).first()
    signed_lia = (
        frame["forward_peak_m"] - frame["lia_target_m"]
    ) / frame["lia_rgi_separation_m_formal"]
    signed_rgi = (
        frame["forward_2000_m"] - frame["rgi_target_m"]
    ) / frame["lia_rgi_separation_m_formal"]
    return pd.DataFrame(
        {
            "lia_unique_id": frame["lia_unique_id"].astype(int),
            "rgi_id": frame["rgi_id"],
            "source_kind": "controller_saved_chain",
            "experiment": frame["score_source"],
            "output_dir": frame["output_dir"],
            "source_dir": frame["output_dir"],
            "spinup_delta_t_c": frame["delta_t"],
            "forward_delta_t_c": frame["delta_t"],
            "delta_gap_spin_minus_forward_c": 0.0,
            "spinup_stability_fraction": frame["spinup_stability_fraction"],
            "lia_target_m": frame["lia_target_m"],
            "rgi_target_m": frame["rgi_target_m"],
            "separation_m": frame["lia_rgi_separation_m_formal"],
            "forward_peak_m": frame["forward_peak_m"],
            "forward_2000_m": frame["forward_2000_m"],
            "signed_lia_sep": signed_lia,
            "signed_rgi_sep": signed_rgi,
            "e_lia_sep": frame["e_lia_sep_formal"],
            "e_rgi_sep": frame["e_rgi_sep_formal"],
            "objective_sep": frame["objective_sep_formal"],
            "e_lia_total": frame["e_lia_total"],
            "e_rgi_total": frame["e_rgi_total"],
            "endpoint_class": [
                _class(float(a), float(b)) for a, b in zip(signed_lia, signed_rgi)
            ],
        }
    )


def main() -> None:
    systems = pd.read_csv(INVENTORY / "system_dual_gate_inventory.csv")
    all_ids = set(systems["lia_unique_id"].astype(int))
    if len(all_ids) != 191:
        raise RuntimeError(f"Expected 191 systems, found {len(all_ids)}")
    frozen_ids = set(pd.read_csv(FROZEN)["lia_unique_id"].astype(int))
    if len(frozen_ids) != 21:
        raise RuntimeError(f"Expected 21 frozen systems, found {len(frozen_ids)}")
    unresolved_ids = all_ids - frozen_ids
    if len(unresolved_ids) != 170:
        raise RuntimeError(f"Expected 170 unresolved systems, found {len(unresolved_ids)}")

    chains = pd.read_csv(INVENTORY / "chain_dual_gate_inventory.csv")
    base = _base_best(chains, frozen_ids)
    split = pd.DataFrame(_split_rows())
    candidates = pd.concat([base, split], ignore_index=True, sort=False)
    candidates = candidates.loc[candidates["lia_unique_id"].isin(unresolved_ids)].copy()
    candidates["objective_sep"] = pd.to_numeric(
        candidates["objective_sep"], errors="coerce"
    )
    current_best = (
        candidates.sort_values(["lia_unique_id", "objective_sep", "source_kind"])
        .groupby("lia_unique_id", as_index=False)
        .first()
    )
    missing = unresolved_ids - set(current_best["lia_unique_id"].astype(int))

    split_only = candidates.loc[candidates["source_kind"].eq("saved_split_forward")].copy()
    gap = pd.to_numeric(split_only["delta_gap_spin_minus_forward_c"], errors="coerce")
    signed_lia = pd.to_numeric(split_only["signed_lia_sep"], errors="coerce")
    e_rgi = pd.to_numeric(split_only["e_rgi_sep"], errors="coerce")
    e_lia_total = pd.to_numeric(split_only["e_lia_total"], errors="coerce")
    e_rgi_total = pd.to_numeric(split_only["e_rgi_total"], errors="coerce")
    direction_helpful = ((signed_lia < -0.2) & (gap < -0.005)) | (
        (signed_lia > 0.2) & (gap > 0.005)
    )
    split_only["taper_direction_helpful"] = direction_helpful
    split_only["taper_eligible"] = (
        direction_helpful
        & (e_rgi <= 0.25)
        & (e_lia_total <= 0.10)
        & (e_rgi_total <= 0.10)
        & (gap.abs() >= 0.005)
    )
    taper = split_only.loc[split_only["taper_eligible"]].copy()
    taper = (
        taper.sort_values(["lia_unique_id", "objective_sep", "e_rgi_sep"])
        .groupby("lia_unique_id", as_index=False)
        .first()
        .sort_values(["objective_sep", "e_rgi_sep"])
        .reset_index(drop=True)
    )
    taper["taper_router_rank"] = range(1, len(taper) + 1)
    taper["selected_representative"] = taper.index < min(3, len(taper))

    current_best["frozen"] = False
    current_best["master_route"] = current_best["endpoint_class"]
    current_best["taper_eligible"] = current_best["lia_unique_id"].isin(
        set(taper["lia_unique_id"].astype(int))
    )
    route_summary = (
        current_best.groupby("master_route", as_index=False)
        .agg(system_count=("lia_unique_id", "size"))
        .sort_values("system_count", ascending=False)
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUTPUT / "all_current_valid_candidates.csv", index=False)
    current_best.to_csv(OUTPUT / "unresolved_master_queue.csv", index=False)
    taper.to_csv(OUTPUT / "taper_eligible_queue.csv", index=False)
    route_summary.to_csv(OUTPUT / "failure_class_summary.csv", index=False)
    summary = {
        "schema_version": 1,
        "total_system_count": len(all_ids),
        "frozen_system_count": len(frozen_ids),
        "unresolved_system_count": len(unresolved_ids),
        "unresolved_with_current_valid_candidate": int(len(current_best)),
        "unresolved_missing_valid_candidate_count": len(missing),
        "unresolved_missing_valid_candidate_ids": sorted(missing),
        "split_forward_valid_candidate_count": int(len(split_only)),
        "taper_eligible_system_count": int(len(taper)),
        "taper_selected_representatives": taper.loc[
            taper["selected_representative"], "lia_unique_id"
        ].astype(int).tolist(),
        "taper_contract": {
            "transition": "linear spinup delta-T to saved forward delta-T from 1470 through 1500",
            "post_1500": "hold saved forward delta-T through 2000",
            "eligibility": (
                "spin-to-forward gap direction moves LIA peak toward target; "
                "RGI separation error <=0.25; both total-length errors <=0.10"
            ),
        },
        "launched_oggm": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(route_summary.to_string(index=False))
    if not taper.empty:
        print(
            taper.loc[
                taper["selected_representative"],
                [
                    "lia_unique_id",
                    "spinup_delta_t_c",
                    "forward_delta_t_c",
                    "signed_lia_sep",
                    "signed_rgi_sep",
                    "objective_sep",
                    "source_dir",
                ],
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
