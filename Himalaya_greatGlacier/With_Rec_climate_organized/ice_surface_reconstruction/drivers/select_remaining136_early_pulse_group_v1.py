"""Select the independent representative for the fixed early-pulse family.

This is a read-only evidence rescreen.  It never starts OGGM and it does not
modify any glacier state.  The selected class has a valid steady/historical
domain, both total-length gates and the modern separation gate already pass,
while the LIA peak is just above the frozen separation-normalized upper gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
VAL = ROOT / "validation_data/results/paleo_surface_validation_v0"
COLLECTOR = VAL / "lia_191_latest_historical_after_accepted55_v1/candidate_results.csv"
ACCEPTED = VAL / "lia_191_accepted_55_registry_v1/accepted_55_curve_metrics.csv"
OUT = VAL / "lia_191_remaining136_early_pulse_group_selection_v1"
STATE = ROOT / "w665steady5180_dtm0p25_v1/lia_equilibrium_state_50y.pkl.gz"


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    accepted = pd.read_csv(ACCEPTED)
    accepted_ids = set(pd.to_numeric(accepted["system_id"]).astype(int))
    if len(accepted_ids) != 55:
        raise RuntimeError(f"accepted-registry-count:{len(accepted_ids)}")

    frame = pd.read_csv(COLLECTOR)
    frame["system_id"] = pd.to_numeric(frame["system_id"], errors="coerce")
    numeric = [
        "historical_peak_length_m", "lia_registered_length_m",
        "rgi_registered_length_m", "e_lia_total_formal",
        "e_rgi_total_formal", "e_lia_separation_formal",
        "e_rgi_separation_formal",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    valid = (
        frame["collection_state"].eq("complete")
        & ~frame["system_id"].isin(accepted_ids)
        & frame["stability_pass"].map(truth)
        & frame["steady_domain_pass"].map(truth)
        & frame["historical_domain_pass"].map(truth)
        & frame["e_lia_total_formal"].le(0.10)
        & frame["e_rgi_total_formal"].le(0.10)
        & frame["e_rgi_separation_formal"].le(0.20)
        & frame["e_lia_separation_formal"].gt(0.20)
    )
    candidates = frame.loc[valid].copy()
    separation = (candidates["lia_registered_length_m"] - candidates["rgi_registered_length_m"]).abs()
    candidates["formal_peak_upper_m"] = pd.concat(
        [
            candidates["lia_registered_length_m"] * 1.10,
            candidates["lia_registered_length_m"] + 0.20 * separation,
        ],
        axis=1,
    ).min(axis=1)
    candidates["peak_excess_m"] = (
        candidates["historical_peak_length_m"] - candidates["formal_peak_upper_m"]
    )
    candidates = candidates.loc[candidates["peak_excess_m"].gt(0)].copy()
    candidates = (
        candidates.sort_values(["peak_excess_m", "e_lia_separation_formal", "candidate_root"])
        .groupby("system_id", as_index=False)
        .first()
        .sort_values(["peak_excess_m", "system_id"])
    )
    if candidates.empty:
        raise RuntimeError("no-early-pulse-analogue")
    selected = candidates.iloc[0]
    selected_id = int(selected["system_id"])
    if selected_id != 5180:
        raise RuntimeError(f"unexpected-best-independent-representative:{selected_id}")
    if not STATE.is_file():
        raise FileNotFoundError(STATE)

    OUT.mkdir(parents=True, exist_ok=False)
    keep = [
        "system_id", "rgi_id", "candidate_root", "historical_peak_length_m",
        "lia_registered_length_m", "formal_peak_upper_m", "peak_excess_m",
        "historical_2000_length_m", "rgi_registered_length_m",
        "e_lia_total_formal", "e_lia_separation_formal",
        "e_rgi_total_formal", "e_rgi_separation_formal",
        "forward_delta_t_start_c", "forward_delta_t_end_c",
        "steady_summary_path",
    ]
    candidates[keep].to_csv(OUT / "early_pulse_group_candidates.csv", index=False)
    result = {
        "schema": "lia-191-remaining136-early-pulse-group-selection-v1",
        "status": "independent-representative-selected",
        "accepted": 55,
        "remaining": 136,
        "candidate_system_count": int(len(candidates)),
        "selection_rule": "smallest positive continuous LIA peak excess among clean steady/historical runs with both total gates and modern separation gate already passing",
        "selected_system_id": selected_id,
        "selected_candidate_root": str(selected["candidate_root"]),
        "baseline_peak_length_m": float(selected["historical_peak_length_m"]),
        "formal_peak_upper_m": float(selected["formal_peak_upper_m"]),
        "peak_excess_m": float(selected["peak_excess_m"]),
        "baseline_2000_length_m": float(selected["historical_2000_length_m"]),
        "frozen_errors": {
            "e_lia_total": float(selected["e_lia_total_formal"]),
            "e_lia_separation": float(selected["e_lia_separation_formal"]),
            "e_rgi_total": float(selected["e_rgi_total_formal"]),
            "e_rgi_separation": float(selected["e_rgi_separation_formal"]),
        },
        "initial_state": {
            "path": str(STATE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(STATE),
            "initial_thickness_modification": False,
        },
        "fixed_family_test": {
            "base_power_gamma": 0.1,
            "pulse_start_year": 1470,
            "pulse_peak_year": 1490,
            "pulse_end_year": 1550,
            "pulse_amplitude_c": 0.03125,
            "amplitude_scan": False,
        },
        "prohibited_source_accessed": False,
        "oggm_started": False,
        "authorized_by_this_file": False,
        "next_required_transaction": "one 1470-1550 system5180 scientific preflight with frozen state, physics, score, and pulse",
    }
    (OUT / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
