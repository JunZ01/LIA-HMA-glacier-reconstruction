"""Refresh the controller audit and register the two new reproducible passes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .summarize_lia_191_remaining_route_v1 import main as refresh_v1


NEW = {
    3495: ("w658hist3495_end0p25_v1", "w661repro3495_linear_end0p25_v1"),
    3808: ("w660hist3808_final_refine_3808_end1p05_v1", "w661repro3808_linear_end1p05_v1"),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    refresh_v1()
    refs = pd.read_csv(validation / "lia_191_controller_v1/core191_length_references.csv")
    refs["lia_unique_id"] = pd.to_numeric(refs["lia_unique_id"]).astype(int)
    refs = refs.set_index("lia_unique_id")
    accepted = pd.read_csv(validation / "lia_191_accepted_38_registry_v1/accepted_38_curve_metrics.csv")
    accepted["system_id"] = pd.to_numeric(accepted["system_id"]).astype(int)
    rows = accepted.to_dict(orient="records")
    for sid, (history_root, reproduction_root) in NEW.items():
        history = load(root / history_root / "historical_summary.json")
        reproduction = load(root / reproduction_root / "reproduction_comparison.json")
        if history.get("status") != "historical-pass" or reproduction.get("status") != "reproduction-pass":
            raise RuntimeError(f"unverified-new-pass:{sid}")
        ref = refs.loc[sid]
        lia = float(ref["lia_registered_length_m"])
        rgi = float(ref["rgi_registered_length_m"])
        separation = abs(lia - rgi)
        rows.append({
            "system_id": sid,
            "rgi_id": str(ref["rgi_id"]),
            "route": "synthetic-system",
            "lia_km": lia / 1000.0,
            "rgi_km": rgi / 1000.0,
            "lia_tol_km": min(0.2 * separation, 0.1 * lia) / 1000.0,
            "rgi_tol_km": min(0.2 * separation, 0.1 * rgi) / 1000.0,
            "e_lia": float(history["e_lia_separation"]),
            "e_rgi": float(history["e_rgi_separation"]),
            "history_root": history_root,
            "reproduction_root": reproduction_root,
        })
    accepted40 = pd.DataFrame(rows).sort_values("system_id")
    if len(accepted40) != 40 or accepted40["system_id"].nunique() != 40:
        raise RuntimeError("accepted-40-registry-size-mismatch")
    registry = validation / "lia_191_accepted_40_registry_v1"
    registry.mkdir(parents=True, exist_ok=True)
    accepted40.to_csv(registry / "accepted_40_curve_metrics.csv", index=False)
    (registry / "summary.json").write_text(json.dumps({"accepted_count": 40, "systems": accepted40["system_id"].astype(int).tolist()}, indent=2) + "\n", encoding="utf-8")

    audit = pd.read_csv(validation / "lia_191_remaining_route_audit_v1/lia_191_route_status.csv")
    audit["lia_unique_id"] = pd.to_numeric(audit["lia_unique_id"]).astype(int)
    accepted_ids = set(accepted40["system_id"].astype(int))
    mask = audit["lia_unique_id"].isin(accepted_ids)
    audit.loc[mask, "accepted"] = True
    audit.loc[mask, "current_route"] = "accepted"
    audit.loc[mask, "current_gate"] = "historical-reproduction-pass"
    corrections = {
        2711: ("legacy-convergence-blocked", "no-delta-convergence-bracket"),
        3584: ("legacy-level-amplitude-hold", "named-configuration-level-amplitude-tradeoff"),
    }
    for sid, (route, gate) in corrections.items():
        row = audit["lia_unique_id"] == sid
        audit.loc[row, "current_route"] = route
        audit.loc[row, "current_gate"] = gate
    route_root = validation / "lia_191_remaining_route_audit_v2"
    route_root.mkdir(parents=True, exist_ok=True)
    audit.to_csv(route_root / "lia_191_route_status.csv", index=False)
    counts = audit.groupby("current_route").size().sort_values(ascending=False).to_dict()
    summary = {
        "population": 191,
        "accepted": int(audit["accepted"].astype(str).str.lower().eq("true").sum()),
        "remaining": 191 - len(accepted_ids),
        "route_counts": {str(k): int(v) for k, v in counts.items()},
        "accepted_registry": str((registry / "accepted_40_curve_metrics.csv").resolve()),
        "route_table": str((route_root / "lia_191_route_status.csv").resolve()),
    }
    (route_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
