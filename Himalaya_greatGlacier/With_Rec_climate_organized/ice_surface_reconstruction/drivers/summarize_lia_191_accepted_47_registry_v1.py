from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_accepted_45_registry_v1/accepted_45_curve_metrics.csv"
REFS = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_length_references.csv"
OUT = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_accepted_47_registry_v1"

ADDITIONS = [
    (2595, "w705power2595_e0p25_g0p1_v1", "w709repro2595_power_e0p25_g0p1_v1"),
    (5524, "w701hist5524_first_5524_end0p16000000000000003_v1", "w709repro5524_linear_e0p16_v1"),
]


def main() -> None:
    base = pd.read_csv(BASE)
    refs = pd.read_csv(REFS).set_index("lia_unique_id")
    rows = []
    for sid, history_root, reproduction_root in ADDITIONS:
        history = json.loads((ROOT / history_root / "historical_summary.json").read_text(encoding="utf-8"))
        comparison = json.loads((ROOT / reproduction_root / "reproduction_comparison.json").read_text(encoding="utf-8"))
        if history.get("status") != "historical-pass" or comparison.get("status") != "reproduction-pass":
            raise RuntimeError(f"system {sid} is not a verified formal reproduction pass")
        ref = refs.loc[sid]
        separation_km = abs(float(ref["lia_registered_length_m"]) - float(ref["rgi_registered_length_m"])) / 1000.0
        rows.append({
            "system_id": sid,
            "rgi_id": ref["rgi_id"],
            "route": "formal-historical-reproduction-pass",
            "lia_km": float(ref["lia_registered_length_m"]) / 1000.0,
            "rgi_km": float(ref["rgi_registered_length_m"]) / 1000.0,
            "lia_tol_km": 0.2 * separation_km,
            "rgi_tol_km": 0.2 * separation_km,
            "e_lia": float(history["e_lia_separation"]),
            "e_rgi": float(history["e_rgi_separation"]),
            "history_root": history_root,
            "reproduction_root": reproduction_root,
        })
    merged = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    merged["system_id"] = pd.to_numeric(merged["system_id"]).astype(int)
    merged = merged.drop_duplicates("system_id", keep="last").sort_values("system_id")
    if len(merged) != 47:
        raise RuntimeError(f"expected 47 unique accepted systems, got {len(merged)}")
    OUT.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT / "accepted_47_curve_metrics.csv", index=False)
    summary = {
        "status": "accepted-registry-complete",
        "accepted_system_count": 47,
        "remaining_system_count": 144,
        "new_systems": [sid for sid, _, _ in ADDITIONS],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
