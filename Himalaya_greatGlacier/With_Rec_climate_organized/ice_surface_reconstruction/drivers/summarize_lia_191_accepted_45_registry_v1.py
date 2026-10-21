from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / "lia_191_accepted_40_registry_v1" / "accepted_40_curve_metrics.csv"
REFS = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / "lia_191_controller_v1" / "core191_length_references.csv"
OUT = ROOT / "validation_data" / "results" / "paleo_surface_validation_v0" / "lia_191_accepted_45_registry_v1"

ADDITIONS = [
    (3010, "w662power_3010_e0p1_g0p25_v1", "w663repro3010_power_e0p1_g0p25_v1"),
    (2748, "w664power_2748_e0p1_g0p25_v1", "w685repro2748_power_e0p1_g0p25_v1"),
    (3137, "w688hist3137_refine_3137_end0p825_v1", "w692repro3137_linear_e0p825_v1"),
    (3914, "w664linear_3914_e0p75_v1", "w685repro3914_linear_e0p75_v1"),
    (3911, "w690hist3911_final_3911_end0p475_v1", "w695repro3911_linear_e0p475_v1"),
]


def main() -> None:
    base = pd.read_csv(BASE)
    refs = pd.read_csv(REFS).set_index("lia_unique_id")
    rows = []
    for sid, history_root, reproduction_root in ADDITIONS:
        h = json.loads((ROOT / history_root / "historical_summary.json").read_text(encoding="utf-8"))
        c = json.loads((ROOT / reproduction_root / "reproduction_comparison.json").read_text(encoding="utf-8"))
        if h.get("status") != "historical-pass" or c.get("status") != "reproduction-pass":
            raise RuntimeError(f"system {sid} is not a verified formal pass")
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
            "e_lia": float(h["e_lia_separation"]),
            "e_rgi": float(h["e_rgi_separation"]),
            "history_root": history_root,
            "reproduction_root": reproduction_root,
        })
    merged = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    merged["system_id"] = pd.to_numeric(merged["system_id"]).astype(int)
    merged = merged.drop_duplicates("system_id", keep="last").sort_values("system_id")
    if len(merged) != 45:
        raise RuntimeError(f"expected 45 unique accepted systems, got {len(merged)}")
    OUT.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT / "accepted_45_curve_metrics.csv", index=False)
    summary = {"status": "accepted-registry-complete", "accepted_system_count": 45, "remaining_system_count": 146, "new_systems": [v[0] for v in ADDITIONS]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
