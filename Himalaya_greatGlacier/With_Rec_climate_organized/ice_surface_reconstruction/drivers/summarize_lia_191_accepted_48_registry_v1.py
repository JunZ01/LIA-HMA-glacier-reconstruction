from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_accepted_47_registry_v1/accepted_47_curve_metrics.csv"
REFS = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_length_references.csv"
OUT = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_accepted_48_registry_v1"
SID = 5465
HISTORY = "w689hist5465_first_5465_end0p7_v1"
REPRO = "w720repro5465_linear_e0p7_v1"


def main() -> None:
    base = pd.read_csv(BASE)
    refs = pd.read_csv(REFS).set_index("lia_unique_id")
    history = json.loads((ROOT / HISTORY / "historical_summary.json").read_text(encoding="utf-8"))
    comparison = json.loads((ROOT / REPRO / "reproduction_comparison.json").read_text(encoding="utf-8"))
    if history.get("status") != "historical-pass" or comparison.get("status") != "reproduction-pass":
        raise RuntimeError("5465 is not a verified formal reproduction pass")
    ref = refs.loc[SID]
    separation_km = abs(float(ref["lia_registered_length_m"]) - float(ref["rgi_registered_length_m"])) / 1000.0
    row = {
        "system_id": SID,
        "rgi_id": ref["rgi_id"],
        "route": "formal-historical-reproduction-pass",
        "lia_km": float(ref["lia_registered_length_m"]) / 1000.0,
        "rgi_km": float(ref["rgi_registered_length_m"]) / 1000.0,
        "lia_tol_km": 0.2 * separation_km,
        "rgi_tol_km": 0.2 * separation_km,
        "e_lia": float(history["e_lia_separation"]),
        "e_rgi": float(history["e_rgi_separation"]),
        "history_root": HISTORY,
        "reproduction_root": REPRO,
    }
    merged = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
    merged["system_id"] = pd.to_numeric(merged["system_id"]).astype(int)
    merged = merged.drop_duplicates("system_id", keep="last").sort_values("system_id")
    if len(merged) != 48:
        raise RuntimeError(f"expected 48 accepted systems, got {len(merged)}")
    OUT.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT / "accepted_48_curve_metrics.csv", index=False)
    summary = {"status": "accepted-registry-complete", "accepted_system_count": 48, "remaining_system_count": 143, "new_systems": [SID]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
