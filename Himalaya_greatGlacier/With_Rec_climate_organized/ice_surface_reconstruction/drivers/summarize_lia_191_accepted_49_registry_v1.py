"""Create accepted-49 registry by adding independently reproduced system 2729."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
VALIDATION = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = VALIDATION / "lia_191_accepted_48_registry_v1/accepted_48_curve_metrics.csv"
HISTORY = ROOT / "w730hist_power_2729_endp0p6_g0p25_from_w728steady_v1/historical_summary.json"
REPRO_CHECK = VALIDATION / "lia_191_reproduction_checks_v1/system_2729_power_reproduction_v1.json"
OUT = VALIDATION / "lia_191_accepted_49_registry_v1"


def main() -> None:
    history = json.loads(HISTORY.read_text(encoding="utf-8"))
    repro = json.loads(REPRO_CHECK.read_text(encoding="utf-8"))
    if history["status"] != "historical-pass" or repro["status"] != "reproduction-pass":
        raise RuntimeError("system-2729-not-formally-reproduced")
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    if any(int(row["system_id"]) == 2729 for row in rows):
        raise RuntimeError("system-2729-already-registered")
    lia_m = 10139.189183844292
    rgi_m = 8295.4144238231
    tolerance_km = 0.2 * abs(lia_m - rgi_m) / 1000.0
    rows.append({
        "system_id": "2729",
        "rgi_id": "RGI60-15.07086",
        "route": "formal-historical-reproduction-pass",
        "lia_km": str(lia_m / 1000.0),
        "rgi_km": str(rgi_m / 1000.0),
        "lia_tol_km": str(tolerance_km),
        "rgi_tol_km": str(tolerance_km),
        "e_lia": str(history["e_lia_separation"]),
        "e_rgi": str(history["e_rgi_separation"]),
        "history_root": HISTORY.parent.name,
        "reproduction_root": "w730hist_power_2729_endp0p6_g0p25_from_w728steady_v1_independent_reproduction_v1",
    })
    rows.sort(key=lambda row: int(row["system_id"]))
    OUT.mkdir(parents=True, exist_ok=False)
    target = OUT / "accepted_49_curve_metrics.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "accepted-registry-complete",
        "accepted_system_count": 49,
        "remaining_system_count": 142,
        "new_systems": [2729],
        "source_registry": str(SOURCE),
        "reproduction_check": str(REPRO_CHECK),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
