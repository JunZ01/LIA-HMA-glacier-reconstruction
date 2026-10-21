"""Promote independently reproduced system 5443 into the accepted registry."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
VAL = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = VAL / "lia_191_accepted_53_registry_v1/accepted_53_curve_metrics.csv"
REFS = VAL / "lia_191_controller_v1/core191_length_references.csv"
OUT = VAL / "lia_191_accepted_54_registry_v1"
SYSTEM_ID = 5443
HISTORY_ROOT = "w739hist_isotonic_5443_endp0p5_v1"
REPRODUCTION_ROOT = "w744hist_isotonic_5443_endp0p5_reproduction_v3"
CHECK = VAL / "lia_191_reproduction_checks_v1/system_5443_isotonic_reproduction_v1.json"


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0])
    with REFS.open(encoding="utf-8-sig", newline="") as f:
        refs = {int(row["lia_unique_id"]): row for row in csv.DictReader(f)}

    existing = {int(row["system_id"]) for row in rows}
    if SYSTEM_ID in existing:
        raise RuntimeError(f"already-registered:{SYSTEM_ID}")

    history = json.loads((ROOT / HISTORY_ROOT / "historical_summary.json").read_text(encoding="utf-8"))
    reproduction = json.loads(CHECK.read_text(encoding="utf-8"))
    if history["status"] != "historical-pass":
        raise RuntimeError(f"historical-not-pass:{SYSTEM_ID}")
    if reproduction["status"] != "reproduction-pass":
        raise RuntimeError(f"reproduction-not-pass:{SYSTEM_ID}")
    if not reproduction["state_arrays_exact"] or not reproduction["timeseries_byte_identical"]:
        raise RuntimeError(f"reproduction-not-exact:{SYSTEM_ID}")

    ref = refs[SYSTEM_ID]
    lia = float(ref["lia_registered_length_m"])
    rgi = float(ref["rgi_registered_length_m"])
    tolerance_km = 0.2 * abs(lia - rgi) / 1000
    rows.append(
        {
            "system_id": str(SYSTEM_ID),
            "rgi_id": ref["rgi_id"],
            "route": "formal-historical-reproduction-pass",
            "lia_km": str(lia / 1000),
            "rgi_km": str(rgi / 1000),
            "lia_tol_km": str(tolerance_km),
            "rgi_tol_km": str(tolerance_km),
            "e_lia": str(history["e_lia_separation"]),
            "e_rgi": str(history["e_rgi_separation"]),
            "history_root": HISTORY_ROOT,
            "reproduction_root": REPRODUCTION_ROOT,
        }
    )
    rows.sort(key=lambda row: int(row["system_id"]))

    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT / "accepted_54_curve_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "accepted-registry-complete",
        "accepted_system_count": 54,
        "remaining_system_count": 137,
        "new_systems": [SYSTEM_ID],
        "source_registry": str(SOURCE),
        "reproduction_checks": [str(CHECK)],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
