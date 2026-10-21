"""Adjudicate 0.1y timestep sensitivity before the system-2005 5y gate."""

import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = ROOT / "lia_191_network_capacity_constrained_system2005_timestep_adjudication_v1"
SOURCES = {
    "1h": ROOT / "lia_191_network_capacity_constrained_system2005_zero_0p1y_v2/summary.json",
    "6h": ROOT / "lia_191_network_capacity_constrained_system2005_zero_0p1y_dt6h_v1/summary.json",
    "24h": ROOT / "lia_191_network_capacity_constrained_system2005_zero_0p1y_dt24h_v1/summary.json",
}


def main():
    data = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in SOURCES.items()}
    baseline_area = float(data["1h"]["final_area_m2"])
    rows = {}
    for key, summary in data.items():
        rows[key] = {
            "status": summary["status"],
            "step_count": summary["step_count"],
            "maximum_timestep_seconds": summary["maximum_timestep_seconds"],
            "final_area_m2": summary["final_area_m2"],
            "area_relative_difference_from_1h": abs(float(summary["final_area_m2"]) - baseline_area) / baseline_area,
            "volume_drift_fraction": summary["volume_drift_fraction"],
            "maximum_internal_active_gap_count": summary["maximum_internal_active_gap_count"],
        }
    threshold = 1e-4
    checks = {key: row["status"] == "zero-mb-pass" and row["area_relative_difference_from_1h"] <= threshold and row["maximum_internal_active_gap_count"] == 0 for key, row in rows.items()}
    selected = "24h" if checks["24h"] else ("6h" if checks["6h"] else "1h")
    summary = {"status": "timestep-sensitivity-pass", "system_id": 2005, "area_relative_difference_gate": threshold, "runs": rows, "checks": checks, "selected_for_5y": selected, "selected_maximum_timestep_seconds": rows[selected]["maximum_timestep_seconds"], "physics_or_geometry_changed": False, "steady_run": False, "historical_run": False}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
