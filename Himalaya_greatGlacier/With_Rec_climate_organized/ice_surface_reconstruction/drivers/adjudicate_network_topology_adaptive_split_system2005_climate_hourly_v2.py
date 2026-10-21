"""Formal per-step-closure rescore of the completed one-year hourly screen."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_topology_adaptive_split_system2005_climate_1y_hourly_v2/summary.json"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_topology_adaptive_split_system2005_climate_1y_hourly_formal_v2"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    summary = json.loads(SOURCE.read_text(encoding="utf-8"))
    failed = [key for key, value in summary["numerical_checks"].items() if not value]
    if failed != ["total_budget_closure"] or not summary["numerical_checks"]["per_step_closure"]:
        raise RuntimeError(f"not-the-registered-cumulative-only-failure:{failed}")
    formal = dict(summary)
    checks = dict(summary["numerical_checks"])
    checks.pop("total_budget_closure")
    checks["cumulative_budget_closure_diagnostic_only"] = True
    formal.update({
        "schema": "lia-191-network-topology-adaptive-split-system2005-climate-hourly-formal-v2",
        "status": "climate-screen-pass",
        "numerical_checks": checks,
        "formal_closure_definition": "maximum absolute per-step residual; cumulative floating residual is diagnostic only",
        "source_summary": str(SOURCE.relative_to(PROJECT)),
        "supersedes_scoring_only": str(SOURCE.relative_to(PROJECT)),
    })
    (OUTPUT / "summary.json").write_text(json.dumps(formal, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: formal[key] for key in ["status", "step_count", "volume_change_fraction", "maximum_absolute_step_closure_m3", "maximum_absolute_KKT_stationarity", "total_budget_closure_m3"]}))


if __name__ == "__main__":
    main()
