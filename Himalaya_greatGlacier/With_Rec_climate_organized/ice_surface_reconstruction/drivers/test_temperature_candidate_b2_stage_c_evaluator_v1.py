from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_temperature_candidate_b2_stage_c_v1 as evaluator
import temperature_candidate_b2_stage_c_core_v1 as core


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
OUT = evaluator.LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_evaluator_preflight_v1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    perfect = evaluator.paired_metrics(
        np.asarray([-1.0, -0.25, 0.5, 1.0]),
        np.asarray([-1.0, -0.25, 0.5, 1.0]),
    )
    years = np.arange(1940, 2001)
    field = np.column_stack(
        [0.02 * (years - 1975), -0.01 * (years - 1975)]
    )
    anomaly = evaluator.own_baseline(field, years)
    baseline = (years >= 1961) & (years <= 1990)
    panel = pd.DataFrame(
        {
            "station_id": ["A"] * 10,
            "year": np.arange(1940, 1950),
            "qcf": np.linspace(-1, 1, 10),
            "candidate_joint": np.linspace(-1, 1, 10),
        }
    )
    decades = evaluator.decadal_pairs(panel, "qcf", "candidate_joint")
    gates = {
        "perfect_rmse_zero": perfect["rmse_degC"] == 0.0,
        "perfect_correlation_one": abs(perfect["correlation_r"] - 1.0) <= 1e-15,
        "perfect_sign_agreement_one": perfect["sign_agreement"] == 1.0,
        "own_1961_1990_baseline_zero": float(np.max(np.abs(anomaly[baseline].mean(axis=0)))) <= 1e-15,
        "decadal_requires_and_accepts_six_or_more_years": len(decades) == 1 and int(decades.iloc[0]["valid_year_count"]) == 10,
        "west_boundary": evaluator.region_group(79.999) == "West",
        "central_boundary": evaluator.region_group(80.0) == "Central",
        "east_boundary": evaluator.region_group(90.0) == "East",
        "early_station_table_has_no_sealed_year": not pd.read_csv(evaluator.EARLY_STATIONS)["year"].between(1921, 1939).any(),
        "modern_station_table_is_1951_2000": bool(pd.read_csv(evaluator.MODERN_STATIONS)["year"].between(1951, 2000).all()),
        "allowed_long_curve_has_no_sealed_year": not pd.read_csv(evaluator.LONG_CURVES)["year"].between(1921, 1939).any(),
        "independent_evidence_sealed_flag_clear": not pd.read_csv(evaluator.EVIDENCE)["sealed_period_values_used"].astype(bool).any(),
        "real_proxy_observations_read": False,
        "enkf_run": False,
    }
    passed = all(
        value
        for key, value in gates.items()
        if key not in {"real_proxy_observations_read", "enkf_run"}
    ) and not gates["real_proxy_observations_read"] and not gates["enkf_run"]
    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "evaluator_preflight_pass": bool(passed),
        "gates": gates,
        "real_proxy_observations_read": False,
        "stage_c_enkf_run": False,
        "sealed_1921_1939_values_used": False,
        "prohibited_source_accessed": False,
        "production_modified": False,
        "oggm_run": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "evaluator": {"path": str(Path(evaluator.__file__).resolve()), "sha256": core.sha256(Path(evaluator.__file__).resolve())},
        "test_driver": {"path": str(DRIVER), "sha256": core.sha256(DRIVER)},
        "early_stations": {"path": str(evaluator.EARLY_STATIONS), "sha256": core.sha256(evaluator.EARLY_STATIONS)},
        "modern_stations": {"path": str(evaluator.MODERN_STATIONS), "sha256": core.sha256(evaluator.MODERN_STATIONS)},
        "allowed_long_curves": {"path": str(evaluator.LONG_CURVES), "sha256": core.sha256(evaluator.LONG_CURVES)},
        "evidence": {"path": str(evaluator.EVIDENCE), "sha256": core.sha256(evaluator.EVIDENCE)},
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("evaluator-preflight-failed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
