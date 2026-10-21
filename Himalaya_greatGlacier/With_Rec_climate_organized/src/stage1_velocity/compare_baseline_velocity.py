"""Compare ITS_LIVE gates with scenario-specific fixed-thickness velocities."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "019f730c-ee32-74d0-8b77-fdb7832f6a06"
)
STAGE0_MB = RUN_ROOT / "stage0_bed_full_farinotti_v0"
STAGE0_MILLAN = RUN_ROOT / "stage0_bed_full_millan_v0"
VELOCITY_WORK = RUN_ROOT / "velocity_work"
GDIR_ROOT = STAGE0_MB / "oggm_workdir" / "per_glacier"
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
SURFACE_TO_SECTION_FACTOR_N3 = 1 / 0.8

SCENARIOS = {
    "farinotti": {
        "eligible_path": STAGE0_MB / "mb_member_parameters.csv",
        "filesuffix": "_f19_fs0_full_v0",
        "output_stem": "velocity_baseline_farinotti",
        "fixed_geometry": "Farinotti 2019 fs=0 Stage 0 inversion output",
    },
    "millan": {
        "eligible_path": STAGE0_MILLAN / "inversion_member_status.csv",
        "filesuffix": "_m22_fs0_full_v0",
        "output_stem": "velocity_baseline_millan",
        "fixed_geometry": "Millan 2022 fs=0 Stage 0 inversion output",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=sorted(SCENARIOS), default="farinotti"
    )
    return parser.parse_args()


def gdir_path(rgi_id: str) -> Path:
    return GDIR_ROOT / rgi_id[:-6] / rgi_id[:-3] / rgi_id


def quantiles(series: pd.Series) -> dict[str, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    return {
        str(q): float(clean.quantile(q))
        for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    }


def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario]
    gates = pd.read_csv(VELOCITY_WORK / "velocity_gate_1205_stats.csv")
    eligible = pd.read_csv(scenario["eligible_path"])
    data = eligible[["lia_unique_id", "rgi_id"]].merge(
        gates, on="rgi_id", how="left", validate="one_to_one"
    )
    if data["qc_status"].isna().any():
        raise ValueError("At least one MB-eligible member has no velocity gate")

    records = []
    failures = []
    for row in data.itertuples(index=False):
        try:
            path = (
                gdir_path(row.rgi_id)
                / f"inversion_output{scenario['filesuffix']}.pkl"
            )
            with gzip.open(path, "rb") as stream:
                output = pickle.load(stream)
            flowline_index = int(row.main_flowline_list_index)
            gate_index = int(row.gate_flowline_index)
            line = output[flowline_index]
            if gate_index >= len(line["flux"]):
                raise IndexError("Gate point is outside inversion output")
            section_m2 = float(line["volume"][gate_index] / line["dx"])
            flux_m3_s = float(line["flux"][gate_index])
            if not np.isfinite(section_m2) or section_m2 <= 0:
                raise ValueError("Invalid fixed-thickness cross-section")
            integrated_speed = flux_m3_s / section_m2 * SECONDS_PER_YEAR
            surface_speed = integrated_speed * SURFACE_TO_SECTION_FACTOR_N3
            observed = float(row.v_m_per_year_median)
            records.append(
                {
                    "lia_unique_id": int(row.lia_unique_id),
                    "rgi_id": row.rgi_id,
                    "qc_status": row.qc_status,
                    "main_flowline_list_index": flowline_index,
                    "gate_flowline_index": gate_index,
                    "fixed_section_m2": section_m2,
                    "flux_m3_s": flux_m3_s,
                    "modeled_integrated_speed_m_per_year": integrated_speed,
                    "modeled_surface_speed_m_per_year": surface_speed,
                    "observed_speed_low_m_per_year": float(
                        row.velocity_observation_low_m_per_year
                    ),
                    "observed_speed_median_m_per_year": observed,
                    "observed_speed_high_m_per_year": float(
                        row.velocity_observation_high_m_per_year
                    ),
                    "observed_to_modeled_surface_ratio": (
                        observed / surface_speed
                        if surface_speed > 0
                        else math.nan
                    ),
                    "modeled_inside_observation_interval": bool(
                        row.velocity_observation_low_m_per_year
                        <= surface_speed
                        <= row.velocity_observation_high_m_per_year
                    ),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "lia_unique_id": int(row.lia_unique_id),
                    "rgi_id": row.rgi_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    frame = pd.DataFrame(records)
    failure_frame = pd.DataFrame(
        failures, columns=["lia_unique_id", "rgi_id", "error_type", "error"]
    )
    output_stem = scenario["output_stem"]
    frame.to_csv(VELOCITY_WORK / f"{output_stem}_stats.csv", index=False)
    failure_frame.to_csv(
        VELOCITY_WORK / f"{output_stem}_failures.csv", index=False
    )

    strong = frame[frame["qc_status"].isin(["usable", "weak-slow"])]
    summary = {
        "eligible_member_count": int(len(data)),
        "successful_comparisons": int(len(frame)),
        "failed_comparisons": int(len(failure_frame)),
        "qc_status_counts": frame["qc_status"].value_counts().to_dict(),
        "modeled_surface_speed_quantiles_m_per_year": quantiles(
            frame["modeled_surface_speed_m_per_year"]
        ),
        "observed_to_modeled_ratio_quantiles": quantiles(
            strong["observed_to_modeled_surface_ratio"]
        ),
        "strong_gate_count": int(len(strong)),
        "strong_gates_inside_observation_interval": int(
            strong["modeled_inside_observation_interval"].sum()
        ),
        "strong_gate_system_count": int(strong["lia_unique_id"].nunique()),
        "scenario": args.scenario,
        "fixed_geometry": scenario["fixed_geometry"],
        "velocity_conversion": (
            "u_integrated = flux / fixed cross-section; "
            "u_surface = u_integrated / 0.8 for Glen n=3 and fs=0"
        ),
        "failures": failures,
    }
    (VELOCITY_WORK / f"{output_stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
