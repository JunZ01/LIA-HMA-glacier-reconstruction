from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import evaluate_temperature_candidate_b2_stage_c_v2 as v2
import temperature_candidate_b2_stage_c_core_v1 as core


DRIVER = Path(__file__).resolve()
V2_EVALUATOR = Path(v2.__file__).resolve()
DEFAULT_INPUT = v2.DEFAULT_INPUT
DEFAULT_OUT = v2.v1.LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_evaluation_v3"
REQUIRED_REGIONS = ("West", "Central", "East")


def main(input_path: Path, output: Path) -> None:
    v2.main(input_path, output)
    panel = pd.read_csv(output / "station_year_candidate_b2_values.csv")
    modern = panel[panel["year"].between(1940, 2000)].copy()
    rows = []
    for region in REQUIRED_REGIONS:
        selected = modern[
            modern["region_group"].eq(region) & modern["qcf"].notna()
        ]
        rows.append(
            {
                "region_group": region,
                "qcf_station_count": int(selected["station_id"].nunique()),
                "qcf_station_year_pair_count": int(len(selected)),
                "direct_station_support": "present" if len(selected) else "absent",
                "adjudication_role": (
                    "direct regional validation"
                    if len(selected)
                    else "not evaluable from admitted GHCN-M panel; use independent evidence matrix without fabricating a station vote"
                ),
            }
        )
    coverage = pd.DataFrame(rows)
    coverage.to_csv(output / "direct_station_regional_coverage.csv", index=False)
    missing = coverage.loc[
        coverage["direct_station_support"].eq("absent"), "region_group"
    ].tolist()

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "experiment": output.name,
            "active_evaluator_version": "v3",
            "required_direct_station_regions": list(REQUIRED_REGIONS),
            "missing_direct_station_regions": missing,
            "direct_station_spatial_coverage_complete": len(missing) == 0,
            "spatial_coverage_boundary": (
                "a missing direct-station region is not silently scored as pass or fail; independent evidence remains separate"
            ),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest_path = output / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluator_v2_dependency"] = {
        "path": str(V2_EVALUATOR),
        "sha256": core.sha256(V2_EVALUATOR),
    }
    manifest["active_evaluator_v3"] = {
        "path": str(DRIVER),
        "sha256": core.sha256(DRIVER),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.input, args.output)
