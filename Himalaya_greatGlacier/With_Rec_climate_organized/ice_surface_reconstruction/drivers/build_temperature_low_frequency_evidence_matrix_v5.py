from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
BASE = LOOP / "temperature_1850_1950_low_frequency_evidence_matrix_v4"
OUT = LOOP / "temperature_1850_1950_low_frequency_evidence_matrix_v5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)

    matrix_path = BASE / "evidence_matrix.csv"
    queue_path = BASE / "author_data_request_queue.csv"
    inputs = [matrix_path, queue_path]
    before = {str(path): sha256(path) for path in inputs}

    rows = read_csv(matrix_path)
    cn = next(row for row in rows if row["evidence_id"] == "CN_PX202_BOREHOLE_GST")
    cn.update(
        {
            "coverage": "five_century_long_ramp_coefficients_pre1500_baseline_to_measurement_1984",
            "target": "rate_of_GST_change_K_per_100a",
            "spatial_relevance": "single_eastern_HMA_edge_site_high_spatial_risk",
            "season_match": "century_scale_GST_not_air_temperature_anomaly",
            "admitted_role": "nineteenth_century_near_zero_direction_gate_only",
            "finding": (
                "the official field is Rate of GST Change (K/100a), not a temperature anomaly; "
                "the nineteenth-century ramp coefficient is +0.023 K/100a, indicating near-zero "
                "whole-century change under the five-ramp inversion, with no supplied coefficient uncertainty"
            ),
            "next_action": (
                "do_not_convert_to_1850-1900_offset_or background weight; retain only as a weak east-edge "
                "gate against very large nineteenth-century trends"
            ),
            "source_url": "https://www.ncei.noaa.gov/pub/data/paleo/contributions_by_author/huang2000/huang2016-CN-PX202.txt",
        }
    )
    write_csv(OUT / "evidence_matrix.csv", rows)
    write_csv(OUT / "author_data_request_queue.csv", read_csv(queue_path))

    summary = {
        "experiment": OUT.name,
        "supersedes": BASE.name,
        "correction": "CN_PX202_field_is_K_per_100a_ramp_rate_not_temperature_anomaly",
        "decision": "no_uniform_HMA_correction_new_direct_routes_prioritized_CN_PX202_direction_only",
        "evidence_rows": len(rows),
        "new_actionable_routes": [
            "World_Weather_Records_1800_1920_official_compilation",
            "India_Monthly_Weather_Review_1891_1920",
            "Schlagintweit_1854_1858",
            "new_site_RCS_data_request",
        ],
        "CN_PX202_can_set_correction_magnitude": False,
        "new_fully_independent_annual_west_central_temperature_series": 0,
        "uniform_historical_warm_bias_supported": False,
        "western_1880_1920_station_residual_supported": True,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "sealed_period_values_used": False,
        "sealed_overlapping_metadata_aggregate_excluded": True,
        "prohibited_source_accessed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = """# HMA 1850-1950 low-frequency temperature evidence matrix v5

## Correction to v4

This version supersedes v4. The CN-PX202 value `+0.023` is not a temperature anomaly relative to pre-1500. Its official field is `Rate of GST Change (K/100a)`: the coefficient of the nineteenth-century linear ramp in a five-century-ramp borehole inversion.

The defensible interpretation is therefore narrow: under that inversion, this single eastern-edge site has a near-zero whole-nineteenth-century GST change (`+0.023 K/100a`). It does not independently resolve 1850-1900, has no published coefficient uncertainty, is smoothed by subsurface heat diffusion, and measures GST rather than air temperature. It cannot set a project offset or a background weight.

## Current decision

`no_uniform_HMA_correction_new_direct_routes_prioritized_CN_PX202_direction_only`

The strongest new practical source is now the Smithsonian 1927 *World Weather Records* first series. It compiles official monthly pressure, temperature and precipitation observations from earliest available records through 1920. It may provide a much faster, standardized extraction route for HMA stations than reading every India Monthly Weather Review volume separately. NOAA confirms that the first issue was assembled from official sources and includes monthly mean temperature through 1920.

The direct-observation hierarchy is now:

1. Extract HMA stations from *World Weather Records* and preserve all station notes about observation practice and configuration.
2. Use the original India monthly reports to resolve values or metadata that are ambiguous in the compilation.
3. Use the 1854-1858 Schlagintweit observations only as short local level anchors after instrument, time-of-day, elevation and location audit.

Records sharing physical observations with GHCN-M or IMD remain processing/transcription checks, not independent climate votes. The RCS chronology mechanism remains open only if genuinely new annual Rara/Cholangpati numeric data are acquired.

## What is now ruled out

- No HMA-wide constant cooling or fixed background mixture is justified.
- CN-PX202 cannot define a correction magnitude.
- Sano Humla is the same physical sample family as project proxy9_199 and cannot be counted as independent validation.
- Chongce, Ngamring, Tiancai and Aweng have already been assessed and are not new evidence.

## Safety

- No original or production file was modified.
- No EnKF or OGGM run occurred.
- No value from 1921-1939 was read, saved, plotted or used.
- A later aggregate exposed in metadata was excluded from all calculations and judgments.
- The permanently prohibited source was not accessed, used or cited.
"""
    (OUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")

    after = {str(path): sha256(path) for path in inputs}
    manifest = {
        "read_only_inputs_before": before,
        "read_only_inputs_after": after,
        "all_inputs_unchanged": before == after,
        "output_directory": str(OUT),
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("A declared read-only input changed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
