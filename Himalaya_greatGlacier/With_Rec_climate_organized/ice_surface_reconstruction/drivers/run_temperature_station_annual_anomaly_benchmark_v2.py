from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import run_temperature_station_annual_anomaly_benchmark_v1 as base


OUTPUT = base.RESULT_ROOT / "temperature_station_annual_anomaly_benchmark_v2"
CONTRACT = base.DRIVERS / "temperature_station_annual_anomaly_benchmark_v2_contract.md"
DRIVER = Path(__file__).resolve()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"independent-output-exists:{OUTPUT}")
    required = {
        "facts": base.FACTS,
        "contract_v1": base.CONTRACT,
        "contract_v2": CONTRACT,
        "driver_v1": base.DRIVER,
        "driver_v2": DRIVER,
        "round277_manifest": base.ROUND277 / "source_manifest.json",
        "round277_station_metadata": base.ROUND277 / "station_metadata.csv",
        "round277_qcf": base.ROUND277 / "prepared/ghcnm_tavg_qcf.data",
        "round277_qcu": base.ROUND277 / "prepared/ghcnm_tavg_qcu.data",
        "member_lineage": base.LINEAGE,
        "background_archive": base.REFERENCE,
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: base.file_state(path) for name, path in required.items()}

    round277_manifest = json.loads(
        (base.ROUND277 / "source_manifest.json").read_text(encoding="utf-8")
    )
    expected_hashes = {
        Path(item["path"]).name: item["sha256"]
        for item in round277_manifest["source_files"]
    }
    for key in ("round277_qcf", "round277_qcu"):
        path = required[key]
        if expected_hashes.get(path.name) != before[key]["sha256"]:
            raise RuntimeError(f"round277-source-hash-mismatch:{path.name}")

    stations = pd.read_csv(base.ROUND277 / "station_metadata.csv")
    expected_columns = {
        "station_id",
        "station_name",
        "station_tier",
        "region",
        "latitude",
        "longitude",
    }
    if not expected_columns.issubset(stations.columns) or len(stations) != 15:
        raise RuntimeError("round277-station-metadata-contract-failed")
    stations = stations.sort_values(
        ["station_tier", "longitude", "station_id"]
    ).reset_index(drop=True)
    stations["region_group"] = stations.longitude.map(base.region_group)
    station_ids = set(stations.station_id)
    poison = base.run_poison_test(stations.station_id.iloc[0])

    audits: dict[str, dict[str, int]] = {}
    monthly_frames = []
    for series, path in (
        ("qcf", required["round277_qcf"]),
        ("qcu", required["round277_qcu"]),
    ):
        audit: dict[str, int] = defaultdict(int)
        with path.open("rb") as handle:
            rows = base.parse_ghcn_lines(handle, station_ids, series, audit)
        audits[series] = dict(audit)
        monthly_frames.append(pd.DataFrame(rows))
    if any(
        audit.get("monthly_cells_accessed", 0) % 12 != 0
        for audit in audits.values()
    ):
        raise RuntimeError("partial-ghcn-row-access")
    if any(
        audit.get("locked_target_rows_skipped_before_value_access", 0) == 0
        for audit in audits.values()
    ):
        raise RuntimeError("locked-year-guard-not-exercised")
    monthly = pd.concat(monthly_frames, ignore_index=True)

    climatology, coverage = base.monthly_climatology(monthly)
    annual = base.station_annual_anomalies(monthly, climatology, coverage)
    annual_wide = (
        annual.pivot(
            index=["station_id", "year"],
            columns="series",
            values="station_annual_anomaly_degC",
        )
        .reset_index()
        .rename(
            columns={
                "qcf": "qcf_station_degC",
                "qcu": "qcu_station_degC",
            }
        )
    )

    lineage = pd.read_csv(base.LINEAGE)
    tas = lineage[
        lineage.variable.eq("tas") & lineage.member.isin(base.PROJECT_MEMBERS)
    ].copy()
    if len(tas) != 6 or set(tas.member) != set(base.PROJECT_MEMBERS):
        raise RuntimeError("six-member-lineage-contract-failed")
    project_tables = []
    member_sources = {}
    for row in tas.itertuples(index=False):
        path = Path(row.source_tif)
        if not path.exists() or base.sha256(path) != row.source_tif_sha256:
            raise RuntimeError(f"project-member-source-failed:{row.member}")
        member_sources[row.member] = base.file_state(path)
        table = base.read_project_member(path, stations).rename(
            columns={"value": f"project_{row.member}_degC"}
        )
        project_tables.append(table)
    project = project_tables[0]
    for table in project_tables[1:]:
        project = project.merge(
            table, on=["station_id", "year"], validate="one_to_one"
        )
    member_columns = [
        f"project_{member}_degC" for member in base.PROJECT_MEMBERS
    ]
    project["project_six_member_degC"] = project[member_columns].mean(axis=1)

    background = base.read_background_products(stations)
    curves = (
        annual_wide.merge(
            project,
            on=["station_id", "year"],
            how="outer",
            validate="one_to_one",
        )
        .merge(
            background,
            on=["station_id", "year"],
            how="outer",
            validate="one_to_one",
        )
        .merge(
            stations[
                [
                    "station_id",
                    "station_name",
                    "station_tier",
                    "region",
                    "region_group",
                    "latitude",
                    "longitude",
                    "elevation_m",
                ]
            ],
            on="station_id",
            how="left",
            validate="many_to_one",
        )
        .sort_values(["station_id", "year"])
    )

    metric_rows = []
    sensitivity_rows = []
    for _, part in curves.groupby("station_id"):
        if part.qcf_station_degC.notna().any():
            for product in member_columns + [
                "project_six_member_degC",
                "berkeley_annual_degC",
                "cru_annual_degC",
            ]:
                metric_rows.extend(
                    base.metrics_for_pair(
                        part, "qcf_station_degC", product, product
                    )
                )
            sensitivity_rows.extend(
                base.metrics_for_pair(
                    part,
                    "qcf_station_degC",
                    "qcu_station_degC",
                    "qcu_vs_qcf",
                )
            )
    metrics = pd.DataFrame(metric_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    regional = base.matched_region_curves(curves)

    temp_dir = Path(
        tempfile.mkdtemp(prefix="temperature_station_benchmark_v2_", dir=base.RESULT_ROOT)
    )
    try:
        climatology.to_csv(
            temp_dir / "station_monthly_climatology_1961_1990.csv", index=False
        )
        coverage.to_csv(temp_dir / "station_baseline_coverage.csv", index=False)
        annual.to_csv(temp_dir / "station_annual_anomalies_long.csv", index=False)
        curves.to_csv(
            temp_dir / "station_annual_anomalies_and_products.csv", index=False
        )
        metrics.to_csv(temp_dir / "station_product_metrics.csv", index=False)
        sensitivity.to_csv(
            temp_dir / "qcu_qcf_sensitivity_metrics.csv", index=False
        )
        regional.to_csv(
            temp_dir / "regional_matched_support_curves.csv", index=False
        )
        base.station_figure(
            curves, stations, temp_dir / "station_annual_anomaly_curves.png"
        )
        base.regional_figure(
            regional, temp_dir / "regional_matched_support_curves.png"
        )

        qcf_eligible = set(
            coverage[
                coverage.series.eq("qcf") & coverage.baseline_eligible
            ].station_id
        )
        qcu_eligible = set(
            coverage[
                coverage.series.eq("qcu") & coverage.baseline_eligible
            ].station_id
        )
        qcf_years = curves.groupby("station_id").qcf_station_degC.count()
        disagreements = (
            int((~sensitivity.trend_sign_agreement).sum())
            if len(sensitivity)
            else 0
        )
        summary = {
            "experiment": OUTPUT.name,
            "status": "complete",
            "scientific_role": "station_anchored_annual_anomaly_benchmark_not_correction",
            "station_count": len(stations),
            "tier_a_count": int(stations.station_tier.eq("A").sum()),
            "tier_b_count": int(stations.station_tier.eq("B").sum()),
            "qcf_baseline_eligible_count": len(qcf_eligible),
            "qcu_baseline_eligible_count": len(qcu_eligible),
            "qcf_complete_historical_station_years": int(
                curves.qcf_station_degC.notna().sum()
            ),
            "qcu_complete_historical_station_years": int(
                curves.qcu_station_degC.notna().sum()
            ),
            "qcf_station_year_count_min_median_max": [
                int(qcf_years.min()),
                float(qcf_years.median()),
                int(qcf_years.max()),
            ],
            "qcu_qcf_segment_trend_sign_disagreement_count": disagreements,
            "unit_preflight": poison,
            "locked_interval": [base.LOCKED_START, base.LOCKED_END],
            "locked_monthly_cells_accessed": 0,
            "requested_historical_intervals": [[1850, 1920], [1940, 1950]],
            "baseline_interval": [1961, 1990],
            "minimum_monthly_baseline_years": base.MIN_MONTHLY_BASELINE_YEARS,
            "project_member_count": 6,
            "project_temporal_resolution": "calendar_annual",
            "station_aggregation": "day_weighted_12_complete_month_anomaly",
            "background_products_count_as_independent_votes": False,
        }
        (temp_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

        after = {name: base.file_state(path) for name, path in required.items()}
        manifest = {
            "experiment": OUTPUT.name,
            "inputs_before": before,
            "project_member_sources": member_sources,
            "inputs_after": after,
            "inputs_unchanged": before == after,
            "round277_v2_manifest_label_mixed": round277_manifest.get("experiment")
            != "round_277_direct_temperature_station_panel_1850_1950_v2",
            "round277_unit_verified_independently": poison,
            "ghcn_parse_audit": audits,
            "year_access_contract": {
                "historical": [[1850, 1920], [1940, 1950]],
                "baseline": [1961, 1990],
                "locked": [1921, 1939],
                "locked_values_accessed": False,
            },
        }
        if not manifest["inputs_unchanged"]:
            raise RuntimeError("source-integrity-failed")
        (temp_dir / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        eligible_names = stations[
            stations.station_id.isin(qcf_eligible)
        ].station_name.tolist()
        ineligible_names = stations[
            ~stations.station_id.isin(qcf_eligible)
        ].station_name.tolist()
        text = "\n".join(
            [
                "# Temperature Station Annual Anomaly Benchmark v2",
                "",
                "## Input closure",
                "",
                f"- GHCN-M `/100` unit poison test passed: `{poison['passed']}`.",
                "- qcf is primary; qcu is the same observation lineage and is used only for homogenization sensitivity.",
                "- Project data are calendar-annual; station values were aggregated from 12 complete monthly anomalies.",
                "- Berkeley and CRU are background processing products, not independent evidence votes.",
                "- Locked 1921-1939 monthly cells accessed: `0`.",
                "",
                "## Baseline coverage",
                "",
                f"- qcf stations passing the baseline gate: `{len(qcf_eligible)}/15`.",
                f"- Eligible qcf stations: `{', '.join(eligible_names) if eligible_names else 'none'}`.",
                f"- Ineligible qcf stations: `{', '.join(ineligible_names) if ineligible_names else 'none'}`.",
                f"- Complete qcf historical station-years: `{summary['qcf_complete_historical_station_years']}`.",
                "",
                "## Interpretation boundary",
                "",
                "This experiment establishes comparable annual anomaly curves. It does not authorize a correction. Stations failing the baseline gate cannot contribute to the primary residual diagnosis. qcu-qcf differences measure homogenization sensitivity; CRU and Berkeley remain source-correlated background context.",
            ]
        ) + "\n"
        (temp_dir / "FINAL_RESULT.md").write_text(text, encoding="utf-8")
        temp_dir.rename(OUTPUT)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
