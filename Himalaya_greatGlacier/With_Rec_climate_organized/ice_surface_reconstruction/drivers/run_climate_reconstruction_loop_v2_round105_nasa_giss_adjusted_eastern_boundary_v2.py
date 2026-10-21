"""Validate the eastern boundary phase with adjusted NASA GISS station CSVs."""

from __future__ import annotations

import calendar
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_climate_reconstruction_loop_v2_round100_darjeeling_temperature_phase_adjudication_v2 as round100
import run_climate_reconstruction_loop_v2_round101_eastern_foreland_temperature_phase_adjudication_v3 as round101


ROUND_ID = "round_105_nasa_giss_adjusted_eastern_boundary_v2"
STATION_IDS = ("IN003041800", "IN003050500", "IN014020800")
MONTH_COLUMNS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
EARLY_YEARS = set(range(1901, 1921))
MODERN_YEARS = set(range(1940, 2001))
ALLOWED_YEARS = EARLY_YEARS | MODERN_YEARS
LOCKED_YEARS = set(range(1921, 1940))
MAX_DISTANCE_KM = 200.0
NETWORK_CORRELATION_GATE = 0.50
SITE_MEDIAN_CORRELATION_GATE = 0.30


def read_csv_header_only(path: Path) -> list[str]:
    with path.open("r", encoding="ascii", newline="") as handle:
        header = next(csv.reader(handle))
    required = {"YEAR", *MONTH_COLUMNS}
    if not required.issubset(header):
        raise RuntimeError(f"station-csv-header-contract-failed:{path.name}:{header}")
    return header


def parse_allowed_station_csv(path: Path, station_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    header = read_csv_header_only(path)
    year_index = header.index("YEAR")
    month_indices = [header.index(month) for month in MONTH_COLUMNS]
    rows: list[dict[str, Any]] = []
    locked_rows_skipped = 0
    outside_allowed_rows_skipped = 0
    allowed_rows_seen = 0
    with path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.reader(handle)
        observed_header = next(reader)
        if observed_header != header:
            raise RuntimeError(f"station-csv-header-changed:{path.name}")
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise RuntimeError(f"station-csv-width-failed:{path.name}:{row_number}")

            # Contract: YEAR is the only converted field before the locked-period branch.
            year = int(row[year_index].strip())
            if year in LOCKED_YEARS:
                locked_rows_skipped += 1
                continue
            if year not in ALLOWED_YEARS:
                outside_allowed_rows_skipped += 1
                continue

            allowed_rows_seen += 1
            period = "early_1901_1920" if year in EARLY_YEARS else "modern_1940_2000"
            for month, value_index in enumerate(month_indices, start=1):
                token = row[value_index].strip()
                if not token or set(token) == {"*"}:
                    continue
                value = float(token)
                if not np.isfinite(value) or abs(value) >= 90.0:
                    continue
                rows.append(
                    {
                        "station_id": station_id,
                        "period": period,
                        "year": year,
                        "month": month,
                        "temperature_degC": value,
                    }
                )
    audit = {
        "station_id": station_id,
        "file": path.name,
        "header": header,
        "year_field_read_before_any_temperature_field": True,
        "locked_rows_skipped_before_temperature_parse": locked_rows_skipped,
        "outside_allowed_rows_skipped_before_temperature_parse": outside_allowed_rows_skipped,
        "allowed_rows_seen": allowed_rows_seen,
        "locked_temperature_values_parsed": 0,
        "locked_temperature_values_retained": 0,
        "locked_temperature_values_output": 0,
    }
    return rows, audit


def complete_annual(records: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame.from_records(records)
    if table.empty or not set(table.year.unique()).issubset(ALLOWED_YEARS):
        raise RuntimeError("allowed-year-station-record-contract-failed")
    table["climatology"] = table.groupby(["station_id", "period", "month"]).temperature_degC.transform("mean")
    table["anomaly"] = table.temperature_degC - table.climatology
    table["days"] = [calendar.monthrange(int(year), int(month))[1] for year, month in zip(table.year, table.month)]
    coverage = table.groupby(["station_id", "period", "year"], as_index=False).month.nunique()
    complete = coverage.loc[coverage.month.eq(12), ["station_id", "period", "year"]]
    table = table.merge(complete, on=["station_id", "period", "year"], how="inner", validate="many_to_one")
    table["weighted"] = table.anomaly * table.days
    annual = table.groupby(["station_id", "period", "year"], as_index=False).agg(
        station_anomaly_degC=("weighted", "sum"), days=("days", "sum")
    )
    annual.station_anomaly_degC /= annual.days
    annual.station_anomaly_degC = annual.groupby(["station_id", "period"]).station_anomaly_degC.transform(
        lambda values: values - values.mean()
    )
    return annual.drop(columns="days")


def read_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_stations(path: Path) -> pd.DataFrame:
    records = read_csv_records(path)
    stations = pd.DataFrame.from_records(records)
    stations = stations[stations.station_id.isin(STATION_IDS)].copy()
    if set(stations.station_id) != set(STATION_IDS):
        raise RuntimeError("station-inventory-contract-failed")
    for column in ("latitude", "longitude", "elevation_m"):
        stations[column] = stations[column].astype(float)
    return stations


def load_member_paths(path: Path) -> dict[str, Path]:
    records = read_csv_records(path)
    selected = {
        row["member"]: Path(row["source_tif"])
        for row in records
        if row.get("variable") == "tas" and row.get("member") in round100.PROJECT_MEMBERS
    }
    if set(selected) != set(round100.PROJECT_MEMBERS):
        raise RuntimeError("temperature-lineage-contract-failed")
    hashes = {row["member"]: row["source_tif_sha256"] for row in records if row.get("member") in selected and row.get("variable") == "tas"}
    for member, source in selected.items():
        if round100.digest(source) != hashes[member]:
            raise RuntimeError(f"source-tif-hash-mismatch:{member}")
    return selected


def make_metric_table(early: pd.DataFrame, modern: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, frame in (("early_1901_1920", early), ("modern_1940_2000", modern)):
        for column in ("project_six_member_degC", "reference_family_balanced_degC"):
            row = round100.metrics(frame, column)
            row["period"] = period
            rows.append(row)
    return pd.DataFrame(rows)


def metric_row(table: pd.DataFrame, period: str, product: str) -> pd.Series:
    rows = table[(table.period == period) & (table["product"] == product)]
    if len(rows) != 1:
        raise RuntimeError(f"metric-row-contract-failed:{period}:{product}")
    return rows.iloc[0]


def main() -> int:
    root = round100.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / ROUND_ID
    source_root = root / "validation_data/sources/nasa_giss_ghcn_v4_station_csv_allowed_parser_v1"
    lineage_path = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_00_production_reproduction/member_lineage.csv"
    baseline_root = loop / "round_101_eastern_foreland_temperature_phase_adjudication_v3"
    reference_path = loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc"
    station_inventory_path = baseline_root / "station_inventory.csv"
    baseline_summary_path = baseline_root / "summary.json"
    contract_path = output / "ROUND_CONTRACT.md"
    station_paths = {station_id: source_root / f"{station_id}.csv" for station_id in STATION_IDS}
    member_paths = load_member_paths(lineage_path)
    paths: dict[str, Path] = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract_path,
        "driver": Path(__file__).resolve(),
        "lineage": lineage_path,
        "baseline_summary": baseline_summary_path,
        "station_inventory": station_inventory_path,
        "reference": reference_path,
        **{f"station_{station_id}": path for station_id, path in station_paths.items()},
        **{f"member_{member}": path for member, path in member_paths.items()},
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if output.name != ROUND_ID or any((output / name).exists() for name in ("summary.json", "FINAL_RESULT.md")):
        raise FileExistsError(f"round105-output-already-populated:{output}")

    before = {name: round100.state(path) for name, path in paths.items()}
    baseline = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    if baseline.get("locked_1921_1939_accessed") is not False:
        raise RuntimeError("round101-locked-period-contract-failed")
    stations = load_stations(station_inventory_path)

    records: list[dict[str, Any]] = []
    parser_audits = []
    for station_id, path in station_paths.items():
        station_records, audit = parse_allowed_station_csv(path, station_id)
        records.extend(station_records)
        parser_audits.append(audit)
    annual = complete_annual(records)
    early_station_ids = sorted(annual.loc[annual.period.eq("early_1901_1920"), "station_id"].unique())
    modern_station_ids = sorted(annual.loc[annual.period.eq("modern_1940_2000"), "station_id"].unique())
    if len(early_station_ids) < 2 or len(modern_station_ids) < 2:
        raise RuntimeError(f"insufficient-real-station-support:{early_station_ids}:{modern_station_ids}")

    lat, lon, reference = round100.read_reference_allowed(reference_path)
    members = {name: round100.read_project_member_allowed(path, lat, lon) for name, path in member_paths.items()}
    project = {
        int(year): np.nanmean(np.stack([members[name][int(year)] for name in round100.PROJECT_MEMBERS]), axis=0)
        for year in sorted(annual.year.unique())
    }
    matched = round101.sample_products(annual, stations, lat, lon, project, reference)
    site_metrics = round101.station_metrics(matched)
    early = round101.network_curve(matched, "early_1901_1920")
    modern = round101.network_curve(matched, "modern_1940_2000")
    if early.empty or modern.empty:
        raise RuntimeError("empty-network-curve")
    curves = pd.concat(
        [early.assign(period="early_1901_1920"), modern.assign(period="modern_1940_2000")],
        ignore_index=True,
    )
    loo = round101.leave_one_station_out(matched, early)
    metrics = make_metric_table(early, modern)
    ep = metric_row(metrics, "early_1901_1920", "project_six_member_degC")
    er = metric_row(metrics, "early_1901_1920", "reference_family_balanced_degC")
    mp = metric_row(metrics, "modern_1940_2000", "project_six_member_degC")
    mr = metric_row(metrics, "modern_1940_2000", "reference_family_balanced_degC")

    modern_site = site_metrics[site_metrics.period.eq("modern_1940_2000")]
    modern_site_median_max = float(modern_site.groupby("product").correlation.median().max())
    maximum_distance = float(matched.station_grid_distance_km.max())
    distance_admissible = bool(maximum_distance <= MAX_DISTANCE_KM)
    minimum_modern_station_count = int(modern.station_count.min())
    adjusted_modern_overlap_coherent = bool(
        max(mp.correlation, mr.correlation) >= NETWORK_CORRELATION_GATE
        and modern_site_median_max >= SITE_MEDIAN_CORRELATION_GATE
        and minimum_modern_station_count >= 2
    )
    baseline_modern_overlap_coherent = bool(baseline["modern_overlap_coherent"])
    modern_inconsistency_repaired = bool(
        not baseline_modern_overlap_coherent and adjusted_modern_overlap_coherent and distance_admissible
    )
    loo_stable = bool(loo.station_curve_correlation.min() >= 0.50)

    project_preferred = bool(
        modern_inconsistency_repaired
        and loo_stable
        and ep.correlation - er.correlation >= 0.15
        and ep.centered_rmse_degC <= 0.90 * er.centered_rmse_degC
        and ep.trend_sign_agreement
    )
    reference_preferred = bool(
        modern_inconsistency_repaired
        and loo_stable
        and er.correlation - ep.correlation >= 0.15
        and er.centered_rmse_degC <= 0.90 * ep.centered_rmse_degC
        and er.trend_sign_agreement
    )
    early_decision = (
        "prefer_project_eastern_boundary_phase"
        if project_preferred
        else "prefer_reference_eastern_boundary_phase"
        if reference_preferred
        else "early_phase_not_credible_without_modern_gate"
        if not modern_inconsistency_repaired
        else "eastern_boundary_phase_inconclusive"
    )
    overall_decision = (
        "adjusted_series_repairs_qcu_modern_inconsistency"
        if modern_inconsistency_repaired
        else "adjusted_series_does_not_repair_qcu_modern_inconsistency"
    )

    output.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output / "adjusted_station_annual_anomalies_and_products.csv", index=False)
    curves.to_csv(output / "adjusted_network_curves.csv", index=False)
    site_metrics.to_csv(output / "adjusted_station_product_metrics.csv", index=False)
    metrics.to_csv(output / "adjusted_network_product_metrics.csv", index=False)
    loo.to_csv(output / "adjusted_early_leave_one_station_out.csv", index=False)
    comparison = pd.DataFrame(
        [
            {
                "station_series": "ghcnm_v4_qcu_round101_v3",
                "modern_overlap_coherent": baseline_modern_overlap_coherent,
                "modern_project_correlation": baseline["modern_project_metrics"]["correlation"],
                "modern_reference_correlation": baseline["modern_reference_metrics"]["correlation"],
                "early_observed_trend_degC_per_century": baseline["early_project_metrics"]["observed_trend_degC_per_century"],
                "early_project_correlation": baseline["early_project_metrics"]["correlation"],
                "early_reference_correlation": baseline["early_reference_metrics"]["correlation"],
            },
            {
                "station_series": "nasa_giss_ghcn_v4_adjusted_cleaned_homogenized",
                "modern_overlap_coherent": adjusted_modern_overlap_coherent,
                "modern_project_correlation": mp.correlation,
                "modern_reference_correlation": mr.correlation,
                "early_observed_trend_degC_per_century": ep.observed_trend_degC_per_century,
                "early_project_correlation": ep.correlation,
                "early_reference_correlation": er.correlation,
            },
        ]
    )
    comparison.to_csv(output / "qcu_vs_adjusted_comparison.csv", index=False)

    locked_audit = {
        "allowed_windows": [[1901, 1920], [1940, 2000]],
        "locked_window": [1921, 1939],
        "parser_order": "header-only pass; per data row convert YEAR; locked YEAR immediately continue; only then parse monthly values",
        "pandas_read_csv_used": False,
        "parser_audits": parser_audits,
        "locked_temperature_values_accessed": False,
        "locked_temperature_values_parsed": 0,
        "locked_temperature_values_retained": 0,
        "locked_temperature_values_output": 0,
    }
    (output / "locked_period_audit.json").write_text(json.dumps(locked_audit, indent=2), encoding="utf-8")

    summary = {
        "status": "complete",
        "round_id": ROUND_ID,
        "station_product": "NASA GISS GHCN v4 adjusted/cleaned/homogenized station CSV",
        "station_ids": list(STATION_IDS),
        "parsed_periods": [[1901, 1920], [1940, 2000]],
        "locked_1921_1939_accessed": False,
        "early_real_value_station_ids": early_station_ids,
        "modern_real_value_station_ids": modern_station_ids,
        "early_network_year_count": int(len(early)),
        "modern_network_year_count": int(len(modern)),
        "minimum_network_station_count": {
            "early": int(early.station_count.min()),
            "modern": minimum_modern_station_count,
        },
        "maximum_station_grid_distance_km": maximum_distance,
        "distance_admissible": distance_admissible,
        "leave_one_station_out_stable": loo_stable,
        "modern_gate": {
            "network_correlation_minimum": NETWORK_CORRELATION_GATE,
            "site_median_correlation_minimum": SITE_MEDIAN_CORRELATION_GATE,
            "maximum_distance_km": MAX_DISTANCE_KM,
            "adjusted_network_best_correlation": float(max(mp.correlation, mr.correlation)),
            "adjusted_site_median_best_correlation": modern_site_median_max,
            "baseline_qcu_modern_overlap_coherent": baseline_modern_overlap_coherent,
            "adjusted_modern_overlap_coherent": adjusted_modern_overlap_coherent,
            "modern_inconsistency_repaired": modern_inconsistency_repaired,
        },
        "early_project_metrics": ep.to_dict(),
        "early_reference_metrics": er.to_dict(),
        "modern_project_metrics": mp.to_dict(),
        "modern_reference_metrics": mr.to_dict(),
        "early_phase_decision": early_decision,
        "decision": overall_decision,
        "scope": "minimal_eastern_highland_foreland_boundary_phase_validation",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), constrained_layout=True)
    for axis, frame, title in zip(
        axes,
        (early, modern),
        ("1901-1920 adjusted-station network", "1940-2000 modern credibility gate"),
    ):
        axis.plot(frame.year, frame.station_anomaly_degC, color="#111111", marker="o", linewidth=2.1, label="NASA GISS adjusted stations")
        axis.plot(frame.year, frame.project_six_member_degC, color="#C23B22", linewidth=1.7, label="Project six-member")
        axis.plot(frame.year, frame.reference_family_balanced_degC, color="#1976A3", linewidth=1.7, label="Reference family-balanced")
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set(title=title, ylabel="Centered annual anomaly (degC)")
        axis.legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Year")
    fig.savefig(output / "nasa_giss_adjusted_eastern_boundary_curves.png", dpi=220)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# NASA GISS Adjusted Eastern Boundary Validation",
                "",
                "- Allowed windows: `1901-1920` and `1940-2000` only.",
                "- Locked-period contract: each row converted `YEAR` first and immediately skipped 1921-1939 before any temperature conversion; zero locked temperature values were parsed, retained, or output.",
                f"- Real-value early stations: `{early_station_ids}`; network years `{len(early)}` with minimum `{int(early.station_count.min())}` stations.",
                f"- Modern adjusted correlations: project `{mp.correlation:.3f}`, reference `{mr.correlation:.3f}`; best station-level median `{modern_site_median_max:.3f}`.",
                f"- Modern credibility gate: `{str(adjusted_modern_overlap_coherent).lower()}`; QCU baseline gate: `{str(baseline_modern_overlap_coherent).lower()}`; inconsistency repaired: `{str(modern_inconsistency_repaired).lower()}`.",
                f"- Maximum station-to-common-valid-grid distance: `{maximum_distance:.1f} km`.",
                f"- Early adjusted observed trend: `{ep.observed_trend_degC_per_century:+.3f} degC/century`.",
                f"- Early project: correlation `{ep.correlation:.3f}`, RMSE `{ep.centered_rmse_degC:.3f} degC`, trend `{ep.trend_degC_per_century:+.3f} degC/century`.",
                f"- Early reference: correlation `{er.correlation:.3f}`, RMSE `{er.centered_rmse_degC:.3f} degC`, trend `{er.trend_degC_per_century:+.3f} degC/century`.",
                "",
                f"Modern decision: `{overall_decision}`.",
                f"Early phase decision: `{early_decision}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    after = {name: round100.state(path) for name, path in paths.items()}
    manifest = {
        "round_id": ROUND_ID,
        "local_sources_before": before,
        "local_sources_after": after,
        "local_source_states_unchanged": before == after,
        "source_csv_sha256": {station_id: before[f"station_{station_id}"]["sha256"] for station_id in STATION_IDS},
        "parsed_periods": [[1901, 1920], [1940, 2000]],
        "locked_1921_1939_accessed": False,
        "pandas_read_csv_used": False,
        "primary_sampling": "round101 nearest common-valid grid cell and exact project six-member/reference fields",
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["local_source_states_unchanged"]:
        raise RuntimeError("local-source-integrity-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
