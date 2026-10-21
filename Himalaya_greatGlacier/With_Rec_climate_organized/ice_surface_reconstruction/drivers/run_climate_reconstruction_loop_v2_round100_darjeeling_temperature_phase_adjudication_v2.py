"""Adjudicate early temperature phase at Darjeeling with modern overlap context."""

from __future__ import annotations

import calendar
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_climate_reconstruction_loop_v2_round97_western_ghcnm_temperature_phase_adjudication_v1 import (
    PROJECT_MEMBERS,
    centered,
    correlation,
    digest,
    interpolate_to_grid,
    slope_century,
    state,
)


STATION_ID = "INXLT811965"
EARLY_YEARS = np.arange(1901, 1921)
MODERN_YEARS = np.arange(1940, 2001)
ALLOWED_YEARS = set(EARLY_YEARS.tolist() + MODERN_YEARS.tolist())
EXPECTED_FACTS_SHA256 = "2e6765a63aa5d3b91b14d0e8768bafb53cb2a9fbc8f8810033c7f9ac8ddebde9"


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def parse_allowed_monthly_values(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="strict").rstrip("\r\n")
            if len(line) < 115 or line[0:11] != STATION_ID or line[15:19] != "TAVG":
                continue
            year = int(line[11:15])
            if year not in ALLOWED_YEARS:
                continue
            period = "early_1901_1920" if year <= 1920 else "modern_1940_2000"
            for month in range(1, 13):
                start = 19 + (month - 1) * 8
                token = line[start : start + 5]
                qc_blank = not line[start + 6 : start + 7].strip()
                if token != "-9999" and qc_blank:
                    rows.append(
                        {
                            "station_id": STATION_ID,
                            "period": period,
                            "year": year,
                            "month": month,
                            "temperature_degC": int(token) / 100.0,
                        }
                    )
    result = pd.DataFrame(rows)
    if result.empty or not set(result.year.unique()).issubset(ALLOWED_YEARS):
        raise RuntimeError("allowed-year-parser-contract-failed")
    return result


def complete_annual_anomalies(monthly: pd.DataFrame) -> pd.DataFrame:
    table = monthly.copy()
    table["monthly_climatology"] = table.groupby(["period", "month"]).temperature_degC.transform("mean")
    table["monthly_anomaly_degC"] = table.temperature_degC - table.monthly_climatology
    table["days"] = [calendar.monthrange(int(y), int(m))[1] for y, m in zip(table.year, table.month)]
    coverage = table.groupby(["period", "year"], as_index=False).month.nunique()
    complete = coverage.loc[coverage.month.eq(12), ["period", "year"]]
    table = table.merge(complete, on=["period", "year"], how="inner", validate="many_to_one")
    table["weighted"] = table.monthly_anomaly_degC * table.days
    annual = table.groupby(["period", "year"], as_index=False).agg(
        station_anomaly_degC=("weighted", "sum"), days=("days", "sum")
    )
    annual.station_anomaly_degC /= annual.days
    annual.station_anomaly_degC = annual.groupby("period").station_anomaly_degC.transform(lambda x: x - x.mean())
    return annual.drop(columns="days")


def read_reference_allowed(path: Path) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    with netcdf_file(path, "r", mmap=True) as dataset:
        years = np.asarray(dataset.variables["year"].data, dtype=int)
        indices = np.flatnonzero(np.isin(years, sorted(ALLOWED_YEARS)))
        selected_years = years[indices].copy()
        if not set(selected_years.tolist()).issubset(ALLOWED_YEARS):
            raise RuntimeError("reference-locked-year-contract-failed")
        lat = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        lon = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
        variable = dataset.variables["tas_reference_family_balanced"]
        raw = np.asarray(variable.data[indices], dtype=float).copy()
        fill = float(getattr(variable, "_FillValue", -9999.0))
    raw[np.isclose(raw, fill) | (raw < -9000)] = np.nan
    return lat, lon, {int(year): raw[i] for i, year in enumerate(selected_years)}


def read_project_member_allowed(path: Path, lat: np.ndarray, lon: np.ndarray) -> dict[int, np.ndarray]:
    years = np.asarray(sorted(ALLOWED_YEARS), dtype=int)
    bands = (years - 999).tolist()
    with rasterio.open(path) as source:
        raw = source.read(bands).astype(float)
        nodata = source.nodata if source.nodata is not None else -9999.0
        raw[np.isclose(raw, nodata) | (raw < -9000)] = np.nan
        source_lat = np.asarray([source.xy(row, 0, offset="center")[1] for row in range(source.height)])
        source_lon = np.asarray([source.xy(0, col, offset="center")[0] for col in range(source.width)])
    field = interpolate_to_grid(raw, source_lat, source_lon, lat, lon)
    return {int(year): field[i] for i, year in enumerate(years)}


def product_at_station(fields: dict[int, np.ndarray], years: np.ndarray, lat_i: int, lon_i: int) -> np.ndarray:
    values = np.asarray([fields[int(year)][lat_i, lon_i] for year in years], dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("nonfinite-product-at-darjeeling")
    return centered(values)


def metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    observed = frame.station_anomaly_degC.to_numpy(float)
    product = frame[column].to_numpy(float)
    rmse = float(np.sqrt(np.mean((product - observed) ** 2)))
    zero_rmse = float(np.sqrt(np.mean(observed**2)))
    observed_trend = slope_century(frame.year.to_numpy(float), observed)
    product_trend = slope_century(frame.year.to_numpy(float), product)
    return {
        "product": column,
        "year_count": int(len(frame)),
        "correlation": correlation(product, observed),
        "centered_rmse_degC": rmse,
        "rmse_skill_vs_zero": 1.0 - rmse / zero_rmse,
        "trend_degC_per_century": product_trend,
        "observed_trend_degC_per_century": observed_trend,
        "trend_sign_agreement": bool(np.sign(product_trend) == np.sign(observed_trend)),
        "opposite_sign_year_count": int(np.sum(product * observed < 0)),
    }


def leave_one_year_out_trends(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for omitted in frame.year:
        kept = frame[frame.year.ne(omitted)]
        row: dict[str, Any] = {"omitted_year": int(omitted)}
        for column in columns:
            row[f"{column}_trend_degC_per_century"] = slope_century(
                kept.year.to_numpy(float), kept[column].to_numpy(float)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_100_darjeeling_temperature_phase_adjudication_v2"
    lineage_path = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_00_production_reproduction/member_lineage.csv"
    station_path = loop / "round_96_ghcnm_1901_1920_temperature_coverage_readiness_v1/final_station_support.csv"
    ghcnm_path = loop / "round_93_20crv3_ghcnm_v4_temperature_value_validation_v4/ghcnm_qcu_temperature.dat"
    reference_path = loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc"
    facts_path = root / "LIA_PROJECT_FACTS.md"
    contract_path = output / "ROUND_CONTRACT.md"
    lineage = pd.read_csv(lineage_path)
    tas = lineage[(lineage.variable.eq("tas")) & lineage.member.isin(PROJECT_MEMBERS)].copy()
    if len(tas) != 6 or set(tas.member) != set(PROJECT_MEMBERS):
        raise RuntimeError("temperature-lineage-contract-failed")
    member_paths = {row.member: Path(row.source_tif) for row in tas.itertuples(index=False)}
    paths = {
        "facts": facts_path,
        "contract": contract_path,
        "lineage": lineage_path,
        "station_support": station_path,
        "ghcnm": ghcnm_path,
        "reference": reference_path,
        "driver": Path(__file__).resolve(),
        **{f"member_{name}": path for name, path in member_paths.items()},
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if digest(facts_path) != EXPECTED_FACTS_SHA256:
        raise RuntimeError(f"facts-hash-conflict:{digest(facts_path)}")
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    for row in tas.itertuples(index=False):
        if before[f"member_{row.member}"]["sha256"] != row.source_tif_sha256:
            raise RuntimeError(f"source-tif-hash-mismatch:{row.member}")

    station = pd.read_csv(station_path).query("station_id == @STATION_ID")
    if len(station) != 1 or station.iloc[0].region != "central_80_90":
        raise RuntimeError("darjeeling-station-contract-failed")
    monthly = parse_allowed_monthly_values(ghcnm_path)
    annual = complete_annual_anomalies(monthly)
    early_count = int((annual.period == "early_1901_1920").sum())
    modern_count = int((annual.period == "modern_1940_2000").sum())
    if early_count != 16 or modern_count < 20:
        raise RuntimeError(f"annual-support-contract-failed:{early_count}:{modern_count}")

    lat, lon, reference = read_reference_allowed(reference_path)
    lat_i = int(np.argmin(np.abs(lat - float(station.iloc[0].grid_latitude))))
    lon_i = int(np.argmin(np.abs(lon - float(station.iloc[0].grid_longitude))))
    member_fields = {name: read_project_member_allowed(path, lat, lon) for name, path in member_paths.items()}
    all_years = np.asarray(sorted(annual.year.unique()), dtype=int)
    project_by_year = {}
    for year in all_years:
        stack = np.stack([member_fields[name][int(year)] for name in PROJECT_MEMBERS])
        project_by_year[int(year)] = np.nanmean(stack, axis=0)

    frames = []
    for period, group in annual.groupby("period", sort=False):
        group = group.sort_values("year").copy()
        years = group.year.to_numpy(int)
        group["project_six_member_degC"] = product_at_station(project_by_year, years, lat_i, lon_i)
        group["reference_family_balanced_degC"] = product_at_station(reference, years, lat_i, lon_i)
        frames.append(group)
    curves = pd.concat(frames, ignore_index=True)
    curves.to_csv(output / "darjeeling_annual_station_and_products.csv", index=False)

    metric_rows = []
    for period, group in curves.groupby("period", sort=False):
        for column in ["project_six_member_degC", "reference_family_balanced_degC"]:
            row = metrics(group, column)
            row["period"] = period
            metric_rows.append(row)
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(output / "product_metrics.csv", index=False)
    early = curves[curves.period.eq("early_1901_1920")].copy()
    loo = leave_one_year_out_trends(
        early,
        ["station_anomaly_degC", "project_six_member_degC", "reference_family_balanced_degC"],
    )
    loo.to_csv(output / "early_leave_one_year_out_trends.csv", index=False)

    early_project = metric_table.query("period == 'early_1901_1920' and product == 'project_six_member_degC'").iloc[0]
    early_reference = metric_table.query("period == 'early_1901_1920' and product == 'reference_family_balanced_degC'").iloc[0]
    modern_project = metric_table.query("period == 'modern_1940_2000' and product == 'project_six_member_degC'").iloc[0]
    modern_reference = metric_table.query("period == 'modern_1940_2000' and product == 'reference_family_balanced_degC'").iloc[0]
    modern_overlap_coherent = bool(max(modern_project.correlation, modern_reference.correlation) >= 0.50)
    project_preferred = bool(
        modern_overlap_coherent
        and early_project.correlation - early_reference.correlation >= 0.15
        and early_project.centered_rmse_degC <= 0.90 * early_reference.centered_rmse_degC
        and early_project.trend_sign_agreement
    )
    reference_preferred = bool(
        modern_overlap_coherent
        and early_reference.correlation - early_project.correlation >= 0.15
        and early_reference.centered_rmse_degC <= 0.90 * early_project.centered_rmse_degC
        and early_reference.trend_sign_agreement
    )
    decision = (
        "prefer_project_at_darjeeling"
        if project_preferred
        else "prefer_reference_at_darjeeling"
        if reference_preferred
        else "darjeeling_single_site_phase_inconclusive"
    )
    station_loo = loo.station_anomaly_degC_trend_degC_per_century.to_numpy(float)
    summary = {
        "status": "complete",
        "station": station.iloc[0].to_dict(),
        "parsed_periods": [[1901, 1920], [1940, 2000]],
        "locked_1921_1939_accessed": False,
        "complete_early_year_count": early_count,
        "complete_modern_year_count": modern_count,
        "modern_overlap_coherent": modern_overlap_coherent,
        "early_station_trend_leave_one_year_out_sign_fraction": float(np.mean(np.sign(station_loo) == np.sign(early_project.observed_trend_degC_per_century))),
        "early_project_metrics": early_project.to_dict(),
        "early_reference_metrics": early_reference.to_dict(),
        "modern_project_metrics": modern_project.to_dict(),
        "modern_reference_metrics": modern_reference.to_dict(),
        "decision": decision,
        "scope": "single_site_phase_constraint_only",
        "source_independence": "GHCN-M QCU is direct station data; gridded reference products may ingest the same station, so preference over the reference is not fully independent.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), constrained_layout=True)
    colors = {"station_anomaly_degC": "#111111", "project_six_member_degC": "#D55E00", "reference_family_balanced_degC": "#0072B2"}
    labels = {"station_anomaly_degC": "GHCN-M QCU Darjeeling", "project_six_member_degC": "Project six-member", "reference_family_balanced_degC": "Reference family-balanced"}
    for axis, period, title in zip(axes, ["early_1901_1920", "modern_1940_2000"], ["Early complete years", "Modern overlap diagnostic"]):
        frame = curves[curves.period.eq(period)]
        for column in colors:
            axis.plot(frame.year, frame[column], marker="o" if column == "station_anomaly_degC" else None, linewidth=2.0 if column == "station_anomaly_degC" else 1.5, color=colors[column], label=labels[column])
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set(title=f"Darjeeling: {title}", ylabel="Centered annual anomaly (degC)")
        axis.legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Year")
    fig.savefig(output / "darjeeling_temperature_phase_curves.png", dpi=220)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Darjeeling Direct Temperature Phase Adjudication",
                "",
                f"- Complete station years: `{early_count}` in 1901-1920 and `{modern_count}` in 1940-2000; locked years were not parsed.",
                f"- Modern overlap correlation: project `{modern_project.correlation:.3f}`, reference `{modern_reference.correlation:.3f}`; overlap coherent: `{str(modern_overlap_coherent).lower()}`.",
                f"- Early station trend: `{early_project.observed_trend_degC_per_century:+.3f} degC/century`; leave-one-year-out sign stability `{summary['early_station_trend_leave_one_year_out_sign_fraction']:.1%}`.",
                f"- Early project: correlation `{early_project.correlation:.3f}`, RMSE `{early_project.centered_rmse_degC:.3f} degC`, trend `{early_project.trend_degC_per_century:+.3f} degC/century`.",
                f"- Early reference: correlation `{early_reference.correlation:.3f}`, RMSE `{early_reference.centered_rmse_degC:.3f} degC`, trend `{early_reference.trend_degC_per_century:+.3f} degC/century`.",
                "- This is one central-HMA station and cannot validate the central/eastern HMA field.",
                "",
                f"Decision: `{decision}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    manifest = {
        "local_sources_before": before,
        "local_sources_after": after,
        "local_source_states_unchanged": before == after,
        "parsed_periods": [[1901, 1920], [1940, 2000]],
        "locked_1921_1939_accessed": False,
        "station_id": STATION_ID,
        "primary_sampling": "same_station_grid_cell_and_same_complete_station_year_support",
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["local_source_states_unchanged"]:
        raise RuntimeError("local-source-integrity-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
