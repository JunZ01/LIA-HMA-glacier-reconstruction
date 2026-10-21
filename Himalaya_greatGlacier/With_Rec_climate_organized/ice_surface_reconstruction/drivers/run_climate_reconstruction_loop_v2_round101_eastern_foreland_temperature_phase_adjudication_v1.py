"""Test early eastern HMA boundary temperature phase with three direct stations."""

from __future__ import annotations

import calendar
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_climate_reconstruction_loop_v2_round93_20crv3_ghcnm_v4_temperature_value_validation_v4 as round93
import run_climate_reconstruction_loop_v2_round100_darjeeling_temperature_phase_adjudication_v2 as round100


STATION_IDS = {"IN003041800", "IN003050500", "IN014020800"}
ALLOWED_YEARS = set(range(1901, 1921)) | set(range(1940, 2001))
EXPECTED_FACTS_SHA256 = "2e6765a63aa5d3b91b14d0e8768bafb53cb2a9fbc8f8810033c7f9ac8ddebde9"


def parse_allowed_monthly(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="strict").rstrip("\r\n")
            if len(line) < 115 or line[0:11] not in STATION_IDS or line[15:19] != "TAVG":
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
                            "station_id": line[0:11],
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


def complete_annual(monthly: pd.DataFrame) -> pd.DataFrame:
    table = monthly.copy()
    table["climatology"] = table.groupby(["station_id", "period", "month"]).temperature_degC.transform("mean")
    table["anomaly"] = table.temperature_degC - table.climatology
    table["days"] = [calendar.monthrange(int(y), int(m))[1] for y, m in zip(table.year, table.month)]
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


def sample_products(
    annual: pd.DataFrame,
    stations: pd.DataFrame,
    lat: np.ndarray,
    lon: np.ndarray,
    project: dict[int, np.ndarray],
    reference: dict[int, np.ndarray],
) -> pd.DataFrame:
    table = annual.merge(
        stations[["station_id", "name", "latitude", "longitude", "elevation_m"]],
        on="station_id",
        validate="many_to_one",
    )
    project_values = []
    reference_values = []
    for row in table.itertuples(index=False):
        lat_i = int(np.argmin(np.abs(lat - row.latitude)))
        lon_i = int(np.argmin(np.abs(lon - row.longitude)))
        project_values.append(float(project[int(row.year)][lat_i, lon_i]))
        reference_values.append(float(reference[int(row.year)][lat_i, lon_i]))
    table["project_six_member_degC"] = project_values
    table["reference_family_balanced_degC"] = reference_values
    for column in ["project_six_member_degC", "reference_family_balanced_degC"]:
        table[column] = table.groupby(["station_id", "period"])[column].transform(lambda values: values - values.mean())
    if not np.isfinite(table[["project_six_member_degC", "reference_family_balanced_degC"]].to_numpy()).all():
        raise RuntimeError("nonfinite-product-at-eastern-station")
    return table


def station_metrics(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (period, station_id), group in table.groupby(["period", "station_id"]):
        for column in ["project_six_member_degC", "reference_family_balanced_degC"]:
            row = round100.metrics(group, column)
            row.update({"period": period, "station_id": station_id, "station_name": group.name.iloc[0]})
            rows.append(row)
    return pd.DataFrame(rows)


def network_curve(table: pd.DataFrame, period: str) -> pd.DataFrame:
    frame = table[table.period.eq(period)].groupby("year", as_index=False).agg(
        station_anomaly_degC=("station_anomaly_degC", "mean"),
        project_six_member_degC=("project_six_member_degC", "mean"),
        reference_family_balanced_degC=("reference_family_balanced_degC", "mean"),
        station_count=("station_id", "nunique"),
    )
    frame = frame[frame.station_count.ge(2)].copy()
    for column in ["station_anomaly_degC", "project_six_member_degC", "reference_family_balanced_degC"]:
        frame[column] = round100.centered(frame[column].to_numpy(float))
    return frame


def leave_one_station_out(table: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station_id in sorted(table.station_id.unique()):
        reduced = network_curve(table[table.station_id.ne(station_id)], "early_1901_1920")
        paired = full.merge(reduced, on="year", suffixes=("_full", "_reduced"))
        rows.append(
            {
                "excluded_station_id": station_id,
                "year_count": int(len(paired)),
                "station_curve_correlation": round100.correlation(
                    paired.station_anomaly_degC_full.to_numpy(float),
                    paired.station_anomaly_degC_reduced.to_numpy(float),
                ),
                "reduced_station_trend_degC_per_century": round100.slope_century(
                    paired.year.to_numpy(float), paired.station_anomaly_degC_reduced.to_numpy(float)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    root = round100.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_101_eastern_foreland_temperature_phase_adjudication_v1"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": output / "ROUND_CONTRACT.md",
        "lineage": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_00_production_reproduction/member_lineage.csv",
        "ghcnm": loop / "round_93_20crv3_ghcnm_v4_temperature_value_validation_v4/ghcnm_qcu_temperature.dat",
        "inventory": loop / "round_93_20crv3_ghcnm_v4_temperature_value_validation_v4/ghcnm_qcu_inventory.inv",
        "reference": loop / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc",
        "driver": Path(__file__).resolve(),
    }
    lineage = pd.read_csv(paths["lineage"])
    tas = lineage[(lineage.variable.eq("tas")) & lineage.member.isin(round100.PROJECT_MEMBERS)].copy()
    if len(tas) != 6:
        raise RuntimeError("temperature-lineage-contract-failed")
    member_paths = {row.member: Path(row.source_tif) for row in tas.itertuples(index=False)}
    paths.update({f"member_{name}": path for name, path in member_paths.items()})
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if round100.digest(paths["facts"]) != EXPECTED_FACTS_SHA256:
        raise RuntimeError(f"facts-hash-conflict:{round100.digest(paths['facts'])}")
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: round100.state(path) for name, path in paths.items()}
    for row in tas.itertuples(index=False):
        if before[f"member_{row.member}"]["sha256"] != row.source_tif_sha256:
            raise RuntimeError(f"source-tif-hash-mismatch:{row.member}")

    inventory = round93.parse_inventory(paths["inventory"])
    stations = inventory[inventory.station_id.isin(STATION_IDS)].copy()
    if set(stations.station_id) != STATION_IDS:
        raise RuntimeError("eastern-station-inventory-contract-failed")
    stations.to_csv(output / "station_inventory.csv", index=False)
    annual = complete_annual(parse_allowed_monthly(paths["ghcnm"]))
    if set(annual.station_id.unique()) != STATION_IDS:
        raise RuntimeError("eastern-station-value-contract-failed")

    lat, lon, reference = round100.read_reference_allowed(paths["reference"])
    members = {name: round100.read_project_member_allowed(path, lat, lon) for name, path in member_paths.items()}
    years = sorted(annual.year.unique())
    project = {
        int(year): np.nanmean(np.stack([members[name][int(year)] for name in round100.PROJECT_MEMBERS]), axis=0)
        for year in years
    }
    matched = sample_products(annual, stations, lat, lon, project, reference)
    matched.to_csv(output / "station_annual_anomalies_and_products.csv", index=False)
    site_metrics = station_metrics(matched)
    site_metrics.to_csv(output / "station_product_metrics.csv", index=False)
    early = network_curve(matched, "early_1901_1920")
    modern = network_curve(matched, "modern_1940_2000")
    curves = pd.concat([early.assign(period="early_1901_1920"), modern.assign(period="modern_1940_2000")])
    curves.to_csv(output / "annual_network_curves.csv", index=False)
    loo = leave_one_station_out(matched, early)
    loo.to_csv(output / "early_leave_one_station_out.csv", index=False)

    metric_rows = []
    for period, frame in [("early_1901_1920", early), ("modern_1940_2000", modern)]:
        for column in ["project_six_member_degC", "reference_family_balanced_degC"]:
            row = round100.metrics(frame, column)
            row["period"] = period
            metric_rows.append(row)
    network_metrics = pd.DataFrame(metric_rows)
    network_metrics.to_csv(output / "network_product_metrics.csv", index=False)
    ep = network_metrics.query("period == 'early_1901_1920' and product == 'project_six_member_degC'").iloc[0]
    er = network_metrics.query("period == 'early_1901_1920' and product == 'reference_family_balanced_degC'").iloc[0]
    mp = network_metrics.query("period == 'modern_1940_2000' and product == 'project_six_member_degC'").iloc[0]
    mr = network_metrics.query("period == 'modern_1940_2000' and product == 'reference_family_balanced_degC'").iloc[0]
    modern_site = site_metrics[site_metrics.period.eq("modern_1940_2000")]
    modern_overlap_coherent = bool(
        max(mp.correlation, mr.correlation) >= 0.50
        and modern_site.groupby("product").correlation.median().max() >= 0.30
    )
    loo_stable = bool(loo.station_curve_correlation.min() >= 0.50)
    project_preferred = bool(
        modern_overlap_coherent and loo_stable and ep.correlation - er.correlation >= 0.15
        and ep.centered_rmse_degC <= 0.90 * er.centered_rmse_degC and ep.trend_sign_agreement
    )
    reference_preferred = bool(
        modern_overlap_coherent and loo_stable and er.correlation - ep.correlation >= 0.15
        and er.centered_rmse_degC <= 0.90 * ep.centered_rmse_degC and er.trend_sign_agreement
    )
    decision = (
        "prefer_project_eastern_boundary_phase" if project_preferred
        else "prefer_reference_eastern_boundary_phase" if reference_preferred
        else "eastern_boundary_phase_inconclusive"
    )
    summary = {
        "status": "complete",
        "station_ids": sorted(STATION_IDS),
        "parsed_periods": [[1901, 1920], [1940, 2000]],
        "locked_1921_1939_accessed": False,
        "early_network_year_count": int(len(early)),
        "modern_network_year_count": int(len(modern)),
        "minimum_network_station_count": {"early": int(early.station_count.min()), "modern": int(modern.station_count.min())},
        "leave_one_station_out_stable": loo_stable,
        "modern_overlap_coherent": modern_overlap_coherent,
        "early_project_metrics": ep.to_dict(),
        "early_reference_metrics": er.to_dict(),
        "modern_project_metrics": mp.to_dict(),
        "modern_reference_metrics": mr.to_dict(),
        "decision": decision,
        "scope": "eastern_highland_foreland_boundary_phase_only",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), constrained_layout=True)
    for axis, frame, title in zip(axes, [early, modern], ["Early eastern boundary network", "Modern overlap diagnostic"]):
        axis.plot(frame.year, frame.station_anomaly_degC, color="#111111", marker="o", linewidth=2.1, label="GHCN-M QCU stations")
        axis.plot(frame.year, frame.project_six_member_degC, color="#D55E00", linewidth=1.7, label="Project six-member")
        axis.plot(frame.year, frame.reference_family_balanced_degC, color="#0072B2", linewidth=1.7, label="Reference family-balanced")
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set(title=title, ylabel="Centered annual anomaly (degC)")
        axis.legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Year")
    fig.savefig(output / "eastern_boundary_temperature_phase_curves.png", dpi=220)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "\n".join([
            "# Eastern Boundary Direct Temperature Phase",
            "",
            f"- Network support: `{len(early)}` early years and `{len(modern)}` modern years; minimum station count `{int(early.station_count.min())}` / `{int(modern.station_count.min())}`.",
            f"- Leave-one-station-out stable: `{str(loo_stable).lower()}`; minimum correlation `{loo.station_curve_correlation.min():.3f}`.",
            f"- Modern overlap: project correlation `{mp.correlation:.3f}`, reference `{mr.correlation:.3f}`; coherent `{str(modern_overlap_coherent).lower()}`.",
            f"- Early observed trend `{ep.observed_trend_degC_per_century:+.3f} degC/century`.",
            f"- Early project: correlation `{ep.correlation:.3f}`, RMSE `{ep.centered_rmse_degC:.3f} degC`, trend `{ep.trend_degC_per_century:+.3f} degC/century`.",
            f"- Early reference: correlation `{er.correlation:.3f}`, RMSE `{er.centered_rmse_degC:.3f} degC`, trend `{er.trend_degC_per_century:+.3f} degC/century`.",
            "- These stations constrain the eastern highland/foreland boundary, not the interior plateau field.",
            "",
            f"Decision: `{decision}`.",
        ]) + "\n",
        encoding="utf-8",
    )
    after = {name: round100.state(path) for name, path in paths.items()}
    manifest = {
        "local_sources_before": before,
        "local_sources_after": after,
        "local_source_states_unchanged": before == after,
        "parsed_periods": [[1901, 1920], [1940, 2000]],
        "locked_1921_1939_accessed": False,
        "station_ids": sorted(STATION_IDS),
        "primary_sampling": "matched_station_cells_and_complete_station_years",
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["local_source_states_unchanged"]:
        raise RuntimeError("local-source-integrity-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
