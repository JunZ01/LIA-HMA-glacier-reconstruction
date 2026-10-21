from __future__ import annotations

import json
from pathlib import Path

import rasterio
import numpy as np
import pandas as pd

import run_temperature_station_annual_anomaly_benchmark_v2 as experiment


OUTPUT = experiment.base.RESULT_ROOT / "temperature_station_annual_anomaly_benchmark_v4"
CONTRACT = experiment.base.DRIVERS / "temperature_station_annual_anomaly_benchmark_v4_contract.md"
DRIVER = Path(__file__).resolve()


def read_project_member_allow_masked(path: Path, stations: pd.DataFrame) -> pd.DataFrame:
    base = experiment.base
    years = np.asarray(sorted(base.VALUE_YEARS), dtype=int)
    bands = [int(year - 999) for year in years]
    with rasterio.open(path) as source:
        if min(bands) < 1 or max(bands) > source.count:
            raise RuntimeError(f"project-band-range-failed:{path.name}")
        raw = source.read(bands).astype(float)
        nodata = source.nodata if source.nodata is not None else -9999.0
        raw[np.isclose(raw, nodata) | (raw < -9000)] = np.nan
        lat = np.asarray(
            [source.xy(row, 0, offset="center")[1] for row in range(source.height)],
            dtype=float,
        )
        lon = np.asarray(
            [source.xy(0, col, offset="center")[0] for col in range(source.width)],
            dtype=float,
        )
    values = base.bilinear_points(
        raw,
        lat,
        lon,
        stations.latitude.to_numpy(float),
        stations.longitude.to_numpy(float),
    )
    rows = []
    baseline_mask = np.isin(years, base.BASELINE_YEARS)
    for station_index, station in enumerate(stations.itertuples(index=False)):
        baseline = values[baseline_mask, station_index]
        finite = np.isfinite(baseline)
        if int(finite.sum()) < 20:
            centered = np.full(len(years), np.nan)
        else:
            centered = values[:, station_index] - float(np.mean(baseline[finite]))
        for year, value in zip(years, centered):
            if int(year) in base.HISTORICAL_YEARS:
                rows.append(
                    {
                        "station_id": station.station_id,
                        "year": int(year),
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def append_coverage() -> None:
    curves_path = OUTPUT / "station_annual_anomalies_and_products.csv"
    summary_path = OUTPUT / "summary.json"
    result_path = OUTPUT / "FINAL_RESULT.md"
    curves = pd.read_csv(curves_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    columns = {
        "project_six_member_degC": "project",
        "berkeley_annual_degC": "berkeley",
        "cru_annual_degC": "cru",
    }
    coverage = {}
    for column, label in columns.items():
        usable = (
            curves.groupby(["station_id", "station_name"])[column]
            .apply(lambda values: bool(values.notna().any()))
            .reset_index(name="usable")
        )
        coverage[label] = {
            "station_count": int(usable.usable.sum()),
            "unavailable_stations": usable.loc[~usable.usable, "station_name"].tolist(),
        }
    summary["product_station_coverage"] = coverage
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["", "## Product spatial coverage", ""]
    for label, record in coverage.items():
        unavailable = ", ".join(record["unavailable_stations"]) or "none"
        lines.append(
            f"- {label}: `{record['station_count']}/15` stations; unavailable: `{unavailable}`."
        )
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


experiment.OUTPUT = OUTPUT
experiment.CONTRACT = CONTRACT
experiment.DRIVER = DRIVER
experiment.base.read_project_member = read_project_member_allow_masked


if __name__ == "__main__":
    code = experiment.main()
    append_coverage()
    raise SystemExit(code)
