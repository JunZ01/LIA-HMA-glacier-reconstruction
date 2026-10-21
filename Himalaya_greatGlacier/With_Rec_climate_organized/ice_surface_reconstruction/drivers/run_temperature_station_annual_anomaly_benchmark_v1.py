from __future__ import annotations

import calendar
import hashlib
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib
import numpy as np
import pandas as pd
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized"
    / "ice_surface_reconstruction/drivers"
)
RESULT_ROOT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
ROUND277 = RESULT_ROOT / "round_277_direct_temperature_station_panel_1850_1950_v2"
OUTPUT = RESULT_ROOT / "temperature_station_annual_anomaly_benchmark_v1"
CONTRACT = DRIVERS / "temperature_station_annual_anomaly_benchmark_v1_contract.md"
DRIVER = Path(__file__).resolve()
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
LINEAGE = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v1"
    / "round_00_production_reproduction/member_lineage.csv"
)
REFERENCE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data"
    r"\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)

PROJECT_MEMBERS = ("AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9")
BACKGROUND_VARIABLES = {
    "berkeley_annual_degC": "tas_obs_bea_05deg",
    "cru_annual_degC": "tas_obs_cru_tmp_05deg",
}
HISTORICAL_SEGMENTS = ((1850, 1879), (1880, 1900), (1901, 1920), (1940, 1950))
HISTORICAL_YEARS = tuple(range(1850, 1921)) + tuple(range(1940, 1951))
BASELINE_YEARS = tuple(range(1961, 1991))
VALUE_YEARS = set(HISTORICAL_YEARS) | set(BASELINE_YEARS)
LOCKED_START = 1921
LOCKED_END = 1939
MIN_MONTHLY_BASELINE_YEARS = 20


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def region_group(longitude: float) -> str:
    if longitude < 80.0:
        return "west_lt80"
    if longitude < 90.0:
        return "central_80_90"
    return "east_ge90"


def synthetic_line(station_id: str, year: int, token: str) -> str:
    cell = f"{token:>5}" + "   "
    return f"{station_id}{year:04d}TAVG" + cell * 12


def parse_ghcn_lines(
    lines: Iterable[bytes | str],
    station_ids: set[str],
    series: str,
    audit: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in lines:
        audit["physical_lines_seen"] += 1
        line = raw.decode("ascii", errors="strict") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if len(line) < 115:
            audit["short_lines_skipped"] += 1
            continue
        station_id = line[0:11]
        element = line[15:19]
        if station_id not in station_ids or element != "TAVG":
            audit["non_target_rows_skipped"] += 1
            continue
        year = int(line[11:15])
        audit["target_year_fields_decoded"] += 1
        if LOCKED_START <= year <= LOCKED_END:
            audit["locked_target_rows_skipped_before_value_access"] += 1
            continue
        if year not in VALUE_YEARS:
            audit["outside_requested_rows_skipped"] += 1
            continue
        for month in range(1, 13):
            audit["monthly_cells_accessed"] += 1
            start = 19 + (month - 1) * 8
            token = line[start : start + 5]
            measurement_flag = line[start + 5 : start + 6]
            quality_flag = line[start + 6 : start + 7]
            source_flag = line[start + 7 : start + 8]
            if token == "-9999":
                value = math.nan
            else:
                value = int(token) / 100.0
            accepted = bool(np.isfinite(value) and not quality_flag.strip())
            rows.append(
                {
                    "station_id": station_id,
                    "series": series,
                    "year": year,
                    "month": month,
                    "temperature_c": value,
                    "quality_accepted": accepted,
                    "measurement_flag": measurement_flag.strip(),
                    "quality_flag": quality_flag.strip(),
                    "source_flag": source_flag.strip(),
                }
            )
    return rows


def run_poison_test(station_id: str) -> dict[str, Any]:
    locked_audit: dict[str, int] = defaultdict(int)
    locked = parse_ghcn_lines(
        [synthetic_line(station_id, 1930, "XXXXX")],
        {station_id},
        "synthetic",
        locked_audit,
    )
    if locked or locked_audit["monthly_cells_accessed"] != 0:
        raise RuntimeError("locked-poison-reached-value-path")
    valid_audit: dict[str, int] = defaultdict(int)
    valid = parse_ghcn_lines(
        [synthetic_line(station_id, 1961, "  123")],
        {station_id},
        "synthetic",
        valid_audit,
    )
    if len(valid) != 12 or valid[0]["temperature_c"] != 1.23:
        raise RuntimeError("ghcn-hundredths-unit-test-failed")
    return {
        "passed": True,
        "integer_scale": 0.01,
        "expected_temperature_c": 1.23,
        "decoded_temperature_c": valid[0]["temperature_c"],
        "locked_monthly_cells_accessed": locked_audit["monthly_cells_accessed"],
        "locked_rows_skipped_before_value_access": locked_audit[
            "locked_target_rows_skipped_before_value_access"
        ],
    }


def monthly_climatology(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = monthly[monthly.quality_accepted].copy()
    baseline = accepted[accepted.year.between(1961, 1990)].copy()
    climatology = (
        baseline.groupby(["station_id", "series", "month"], as_index=False)
        .agg(
            baseline_temperature_c=("temperature_c", "mean"),
            baseline_year_count=("year", "nunique"),
        )
    )
    expected = pd.MultiIndex.from_product(
        [
            sorted(monthly.station_id.unique()),
            sorted(monthly.series.unique()),
            range(1, 13),
        ],
        names=["station_id", "series", "month"],
    ).to_frame(index=False)
    climatology = expected.merge(
        climatology, on=["station_id", "series", "month"], how="left"
    )
    climatology["baseline_year_count"] = (
        climatology.baseline_year_count.fillna(0).astype(int)
    )
    coverage = (
        climatology.groupby(["station_id", "series"], as_index=False)
        .agg(
            minimum_month_baseline_years=("baseline_year_count", "min"),
            maximum_month_baseline_years=("baseline_year_count", "max"),
            complete_calendar_months=(
                "baseline_year_count",
                lambda x: int(np.sum(np.asarray(x) >= MIN_MONTHLY_BASELINE_YEARS)),
            ),
        )
    )
    coverage["baseline_eligible"] = coverage.complete_calendar_months.eq(12)
    return climatology, coverage


def station_annual_anomalies(
    monthly: pd.DataFrame,
    climatology: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    historical = monthly[
        monthly.year.isin(HISTORICAL_YEARS) & monthly.quality_accepted
    ].copy()
    historical = historical.merge(
        climatology,
        on=["station_id", "series", "month"],
        how="left",
        validate="many_to_one",
    ).merge(
        coverage[["station_id", "series", "baseline_eligible"]],
        on=["station_id", "series"],
        how="left",
        validate="many_to_one",
    )
    historical = historical[
        historical.baseline_eligible & np.isfinite(historical.baseline_temperature_c)
    ].copy()
    historical["monthly_anomaly_degC"] = (
        historical.temperature_c - historical.baseline_temperature_c
    )
    historical["days"] = [
        calendar.monthrange(int(year), int(month))[1]
        for year, month in zip(historical.year, historical.month)
    ]
    counts = (
        historical.groupby(["station_id", "series", "year"], as_index=False)
        .month.nunique()
        .rename(columns={"month": "valid_months"})
    )
    complete = counts[counts.valid_months.eq(12)][["station_id", "series", "year"]]
    historical = historical.merge(
        complete,
        on=["station_id", "series", "year"],
        how="inner",
        validate="many_to_one",
    )
    historical["weighted_anomaly"] = historical.monthly_anomaly_degC * historical.days
    annual = (
        historical.groupby(["station_id", "series", "year"], as_index=False)
        .agg(weighted_anomaly=("weighted_anomaly", "sum"), days=("days", "sum"))
    )
    annual["station_annual_anomaly_degC"] = annual.weighted_anomaly / annual.days
    return annual[["station_id", "series", "year", "station_annual_anomaly_degC"]]


def bilinear_points(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    station_lat: np.ndarray,
    station_lon: np.ndarray,
) -> np.ndarray:
    data = np.asarray(field, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat[1] < lat[0]:
        lat = lat[::-1]
        data = data[:, ::-1, :]
    if lon[1] < lon[0]:
        lon = lon[::-1]
        data = data[:, :, ::-1]
    fy = (station_lat - lat[0]) / (lat[1] - lat[0])
    fx = (station_lon - lon[0]) / (lon[1] - lon[0])
    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    if np.any(y0 < 0) or np.any(y0 >= len(lat) - 1) or np.any(x0 < 0) or np.any(x0 >= len(lon) - 1):
        raise RuntimeError("station-outside-product-grid")
    wy = fy - y0
    wx = fx - x0
    corners = np.stack(
        [
            data[:, y0, x0],
            data[:, y0, x0 + 1],
            data[:, y0 + 1, x0],
            data[:, y0 + 1, x0 + 1],
        ],
        axis=0,
    )
    weights = np.stack(
        [(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx],
        axis=0,
    )
    valid = np.isfinite(corners)
    denominator = np.sum(weights[:, None, :] * valid, axis=0)
    numerator = np.nansum(corners * weights[:, None, :], axis=0)
    return numerator / np.where(denominator > 0, denominator, np.nan)


def source_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as source:
        lat = np.asarray(
            [source.xy(row, 0, offset="center")[1] for row in range(source.height)],
            dtype=float,
        )
        lon = np.asarray(
            [source.xy(0, col, offset="center")[0] for col in range(source.width)],
            dtype=float,
        )
    return lat, lon


def read_project_member(
    path: Path,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    years = np.asarray(sorted(VALUE_YEARS), dtype=int)
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
    values = bilinear_points(
        raw,
        lat,
        lon,
        stations.latitude.to_numpy(float),
        stations.longitude.to_numpy(float),
    )
    rows = []
    for station_index, station in enumerate(stations.itertuples(index=False)):
        baseline = values[np.isin(years, BASELINE_YEARS), station_index]
        if int(np.isfinite(baseline).sum()) != len(BASELINE_YEARS):
            raise RuntimeError(f"project-baseline-incomplete:{path.name}:{station.station_id}")
        centered = values[:, station_index] - float(np.mean(baseline))
        for year, value in zip(years, centered):
            if int(year) in HISTORICAL_YEARS:
                rows.append(
                    {
                        "station_id": station.station_id,
                        "year": int(year),
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def read_background_products(stations: pd.DataFrame) -> pd.DataFrame:
    with h5py.File(REFERENCE, "r") as handle:
        years_all = np.asarray(handle["time"][:], dtype=int)
        indices = np.flatnonzero(np.isin(years_all, sorted(VALUE_YEARS)))
        years = years_all[indices]
        if set(years.tolist()) != VALUE_YEARS:
            raise RuntimeError("background-requested-year-coverage-failed")
        lat = np.asarray(handle["lat"][:], dtype=float)
        lon = np.asarray(handle["lon"][:], dtype=float)
        sampled: dict[str, np.ndarray] = {}
        for output_name, variable in BACKGROUND_VARIABLES.items():
            raw = np.asarray(handle[variable][indices, :, :], dtype=float)
            raw[(raw < -9000) | (raw > 9000)] = np.nan
            sampled[output_name] = bilinear_points(
                raw,
                lat,
                lon,
                stations.latitude.to_numpy(float),
                stations.longitude.to_numpy(float),
            )
    rows = []
    for station_index, station in enumerate(stations.itertuples(index=False)):
        for output_name, values in sampled.items():
            baseline = values[np.isin(years, BASELINE_YEARS), station_index]
            finite = np.isfinite(baseline)
            if int(finite.sum()) < 20:
                centered = np.full(len(years), np.nan)
            else:
                centered = values[:, station_index] - float(np.mean(baseline[finite]))
            for year, value in zip(years, centered):
                if int(year) in HISTORICAL_YEARS:
                    rows.append(
                        {
                            "station_id": station.station_id,
                            "year": int(year),
                            "product": output_name,
                            "value": float(value),
                        }
                    )
    return pd.DataFrame(rows).pivot(
        index=["station_id", "year"], columns="product", values="value"
    ).reset_index()


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return math.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def trend(years: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(years) & np.isfinite(values)
    if int(valid.sum()) < 5:
        return math.nan
    return float(np.polyfit(years[valid], values[valid], 1)[0] * 100.0)


def metrics_for_pair(
    frame: pd.DataFrame,
    observed: str,
    modeled: str,
    label: str,
) -> list[dict[str, Any]]:
    rows = []
    for start, end in HISTORICAL_SEGMENTS:
        part = frame[frame.year.between(start, end)].copy()
        valid = np.isfinite(part[observed]) & np.isfinite(part[modeled])
        part = part[valid]
        if len(part) < 5:
            continue
        obs = part[observed].to_numpy(float)
        mod = part[modeled].to_numpy(float)
        obs_trend = trend(part.year.to_numpy(float), obs)
        mod_trend = trend(part.year.to_numpy(float), mod)
        rows.append(
            {
                "station_id": str(frame.station_id.iloc[0]),
                "segment": f"{start}_{end}",
                "product": label,
                "year_count": len(part),
                "correlation": correlation(obs, mod),
                "bias_degC": float(np.mean(mod - obs)),
                "rmse_degC": float(np.sqrt(np.mean((mod - obs) ** 2))),
                "station_trend_degC_century": obs_trend,
                "product_trend_degC_century": mod_trend,
                "trend_sign_agreement": bool(
                    np.isfinite(obs_trend)
                    and np.isfinite(mod_trend)
                    and np.sign(obs_trend) == np.sign(mod_trend)
                ),
            }
        )
    return rows


def plot_segments(axis: plt.Axes, frame: pd.DataFrame, column: str, **kwargs: Any) -> None:
    for start, end in ((1850, 1920), (1940, 1950)):
        part = frame[frame.year.between(start, end)]
        if part[column].notna().any():
            axis.plot(part.year, part[column], **kwargs)
        kwargs.pop("label", None)


def station_figure(curves: pd.DataFrame, stations: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(5, 3, figsize=(17, 20), sharex=True, sharey=True, constrained_layout=True)
    for axis, station in zip(axes.flat, stations.itertuples(index=False)):
        part = curves[curves.station_id.eq(station.station_id)].copy()
        plot_segments(axis, part, "qcf_station_degC", color="#111111", linewidth=1.8, label="GHCN qcf")
        plot_segments(axis, part, "qcu_station_degC", color="#777777", linewidth=1.1, linestyle="--", label="GHCN qcu")
        for member in PROJECT_MEMBERS:
            plot_segments(axis, part, f"project_{member}_degC", color="#D55E00", alpha=0.18, linewidth=0.7)
        plot_segments(axis, part, "project_six_member_degC", color="#D55E00", linewidth=1.5, label="Project mean")
        plot_segments(axis, part, "berkeley_annual_degC", color="#0072B2", linewidth=1.1, label="Berkeley")
        plot_segments(axis, part, "cru_annual_degC", color="#009E73", linewidth=1.1, label="CRU")
        axis.axhline(0, color="#AAAAAA", linewidth=0.6)
        axis.set_title(f"{station.station_name} | {station.region_group}", fontsize=10)
        axis.grid(alpha=0.18)
    axes.flat[0].legend(frameon=False, ncol=3, fontsize=8)
    fig.supxlabel("Year")
    fig.supylabel("Annual anomaly relative to 1961-1990 (degC)")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def matched_region_curves(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    products = [
        "project_six_member_degC",
        "berkeley_annual_degC",
        "cru_annual_degC",
    ]
    for region in ("west_lt80", "central_80_90", "east_ge90"):
        region_data = curves[curves.region_group.eq(region)]
        for year, year_data in region_data.groupby("year"):
            observed = year_data[np.isfinite(year_data.qcf_station_degC)]
            base = {
                "region_group": region,
                "year": int(year),
                "qcf_station_degC": float(observed.qcf_station_degC.mean()) if len(observed) else math.nan,
                "station_count": int(observed.station_id.nunique()),
            }
            for product in products:
                support = observed[np.isfinite(observed[product])]
                base[product] = float(support[product].mean()) if len(support) else math.nan
                base[f"{product}_matched_station_count"] = int(support.station_id.nunique())
            rows.append(base)
    return pd.DataFrame(rows)


def regional_figure(curves: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    for axis, region in zip(axes, ("west_lt80", "central_80_90", "east_ge90")):
        part = curves[curves.region_group.eq(region)]
        plot_segments(axis, part, "qcf_station_degC", color="#111111", linewidth=2.0, label="GHCN qcf matched network")
        plot_segments(axis, part, "project_six_member_degC", color="#D55E00", linewidth=1.8, label="Project mean")
        plot_segments(axis, part, "berkeley_annual_degC", color="#0072B2", linewidth=1.3, label="Berkeley background")
        plot_segments(axis, part, "cru_annual_degC", color="#009E73", linewidth=1.3, label="CRU background")
        axis.axhline(0, color="#999999", linewidth=0.7)
        axis.set_title(region)
        axis.set_ylabel("Anomaly (degC)")
        axis.grid(alpha=0.2)
        count_axis = axis.twinx()
        count_axis.fill_between(part.year, 0, part.station_count, color="#999999", alpha=0.10)
        count_axis.set_ylabel("Stations", color="#777777")
    axes[0].legend(frameon=False, ncol=4)
    axes[-1].set_xlabel("Year")
    fig.savefig(path, dpi=210)
    plt.close(fig)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"independent-output-exists:{OUTPUT}")
    required = {
        "facts": FACTS,
        "contract": CONTRACT,
        "driver": DRIVER,
        "round277_manifest": ROUND277 / "source_manifest.json",
        "round277_station_metadata": ROUND277 / "station_metadata.csv",
        "round277_qcf": ROUND277 / "prepared/ghcnm_tavg_qcf.data",
        "round277_qcu": ROUND277 / "prepared/ghcnm_tavg_qcu.data",
        "member_lineage": LINEAGE,
        "background_archive": REFERENCE,
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: file_state(path) for name, path in required.items()}

    round277_manifest = json.loads((ROUND277 / "source_manifest.json").read_text(encoding="utf-8"))
    expected_hashes = {
        Path(item["path"]).name: item["sha256"]
        for item in round277_manifest["source_files"]
    }
    for key in ("round277_qcf", "round277_qcu"):
        path = required[key]
        if expected_hashes.get(path.name) != before[key]["sha256"]:
            raise RuntimeError(f"round277-source-hash-mismatch:{path.name}")

    stations = pd.read_csv(ROUND277 / "station_metadata.csv")
    expected_columns = {"station_id", "station_name", "station_tier", "region", "latitude", "longitude"}
    if not expected_columns.issubset(stations.columns) or len(stations) != 15:
        raise RuntimeError("round277-station-metadata-contract-failed")
    stations = stations.sort_values(["station_tier", "longitude", "station_id"]).reset_index(drop=True)
    stations["region_group"] = stations.longitude.map(region_group)
    station_ids = set(stations.station_id)
    poison = run_poison_test(stations.station_id.iloc[0])

    audits: dict[str, dict[str, int]] = {}
    monthly_frames = []
    for series, path in (("qcf", required["round277_qcf"]), ("qcu", required["round277_qcu"])):
        audit: dict[str, int] = defaultdict(int)
        with path.open("rb") as handle:
            rows = parse_ghcn_lines(handle, station_ids, series, audit)
        audits[series] = dict(audit)
        monthly_frames.append(pd.DataFrame(rows))
    monthly = pd.concat(monthly_frames, ignore_index=True)
    if any(audit.get("monthly_cells_accessed", 0) != 15 * len(VALUE_YEARS) * 12 for audit in audits.values()):
        raise RuntimeError("unexpected-target-row-coverage")

    climatology, coverage = monthly_climatology(monthly)
    annual = station_annual_anomalies(monthly, climatology, coverage)
    annual_wide = annual.pivot(
        index=["station_id", "year"],
        columns="series",
        values="station_annual_anomaly_degC",
    ).reset_index().rename(columns={"qcf": "qcf_station_degC", "qcu": "qcu_station_degC"})

    lineage = pd.read_csv(LINEAGE)
    tas = lineage[(lineage.variable.eq("tas")) & lineage.member.isin(PROJECT_MEMBERS)].copy()
    if len(tas) != 6 or set(tas.member) != set(PROJECT_MEMBERS):
        raise RuntimeError("six-member-lineage-contract-failed")
    project_tables = []
    member_sources: dict[str, dict[str, Any]] = {}
    for row in tas.itertuples(index=False):
        path = Path(row.source_tif)
        if not path.exists() or sha256(path) != row.source_tif_sha256:
            raise RuntimeError(f"project-member-source-failed:{row.member}")
        member_sources[row.member] = file_state(path)
        table = read_project_member(path, stations).rename(columns={"value": f"project_{row.member}_degC"})
        project_tables.append(table)
    project = project_tables[0]
    for table in project_tables[1:]:
        project = project.merge(table, on=["station_id", "year"], validate="one_to_one")
    member_columns = [f"project_{member}_degC" for member in PROJECT_MEMBERS]
    project["project_six_member_degC"] = project[member_columns].mean(axis=1)

    background = read_background_products(stations)
    curves = (
        annual_wide.merge(project, on=["station_id", "year"], how="outer", validate="one_to_one")
        .merge(background, on=["station_id", "year"], how="outer", validate="one_to_one")
        .merge(
            stations[["station_id", "station_name", "station_tier", "region", "region_group", "latitude", "longitude", "elevation_m"]],
            on="station_id",
            how="left",
            validate="many_to_one",
        )
        .sort_values(["station_id", "year"])
    )

    metric_rows = []
    sensitivity_rows = []
    for station_id, part in curves.groupby("station_id"):
        if part.qcf_station_degC.notna().any():
            for product in member_columns + ["project_six_member_degC", "berkeley_annual_degC", "cru_annual_degC"]:
                metric_rows.extend(metrics_for_pair(part, "qcf_station_degC", product, product))
            sensitivity_rows.extend(metrics_for_pair(part, "qcf_station_degC", "qcu_station_degC", "qcu_vs_qcf"))
    metrics = pd.DataFrame(metric_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    regional = matched_region_curves(curves)

    temp_dir = Path(tempfile.mkdtemp(prefix="temperature_station_benchmark_", dir=RESULT_ROOT))
    try:
        climatology.to_csv(temp_dir / "station_monthly_climatology_1961_1990.csv", index=False)
        coverage.to_csv(temp_dir / "station_baseline_coverage.csv", index=False)
        annual.to_csv(temp_dir / "station_annual_anomalies_long.csv", index=False)
        curves.to_csv(temp_dir / "station_annual_anomalies_and_products.csv", index=False)
        metrics.to_csv(temp_dir / "station_product_metrics.csv", index=False)
        sensitivity.to_csv(temp_dir / "qcu_qcf_sensitivity_metrics.csv", index=False)
        regional.to_csv(temp_dir / "regional_matched_support_curves.csv", index=False)
        station_figure(curves, stations, temp_dir / "station_annual_anomaly_curves.png")
        regional_figure(regional, temp_dir / "regional_matched_support_curves.png")

        qcf_eligible = set(coverage[(coverage.series.eq("qcf")) & coverage.baseline_eligible].station_id)
        qcu_eligible = set(coverage[(coverage.series.eq("qcu")) & coverage.baseline_eligible].station_id)
        qcf_years = curves.groupby("station_id").qcf_station_degC.count()
        qcu_qcf_disagreements = int((~sensitivity.trend_sign_agreement).sum()) if len(sensitivity) else 0
        summary = {
            "experiment": OUTPUT.name,
            "status": "complete",
            "scientific_role": "station_anchored_annual_anomaly_benchmark_not_correction",
            "station_count": len(stations),
            "tier_a_count": int(stations.station_tier.eq("A").sum()),
            "tier_b_count": int(stations.station_tier.eq("B").sum()),
            "qcf_baseline_eligible_count": len(qcf_eligible),
            "qcu_baseline_eligible_count": len(qcu_eligible),
            "qcf_complete_historical_station_years": int(curves.qcf_station_degC.notna().sum()),
            "qcu_complete_historical_station_years": int(curves.qcu_station_degC.notna().sum()),
            "qcf_station_year_count_min_median_max": [
                int(qcf_years.min()),
                float(qcf_years.median()),
                int(qcf_years.max()),
            ],
            "qcu_qcf_segment_trend_sign_disagreement_count": qcu_qcf_disagreements,
            "unit_preflight": poison,
            "locked_interval": [LOCKED_START, LOCKED_END],
            "locked_monthly_cells_accessed": 0,
            "requested_historical_intervals": [[1850, 1920], [1940, 1950]],
            "baseline_interval": [1961, 1990],
            "minimum_monthly_baseline_years": MIN_MONTHLY_BASELINE_YEARS,
            "project_member_count": 6,
            "project_temporal_resolution": "calendar_annual",
            "station_aggregation": "day_weighted_12_complete_month_anomaly",
            "background_products_count_as_independent_votes": False,
            "source_files_unchanged": True,
        }
        (temp_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        manifest = {
            "experiment": OUTPUT.name,
            "inputs_before": before,
            "project_member_sources": member_sources,
            "inputs_after": {name: file_state(path) for name, path in required.items()},
            "round277_v2_manifest_label_mixed": round277_manifest.get("experiment") != "round_277_direct_temperature_station_panel_1850_1950_v2",
            "round277_unit_verified_independently": poison,
            "ghcn_parse_audit": audits,
            "year_access_contract": {
                "historical": [[1850, 1920], [1940, 1950]],
                "baseline": [1961, 1990],
                "locked": [1921, 1939],
                "locked_values_accessed": False,
            },
        }
        manifest["inputs_unchanged"] = manifest["inputs_before"] == manifest["inputs_after"]
        if not manifest["inputs_unchanged"]:
            raise RuntimeError("source-integrity-failed")
        (temp_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        eligible_names = stations[stations.station_id.isin(qcf_eligible)].station_name.tolist()
        ineligible_names = stations[~stations.station_id.isin(qcf_eligible)].station_name.tolist()
        result_text = "\n".join(
            [
                "# Temperature Station Annual Anomaly Benchmark v1",
                "",
                "## Input closure",
                "",
                f"- GHCN-M `/100` unit poison test passed: `{poison['passed']}`.",
                "- qcf is primary; qcu is the same observation lineage and is used only for homogenization sensitivity.",
                "- Project data are calendar-annual; station values were therefore aggregated from 12 complete monthly anomalies.",
                "- Berkeley and CRU are background processing products, not independent evidence votes.",
                "- Locked 1921-1939 monthly cells accessed: `0`.",
                "",
                "## Baseline coverage",
                "",
                f"- qcf stations passing the 20-values-per-calendar-month 1961-1990 gate: `{len(qcf_eligible)}/15`.",
                f"- Eligible qcf stations: `{', '.join(eligible_names) if eligible_names else 'none'}`.",
                f"- Ineligible qcf stations: `{', '.join(ineligible_names) if ineligible_names else 'none'}`.",
                f"- Complete qcf historical station-years: `{summary['qcf_complete_historical_station_years']}`.",
                "",
                "## Interpretation boundary",
                "",
                "This experiment establishes comparable annual anomaly curves. It does not yet authorize a temperature correction. Stations that fail the baseline gate cannot contribute to the primary residual diagnosis. qcu-qcf differences measure homogenization sensitivity, while CRU/Berkeley agreement remains source-correlated background context.",
            ]
        ) + "\n"
        (temp_dir / "FINAL_RESULT.md").write_text(result_text, encoding="utf-8")
        temp_dir.rename(OUTPUT)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
