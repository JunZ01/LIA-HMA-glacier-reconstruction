"""Test seasonal precipitation skill for documented-moisture proxy groups."""

from __future__ import annotations

import calendar
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1940, 2001, dtype=int)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
SEASON_MONTHS = {
    "JJA": (6, 7, 8),
    "Jun-Sep": (6, 7, 8, 9),
    "JJAS_boreal_summer": (6, 7, 8, 9),
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def nearest(source: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.asarray([int(np.argmin(np.abs(source - value))) for value in targets])


def read_netcdf3_points(
    path: Path,
    variable: str,
    source_start_year: int,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    start = (1940 - source_start_year) * 12
    stop = (2001 - source_start_year) * 12
    expected = len(YEARS) * 12
    result = np.full((expected, len(latitudes)), np.nan)
    with netcdf_file(path, "r", mmap=True) as handle:
        source_lat = np.array(handle.variables["lat"][:], dtype=float, copy=True)
        source_lon = np.array(handle.variables["lon"][:], dtype=float, copy=True)
        iy = nearest(source_lat, latitudes)
        ix = nearest(source_lon, longitudes)
        source = handle.variables[variable]
        fill = float(source._attributes.get("_FillValue", -99999.99))
        for column, (row_index, col_index) in enumerate(zip(iy, ix)):
            values = np.array(source[start:stop, row_index, col_index], dtype=float, copy=True)
            if len(values) != expected:
                raise RuntimeError(f"monthly-length-failed:{path.name}:{len(values)}")
            values[(values == fill) | (values <= -9000) | (values >= 1e20)] = np.nan
            result[:, column] = values
    return result


def read_berkeley_points(path: Path, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    start = (1940 - 1850) * 12
    stop = (2001 - 1850) * 12
    expected = len(YEARS) * 12
    result = np.full((expected, len(latitudes)), np.nan)
    with h5py.File(path, "r") as handle:
        source_lat = np.asarray(handle["latitude"], dtype=float)
        source_lon = np.asarray(handle["longitude"], dtype=float)
        iy = nearest(source_lat, latitudes)
        ix = nearest(source_lon, longitudes)
        for column, (row_index, col_index) in enumerate(zip(iy, ix)):
            values = np.asarray(handle["temperature"][start:stop, row_index, col_index], dtype=float)
            if len(values) != expected:
                raise RuntimeError(f"berkeley-monthly-length-failed:{len(values)}")
            values[(values <= -9000) | (values >= 1e20)] = np.nan
            result[:, column] = values
    return result


def aggregate_temperature(monthly: np.ndarray, months_by_record: list[tuple[int, ...]]) -> np.ndarray:
    result = np.full((len(YEARS), monthly.shape[1]), np.nan)
    for column, months in enumerate(months_by_record):
        month_index = np.asarray(months, dtype=int) - 1
        values = monthly[:, column].reshape(len(YEARS), 12)[:, month_index]
        complete = np.all(np.isfinite(values), axis=1)
        result[complete, column] = np.mean(values[complete], axis=1)
    return result


def aggregate_precipitation(monthly: np.ndarray, months_by_record: list[tuple[int, ...]]) -> np.ndarray:
    result = np.full((len(YEARS), monthly.shape[1]), np.nan)
    for column, months in enumerate(months_by_record):
        month_index = np.asarray(months, dtype=int) - 1
        values = monthly[:, column].reshape(len(YEARS), 12)[:, month_index]
        for row, year in enumerate(YEARS):
            if np.all(np.isfinite(values[row])):
                days = sum(calendar.monthrange(int(year), int(month))[1] for month in months)
                result[row, column] = float(np.sum(values[row]) / days)
    return result


def center(values: np.ndarray) -> np.ndarray:
    reference = (YEARS >= 1961) & (YEARS <= 1990)
    return values - np.nanmean(values[reference], axis=0, keepdims=True)


def fit(response: np.ndarray, temperature: np.ndarray, precipitation: np.ndarray | None, train: np.ndarray):
    valid = train & np.isfinite(response) & np.isfinite(temperature)
    if precipitation is not None:
        valid &= np.isfinite(precipitation)
    if int(valid.sum()) < 30:
        return None
    columns = [np.ones(int(valid.sum())), temperature[valid]]
    if precipitation is not None:
        columns.append(precipitation[valid])
    return np.linalg.lstsq(np.column_stack(columns), response[valid], rcond=None)[0]


def predict(coefficients, temperature: np.ndarray, precipitation: np.ndarray | None, test: np.ndarray) -> np.ndarray:
    result = np.full(len(YEARS), np.nan)
    valid = test & np.isfinite(temperature)
    if precipitation is not None:
        valid &= np.isfinite(precipitation)
    if coefficients is not None:
        columns = [np.ones(int(valid.sum())), temperature[valid]]
        if precipitation is not None:
            columns.append(precipitation[valid])
        result[valid] = np.column_stack(columns) @ coefficients
    return result


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if int(valid.sum()) < 30:
        return float("nan")
    return float(np.sqrt(np.mean((observed[valid] - predicted[valid]) ** 2)))


def correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if int(valid.sum()) < 3 or np.std(observed[valid]) == 0 or np.std(predicted[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(observed[valid], predicted[valid])[0, 1])


def main() -> int:
    root = project_root()
    driver = Path(__file__).resolve().parent
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_118_documented_moisture_seasonal_incremental_skill_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "proxy9": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "ledger": loop / "round_110_unresolved_proxy_metadata_recovery_v4/proxy_evidence_master_ledger_metadata_recovered.csv",
        "berkeley_temperature": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "cru_temperature": phyda / "input-recon/cruts.2004151855.v4.04/tmp/cru_ts4.04.1901.2019.tmp.dat.nc",
        "cru_precipitation": phyda / "input-recon/cruts.2004151855.v4.04/pre/cru_ts4.04.1901.2019.pre.dat.nc",
        "gpcc_precipitation": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "contract": driver / "round118_documented_moisture_seasonal_incremental_skill_v1_contract.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    shutil.copy2(paths["contract"], output / "ROUND_CONTRACT.md")
    before = {name: state(path) for name, path in paths.items()}
    (output / "config.json").write_text(
        json.dumps(
            {
                "experiment_id": "round_118_documented_moisture_seasonal_incremental_skill_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "years_read": [1940, 2000],
                "locked_1921_1939_accessed": False,
                "early_1901_1920_accessed": False,
                "paleo_reserve_opened": False,
                "blocks": BLOCKS,
                "season_mapping": {name: list(months) for name, months in SEASON_MONTHS.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    proxy_mat = scipy.io.loadmat(paths["proxy9"], variable_names=["years", "observations", "locations"])
    proxy_years = np.asarray(proxy_mat["years"], dtype=int).reshape(-1)
    proxy_all = np.asarray(proxy_mat["observations"], dtype=float)
    locations = np.asarray(proxy_mat["locations"], dtype=float)
    if not np.array_equal(proxy_years, YEARS) or proxy_all.shape != (61, 297) or locations.shape != (297, 2):
        raise RuntimeError("proxy9-input-contract-failed")
    ledger = pd.read_csv(paths["ledger"], low_memory=False)
    proxy_ledger = ledger[ledger.network.eq("proxy9")].copy()
    proxy_ledger["network_record_1based"] = proxy_ledger.network_record_1based.astype(int)
    proxy_ledger = proxy_ledger.sort_values("network_record_1based")
    selected = (
        proxy_ledger.documented_network_role_v4.eq("documented_moisture")
        & proxy_ledger.operator_metadata_ready_v4.astype(str).str.lower().eq("true")
    ).to_numpy()
    if len(proxy_ledger) != 297 or int(selected.sum()) != 25:
        raise RuntimeError("documented-moisture-selection-failed")
    selected_ledger = proxy_ledger.loc[selected].copy()
    seasons = selected_ledger.canonical_seasonality_v4.astype(str).tolist()
    if any(season not in SEASON_MONTHS for season in seasons):
        raise RuntimeError(f"unsupported-documented-season:{seasons}")
    months_by_record = [SEASON_MONTHS[season] for season in seasons]
    response = proxy_all[:, selected]
    latitudes = locations[selected, 0]
    longitudes = locations[selected, 1]

    bea_monthly = read_berkeley_points(paths["berkeley_temperature"], latitudes, longitudes)
    cru_t_monthly = read_netcdf3_points(paths["cru_temperature"], "tmp", 1901, latitudes, longitudes)
    cru_p_monthly = read_netcdf3_points(paths["cru_precipitation"], "pre", 1901, latitudes, longitudes)
    gpcc_monthly = read_netcdf3_points(paths["gpcc_precipitation"], "precip", 1891, latitudes, longitudes)
    climate = {
        "BEA_GPCC": {
            "temperature": center(aggregate_temperature(bea_monthly, months_by_record)),
            "precipitation": center(aggregate_precipitation(gpcc_monthly, months_by_record)),
        },
        "CRU": {
            "temperature": center(aggregate_temperature(cru_t_monthly, months_by_record)),
            "precipitation": center(aggregate_precipitation(cru_p_monthly, months_by_record)),
        },
    }

    paths_to_score = ("bea_self", "cru_self", "bea_to_cru", "cru_to_bea")
    record_rows = []
    coefficient_rows = []
    for column in range(response.shape[1]):
        predictions = {
            path: {"temperature_only": np.full(len(YEARS), np.nan), "bivariate": np.full(len(YEARS), np.nan)}
            for path in paths_to_score
        }
        for outer_fold, (start, end) in enumerate(BLOCKS, start=1):
            test = (YEARS >= start) & (YEARS <= end)
            train = ~test
            coefficients = {}
            for product in ("BEA_GPCC", "CRU"):
                t = climate[product]["temperature"][:, column]
                p = climate[product]["precipitation"][:, column]
                coefficients[(product, "temperature_only")] = fit(response[:, column], t, None, train)
                coefficients[(product, "bivariate")] = fit(response[:, column], t, p, train)
            for product in ("BEA_GPCC", "CRU"):
                bivariate = coefficients[(product, "bivariate")]
                coefficient_rows.append(
                    {
                        "record": int(column + 1),
                        "evidence_group_id": selected_ledger.iloc[column].evidence_group_id,
                        "season": seasons[column],
                        "outer_fold": int(outer_fold),
                        "product": product,
                        "precipitation_coefficient": float(bivariate[2]) if bivariate is not None else np.nan,
                    }
                )
            mapping = {
                "bea_self": ("BEA_GPCC", "BEA_GPCC"),
                "cru_self": ("CRU", "CRU"),
                "bea_to_cru": ("BEA_GPCC", "CRU"),
                "cru_to_bea": ("CRU", "BEA_GPCC"),
            }
            for path_name, (fit_product, predictor_product) in mapping.items():
                t = climate[predictor_product]["temperature"][:, column]
                p = climate[predictor_product]["precipitation"][:, column]
                predictions[path_name]["temperature_only"][test] = predict(
                    coefficients[(fit_product, "temperature_only")], t, None, test
                )[test]
                predictions[path_name]["bivariate"][test] = predict(
                    coefficients[(fit_product, "bivariate")], t, p, test
                )[test]
        row = {
            "record": int(column + 1),
            "network_record_1based": int(selected_ledger.iloc[column].network_record_1based),
            "evidence_group_id": selected_ledger.iloc[column].evidence_group_id,
            "season": seasons[column],
            "latitude": float(latitudes[column]),
            "longitude": float(longitudes[column]),
        }
        all_improve = True
        for path_name in paths_to_score:
            temperature_rmse = rmse(response[:, column], predictions[path_name]["temperature_only"])
            bivariate_rmse = rmse(response[:, column], predictions[path_name]["bivariate"])
            improvement = 1.0 - bivariate_rmse / temperature_rmse
            row[f"{path_name}_temperature_rmse"] = temperature_rmse
            row[f"{path_name}_bivariate_rmse"] = bivariate_rmse
            row[f"{path_name}_improvement"] = improvement
            row[f"{path_name}_bivariate_correlation"] = correlation(
                response[:, column], predictions[path_name]["bivariate"]
            )
            all_improve &= bool(np.isfinite(improvement) and improvement > 0)
        row["all_four_paths_improve"] = bool(
            all_improve
            and row["bea_self_bivariate_correlation"] > 0
            and row["cru_self_bivariate_correlation"] > 0
        )
        record_rows.append(row)

    records = pd.DataFrame(record_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    paired = coefficients.pivot_table(
        index=["record", "outer_fold"], columns="product", values="precipitation_coefficient"
    ).dropna()
    sign_agreement = float(
        np.mean(np.sign(paired["BEA_GPCC"].to_numpy()) == np.sign(paired["CRU"].to_numpy()))
    )
    medians = {path: float(np.nanmedian(records[f"{path}_improvement"])) for path in paths_to_score}
    migration_bea_to_cru = float(
        np.nanmedian(records.bea_to_cru_bivariate_rmse / records.bea_self_bivariate_rmse - 1.0)
    )
    migration_cru_to_bea = float(
        np.nanmedian(records.cru_to_bea_bivariate_rmse / records.cru_self_bivariate_rmse - 1.0)
    )
    gates = {
        **{f"median_{path}_incremental_improvement_at_least_0_01": bool(value >= 0.01) for path, value in medians.items()},
        "fraction_records_all_four_paths_improve_at_least_0_50": bool(np.mean(records.all_four_paths_improve) >= 0.50),
        "cross_product_precipitation_coefficient_sign_agreement_at_least_0_70": bool(sign_agreement >= 0.70),
        "bea_fit_to_cru_median_migration_loss_at_most_0_10": bool(migration_bea_to_cru <= 0.10),
        "cru_fit_to_bea_median_migration_loss_at_most_0_10": bool(migration_cru_to_bea <= 0.10),
    }
    passes = bool(all(gates.values()))
    records.to_csv(output / "record_seasonal_incremental_skill.csv", index=False)
    coefficients.to_csv(output / "fold_precipitation_coefficients.csv", index=False)
    selected_ledger.to_csv(output / "selected_documented_moisture_groups.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    data = [records[f"{path}_improvement"].to_numpy() for path in paths_to_score]
    axes[0].boxplot(data, labels=paths_to_score, showfliers=False)
    axes[0].axhline(0.01, color="#D55E00", linewidth=1.0)
    axes[0].axhline(0, color="#777777", linewidth=0.7)
    axes[0].set_ylabel("RMSE improvement over temperature-only")
    axes[0].tick_params(axis="x", rotation=25)
    counts = selected_ledger.canonical_seasonality_v4.value_counts()
    axes[1].bar(counts.index, counts.values, color=["#0072B2", "#009E73", "#CC79A7"][: len(counts)])
    axes[1].set_ylabel("Evidence groups")
    axes[1].set_title("Frozen documented seasons")
    axes[1].tick_params(axis="x", rotation=25)
    fig.savefig(output / "documented_moisture_seasonal_incremental_skill.png", dpi=220)
    plt.close(fig)

    after = {name: state(path) for name, path in paths.items()}
    unchanged = bool(before == after)
    if not unchanged:
        raise RuntimeError("source-changed-during-round118")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "experiment_id": "round_118_documented_moisture_seasonal_incremental_skill_v1",
        "status": "complete",
        "passes_all_seasonal_promotion_gates": passes,
        "decision": "advance_to_season_aware_precipitation_field" if passes else "close_current_documented_moisture_psm_route",
        "locked_1921_1939_accessed": False,
        "early_1901_1920_accessed": False,
        "paleo_reserve_opened": False,
        "source_hashes_unchanged": unchanged,
        "documented_moisture_group_count": int(len(records)),
        "median_incremental_improvements": medians,
        "fraction_records_all_four_paths_improve": float(np.mean(records.all_four_paths_improve)),
        "cross_product_precipitation_coefficient_sign_agreement": sign_agreement,
        "migration_loss": {
            "bea_fit_to_cru": migration_bea_to_cru,
            "cru_fit_to_bea": migration_cru_to_bea,
        },
        "gates": gates,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Round 118 Documented-Moisture Seasonal Incremental Skill",
        "",
        f"- Decision: `{summary['decision']}`.",
        f"- Passes every seasonal promotion gate: `{str(passes).lower()}`.",
        f"- Median improvements: `{medians}`.",
        f"- Records improving on all four paths: `{summary['fraction_records_all_four_paths_improve']:.3%}`.",
        f"- Cross-product precipitation-coefficient sign agreement: `{sign_agreement:.3%}`.",
        f"- Migration losses, BEA->CRU / CRU->BEA: `{migration_bea_to_cru:.3%}` / `{migration_cru_to_bea:.3%}`.",
        "- 1901-1920 accessed: `false`.",
        "- Locked 1921-1939 accessed: `false`.",
        "- Paleo reserve opened: `false`.",
        f"- Source hashes unchanged: `{str(unchanged).lower()}`.",
        "",
        "## Gates",
        "",
        *[f"- `{name}`: `{str(value).lower()}`" for name, value in gates.items()],
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
