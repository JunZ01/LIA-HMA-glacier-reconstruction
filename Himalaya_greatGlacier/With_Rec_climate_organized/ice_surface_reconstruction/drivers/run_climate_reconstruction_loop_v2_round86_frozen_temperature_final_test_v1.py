"""Execute the one-time 1921-1939 frozen temperature final test."""

from __future__ import annotations

import calendar
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

import acquire_round67_mode_ra_allowed_modern_subset_v2 as access
import run_climate_reconstruction_loop_v2_round30_temperature_reference_spatial_consensus_v1 as old
import run_climate_reconstruction_loop_v2_round68_mode_ra_modern_validation_v1 as modern
import run_climate_reconstruction_loop_v2_round71_mode_ra_lia_spatial_consensus_v1 as spatial


YEARS = np.arange(1921, 1940)
EXPECTED_CANDIDATE_SHA256 = "72615401b2c3df6ac7e6dc9e8204aff54aae40b9a295c1f6ba86e206eb864517"
FAMILY_NAMES = ("local_enkf", "lmr", "neukom", "phyda", "mode_ra")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def clean_bbox(
    dataset: h5py.Dataset,
    time_indices: np.ndarray,
    lat_slice: slice,
    lon_slice: slice,
) -> np.ndarray:
    values = np.asarray(dataset[time_indices, lat_slice, lon_slice], dtype=float)
    fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).reshape(-1)[0])
    values[np.isclose(values, fill) | (values < -9000)] = np.nan
    return values


def finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


class AuthorizedLockedRangeSource(access.ExactRangeSource):
    """Permit only the explicitly unlocked 1921-1939 temperature records."""

    def read(
        self,
        start: int,
        end: int,
        purpose: str,
        years: tuple[int, int] | None,
    ) -> bytes:
        if start < 0 or end < start or end >= self.content_length:
            raise RuntimeError(f"invalid-byte-range:{self.label}:{start}:{end}")
        overlap = False
        if years is not None:
            first, last = years
            overlap = first <= 1939 and last >= 1921
            if overlap and (first < 1921 or last > 1939):
                raise RuntimeError(f"authorized-range-exceeds-locked-test:{first}:{last}")
            if purpose == "locked_temperature_records" and not (first >= 1921 and last <= 1939):
                raise RuntimeError(f"unexpected-final-test-record-range:{first}:{last}")
        response = self.session.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=180,
        )
        try:
            if response.status_code != 206:
                raise RuntimeError(f"range-status-not-206:{self.label}:{response.status_code}")
            expected_header = f"bytes {start}-{end}/{self.content_length}"
            if response.headers.get("Content-Range") != expected_header:
                raise RuntimeError(
                    f"content-range-mismatch:{self.label}:"
                    f"{response.headers.get('Content-Range')}:{expected_header}"
                )
            expected_size = end - start + 1
            payload = response.raw.read(expected_size + 1)
            if len(payload) != expected_size:
                raise RuntimeError(
                    f"range-size-mismatch:{self.label}:{len(payload)}:{expected_size}"
                )
        finally:
            response.close()
        self.log.append(
            {
                "source": self.label,
                "purpose": purpose,
                "byte_start": start,
                "byte_end": end,
                "byte_count": end - start + 1,
                "first_year": "" if years is None else years[0],
                "last_year": "" if years is None else years[1],
                "locked_period_overlap": overlap,
                "http_status": 206,
            }
        )
        return payload


def standard_calendar_month_days(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def acquire_mode_ra_locked() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    request_log: list[dict[str, Any]] = []
    source = AuthorizedLockedRangeSource(access.SOURCES["temp2"], "temp2", request_log)
    remote = {
        "url": source.url,
        "content_length": source.content_length,
        "etag": source.etag,
        "last_modified": source.last_modified,
    }
    try:
        header_payload = source.read(0, access.HEADER_BYTES - 1, "classic_netcdf_header", None)
        header = access.parse_header(header_payload)
        lat_var = access.coordinate_variable(header, ("lat", "latitude", "y"))
        lon_var = access.coordinate_variable(header, ("lon", "longitude", "x"))
        time_var = access.coordinate_variable(header, ("time", "t"))
        data_var = header.variables["temp2"]
        lat = access.read_fixed_variable(source, header, lat_var)
        lon = access.normalize_longitude(access.read_fixed_variable(source, header, lon_var))
        lat_indices = np.flatnonzero(
            (lat >= access.HMA_BBOX["south"]) & (lat <= access.HMA_BBOX["north"])
        )
        lon_indices = np.flatnonzero(
            (lon >= access.HMA_BBOX["west"]) & (lon <= access.HMA_BBOX["east"])
        )
        spatial_dims = list(data_var.dimensions[1:])
        lat_axis = spatial_dims.index(lat_var.dimensions[0])
        lon_axis = spatial_dims.index(lon_var.dimensions[0])
        if len(spatial_dims) != 2 or (lat_axis, lon_axis) not in ((0, 1), (1, 0)):
            raise RuntimeError(f"unexpected-temperature-spatial-dimensions:{spatial_dims}")

        first_record = (1921 - access.START_YEAR) * 12
        last_record = (1939 - access.START_YEAR + 1) * 12 - 1
        byte_start = header.record_base + first_record * header.record_size
        byte_end = header.record_base + (last_record + 1) * header.record_size - 1
        payload = source.read(
            byte_start,
            byte_end,
            "locked_temperature_records",
            (1921, 1939),
        )
        annual = np.full((len(YEARS), len(lat_indices), len(lon_indices)), np.nan)
        time_values: list[float] = []
        for year in YEARS:
            monthly = []
            weights = []
            for month in range(1, 13):
                local_record = (int(year) - 1921) * 12 + month - 1
                time_raw = access.record_segment(payload, header, time_var, local_record)
                time_value = access.decode_numeric(
                    time_raw, time_var, header.shape(time_var)[1:] or (1,)
                ).reshape(-1)[0]
                time_values.append(float(time_value))
                data_raw = access.record_segment(payload, header, data_var, local_record)
                field = access.decode_numeric(data_raw, data_var, header.shape(data_var)[1:])
                if (lat_axis, lon_axis) != (0, 1):
                    field = np.moveaxis(field, (lat_axis, lon_axis), (0, 1))
                monthly.append(field[np.ix_(lat_indices, lon_indices)])
                weights.append(standard_calendar_month_days(int(year), month))
            month_stack = np.stack(monthly)
            month_weights = np.asarray(weights, dtype=float)
            complete = np.all(np.isfinite(month_stack), axis=0)
            annual[int(year) - 1921] = np.where(
                complete,
                np.sum(month_stack * month_weights[:, None, None], axis=0) / month_weights.sum(),
                np.nan,
            )
    finally:
        source.close()

    time_values_array = np.asarray(time_values)
    if len(time_values_array) != 19 * 12 or not np.all(np.diff(time_values_array) > 0):
        raise RuntimeError("mode-ra-locked-month-axis-failed")
    lat_subset = lat[lat_indices]
    lon_subset = lon[lon_indices]
    lat_order = np.argsort(lat_subset)
    lon_order = np.argsort(lon_subset)
    return (
        lat_subset[lat_order],
        lon_subset[lon_order],
        annual[:, lat_order, :][:, :, lon_order],
        request_log,
        remote,
    )


def metric_record(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.endswith("_series")}


def draw_figure(
    output: Path,
    curve_table: pd.DataFrame,
    score_table: pd.DataFrame,
    comparison: pd.DataFrame,
    accepted: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), constrained_layout=True)
    ax = axes[0, 0]
    pivot = curve_table.pivot(index="year", columns="observation", values="observed")
    ax.fill_between(pivot.index, pivot.min(axis=1), pivot.max(axis=1), color="#b8c4cc", alpha=0.35, label="Observation range")
    ax.plot(pivot.index, pivot.mean(axis=1), color="#27323a", lw=2.0, label="Observation mean")
    first = curve_table[curve_table.observation == curve_table.observation.iloc[0]]
    ax.plot(first.year, first.four_family, color="#d55e00", lw=2.0, label="Four-family control")
    ax.plot(first.year, first.five_family, color="#0072b2", lw=2.5, label="Five-family candidate")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set(title="HMA annual temperature anomaly", ylabel="degC", xlabel="Year")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[0, 1]
    five = score_table[score_table.candidate == "five_family"].copy()
    region_order = list(modern.REGIONS)
    product_order = list(modern.TEMPERATURE_OBSERVATIONS)
    matrix = five.pivot(index="region", columns="observation", values="regional_skill").loc[region_order, product_order]
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(product_order)), product_order, rotation=35, ha="right")
    ax.set_yticks(range(len(region_order)), region_order)
    ax.set_title("Five-family regional RMSE skill")
    fig.colorbar(image, ax=ax, shrink=0.78, label="Skill vs zero anomaly")

    ax = axes[1, 0]
    relative = comparison.pivot(index="region", columns="observation", values="regional_rmse_improvement_over_four_family").loc[region_order, product_order]
    image = ax.imshow(relative.to_numpy(), cmap="PiYG", vmin=-0.15, vmax=0.15, aspect="auto")
    for i in range(relative.shape[0]):
        for j in range(relative.shape[1]):
            ax.text(j, i, f"{100 * relative.iloc[i, j]:+.1f}%", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(product_order)), product_order, rotation=35, ha="right")
    ax.set_yticks(range(len(region_order)), region_order)
    ax.set_title("Five-family RMSE change vs four-family")
    fig.colorbar(image, ax=ax, shrink=0.78, label="Relative improvement")

    ax = axes[1, 1]
    hma = five[five.region == "hma"].set_index("observation").loc[product_order]
    x = np.arange(len(product_order))
    ax.bar(x - 0.18, hma.spatial_residual_skill, width=0.36, color="#009e73", label="Spatial residual skill")
    ax.bar(x + 0.18, hma.median_annual_spatial_pattern_correlation, width=0.36, color="#cc79a7", label="Pattern correlation")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(x, product_order, rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"HMA spatial test: {'PASS' if accepted else 'FAIL'}")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Frozen 1921-1939 HMA Temperature Final Test", fontsize=15)
    fig.savefig(output / "frozen_temperature_final_test.png", dpi=220)
    plt.close(fig)


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_86_frozen_temperature_final_test_v1"
    round67 = loop / "round_67_mode_ra_allowed_modern_subset_acquisition_v2"
    round72 = loop / "round_72_mode_ra_five_family_temperature_reference_v1"
    round79 = loop / "round_79_frozen_temperature_candidate_final_test_readiness_v1"
    public = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
    paths = {
        "public_reference_source": public,
        "hma_mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "mode_ra_modern_baseline": round67 / "mode_ra_ensmean_1940_2000_hma_bbox.nc",
        "frozen_candidate": round72 / "temperature_reference_five_family_coarse_allowed_v1.nc",
        "readiness": round79 / "readiness.json",
        "frozen_contract": round79 / "ROUND_CONTRACT.md",
        "unlock_record": output / "USER_UNLOCK_RECORD.md",
        "round_contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    for name in (
        "summary.json",
        "temperature_final_test_1921_1939.nc",
        "final_test_scores.csv",
    ):
        if (output / name).exists():
            raise FileExistsError(output / name)
    if "允许解锁温度" not in paths["unlock_record"].read_text(encoding="utf-8"):
        raise RuntimeError("explicit-temperature-unlock-not-recorded")

    before = {name: state(path) for name, path in paths.items()}
    if before["frozen_candidate"]["sha256"] != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("frozen-candidate-hash-conflict")
    readiness = json.loads(paths["readiness"].read_text(encoding="utf-8"))
    if not readiness["test_contract_frozen"] or readiness["locked_numeric_values_accessed"]:
        raise RuntimeError("round79-readiness-contract-conflict")
    (output / "execution_started.json").write_text(
        json.dumps(
            {
                "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "explicit_unlock": "允许解锁温度",
                "locked_years": [1921, 1939],
                "candidate_sha256_verified": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with h5py.File(paths["frozen_candidate"], "r") as handle:
        candidate_years = np.asarray(handle["time"], dtype=int)
        coarse_lat = np.asarray(handle["lat"], dtype=float)
        coarse_lon = np.asarray(handle["lon"], dtype=float)
        fine_counts = np.asarray(handle["fine_hma_cell_count"], dtype=int)
    if np.any((candidate_years >= 1921) & (candidate_years <= 1939)):
        raise RuntimeError("locked-years-present-in-frozen-candidate")
    if len(coarse_lat) != 140:
        raise RuntimeError(f"frozen-support-count-conflict:{len(coarse_lat)}")
    grid_lat = np.unique(coarse_lat)
    grid_lon = np.unique(coarse_lon)
    support = pd.DataFrame(
        {"latitude": coarse_lat, "longitude": coarse_lon, "fine_hma_cell_count": fine_counts}
    )

    full_mask = np.load(paths["hma_mask"]).astype(bool)
    with h5py.File(paths["public_reference_source"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).ravel()
        full_lat = np.asarray(handle["lat"], dtype=float).ravel()
        full_lon = np.asarray(handle["lon"], dtype=float).ravel()
        if not np.array_equal(source_years, np.arange(1000, 2001)):
            raise RuntimeError("public-source-year-axis-failed")
        if full_mask.shape != (len(full_lat), len(full_lon)):
            raise RuntimeError("public-mask-shape-failed")
        locked_indices = np.flatnonzero((source_years >= 1921) & (source_years <= 1939))
        baseline_indices = np.flatnonzero((source_years >= 1961) & (source_years <= 1990))
        if not np.array_equal(source_years[locked_indices], YEARS) or len(baseline_indices) != 30:
            raise RuntimeError("public-source-selection-failed")
        ii, jj = np.where(full_mask)
        lat_slice = slice(int(ii.min()), int(ii.max()) + 1)
        lon_slice = slice(int(jj.min()), int(jj.max()) + 1)
        fine_lat = full_lat[lat_slice]
        fine_lon = full_lon[lon_slice]
        mask = full_mask[lat_slice, lon_slice]

        def locked_anomaly(variable: str) -> np.ndarray:
            locked = clean_bbox(handle[variable], locked_indices, lat_slice, lon_slice)
            baseline = clean_bbox(handle[variable], baseline_indices, lat_slice, lon_slice)
            return locked - np.nanmean(baseline, axis=0)[None]

        member_fine = {name: locked_anomaly(variable) for name, variable in old.MEMBERS.items()}
        observation_fine = {
            name: locked_anomaly(variable)
            for name, variable in modern.TEMPERATURE_OBSERVATIONS.items()
        }

    member_coarse = {
        name: spatial.aggregate_reference(
            values, fine_lat, fine_lon, mask, support, grid_lat, grid_lon
        )
        for name, values in member_fine.items()
    }
    observations = {
        name: spatial.aggregate_reference(
            values, fine_lat, fine_lon, mask, support, grid_lat, grid_lon
        )
        for name, values in observation_fine.items()
    }
    families = {
        "local_enkf": finite_mean([member_coarse["HGAO"], member_coarse["O"]]),
        "lmr": member_coarse["LMR"],
        "neukom": finite_mean(
            [values for name, values in member_coarse.items() if name.startswith("Nk_")]
        ),
        "phyda": member_coarse["PHYDA"],
    }
    four_family = finite_mean([families[name] for name in FAMILY_NAMES[:4]])

    mode_lat, mode_lon, mode_source_locked, request_log, remote = acquire_mode_ra_locked()
    with h5py.File(paths["mode_ra_modern_baseline"], "r") as handle:
        baseline_years = np.asarray(handle["time"], dtype=int)
        saved_lat = np.asarray(handle["lat"], dtype=float)
        saved_lon = np.asarray(handle["lon"], dtype=float)
        baseline_source = np.asarray(handle["temp2_anomaly"], dtype=float)
    if not np.array_equal(baseline_years, np.arange(1940, 2001)):
        raise RuntimeError("mode-ra-baseline-axis-failed")
    if not np.array_equal(mode_lat, saved_lat) or not np.array_equal(mode_lon, saved_lon):
        raise RuntimeError("mode-ra-locked-grid-conflict")
    baseline = (baseline_years >= 1961) & (baseline_years <= 1990)
    mode_locked = mode_source_locked - np.nanmean(baseline_source[baseline], axis=0)[None]
    temp_cells = modern.build_coarse_support(fine_lat, fine_lon, mask, mode_lat, mode_lon)
    rebuilt_support = pd.DataFrame(
        [
            {
                "latitude": cell["latitude"],
                "longitude": cell["longitude"],
                "fine_hma_cell_count": cell["fine_cell_count"],
            }
            for cell in temp_cells
        ]
    )
    if (
        len(rebuilt_support) != 140
        or not np.allclose(rebuilt_support.latitude, coarse_lat)
        or not np.allclose(rebuilt_support.longitude, coarse_lon)
        or not np.array_equal(rebuilt_support.fine_hma_cell_count, fine_counts)
    ):
        raise RuntimeError("frozen-140-cell-support-conflict")
    mode_coarse = modern.extract_coarse(mode_locked, temp_cells)
    families["mode_ra"] = mode_coarse
    five_family = (4.0 * four_family + mode_coarse) / 5.0
    zero = np.zeros_like(five_family)
    candidates = {
        "zero_anomaly": zero,
        "four_family": four_family,
        "five_family": five_family,
    }

    weights = np.cos(np.deg2rad(coarse_lat)) * fine_counts.astype(float)
    score_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for observation_name, observation in observations.items():
        for region, (lower, upper) in modern.REGIONS.items():
            selected = (coarse_lon >= lower) & (coarse_lon < upper)
            for candidate_name, candidate in candidates.items():
                metrics = modern.pair_metrics(
                    candidate[:, selected], observation[:, selected], YEARS, weights[selected]
                )
                score_rows.append(
                    {
                        "observation": observation_name,
                        "region": region,
                        "candidate": candidate_name,
                        **metric_record(metrics),
                    }
                )
                if region == "hma" and candidate_name == "five_family":
                    control_metrics = modern.pair_metrics(
                        four_family[:, selected], observation[:, selected], YEARS, weights[selected]
                    )
                    for index, year in enumerate(YEARS):
                        curve_rows.append(
                            {
                                "observation": observation_name,
                                "year": int(year),
                                "five_family": float(metrics["candidate_series"][index]),
                                "four_family": float(control_metrics["candidate_series"][index]),
                                "observed": float(metrics["observation_series"][index]),
                            }
                        )
    scores = pd.DataFrame(score_rows)
    curves = pd.DataFrame(curve_rows)
    comparisons: list[dict[str, Any]] = []
    for (observation, region), group in scores.groupby(["observation", "region"]):
        indexed = group.set_index("candidate")
        control = indexed.loc["four_family"]
        candidate = indexed.loc["five_family"]
        comparisons.append(
            {
                "observation": observation,
                "region": region,
                "regional_rmse_improvement_over_four_family": float(
                    1.0 - candidate.regional_rmse / control.regional_rmse
                ),
                "spatial_residual_rmse_improvement_over_four_family": float(
                    1.0 - candidate.spatial_residual_rmse / control.spatial_residual_rmse
                ),
                "annual_correlation_change": float(
                    candidate.annual_correlation - control.annual_correlation
                ),
                "spatial_pattern_correlation_change": float(
                    candidate.median_annual_spatial_pattern_correlation
                    - control.median_annual_spatial_pattern_correlation
                ),
            }
        )
    comparison = pd.DataFrame(comparisons)
    five_scores = scores[scores.candidate == "five_family"]
    five_hma = five_scores[five_scores.region == "hma"]
    relative_hma = comparison[comparison.region == "hma"]
    gates = {
        "relative_regional_median_improvement_positive": float(
            comparison.regional_rmse_improvement_over_four_family.median()
        ) > 0,
        "relative_regional_worst_regression_at_most_5pct": float(
            comparison.regional_rmse_improvement_over_four_family.min()
        ) >= -0.05,
        "relative_regional_at_least_75pct_improve": float(
            np.mean(comparison.regional_rmse_improvement_over_four_family > 0)
        ) >= 0.75,
        "relative_spatial_median_improvement_positive": float(
            relative_hma.spatial_residual_rmse_improvement_over_four_family.median()
        ) > 0,
        "relative_spatial_worst_regression_at_most_5pct": float(
            relative_hma.spatial_residual_rmse_improvement_over_four_family.min()
        ) >= -0.05,
        "relative_spatial_at_least_75pct_improve": float(
            np.mean(relative_hma.spatial_residual_rmse_improvement_over_four_family > 0)
        ) >= 0.75,
        "absolute_median_regional_skill_at_least_5pct": float(
            five_scores.regional_skill.median()
        ) >= 0.05,
        "absolute_worst_regional_skill_at_least_minus_5pct": float(
            five_scores.regional_skill.min()
        ) >= -0.05,
        "absolute_at_least_75pct_positive_regional_paths": float(
            np.mean(five_scores.regional_skill > 0)
        ) >= 0.75,
        "absolute_median_annual_correlation_at_least_0_30": float(
            five_scores.annual_correlation.median()
        ) >= 0.30,
        "absolute_median_hma_spatial_skill_positive": float(
            five_hma.spatial_residual_skill.median()
        ) > 0,
        "absolute_worst_hma_spatial_skill_at_least_minus_5pct": float(
            five_hma.spatial_residual_skill.min()
        ) >= -0.05,
        "absolute_at_least_75pct_positive_hma_spatial_paths": float(
            np.mean(five_hma.spatial_residual_skill > 0)
        ) >= 0.75,
        "absolute_median_spatial_pattern_correlation_at_least_0_20": float(
            five_hma.median_annual_spatial_pattern_correlation.median()
        ) >= 0.20,
    }
    accepted = bool(all(gates.values()))

    scores.to_csv(output / "final_test_scores.csv", index=False)
    comparison.to_csv(output / "five_vs_four_comparison.csv", index=False)
    curves.to_csv(output / "hma_annual_curves.csv", index=False)
    with (output / "mode_ra_exact_http_range_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(request_log[0]))
        writer.writeheader()
        writer.writerows(request_log)

    artifact = output / "temperature_final_test_1921_1939.nc"
    with h5py.File(artifact, "w") as handle:
        handle.attrs["title"] = "Frozen HMA temperature final test"
        handle.attrs["years"] = "1921-1939"
        handle.attrs["candidate_sha256"] = EXPECTED_CANDIDATE_SHA256
        handle.attrs["user_unlock_recorded"] = True
        handle.attrs["production_field"] = False
        handle.create_dataset("time", data=YEARS.astype(np.int16))
        handle.create_dataset("lat", data=coarse_lat)
        handle.create_dataset("lon", data=coarse_lon)
        handle.create_dataset("fine_hma_cell_count", data=fine_counts.astype(np.int16))
        for name, values in candidates.items():
            handle.create_dataset(name, data=values.astype(np.float32), compression="gzip", shuffle=True)
        for name, values in observations.items():
            handle.create_dataset(
                f"observation_{name}", data=values.astype(np.float32), compression="gzip", shuffle=True
            )

    summary = {
        "status": "complete",
        "execution": "one_time_frozen_final_test",
        "locked_temperature_years_accessed": [1921, 1939],
        "precipitation_locked_data_accessed": False,
        "user_unlock": "允许解锁温度",
        "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
        "family_definition": list(FAMILY_NAMES),
        "regional_path_count": int(len(comparison)),
        "spatial_path_count": int(len(relative_hma)),
        "relative": {
            "median_regional_rmse_improvement": float(
                comparison.regional_rmse_improvement_over_four_family.median()
            ),
            "worst_regional_rmse_improvement": float(
                comparison.regional_rmse_improvement_over_four_family.min()
            ),
            "positive_regional_fraction": float(
                np.mean(comparison.regional_rmse_improvement_over_four_family > 0)
            ),
            "median_hma_spatial_rmse_improvement": float(
                relative_hma.spatial_residual_rmse_improvement_over_four_family.median()
            ),
            "worst_hma_spatial_rmse_improvement": float(
                relative_hma.spatial_residual_rmse_improvement_over_four_family.min()
            ),
            "positive_hma_spatial_fraction": float(
                np.mean(relative_hma.spatial_residual_rmse_improvement_over_four_family > 0)
            ),
        },
        "absolute": {
            "median_regional_skill": float(five_scores.regional_skill.median()),
            "worst_regional_skill": float(five_scores.regional_skill.min()),
            "positive_regional_fraction": float(np.mean(five_scores.regional_skill > 0)),
            "median_annual_correlation": float(five_scores.annual_correlation.median()),
            "median_hma_spatial_skill": float(five_hma.spatial_residual_skill.median()),
            "worst_hma_spatial_skill": float(five_hma.spatial_residual_skill.min()),
            "positive_hma_spatial_fraction": float(np.mean(five_hma.spatial_residual_skill > 0)),
            "median_hma_spatial_pattern_correlation": float(
                five_hma.median_annual_spatial_pattern_correlation.median()
            ),
        },
        "gates": gates,
        "passes_frozen_final_test": accepted,
        "decision": (
            "validate_frozen_five_family_as_external_coarse_hma_temperature_reference"
            if accepted
            else "reject_frozen_five_family_as_external_coarse_hma_temperature_reference"
        ),
        "interpretation_limits": {
            "proxy_evidence_independent": False,
            "legacy_project_reconstruction_validated": False,
            "oggm_forcing_authorized": False,
            "post_test_tuning_or_candidate_replacement_allowed": False,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_figure(output, curves, scores, comparison, accepted)
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Frozen 1921-1939 Temperature Final Test",
                "",
                f"- Final decision: `{'PASS' if accepted else 'FAIL'}`.",
                f"- Relative regional median / worst / positive fraction: `{summary['relative']['median_regional_rmse_improvement']:.3f}` / `{summary['relative']['worst_regional_rmse_improvement']:.3f}` / `{summary['relative']['positive_regional_fraction']:.1%}`.",
                f"- Absolute regional median / worst skill: `{summary['absolute']['median_regional_skill']:.3f}` / `{summary['absolute']['worst_regional_skill']:.3f}`.",
                f"- Median annual correlation: `{summary['absolute']['median_annual_correlation']:.3f}`.",
                f"- Relative HMA spatial median / worst / positive fraction: `{summary['relative']['median_hma_spatial_rmse_improvement']:.3f}` / `{summary['relative']['worst_hma_spatial_rmse_improvement']:.3f}` / `{summary['relative']['positive_hma_spatial_fraction']:.1%}`.",
                f"- Absolute HMA spatial median / worst skill: `{summary['absolute']['median_hma_spatial_skill']:.3f}` / `{summary['absolute']['worst_hma_spatial_skill']:.3f}`.",
                f"- Median annual spatial pattern correlation: `{summary['absolute']['median_hma_spatial_pattern_correlation']:.3f}`.",
                "",
                f"Decision: `{summary['decision']}`.",
                "",
                "This result is final for the frozen candidate. No locked-outcome tuning or candidate replacement is permitted.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    manifest = {
        "local_sources_before": before,
        "local_sources_after": after,
        "local_source_states_unchanged": unchanged,
        "remote_source": remote,
        "remote_exact_range_request_count": len(request_log),
        "remote_exact_range_bytes_read": int(sum(row["byte_count"] for row in request_log)),
        "remote_locked_record_requests": [
            row for row in request_log if row["purpose"] == "locked_temperature_records"
        ],
        "output_artifact": state(artifact),
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("local-source-state-changed")
    if len(manifest["remote_locked_record_requests"]) != 1:
        raise RuntimeError("locked-remote-request-count-failed")
    locked_request = manifest["remote_locked_record_requests"][0]
    if locked_request["first_year"] != 1921 or locked_request["last_year"] != 1939:
        raise RuntimeError("locked-remote-request-year-range-failed")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
