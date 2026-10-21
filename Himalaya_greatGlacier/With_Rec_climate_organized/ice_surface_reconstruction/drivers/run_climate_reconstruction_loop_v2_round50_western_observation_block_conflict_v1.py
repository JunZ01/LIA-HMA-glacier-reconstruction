"""Localize western precipitation disagreement among UDel, CRU, and GPCC."""

from __future__ import annotations

import calendar
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1940, 2001)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
OBSERVATION_VARIABLES = {
    "CRU": "pr_obs_cru_pre_05deg",
    "GPCC": "pr_obs_gpcp_05deg",
}
PRODUCTS = ("UDel", "CRU", "GPCC")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
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


def clean(dataset: h5py.Dataset, selection: slice) -> np.ndarray:
    values = np.asarray(dataset[selection], dtype=float)
    fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).squeeze())
    values[np.isclose(values, fill) | (values < -9000)] = np.nan
    return values


def decode_udel_dates(handle: h5py.File) -> list[datetime]:
    units = np.asarray(handle["time"].attrs["units"]).reshape(-1)[0]
    units = units.decode() if isinstance(units, bytes) else str(units)
    if units != "days since 1900-01-01 00:00:00":
        raise RuntimeError(f"unexpected-udel-time-units:{units}")
    return [datetime(1900, 1, 1) + timedelta(days=float(value)) for value in handle["time"]]


def annualize_udel(path: Path, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        dates = decode_udel_dates(handle)
        expected = [(int(year), month) for year in YEARS for month in range(1, 13)]
        if [(date.year, date.month) for date in dates] != expected:
            raise RuntimeError("udel-month-axis-failed")
        if any(1921 <= date.year <= 1939 for date in dates):
            raise RuntimeError("udel-locked-period-present")
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        lat_index = np.asarray(
            [np.flatnonzero(np.isclose(lat, value, atol=1e-7))[0] for value in target_lat],
            dtype=int,
        )
        lon_index = np.asarray(
            [np.flatnonzero(np.isclose(lon, value, atol=1e-7))[0] for value in target_lon],
            dtype=int,
        )
        monthly = np.asarray(handle["precip"][:, lat_index, :][:, :, lon_index], dtype=float)
    annual = np.full((len(YEARS), len(target_lat), len(target_lon)), np.nan)
    for index, year in enumerate(YEARS):
        values = monthly[index * 12 : (index + 1) * 12]
        complete = np.all(np.isfinite(values), axis=0)
        annual[index] = np.where(
            complete,
            np.nansum(values, axis=0) * 10.0 / (366 if calendar.isleap(int(year)) else 365),
            np.nan,
        )
    return annual


def weighted_series(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(values)
    return np.nansum(values * weights[None], axis=(1, 2)) / np.where(
        np.sum(valid * weights[None], axis=(1, 2)) == 0,
        np.nan,
        np.sum(valid * weights[None], axis=(1, 2)),
    )


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return float(np.corrcoef(left[valid], right[valid])[0, 1]) if valid.sum() >= 3 else np.nan


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_50_western_observation_block_conflict_v1"
    round49 = loop / "round_49_western_udel_modern_adjudication_v1"
    paths = {
        "udel": loop / "round_44_udel_allowed_subset_acquisition_v2/udel_precip_v501_1940_2000_hma_bbox.nc",
        "public_reference_source": Path(
            r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
        ),
        "hma_mask": project
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "round49_blocks": round49 / "three_product_western_block_scores.csv",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    mask = np.load(paths["hma_mask"]).astype(bool)
    baseline = (YEARS >= 1961) & (YEARS <= 1990)
    with h5py.File(paths["public_reference_source"], "r") as handle:
        years = np.asarray(handle["time"], dtype=int).ravel()
        indices = np.flatnonzero((years >= 1940) & (years <= 2000))
        if not np.array_equal(years[indices], YEARS):
            raise RuntimeError("public-year-axis-failed")
        if np.any((years[indices] >= 1921) & (years[indices] <= 1939)):
            raise RuntimeError("locked-period-index-requested")
        selection = slice(int(indices[0]), int(indices[-1]) + 1)
        latitudes = np.asarray(handle["lat"], dtype=float).ravel()
        longitudes = np.asarray(handle["lon"], dtype=float).ravel()
        observations = {}
        for product, variable in OBSERVATION_VARIABLES.items():
            raw = clean(handle[variable], selection)
            observations[product] = raw - np.nanmean(raw[baseline], axis=0)[None]
    udel_raw = annualize_udel(paths["udel"], latitudes, longitudes)
    observations["UDel"] = udel_raw - np.nanmean(udel_raw[baseline], axis=0)[None]

    west = mask & (np.broadcast_to(longitudes[None], mask.shape) < 80.0)
    common = west.copy()
    for values in observations.values():
        common &= np.all(np.isfinite(values), axis=0)
    if not np.any(common):
        raise RuntimeError("empty-three-product-west-support")
    series = {
        product: weighted_series(observations[product], common, latitudes)
        for product in PRODUCTS
    }
    series_frame = pd.DataFrame({"year": YEARS, **series})
    series_frame.to_csv(output / "western_observation_series.csv", index=False)

    pair_rows: list[dict[str, object]] = []
    outlier_rows: list[dict[str, object]] = []
    block_outliers: dict[int, str | None] = {}
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        selected = (YEARS >= start) & (YEARS <= end)
        for index, left in enumerate(PRODUCTS):
            for right in PRODUCTS[index + 1 :]:
                pair_rows.append(
                    {
                        "fold": fold,
                        "start_year": start,
                        "end_year": end,
                        "left": left,
                        "right": right,
                        "correlation": correlation(series[left][selected], series[right][selected]),
                        "rmse_mm_day": rmse(series[left][selected], series[right][selected]),
                    }
                )
        ratios = {}
        for product in PRODUCTS:
            others = [name for name in PRODUCTS if name != product]
            product_distance = rmse(
                series[product][selected],
                0.5 * (series[others[0]][selected] + series[others[1]][selected]),
            )
            other_distance = rmse(series[others[0]][selected], series[others[1]][selected])
            ratio = float(product_distance / other_distance) if other_distance > 0 else np.inf
            ratios[product] = ratio
            outlier_rows.append(
                {
                    "fold": fold,
                    "start_year": start,
                    "end_year": end,
                    "product": product,
                    "rmse_to_other_product_mean_mm_day": product_distance,
                    "other_product_pair_rmse_mm_day": other_distance,
                    "outlier_ratio": ratio,
                }
            )
        largest = max(ratios, key=ratios.get)
        block_outliers[fold] = largest if ratios[largest] > 1.5 else None

    pairs = pd.DataFrame(pair_rows)
    outliers = pd.DataFrame(outlier_rows)
    pairs.to_csv(output / "western_observation_pairwise_block_metrics.csv", index=False)
    outliers.to_csv(output / "western_observation_outlier_ratios.csv", index=False)
    round49_blocks = pd.read_csv(paths["round49_blocks"])
    o_scores = round49_blocks[round49_blocks.candidate == "O"][
        ["observation", "fold", "start_year", "end_year", "regional_rmse_improvement_over_zero"]
    ].copy()
    o_scores["block_observation_outlier"] = o_scores.apply(
        lambda row: block_outliers[int(row.fold)] == row.observation,
        axis=1,
    )
    o_scores.to_csv(output / "o_candidate_failure_context.csv", index=False)
    failed_o = o_scores[o_scores.regional_rmse_improvement_over_zero < -0.05]
    if len(failed_o) != 1:
        raise RuntimeError("expected-one-o-severe-regression")
    failed_row = failed_o.iloc[0]
    failed_fold = int(failed_row.fold)
    failed_product = str(failed_row.observation)
    coincides = bool(block_outliers[failed_fold] == failed_product)
    failed_outlier_ratio = float(
        outliers[(outliers.fold == failed_fold) & (outliers["product"] == failed_product)]
        .iloc[0]
        .outlier_ratio
    )
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "common_western_cell_count": int(common.sum()),
        "block_specific_outlier_by_fold": {
            str(fold): product for fold, product in block_outliers.items()
        },
        "o_severe_regression_fold": failed_fold,
        "o_severe_regression_years": [int(failed_row.start_year), int(failed_row.end_year)],
        "o_severe_regression_observation": failed_product,
        "failed_product_outlier_ratio": failed_outlier_ratio,
        "o_failure_coincides_with_same_product_observation_outlier": coincides,
        "decision": (
            "route_o_candidate_to_fourth_product_adjudication_without_relaxing_round48"
            if coincides
            else "close_o_candidate_as_modernly_unstable"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 1, figsize=(12, 7.5), constrained_layout=True)
    colors = {"UDel": "#111111", "CRU": "#d55e00", "GPCC": "#009e73"}
    for product in PRODUCTS:
        axes[0].plot(YEARS, series[product], color=colors[product], label=product)
    axes[0].axvspan(1976, 1987, color="#d9d9d9", alpha=0.65)
    axes[0].axhline(0, color="#999999", linewidth=0.8)
    axes[0].set(title="Western HMA observation products", ylabel="Anomaly (mm/day)")
    axes[0].legend(frameon=False, ncol=3)
    pivot = outliers.pivot(index="fold", columns="product", values="outlier_ratio").reindex(columns=PRODUCTS)
    pivot.plot.bar(ax=axes[1], color=[colors[product] for product in PRODUCTS])
    axes[1].axhline(1.5, color="#b2182b", linestyle="--", linewidth=1.0)
    axes[1].set(
        title="Block-specific product outlier ratio",
        xlabel="Frozen block",
        ylabel="RMSE to other-product mean / other-pair RMSE",
    )
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(frameon=False)
    figure.savefig(output / "western_observation_block_conflict.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Western Observation Block Conflict",
        "",
        f"- O severe regression: `{failed_product}`, fold `{failed_fold}` ({int(failed_row.start_year)}-{int(failed_row.end_year)}).",
        f"- Same-product observation outlier ratio: `{failed_outlier_ratio:.3f}`.",
        f"- Failure coincides with observation outlier: `{str(coincides).lower()}`.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "sources_before": before,
                "sources_after": after,
                "source_states_unchanged": unchanged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
