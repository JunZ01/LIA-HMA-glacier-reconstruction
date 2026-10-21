"""Validate NOAA PREC/L and adjudicate frozen western precipitation candidates."""

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


YEARS = np.arange(1948, 2001)
BLOCKS = ((1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
CHUNKS = ((1948, 1957), (1958, 1967), (1968, 1977), (1978, 1987), (1988, 1997), (1998, 2000))
MEMBERS = {
    "HGAO": "pr_rec_ref_HGAO_c05d",
    "LMR": "pr_rec_ref_LMR_c05d",
    "O": "pr_rec_ref_O_c05d",
}
OBSERVATIONS = {
    "CRU": "pr_obs_cru_pre_05deg",
    "GPCC": "pr_obs_gpcp_05deg",
}


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


def text_attr(value: object) -> str:
    item = np.asarray(value).reshape(-1)[0]
    return item.decode() if isinstance(item, bytes) else str(item)


def clean(dataset: h5py.Dataset, selection: slice) -> np.ndarray:
    values = np.asarray(dataset[selection], dtype=float)
    fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).squeeze())
    values[np.isclose(values, fill) | (values < -9000)] = np.nan
    return values


def finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


def decode_precl_time(values: np.ndarray, units: str) -> list[datetime]:
    if units != "hours since 1800-1-1 00:00:0.0":
        raise RuntimeError(f"unexpected-precl-time-units:{units}")
    origin = datetime(1800, 1, 1)
    return [origin + timedelta(hours=float(value)) for value in values]


def annualize_precl(paths: list[Path], target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    all_dates: list[datetime] = []
    all_values: list[np.ndarray] = []
    reference_lat: np.ndarray | None = None
    reference_lon: np.ndarray | None = None
    for path in paths:
        with h5py.File(path, "r") as handle:
            lat = np.asarray(handle["lat"], dtype=float)
            lon = np.asarray(handle["lon"], dtype=float)
            dates = decode_precl_time(
                np.asarray(handle["time"], dtype=float),
                text_attr(handle["time"].attrs["units"]),
            )
            dataset = handle["precip"]
            if text_attr(dataset.attrs.get("units", "")) != "mm/day":
                raise RuntimeError(f"unexpected-precl-units:{path.name}")
            values = np.asarray(dataset, dtype=float)
            fill = float(np.asarray(dataset.attrs.get("missing_value", -9.96921e36)).reshape(-1)[0])
            values[np.isclose(values, fill) | (values < -1e20)] = np.nan
        if reference_lat is None:
            reference_lat, reference_lon = lat, lon
        elif not np.array_equal(reference_lat, lat) or not np.array_equal(reference_lon, lon):
            raise RuntimeError(f"precl-grid-varies:{path.name}")
        all_dates.extend(dates)
        all_values.append(values)
    assert reference_lat is not None and reference_lon is not None
    expected = [(int(year), month) for year in YEARS for month in range(1, 13)]
    if [(date.year, date.month) for date in all_dates] != expected:
        raise RuntimeError("precl-month-axis-failed")
    if any(1921 <= date.year <= 1939 for date in all_dates):
        raise RuntimeError("precl-locked-period-present")
    lat_index = np.asarray(
        [np.flatnonzero(np.isclose(reference_lat, value, atol=1e-7))[0] for value in target_lat],
        dtype=int,
    )
    lon_index = np.asarray(
        [np.flatnonzero(np.isclose(reference_lon, value, atol=1e-7))[0] for value in target_lon],
        dtype=int,
    )
    monthly = np.concatenate(all_values, axis=0)[:, lat_index, :][:, :, lon_index]
    annual = np.full((len(YEARS), len(target_lat), len(target_lon)), np.nan)
    offset = 0
    for index, year in enumerate(YEARS):
        values = monthly[offset : offset + 12]
        weights = np.asarray([calendar.monthrange(int(year), month)[1] for month in range(1, 13)])
        complete = np.all(np.isfinite(values), axis=0)
        annual[index] = np.where(
            complete,
            np.sum(values * weights[:, None, None], axis=0) / weights.sum(),
            np.nan,
        )
        offset += 12
    return annual


def decode_udel_dates(handle: h5py.File) -> list[datetime]:
    units = text_attr(handle["time"].attrs["units"])
    if units != "days since 1900-01-01 00:00:00":
        raise RuntimeError(f"unexpected-udel-time-units:{units}")
    return [datetime(1900, 1, 1) + timedelta(days=float(value)) for value in handle["time"]]


def annualize_udel(path: Path, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        dates = decode_udel_dates(handle)
        selected = [index for index, date in enumerate(dates) if 1948 <= date.year <= 2000]
        actual = [(dates[index].year, dates[index].month) for index in selected]
        expected = [(int(year), month) for year in YEARS for month in range(1, 13)]
        if actual != expected:
            raise RuntimeError("udel-month-axis-failed")
        if any(1921 <= dates[index].year <= 1939 for index in selected):
            raise RuntimeError("udel-locked-period-requested")
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        lat_index = np.asarray([np.flatnonzero(np.isclose(lat, value, atol=1e-7))[0] for value in target_lat])
        lon_index = np.asarray([np.flatnonzero(np.isclose(lon, value, atol=1e-7))[0] for value in target_lon])
        monthly = np.asarray(handle["precip"][selected][:, lat_index, :][:, :, lon_index], dtype=float)
    annual = np.full((len(YEARS), len(target_lat), len(target_lon)), np.nan)
    for index, year in enumerate(YEARS):
        values = monthly[index * 12 : (index + 1) * 12]
        complete = np.all(np.isfinite(values), axis=0)
        annual[index] = np.where(
            complete,
            np.sum(values, axis=0) * 10.0 / (366 if calendar.isleap(int(year)) else 365),
            np.nan,
        )
    return annual


def weighted_series(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(values)
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return np.nansum(values * weights[None], axis=(1, 2)) / np.where(denominator == 0, np.nan, denominator)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def smooth(values: np.ndarray, width: int = 15) -> np.ndarray:
    result = np.full(len(values), np.nan)
    half = width // 2
    for index in range(half, len(values) - half):
        window = values[index - half : index + half + 1]
        if np.all(np.isfinite(window)):
            result[index] = float(np.mean(window))
    return result


def recenter(values: np.ndarray) -> np.ndarray:
    baseline = (YEARS >= 1961) & (YEARS <= 1990)
    with np.errstate(invalid="ignore"):
        center = np.nanmean(values[baseline], axis=0)
    return values - center[None]


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_58_noaa_precl_western_adjudication_v1"
    round57 = loop / "round_57_noaa_precl_allowed_subset_acquisition_v3"
    round47 = loop / "round_47_public_reconstruction_western_early_constraint_v1"
    precl_paths = [round57 / f"precl_{start}_{end}.nc" for start, end in CHUNKS]
    paths = {
        **{f"precl_{start}_{end}": path for (start, end), path in zip(CHUNKS, precl_paths)},
        "round57_summary": round57 / "summary.json",
        "round47_summary": round47 / "summary.json",
        "udel": loop / "round_44_udel_allowed_subset_acquisition_v2/udel_precip_v501_1940_2000_hma_bbox.nc",
        "public_reference_source": Path(
            r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
        ),
        "hma_mask": project
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    if not json.loads(paths["round57_summary"].read_text(encoding="utf-8"))["passes_acquisition_preflight"]:
        raise RuntimeError("round57-preflight-did-not-pass")

    mask = np.load(paths["hma_mask"]).astype(bool)
    with h5py.File(paths["public_reference_source"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).ravel()
        indices = np.flatnonzero((source_years >= 1948) & (source_years <= 2000))
        if not np.array_equal(source_years[indices], YEARS):
            raise RuntimeError("public-year-axis-failed")
        if np.any((source_years[indices] >= 1921) & (source_years[indices] <= 1939)):
            raise RuntimeError("locked-period-index-requested")
        selection = slice(int(indices[0]), int(indices[-1]) + 1)
        latitudes = np.asarray(handle["lat"], dtype=float).ravel()
        longitudes = np.asarray(handle["lon"], dtype=float).ravel()
        if mask.shape != (len(latitudes), len(longitudes)):
            raise RuntimeError("hma-mask-shape-failed")
        observations = {
            name: recenter(clean(handle[variable], selection))
            for name, variable in OBSERVATIONS.items()
        }
        members = {
            name: recenter(clean(handle[variable], selection))
            for name, variable in MEMBERS.items()
        }

    observations["UDel"] = recenter(annualize_udel(paths["udel"], latitudes, longitudes))
    observations["PREC/L"] = recenter(annualize_precl(precl_paths, latitudes, longitudes))
    all_candidates = {
        "HGAO": members["HGAO"],
        "O": members["O"],
        "LMR": members["LMR"],
        "local_enkf": finite_mean([members["HGAO"], members["O"]]),
        "member_equal": finite_mean([members["HGAO"], members["LMR"], members["O"]]),
        "family_balanced": finite_mean([finite_mean([members["HGAO"], members["O"]]), members["LMR"]]),
    }
    round47_summary = json.loads(paths["round47_summary"].read_text(encoding="utf-8"))
    candidate_names = list(round47_summary["passing_candidates"])
    candidates = {name: all_candidates[name] for name in candidate_names}

    west = mask & (np.broadcast_to(longitudes[None], mask.shape) < 80.0)
    common = west.copy()
    for values in observations.values():
        common &= np.all(np.isfinite(values), axis=0)
    if common.sum() < 50:
        raise RuntimeError(f"insufficient-four-product-common-west-support:{int(common.sum())}")
    observation_series = {
        name: weighted_series(values, common, latitudes)
        for name, values in observations.items()
    }

    compatibility_rows: list[dict[str, object]] = []
    compatibility_gates: dict[str, bool] = {}
    precl_series = observation_series["PREC/L"]
    for product in ("CRU", "GPCC", "UDel"):
        other = observation_series[product]
        annual_corr = correlation(precl_series, other)
        lowpass_corr = correlation(smooth(precl_series), smooth(other))
        std_ratio = float(np.std(precl_series, ddof=1) / np.std(other, ddof=1))
        compatibility_rows.append(
            {
                "comparison_product": product,
                "common_cell_count": int(common.sum()),
                "annual_correlation": annual_corr,
                "correlation_15year": lowpass_corr,
                "precl_to_product_std_ratio": std_ratio,
            }
        )
        compatibility_gates[f"annual_correlation_with_{product}_at_least_0_50"] = annual_corr >= 0.50
        compatibility_gates[f"correlation_15year_with_{product}_at_least_0_50"] = lowpass_corr >= 0.50
        compatibility_gates[f"std_ratio_with_{product}_between_0_5_and_1_5"] = 0.5 <= std_ratio <= 1.5
    passes_compatibility = bool(all(compatibility_gates.values()))
    pd.DataFrame(compatibility_rows).to_csv(output / "precl_western_compatibility.csv", index=False)
    pd.DataFrame({"year": YEARS, **observation_series}).to_csv(
        output / "western_observation_series.csv", index=False
    )

    candidate_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    series_rows: list[dict[str, object]] = []
    candidate_results: dict[str, dict[str, object]] = {}
    if passes_compatibility:
        for name, values in candidates.items():
            candidate_series = weighted_series(values, common, latitudes)
            for index, year in enumerate(YEARS):
                series_rows.append(
                    {
                        "year": int(year),
                        "candidate": name,
                        "candidate_anomaly_mm_day": float(candidate_series[index]),
                        "precl_anomaly_mm_day": float(precl_series[index]),
                    }
                )
            improvements: list[float] = []
            for fold, (start, end) in enumerate(BLOCKS, start=2):
                selected = (YEARS >= start) & (YEARS <= end)
                rmse = float(np.sqrt(np.mean((candidate_series[selected] - precl_series[selected]) ** 2)))
                zero_rmse = float(np.sqrt(np.mean(precl_series[selected] ** 2)))
                improvement = float(1.0 - rmse / zero_rmse)
                improvements.append(improvement)
                block_rows.append(
                    {
                        "candidate": name,
                        "observation": "PREC/L",
                        "fold": fold,
                        "start_year": start,
                        "end_year": end,
                        "common_cell_count": int(common.sum()),
                        "regional_rmse_mm_day": rmse,
                        "zero_regional_rmse_mm_day": zero_rmse,
                        "regional_rmse_improvement_over_zero": improvement,
                    }
                )
            annual_corr = correlation(candidate_series, precl_series)
            lowpass_corr = correlation(smooth(candidate_series), smooth(precl_series))
            gates = {
                "median_block_improvement_at_least_5pct": float(np.median(improvements)) >= 0.05,
                "maximum_block_regression_at_most_5pct": float(np.max(-np.asarray(improvements))) <= 0.05,
                "at_least_3_of_4_blocks_improve": int(np.sum(np.asarray(improvements) > 0)) >= 3,
                "annual_correlation_at_least_0_30": annual_corr >= 0.30,
                "correlation_15year_at_least_0_30": lowpass_corr >= 0.30,
            }
            result = {
                "median_block_improvement_over_zero": float(np.median(improvements)),
                "maximum_block_regression": float(np.max(-np.asarray(improvements))),
                "improving_fold_count": int(np.sum(np.asarray(improvements) > 0)),
                "annual_correlation": annual_corr,
                "correlation_15year": lowpass_corr,
                "gates": gates,
                "passes_precl_western_diagnostic": bool(all(gates.values())),
            }
            candidate_results[name] = result
            candidate_rows.append({"candidate": name, **{key: value for key, value in result.items() if key != "gates"}})
    pd.DataFrame(block_rows).to_csv(output / "precl_western_candidate_block_scores.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output / "precl_western_candidate_summary.csv", index=False)
    pd.DataFrame(series_rows).to_csv(output / "precl_western_candidate_series.csv", index=False)

    passing_candidates = [
        name for name, result in candidate_results.items() if result["passes_precl_western_diagnostic"]
    ]
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "years": [1948, 2000],
        "candidate_blocks": [list(block) for block in BLOCKS],
        "common_western_cell_count": int(common.sum()),
        "precl_compatibility": compatibility_rows,
        "precl_compatibility_gates": compatibility_gates,
        "passes_precl_western_modern_compatibility": passes_compatibility,
        "candidate_results": candidate_results,
        "precl_passing_candidates": passing_candidates,
        "candidate_scoring_executed": passes_compatibility,
        "decision": (
            "use_precl_for_fixed_candidate_triangulation"
            if passes_compatibility and passing_candidates
            else "precl_supports_no_fixed_candidate"
            if passes_compatibility
            else "do_not_use_precl_for_candidate_adjudication"
        ),
        "round48_and_round50_decisions_unchanged": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 1, figsize=(12, 8.2), constrained_layout=True)
    colors = {"CRU": "#d55e00", "GPCC": "#009e73", "UDel": "#0072b2", "PREC/L": "#111111"}
    for product, values in observation_series.items():
        axes[0].plot(YEARS, values, color=colors[product], linewidth=2.0 if product == "PREC/L" else 1.2, label=product)
    axes[0].axhline(0, color="#999999", linewidth=0.8)
    axes[0].set(title="Western HMA observation products on common support", ylabel="Anomaly (mm/day)")
    axes[0].legend(frameon=False, ncol=4)
    if passes_compatibility:
        for name, values in candidates.items():
            axes[1].plot(YEARS, weighted_series(values, common, latitudes), linewidth=1.1, label=name)
        axes[1].plot(YEARS, precl_series, color="#111111", linewidth=2.3, label="PREC/L")
        axes[1].legend(frameon=False, ncol=3, fontsize=8)
        axes[1].set_title("Frozen western candidates versus NOAA PREC/L")
    else:
        for product, values in observation_series.items():
            axes[1].plot(YEARS, smooth(values), color=colors[product], linewidth=1.8, label=product)
        axes[1].legend(frameon=False, ncol=4)
        axes[1].set_title("15-year observation-product curves: compatibility failed")
    axes[1].axhline(0, color="#999999", linewidth=0.8)
    axes[1].set(xlabel="Year", ylabel="Anomaly (mm/day)")
    figure.savefig(output / "noaa_precl_western_adjudication.png", dpi=180)
    plt.close(figure)

    lines = ["# NOAA PREC/L Western Adjudication", ""]
    lines.append(f"- Shared western cells: `{int(common.sum())}`.")
    for row in compatibility_rows:
        lines.append(
            f"- PREC/L vs {row['comparison_product']}: annual/15-year correlations "
            f"`{row['annual_correlation']:.3f}/{row['correlation_15year']:.3f}`, "
            f"standard-deviation ratio `{row['precl_to_product_std_ratio']:.3f}`."
        )
    lines.append(f"- PREC/L compatibility passes: `{str(passes_compatibility).lower()}`.")
    if passes_compatibility:
        for name, result in candidate_results.items():
            lines.append(
                f"- {name}: median skill `{result['median_block_improvement_over_zero']:.2%}`, "
                f"worst regression `{result['maximum_block_regression']:.2%}`, "
                f"improving blocks `{result['improving_fold_count']}/4`, "
                f"annual/15-year correlations `{result['annual_correlation']:.3f}/{result['correlation_15year']:.3f}`, "
                f"passes `{str(result['passes_precl_western_diagnostic']).lower()}`."
            )
    lines.extend(["", f"Decision: `{summary['decision']}`.", "Round 48 and Round 50 decisions remain unchanged."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
