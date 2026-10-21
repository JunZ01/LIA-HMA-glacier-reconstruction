"""Adjudicate western modern precipitation skill with UDel as a third product."""

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
MEMBERS = {
    "HGAO": "pr_rec_ref_HGAO_c05d",
    "LMR": "pr_rec_ref_LMR_c05d",
    "O": "pr_rec_ref_O_c05d",
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


def clean(dataset: h5py.Dataset, selection: slice) -> np.ndarray:
    values = np.asarray(dataset[selection], dtype=float)
    fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).squeeze())
    values[np.isclose(values, fill) | (values < -9000)] = np.nan
    return values


def finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays)
    count = np.isfinite(stack).sum(axis=0)
    return np.nansum(stack, axis=0) / np.where(count == 0, np.nan, count)


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
        total_mm = np.nansum(values, axis=0) * 10.0
        days = 366 if calendar.isleap(int(year)) else 365
        annual[index] = np.where(complete, total_mm / days, np.nan)
    return annual


def weighted_series(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(values)
    numerator = np.nansum(values * weights[None], axis=(1, 2))
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


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


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_49_western_udel_modern_adjudication_v1"
    round31 = loop / "round_31_precipitation_fixed_reference_candidate_skill_v1"
    round47 = loop / "round_47_public_reconstruction_western_early_constraint_v1"
    paths = {
        "udel": loop / "round_44_udel_allowed_subset_acquisition_v2/udel_precip_v501_1940_2000_hma_bbox.nc",
        "public_reference_source": Path(
            r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
        ),
        "hma_mask": project
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "round31_blocks": round31 / "fixed_candidate_block_scores.csv",
        "round47_summary": round47 / "summary.json",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    round47_summary = json.loads(paths["round47_summary"].read_text(encoding="utf-8"))
    candidates_to_test = list(round47_summary["passing_candidates"])
    if not candidates_to_test:
        raise RuntimeError("no-round47-candidates")
    mask = np.load(paths["hma_mask"]).astype(bool)
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
        if mask.shape != (len(latitudes), len(longitudes)):
            raise RuntimeError("hma-mask-shape-failed")
        members = {}
        baseline = (YEARS >= 1961) & (YEARS <= 1990)
        for name, variable in MEMBERS.items():
            raw = clean(handle[variable], selection)
            members[name] = raw - np.nanmean(raw[baseline], axis=0)[None]

    all_candidates = {
        "HGAO": members["HGAO"],
        "O": members["O"],
        "LMR": members["LMR"],
        "local_enkf": finite_mean([members["HGAO"], members["O"]]),
        "member_equal": finite_mean([members["HGAO"], members["LMR"], members["O"]]),
        "family_balanced": finite_mean(
            [finite_mean([members["HGAO"], members["O"]]), members["LMR"]]
        ),
    }
    candidates = {name: all_candidates[name] for name in candidates_to_test}
    udel_raw = annualize_udel(paths["udel"], latitudes, longitudes)
    baseline = (YEARS >= 1961) & (YEARS <= 1990)
    udel = udel_raw - np.nanmean(udel_raw[baseline], axis=0)[None]
    west_mask = mask & (np.broadcast_to(longitudes[None], mask.shape) < 80.0)

    block_rows: list[dict[str, object]] = []
    full_rows: list[dict[str, object]] = []
    series_rows: list[dict[str, object]] = []
    result_map: dict[str, dict[str, object]] = {}
    for candidate_name, values in candidates.items():
        common = west_mask & np.all(np.isfinite(values), axis=0) & np.all(np.isfinite(udel), axis=0)
        if not np.any(common):
            raise RuntimeError(f"empty-west-support:{candidate_name}")
        candidate_series = weighted_series(values, common, latitudes)
        udel_series = weighted_series(udel, common, latitudes)
        for index, year in enumerate(YEARS):
            series_rows.append(
                {
                    "year": int(year),
                    "candidate": candidate_name,
                    "candidate_anomaly_mm_day": float(candidate_series[index]),
                    "udel_anomaly_mm_day": float(udel_series[index]),
                }
            )
        improvements: list[float] = []
        for fold, (start, end) in enumerate(BLOCKS, start=1):
            selected = (YEARS >= start) & (YEARS <= end)
            candidate_rmse = float(
                np.sqrt(np.mean((candidate_series[selected] - udel_series[selected]) ** 2))
            )
            zero_rmse = float(np.sqrt(np.mean(udel_series[selected] ** 2)))
            improvement = float(1.0 - candidate_rmse / zero_rmse)
            improvements.append(improvement)
            block_rows.append(
                {
                    "candidate": candidate_name,
                    "observation": "UDel",
                    "fold": fold,
                    "start_year": start,
                    "end_year": end,
                    "common_cell_count": int(common.sum()),
                    "regional_rmse_mm_day": candidate_rmse,
                    "zero_regional_rmse_mm_day": zero_rmse,
                    "regional_rmse_improvement_over_zero": improvement,
                }
            )
        annual_corr = correlation(candidate_series, udel_series)
        lowpass_corr = correlation(smooth(candidate_series), smooth(udel_series))
        gates = {
            "median_block_improvement_at_least_5pct": float(np.median(improvements)) >= 0.05,
            "maximum_block_regression_at_most_5pct": float(np.max(-np.asarray(improvements))) <= 0.05,
            "at_least_4_of_5_blocks_improve": int(np.sum(np.asarray(improvements) > 0)) >= 4,
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
            "passes_udel_western_modern_diagnostic": bool(all(gates.values())),
        }
        result_map[candidate_name] = result
        full_rows.append({"candidate": candidate_name, **{key: value for key, value in result.items() if key != "gates"}})

    udel_blocks = pd.DataFrame(block_rows)
    udel_full = pd.DataFrame(full_rows)
    series_frame = pd.DataFrame(series_rows)
    archived = pd.read_csv(paths["round31_blocks"])
    archived = archived[
        (archived.region == "west_lt80")
        & archived.candidate.isin(candidates_to_test)
        & archived.observation.isin(["CRU", "GPCC"])
    ][
        [
            "candidate",
            "observation",
            "fold",
            "start_year",
            "end_year",
            "regional_rmse_mm_day",
            "zero_regional_rmse_mm_day",
            "regional_rmse_improvement_over_zero",
        ]
    ]
    if len(archived) != len(candidates_to_test) * 2 * 5:
        raise RuntimeError("archived-round31-row-count-failed")
    all_blocks = pd.concat([archived, udel_blocks], ignore_index=True)
    udel_blocks.to_csv(output / "udel_western_block_scores.csv", index=False)
    all_blocks.to_csv(output / "three_product_western_block_scores.csv", index=False)
    udel_full.to_csv(output / "udel_western_candidate_summary.csv", index=False)
    series_frame.to_csv(output / "udel_western_candidate_series.csv", index=False)

    passing = [
        name
        for name, result in result_map.items()
        if result["passes_udel_western_modern_diagnostic"]
    ]
    product_fold_counts = (
        all_blocks.assign(improves=all_blocks.regional_rmse_improvement_over_zero > 0)
        .groupby(["candidate", "observation"], as_index=False)
        .agg(
            improving_fold_count=("improves", "sum"),
            worst_regression=("regional_rmse_improvement_over_zero", lambda values: float(np.max(-values))),
            median_improvement=("regional_rmse_improvement_over_zero", "median"),
        )
    )
    product_fold_counts.to_csv(output / "three_product_failure_localization.csv", index=False)
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "candidate_results": result_map,
        "udel_passing_candidates": passing,
        "failure_localization": product_fold_counts.to_dict(orient="records"),
        "decision": (
            "western_failure_is_observation_product_specific_for_at_least_one_candidate"
            if passing
            else "western_candidate_instability_is_confirmed_by_udel"
        ),
        "round48_gates_remain_unchanged": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for candidate_name in candidates_to_test:
        selected = series_frame[series_frame.candidate == candidate_name]
        axes[0].plot(
            selected.year,
            selected.candidate_anomaly_mm_day,
            linewidth=1.2,
            label=candidate_name,
        )
    first = series_frame[series_frame.candidate == candidates_to_test[0]]
    axes[0].plot(first.year, first.udel_anomaly_mm_day, color="#111111", linewidth=2.3, label="UDel")
    axes[0].axhline(0, color="#aaaaaa", linewidth=0.8)
    axes[0].set(
        title="Western HMA modern precipitation",
        xlabel="Year",
        ylabel="Anomaly (mm/day)",
    )
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    pivot = product_fold_counts.pivot(
        index="candidate", columns="observation", values="median_improvement"
    ).reindex(candidates_to_test)[["CRU", "GPCC", "UDel"]]
    pivot.mul(100).plot.bar(ax=axes[1], color=["#d55e00", "#009e73", "#222222"])
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].axhline(5, color="#b2182b", linestyle="--", linewidth=1.0)
    axes[1].set(
        title="Median five-block skill by observation",
        ylabel="RMSE improvement over zero anomaly (%)",
        xlabel="",
    )
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False)
    figure.savefig(output / "western_udel_modern_adjudication.png", dpi=180)
    plt.close(figure)

    lines = ["# Western UDel Modern Adjudication", ""]
    for name, result in result_map.items():
        lines.append(
            f"- {name}: UDel median skill `{result['median_block_improvement_over_zero']:.2%}`, "
            f"worst regression `{result['maximum_block_regression']:.2%}`, "
            f"improving folds `{result['improving_fold_count']}/5`, "
            f"annual/15-year correlations `{result['annual_correlation']:.3f}/{result['correlation_15year']:.3f}`, "
            f"passes `{str(result['passes_udel_western_modern_diagnostic']).lower()}`."
        )
    lines.extend(
        [
            "",
            f"UDel-passing candidates: `{passing}`.",
            "",
            f"Decision: `{summary['decision']}`.",
            "Round 48 gates remain unchanged.",
        ]
    )
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
