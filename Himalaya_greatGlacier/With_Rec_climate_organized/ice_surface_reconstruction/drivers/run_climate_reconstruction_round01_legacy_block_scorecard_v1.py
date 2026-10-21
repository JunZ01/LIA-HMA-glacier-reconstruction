"""Round 01: common-mask temporal-block scorecard for the six legacy members."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_ID = "climate_reconstruction_improvement_loop_v1/round_01_blocked_cv_baseline"
YEARS = np.arange(1940, 2001, dtype=int)
LOCKED_PERIOD = (1921, 1939)
BLOCKS = {
    "block_1_1940_1951": (1940, 1951),
    "block_2_1952_1963": (1952, 1963),
    "block_3_1964_1975": (1964, 1975),
    "block_4_1976_1987": (1976, 1987),
    "block_5_1988_2000": (1988, 2000),
    "all_1940_2000": (1940, 2000),
}
NETCDF = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
MEMBERS = {
    "AO_6": "AO_6_m_c05d",
    "AO_9": "AO_9_m_c05d",
    "HGAO_6": "HGAO_6_m_c05d",
    "HGAO_9": "HGAO_9_m_c05d",
    "O_6": "O_6_m_c05d",
    "O_9": "O_9_m_c05d",
}
OBSERVATIONS = {
    "tas": {
        "Berkeley Earth": "tas_obs_bea_05deg",
        "CRU": "tas_obs_cru_tmp_05deg",
        "GISTEMP": "tas_obs_GIST_05deg",
        "HadCRUT5": "tas_obs_had5_05deg",
        "NOAA": "tas_obs_NOAA_05deg",
    },
    "pr": {
        "CRU": "pr_obs_cru_pre_05deg",
        "GPCC": "pr_obs_gpcp_05deg",
    },
}
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state(path: Path) -> dict[str, object]:
    item = path.stat()
    return {"path": str(path), "size_bytes": int(item.st_size), "mtime_ns": int(item.st_mtime_ns)}


def clean(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result[(result <= -9000) | (result >= 1e20)] = np.nan
    return result


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.nanstd(left[valid]) == 0 or np.nanstd(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def area_series(field: np.ndarray, latitudes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(field) & mask[None, :, :]
    numerator = np.sum(np.where(valid, field, 0.0) * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return np.divide(numerator, denominator, out=np.full(field.shape[0], np.nan), where=denominator > 0)


def score(reconstruction: np.ndarray, observation: np.ndarray, latitudes: np.ndarray, mask: np.ndarray, years: np.ndarray) -> dict[str, object]:
    spatial_mask = mask[None, :, :]
    valid = np.isfinite(reconstruction) & np.isfinite(observation) & spatial_mask
    delta = reconstruction[valid] - observation[valid]
    rec_series = area_series(np.where(valid, reconstruction, np.nan), latitudes, mask)
    obs_series = area_series(np.where(valid, observation, np.nan), latitudes, mask)
    series_valid = np.isfinite(rec_series) & np.isfinite(obs_series)
    if int(valid.sum()) == 0 or int(series_valid.sum()) < 3:
        return {
            "valid_value_count": int(valid.sum()), "valid_year_count": int(series_valid.sum()),
            "bias": float("nan"), "rmse": float("nan"), "field_correlation": float("nan"),
            "annual_mean_correlation": float("nan"), "annual_mean_rmse": float("nan"),
            "reconstruction_trend_per_century": float("nan"), "observation_trend_per_century": float("nan"),
            "absolute_trend_difference_per_century": float("nan"),
        }
    rec_trend = float(np.polyfit(years[series_valid], rec_series[series_valid], 1)[0] * 100.0)
    obs_trend = float(np.polyfit(years[series_valid], obs_series[series_valid], 1)[0] * 100.0)
    return {
        "valid_value_count": int(valid.sum()),
        "valid_year_count": int(series_valid.sum()),
        "bias": float(np.mean(delta)),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "field_correlation": correlation(reconstruction[valid], observation[valid]),
        "annual_mean_correlation": correlation(rec_series, obs_series),
        "annual_mean_rmse": float(np.sqrt(np.mean(np.square(rec_series[series_valid] - obs_series[series_valid])))),
        "reconstruction_trend_per_century": rec_trend,
        "observation_trend_per_century": obs_trend,
        "absolute_trend_difference_per_century": abs(rec_trend - obs_trend),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0" / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    round00 = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_00_production_reproduction"
    mask_file = round00 / "source_valid_domain_masks_05deg.npz"
    hma_mask_file = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
    lock_contract = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_instrumental_baseline_v1/LOCKED_FINAL_PERIOD.json"
    facts = root / "LIA_PROJECT_FACTS.md"
    driver = Path(__file__).resolve()
    sources = [NETCDF, mask_file, hma_mask_file, lock_contract, facts]
    if missing := [str(path) for path in sources if not path.exists()]:
        raise FileNotFoundError("missing-round01-inputs:" + ";".join(missing))
    locked = json.loads(lock_contract.read_text(encoding="utf-8"))
    if locked.get("status") != "locked_not_evaluated":
        raise RuntimeError("locked-period-contract-failed")
    if np.any((YEARS >= LOCKED_PERIOD[0]) & (YEARS <= LOCKED_PERIOD[1])):
        raise RuntimeError("locked-period-requested")

    before = {str(path): state(path) for path in sources}
    hashes = {str(path): digest(path) for path in sources}
    masks = np.load(mask_file)
    hma_mask = np.load(hma_mask_file).astype(bool)
    if hma_mask.shape != (43, 78):
        raise RuntimeError("hma-mask-shape-failed")

    fields: dict[str, dict[str, np.ndarray]] = {"tas": {}, "pr": {}}
    observations: dict[str, dict[str, np.ndarray]] = {"tas": {}, "pr": {}}
    with h5py.File(NETCDF, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        indices = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        latitudes = np.asarray(handle["lat"], dtype=float)
        longitudes = np.asarray(handle["lon"], dtype=float)
        for variable in ("tas", "pr"):
            for label, suffix in MEMBERS.items():
                fields[variable][label] = clean(np.asarray(handle[f"{variable}_rec_result_{suffix}"][indices], dtype=float))
            fields[variable]["equal_six_mean"] = clean(np.asarray(handle[f"{variable}_ensemble_mean_rec_result"][indices], dtype=float))
            for label, name in OBSERVATIONS[variable].items():
                observations[variable][label] = clean(np.asarray(handle[name][indices], dtype=float))

    source_masks = {
        "tas": np.asarray(masks["tas_common_six_member"], dtype=bool),
        "pr": np.asarray(masks["pr_common_six_member"], dtype=bool),
    }
    if any(np.any(hma_mask & ~source_masks[variable]) for variable in source_masks):
        raise RuntimeError("round00-source-mask-excludes-hma-cells")
    lon_grid = np.broadcast_to(longitudes[None, :], hma_mask.shape)
    region_masks = {
        variable: {
            region: hma_mask & source_masks[variable] & (lon_grid >= lower) & (lon_grid < upper)
            for region, (lower, upper) in REGIONS.items()
        }
        for variable in ("tas", "pr")
    }

    score_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    curves: dict[str, dict[str, dict[str, np.ndarray]]] = {"tas": {}, "pr": {}}
    for variable in ("tas", "pr"):
        for region, region_mask in region_masks[variable].items():
            curves[variable][region] = {}
            for branch, field in fields[variable].items():
                series = area_series(field, latitudes, region_mask)
                curves[variable][region][branch] = series
                for year, value in zip(YEARS, series):
                    curve_rows.append({"year": int(year), "variable": variable, "region": region, "series_type": "reconstruction", "series": branch, "value": float(value)})
            for observation, field in observations[variable].items():
                series = area_series(field, latitudes, region_mask)
                curves[variable][region][observation] = series
                for year, value in zip(YEARS, series):
                    curve_rows.append({"year": int(year), "variable": variable, "region": region, "series_type": "observation", "series": observation, "value": float(value)})
            for block, (start, end) in BLOCKS.items():
                selected = (YEARS >= start) & (YEARS <= end)
                for branch, field in fields[variable].items():
                    for observation, observed in observations[variable].items():
                        score_rows.append({
                            "variable": variable,
                            "branch": branch,
                            "observation": observation,
                            "region": region,
                            "block": block,
                            "start_year": start,
                            "end_year": end,
                            "region_gridcell_count": int(region_mask.sum()),
                            **score(field[selected], observed[selected], latitudes, region_mask, YEARS[selected]),
                        })

    all_rows = [row for row in score_rows if row["block"] == "all_1940_2000"]
    rankings: list[dict[str, object]] = []
    for variable in ("tas", "pr"):
        for branch in [*MEMBERS, "equal_six_mean"]:
            selected = [row for row in all_rows if row["variable"] == variable and row["branch"] == branch]
            rankings.append({
                "variable": variable,
                "branch": branch,
                "median_rmse_across_region_reference_pairs": float(np.nanmedian([row["rmse"] for row in selected])),
                "median_annual_mean_rmse": float(np.nanmedian([row["annual_mean_rmse"] for row in selected])),
                "median_annual_mean_correlation": float(np.nanmedian([row["annual_mean_correlation"] for row in selected])),
                "median_absolute_trend_difference": float(np.nanmedian([row["absolute_trend_difference_per_century"] for row in selected])),
            })
    ranking_by_variable = {
        variable: sorted([row for row in rankings if row["variable"] == variable], key=lambda row: row["median_rmse_across_region_reference_pairs"])
        for variable in ("tas", "pr")
    }
    equal_baseline = {variable: next(row for row in rankings if row["variable"] == variable and row["branch"] == "equal_six_mean") for variable in ("tas", "pr")}

    output.mkdir(parents=True)
    write_csv(output / "temporal_block_scores.csv", score_rows)
    write_csv(output / "regional_annual_curves.csv", curve_rows)
    write_csv(output / "legacy_branch_ranking.csv", rankings)

    colors = {"AO_6": "#0072B2", "AO_9": "#56B4E9", "HGAO_6": "#009E73", "HGAO_9": "#E69F00", "O_6": "#CC79A7", "O_9": "#D55E00"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for axis, variable in zip(axes, ("tas", "pr")):
        for branch, color in colors.items():
            axis.plot(YEARS, curves[variable]["hma"][branch], color=color, linewidth=1.1, alpha=0.72, label=branch)
        axis.plot(YEARS, curves[variable]["hma"]["equal_six_mean"], color="#111111", linewidth=2.8, label="Equal six-member mean")
        for index, observation in enumerate(OBSERVATIONS[variable]):
            axis.plot(YEARS, curves[variable]["hma"][observation], linewidth=1.2, linestyle="--", color=plt.cm.Greys(0.35 + 0.12 * index), label=observation)
        axis.axhline(0, color="#AAAAAA", linewidth=0.7)
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel("Temperature anomaly (degC)" if variable == "tas" else "Precipitation anomaly (mm/day)")
        axis.set_title("HMA temperature" if variable == "tas" else "HMA precipitation")
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    axes[1].set_xlabel("Year")
    fig.savefig(output / "figure_hma_legacy_six_members_1940_2000.png", dpi=220)
    plt.close(fig)

    after = {str(path): state(path) for path in sources}
    if before != after:
        raise RuntimeError("source-changed-during-round01")
    config = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_period": [1940, 2000],
        "temporal_blocks": BLOCKS,
        "locked_period": list(LOCKED_PERIOD),
        "calibration_status": "legacy members were calibrated in this period; scores are baseline controls, not out-of-fold acceptance evidence",
        "promotion_thresholds_for_round02": {
            "primary": "at least 5 percent reduction in median region-reference RMSE against equal_six_mean under nested blocked CV",
            "no_regression": "no region-reference RMSE may worsen by more than 5 percent",
            "locked_period_use": "forbidden until Gate 5B",
        },
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "sha256": hashes, "driver_sha256": digest(driver)}, indent=2), encoding="utf-8")
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "legacy_temporal_block_baseline_complete",
        "locked_1921_1939_accessed": False,
        "hma_gridcell_count": int(hma_mask.sum()),
        "round00_invalid_cells_inside_hma": {variable: int(np.sum(hma_mask & ~source_masks[variable])) for variable in source_masks},
        "equal_six_member_baseline": equal_baseline,
        "rankings": ranking_by_variable,
        "interpretation_boundary": "This is a common-mask temporal-block scorecard of members calibrated over 1940-2000. It freezes the legacy control but is not OOF evidence for candidate promotion.",
        "promotion_thresholds_frozen": config["promotion_thresholds_for_round02"],
        "all_declared_sources_unchanged": before == after,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# Round 01 Contract\n\nDecision: freeze a common-mask legacy control and identify which variable/member conflict merits the first nested-CV repair.\n\nOnly 1940-2000 is read. These legacy reconstructions were calibrated in this interval, so the block scores are controls rather than OOF promotion evidence.\n",
        encoding="utf-8",
    )
    best_tas = ranking_by_variable["tas"][0]
    best_pr = ranking_by_variable["pr"][0]
    (output / "FINAL_RESULT.md").write_text(
        "# Round 01 Legacy Baseline\n\n"
        f"- Temperature equal-mean median RMSE: `{equal_baseline['tas']['median_rmse_across_region_reference_pairs']:.4f}`; best branch `{best_tas['branch']}` at `{best_tas['median_rmse_across_region_reference_pairs']:.4f}`.\n"
        f"- Precipitation equal-mean median RMSE: `{equal_baseline['pr']['median_rmse_across_region_reference_pairs']:.4f}`; best branch `{best_pr['branch']}` at `{best_pr['median_rmse_across_region_reference_pairs']:.4f}`.\n"
        "- The 5% improvement and 5% regional no-regression thresholds are frozen for nested blocked CV.\n"
        "- Locked 1921-1939 values accessed: `false`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
