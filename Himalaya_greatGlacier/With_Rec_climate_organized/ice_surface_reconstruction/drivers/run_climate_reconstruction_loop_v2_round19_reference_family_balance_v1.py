"""Compare member-balanced and evidence-family-balanced reference ensembles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1940, 2001)
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
MEMBERS = {
    "tas": {
        "HGAO": "tas_rec_ref_HGAO_c05d",
        "LMR": "tas_rec_ref_LMR_c05d",
        "Nk_AM": "tas_rec_ref_Nk_AM_c05d",
        "Nk_CCA": "tas_rec_ref_Nk_CCA_c05d",
        "Nk_CPS": "tas_rec_ref_Nk_CPS_c05d",
        "Nk_DA": "tas_rec_ref_Nk_DA_c05d",
        "Nk_GraphEM": "tas_rec_ref_Nk_GraphEM_c05d",
        "Nk_PCR": "tas_rec_ref_Nk_PCR_c05d",
        "O": "tas_rec_ref_O_c05d",
        "PHYDA": "tas_rec_ref_PHYDA_c05d",
    },
    "pr": {
        "HGAO": "pr_rec_ref_HGAO_c05d",
        "LMR": "pr_rec_ref_LMR_c05d",
        "O": "pr_rec_ref_O_c05d",
    },
}
STORED_ENSEMBLE = {
    "tas": "tas_ensemble_mean_rec_ref",
    "pr": "pr_ensemble_mean_rec_ref",
}
OBSERVATIONS = {
    "tas": {
        "Berkeley": "tas_obs_bea_05deg",
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


def clean(dataset: h5py.Dataset, selection: slice) -> np.ndarray:
    values = np.asarray(dataset[selection, :, :], dtype=float)
    fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).squeeze())
    values[np.isclose(values, fill) | (values < -9000)] = np.nan
    return values


def recenter(values: np.ndarray) -> np.ndarray:
    baseline = (YEARS >= 1961) & (YEARS <= 1990)
    return values - np.nanmean(values[baseline], axis=0)[None, :, :]


def finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(arrays)
    count = np.isfinite(stacked).sum(axis=0)
    result = np.nansum(stacked, axis=0)
    result = result / np.where(count == 0, np.nan, count)
    return result


def family_balance(variable: str, members: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if variable == "tas":
        families = {
            "local_enkf": finite_mean([members["HGAO"], members["O"]]),
            "lmr": members["LMR"],
            "neukom": finite_mean([members[name] for name in members if name.startswith("Nk_")]),
            "phyda": members["PHYDA"],
        }
    else:
        families = {
            "local_enkf": finite_mean([members["HGAO"], members["O"]]),
            "lmr": members["LMR"],
        }
    return finite_mean(list(families.values())), families


def weighted_series(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(values)
    numerator = np.nansum(values * weights[None, :, :], axis=(1, 2))
    denominator = np.sum(valid * weights[None, :, :], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3:
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def centered_smooth(values: np.ndarray, width: int = 15) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    half = width // 2
    for index in range(half, len(values) - half):
        window = values[index - half:index + half + 1]
        if np.all(np.isfinite(window)):
            result[index] = float(np.mean(window))
    return result


def trend(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if int(valid.sum()) < 10:
        return np.nan
    return float(np.polyfit(YEARS[valid], values[valid], 1)[0] * 100)


def pair_metrics(candidate: np.ndarray, observation: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> dict[str, float]:
    common = mask & np.all(np.isfinite(candidate), axis=0) & np.all(np.isfinite(observation), axis=0)
    if int(common.sum()) == 0:
        return {key: np.nan for key in (
            "common_cell_count", "regional_rmse", "regional_correlation", "regional_lowpass_correlation_15y",
            "regional_trend_difference_per_century", "median_pixel_rmse",
        )}
    candidate_series = weighted_series(candidate, common, latitudes)
    observation_series = weighted_series(observation, common, latitudes)
    regional_rmse = float(np.sqrt(np.mean((candidate_series - observation_series) ** 2)))
    pixel_rmse = np.sqrt(np.mean((candidate[:, common] - observation[:, common]) ** 2, axis=0))
    return {
        "common_cell_count": int(common.sum()),
        "regional_rmse": regional_rmse,
        "regional_correlation": correlation(candidate_series, observation_series),
        "regional_lowpass_correlation_15y": correlation(centered_smooth(candidate_series), centered_smooth(observation_series)),
        "regional_trend_difference_per_century": trend(candidate_series) - trend(observation_series),
        "median_pixel_rmse": float(np.median(pixel_rmse)),
    }


def aggregate(variable: str, comparison: pd.DataFrame) -> dict[str, object]:
    values = comparison[comparison.variable == variable].copy()
    hma = values[values.region == "hma"]
    median_regional = float(values.regional_rmse_improvement.median())
    maximum_regression = float(np.max(-values.regional_rmse_improvement.to_numpy(float)))
    positive_fraction = float(np.mean(values.regional_rmse_improvement > 0))
    median_pixel = float(hma.pixel_rmse_improvement.median())
    median_lowpass_change = float(values.lowpass_correlation_change.median())
    gates = {
        "median_regional_rmse_improvement_at_least_2pct": median_regional >= 0.02,
        "maximum_pair_regression_at_most_5pct": maximum_regression <= 0.05,
        "more_than_half_pairs_improve": positive_fraction > 0.5,
        "median_hma_pixel_rmse_does_not_worsen": median_pixel >= 0,
        "median_regional_lowpass_correlation_does_not_decrease": median_lowpass_change >= 0,
    }
    return {
        "median_regional_rmse_improvement": median_regional,
        "maximum_pair_regression": maximum_regression,
        "positive_pair_fraction": positive_fraction,
        "median_hma_pixel_rmse_improvement": median_pixel,
        "median_regional_lowpass_correlation_change": median_lowpass_change,
        "gates": gates,
        "passes": bool(all(gates.values())),
    }


def main() -> int:
    project = project_root()
    output = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_19_reference_family_balance_v1"
    netcdf = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
    mask_path = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
    paths = {"netcdf": netcdf, "hma_mask": mask_path, "contract": output / "ROUND_CONTRACT.md", "driver": Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    hma_mask = np.load(mask_path).astype(bool)
    score_rows: list[dict[str, object]] = []
    curve_rows = {"year": YEARS}
    closure: dict[str, float] = {}
    family_counts: dict[str, int] = {}
    with h5py.File(netcdf, "r") as handle:
        all_years = np.asarray(handle["time"], dtype=int).ravel()
        indices = np.flatnonzero((all_years >= 1940) & (all_years <= 2000))
        if not np.array_equal(all_years[indices], YEARS) or np.any((all_years[indices] >= 1921) & (all_years[indices] <= 1939)):
            raise RuntimeError("year-selection-contract-failed")
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise RuntimeError("year-selection-not-contiguous")
        selection = slice(int(indices[0]), int(indices[-1]) + 1)
        latitudes = np.asarray(handle["lat"], dtype=float).ravel()
        longitudes = np.asarray(handle["lon"], dtype=float).ravel()
        if hma_mask.shape != (latitudes.size, longitudes.size):
            raise RuntimeError("hma-mask-shape-failed")
        longitude_grid = np.broadcast_to(longitudes[None, :], hma_mask.shape)

        for variable in ("tas", "pr"):
            members = {name: recenter(clean(handle[dataset], selection)) for name, dataset in MEMBERS[variable].items()}
            observations = {name: recenter(clean(handle[dataset], selection)) for name, dataset in OBSERVATIONS[variable].items()}
            stored = recenter(clean(handle[STORED_ENSEMBLE[variable]], selection))
            equal_member = finite_mean(list(members.values()))
            family_candidate, families = family_balance(variable, members)
            comparable = np.isfinite(stored) & np.isfinite(equal_member)
            closure[variable] = float(np.max(np.abs(stored[comparable] - equal_member[comparable])))
            family_counts[variable] = len(families)
            if closure[variable] > 1e-5:
                raise RuntimeError(f"stored-ensemble-closure-failed:{variable}:{closure[variable]}")

            for label, values in (("member_equal", equal_member), ("family_balanced", family_candidate)):
                curve_rows[f"{variable}_{label}"] = weighted_series(values, hma_mask, latitudes)
            for observation_name, observation in observations.items():
                curve_rows[f"{variable}_obs_{observation_name}"] = weighted_series(observation, hma_mask, latitudes)
                for region, (lower, upper) in REGIONS.items():
                    region_mask = hma_mask & (longitude_grid >= lower) & (longitude_grid < upper)
                    for candidate_name, candidate in (("member_equal", equal_member), ("family_balanced", family_candidate)):
                        metrics = pair_metrics(candidate, observation, region_mask, latitudes)
                        score_rows.append({
                            "variable": variable,
                            "observation": observation_name,
                            "region": region,
                            "candidate": candidate_name,
                            **metrics,
                        })

    scores = pd.DataFrame(score_rows)
    scores.to_csv(output / "candidate_scores.csv", index=False)
    comparisons: list[dict[str, object]] = []
    for keys, group in scores.groupby(["variable", "observation", "region"]):
        variable, observation, region = keys
        indexed = group.set_index("candidate")
        control = indexed.loc["member_equal"]
        candidate = indexed.loc["family_balanced"]
        comparisons.append({
            "variable": variable,
            "observation": observation,
            "region": region,
            "regional_rmse_improvement": float(1 - candidate.regional_rmse / control.regional_rmse),
            "pixel_rmse_improvement": float(1 - candidate.median_pixel_rmse / control.median_pixel_rmse),
            "annual_correlation_change": float(candidate.regional_correlation - control.regional_correlation),
            "lowpass_correlation_change": float(candidate.regional_lowpass_correlation_15y - control.regional_lowpass_correlation_15y),
            "absolute_trend_error_change": float(abs(candidate.regional_trend_difference_per_century) - abs(control.regional_trend_difference_per_century)),
            "common_cell_count": int(candidate.common_cell_count),
        })
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "family_balance_comparison.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "hma_regional_curves.csv", index=False)

    variable_summary = {variable: aggregate(variable, comparison) for variable in ("tas", "pr")}
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "stored_member_mean_closure_max_abs": closure,
        "evidence_family_counts": family_counts,
        "variables": variable_summary,
        "any_family_balanced_candidate_passes": bool(any(item["passes"] for item in variable_summary.values())),
        "decision": {
            variable: "permit_versioned_family_balanced_reference_candidate" if item["passes"] else "retain_member_equal_reference_as_comparison_only"
            for variable, item in variable_summary.items()
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    curves = pd.DataFrame(curve_rows)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for axis, variable, ylabel in ((axes[0], "tas", "Temperature anomaly"), (axes[1], "pr", "Precipitation anomaly")):
        axis.plot(YEARS, curves[f"{variable}_member_equal"], color="#333333", linewidth=1.8, label="Member equal")
        axis.plot(YEARS, curves[f"{variable}_family_balanced"], color="#0072B2", linewidth=2.0, label="Family balanced")
        for column in [name for name in curves if name.startswith(f"{variable}_obs_")]:
            axis.plot(YEARS, curves[column], linewidth=0.8, alpha=0.45, label=column.replace(f"{variable}_obs_", ""))
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.set_ylabel(ylabel)
        axis.legend(ncol=4, fontsize=8, frameon=False)
    axes[1].set_xlabel("Year")
    axes[0].set_title("HMA reference ensemble family-balance validation")
    fig.savefig(output / "reference_family_balance.png", dpi=180)
    plt.close(fig)

    lines = ["# Reference Evidence-Family Balance", ""]
    for variable, item in variable_summary.items():
        lines.extend([
            f"## {variable}",
            f"- Median regional RMSE improvement: `{item['median_regional_rmse_improvement']:.3%}`.",
            f"- Maximum pair regression: `{item['maximum_pair_regression']:.3%}`.",
            f"- Positive pair fraction: `{item['positive_pair_fraction']:.3%}`.",
            f"- Median HMA pixel RMSE improvement: `{item['median_hma_pixel_rmse_improvement']:.3%}`.",
            f"- Median 15-year correlation change: `{item['median_regional_lowpass_correlation_change']:+.3f}`.",
            f"- Passes: `{str(item['passes']).lower()}`.",
            "",
        ])
    lines.append(f"Decision: `{summary['decision']}`.")
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": before == after}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
