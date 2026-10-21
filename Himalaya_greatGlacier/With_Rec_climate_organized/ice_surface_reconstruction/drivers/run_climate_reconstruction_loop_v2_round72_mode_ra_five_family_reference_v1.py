"""Construct and modern-validate a coarse five-family temperature reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

import run_climate_reconstruction_loop_v2_round68_mode_ra_modern_validation_v1 as modern
import run_climate_reconstruction_loop_v2_round71_mode_ra_lia_spatial_consensus_v1 as spatial

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALLOWED_YEARS = np.concatenate((np.arange(1421, 1921), np.arange(1940, 2001)))


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


def load_reference_allowed(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with netcdf_file(path, "r", mmap=False) as handle:
        years = np.asarray(handle.variables["year"][:], dtype=int)
        lat = np.asarray(handle.variables["lat"][:], dtype=float)
        lon = np.asarray(handle.variables["lon"][:], dtype=float)
        mask = np.asarray(handle.variables["hma_mask"][:], dtype=int).astype(bool)
        variable = handle.variables["tas_reference_family_balanced"]
        fill = float(variable._FillValue)
        parts = []
        for start, end in ((1421, 1920), (1940, 2000)):
            indices = np.flatnonzero((years >= start) & (years <= end))
            if not np.array_equal(years[indices], np.arange(start, end + 1)):
                raise RuntimeError(f"reference-allowed-axis-failed:{start}:{end}")
            if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
                raise RuntimeError(f"reference-allowed-selection-not-contiguous:{start}:{end}")
            parts.append(np.asarray(variable[slice(int(indices[0]), int(indices[-1]) + 1)], dtype=float))
    values = np.concatenate(parts)
    values[np.isclose(values, fill) | (values < -1e20)] = np.nan
    return ALLOWED_YEARS.copy(), lat, lon, mask, values


def load_mode_allowed(paleo_path: Path, modern_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(paleo_path, "r") as handle:
        paleo_years = np.asarray(handle["time"], dtype=int)
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        paleo = np.asarray(handle["temp2_anomaly_1961_1990"], dtype=float)
    with h5py.File(modern_path, "r") as handle:
        modern_years = np.asarray(handle["time"], dtype=int)
        modern_lat = np.asarray(handle["lat"], dtype=float)
        modern_lon = np.asarray(handle["lon"], dtype=float)
        source = np.asarray(handle["temp2_anomaly"], dtype=float)
    if not np.array_equal(paleo_years, np.arange(1421, 1921)):
        raise RuntimeError("mode-paleo-year-axis-failed")
    if not np.array_equal(modern_years, np.arange(1940, 2001)):
        raise RuntimeError("mode-modern-year-axis-failed")
    if not np.array_equal(lat, modern_lat) or not np.array_equal(lon, modern_lon):
        raise RuntimeError("mode-paleo-modern-grid-conflict")
    baseline = (modern_years >= 1961) & (modern_years <= 1990)
    modern = source - np.mean(source[baseline], axis=0)[None]
    return ALLOWED_YEARS.copy(), lat, lon, np.concatenate((paleo, modern))


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_72_mode_ra_five_family_temperature_reference_v1"
    round21 = loop / "round_21_temperature_reference_field_v1"
    round67 = loop / "round_67_mode_ra_allowed_modern_subset_acquisition_v2"
    round68 = loop / "round_68_mode_ra_modern_independent_validation_v1"
    round69 = loop / "round_69_mode_ra_allowed_paleo_temperature_acquisition_v3"
    round71 = loop / "round_71_mode_ra_lia_spatial_consensus_v1"
    public = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
    paths = {
        "reference_field": round21 / "temperature_reference_family_balanced_allowed_v1.nc",
        "mode_ra_paleo": round69 / "mode_ra_temp2_1421_1920_hma_bbox.nc",
        "mode_ra_modern": round67 / "mode_ra_ensmean_1940_2000_hma_bbox.nc",
        "round71_summary": round71 / "summary.json",
        "round68_summary": round68 / "summary.json",
        "coarse_support": round68 / "mode_ra_hma_coarse_support.csv",
        "public_observations": public,
        "round68_driver": Path(modern.__file__).resolve(),
        "round71_driver": Path(spatial.__file__).resolve(),
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    target = output / "temperature_reference_five_family_coarse_allowed_v1.nc"
    if target.exists() or (output / "summary.json").exists():
        raise FileExistsError(target)
    before = {name: state(path) for name, path in paths.items()}
    if not json.loads(paths["round71_summary"].read_text(encoding="utf-8"))[
        "passes_lia_spatial_consensus"
    ]:
        raise RuntimeError("round71-spatial-consensus-did-not-pass")

    reference_years, fine_lat, fine_lon, hma_mask, reference_fine = load_reference_allowed(
        paths["reference_field"]
    )
    mode_years, mode_lat, mode_lon, mode_field = load_mode_allowed(
        paths["mode_ra_paleo"], paths["mode_ra_modern"]
    )
    if not np.array_equal(reference_years, ALLOWED_YEARS) or not np.array_equal(mode_years, ALLOWED_YEARS):
        raise RuntimeError("allowed-year-axis-conflict")
    if np.any((ALLOWED_YEARS >= 1921) & (ALLOWED_YEARS <= 1939)):
        raise RuntimeError("locked-years-in-candidate-axis")
    support_table = pd.read_csv(paths["coarse_support"])
    cells = modern.build_coarse_support(fine_lat, fine_lon, hma_mask, mode_lat, mode_lon)
    if len(cells) != len(support_table):
        raise RuntimeError("coarse-support-count-conflict")
    for cell, row in zip(cells, support_table.itertuples(index=False)):
        if (
            not np.isclose(cell["latitude"], row.latitude)
            or not np.isclose(cell["longitude"], row.longitude)
            or int(cell["fine_cell_count"]) != int(row.fine_hma_cell_count)
        ):
            raise RuntimeError("coarse-support-identity-conflict")
    four_family = spatial.aggregate_reference(
        reference_fine, fine_lat, fine_lon, hma_mask, support_table, mode_lat, mode_lon
    )
    mode_coarse = modern.extract_coarse(mode_field, cells)
    five_family = (4.0 * four_family + mode_coarse) / 5.0
    baseline = (ALLOWED_YEARS >= 1961) & (ALLOWED_YEARS <= 1990)
    closure = {
        "four_family": float(np.max(np.abs(np.mean(four_family[baseline], axis=0)))),
        "mode_ra": float(np.max(np.abs(np.mean(mode_coarse[baseline], axis=0)))),
        "five_family": float(np.max(np.abs(np.mean(five_family[baseline], axis=0)))),
    }

    with h5py.File(target, "w") as handle:
        handle.attrs["title"] = "Allowed-period HMA coarse five-family temperature reference candidate"
        handle.attrs["family_definition"] = "equal mean of local EnKF, LMR, Neukom, PHYDA, and ModE-RA"
        handle.attrs["locked_1921_1939_accessed"] = False
        handle.attrs["production_field"] = False
        handle.create_dataset("time", data=ALLOWED_YEARS.astype(np.int16))
        handle["time"].attrs["units"] = "year CE"
        handle.create_dataset("cell", data=np.arange(len(cells), dtype=np.int16))
        handle.create_dataset("lat", data=support_table.latitude.to_numpy(float))
        handle.create_dataset("lon", data=support_table.longitude.to_numpy(float))
        handle.create_dataset(
            "fine_hma_cell_count", data=support_table.fine_hma_cell_count.to_numpy(np.int16)
        )
        for name, values in (
            ("temperature_reference_four_family", four_family),
            ("temperature_mode_ra", mode_coarse),
            ("temperature_reference_five_family", five_family),
        ):
            handle.create_dataset(
                name,
                data=values.astype(np.float32),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fillvalue=np.float32(np.nan),
            )
            handle[name].attrs["units"] = "degC"
            handle[name].attrs["reference_period"] = "1961-1990"

    with h5py.File(public, "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).ravel()
        indices = np.flatnonzero((source_years >= 1940) & (source_years <= 2000))
        if not np.array_equal(source_years[indices], np.arange(1940, 2001)):
            raise RuntimeError("observation-modern-year-axis-failed")
        selection = slice(int(indices[0]), int(indices[-1]) + 1)
        observations = {
            name: modern.recenter(modern.clean(handle[variable], selection), np.arange(1940, 2001))
            for name, variable in modern.TEMPERATURE_OBSERVATIONS.items()
        }
    modern_selected = ALLOWED_YEARS >= 1940
    four_modern = four_family[modern_selected]
    five_modern = five_family[modern_selected]
    cell_lat = support_table.latitude.to_numpy(float)
    cell_lon = support_table.longitude.to_numpy(float)
    weights = (
        np.cos(np.deg2rad(cell_lat)) * support_table.fine_hma_cell_count.to_numpy(float)
    )
    score_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for product, fine_values in observations.items():
        observation = modern.recenter(
            modern.aggregate_fine(fine_values, fine_lat, cells), np.arange(1940, 2001)
        )
        for region, (lower, upper) in modern.REGIONS.items():
            selected_cells = (cell_lon >= lower) & (cell_lon < upper)
            for candidate_name, candidate in (
                ("four_family", four_modern),
                ("five_family", five_modern),
            ):
                metrics = modern.pair_metrics(
                    candidate[:, selected_cells],
                    observation[:, selected_cells],
                    np.arange(1940, 2001),
                    weights[selected_cells],
                )
                score_rows.append(
                    {
                        "observation": product,
                        "region": region,
                        "candidate": candidate_name,
                        **{key: value for key, value in metrics.items() if not key.endswith("_series")},
                    }
                )
                if candidate_name == "five_family":
                    for block_index, (start, end) in enumerate(modern.BLOCKS, start=1):
                        selected_years = (np.arange(1940, 2001) >= start) & (
                            np.arange(1940, 2001) <= end
                        )
                        cand_series = metrics["candidate_series"][selected_years]
                        obs_series = metrics["observation_series"][selected_years]
                        rmse = float(np.sqrt(np.mean((cand_series - obs_series) ** 2)))
                        zero_rmse = float(np.sqrt(np.mean(obs_series**2)))
                        block_rows.append(
                            {
                                "variable": "temperature",
                                "observation": product,
                                "region": region,
                                "block": block_index,
                                "regional_skill": float(1.0 - rmse / zero_rmse),
                            }
                        )
                if region == "hma" and candidate_name == "five_family":
                    for index, year in enumerate(range(1940, 2001)):
                        curve_rows.append(
                            {
                                "observation": product,
                                "year": year,
                                "five_family": float(metrics["candidate_series"][index]),
                                "observed": float(metrics["observation_series"][index]),
                            }
                        )
    scores = pd.DataFrame(score_rows)
    blocks = pd.DataFrame(block_rows)
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
                "correlation_15year_change": float(
                    candidate.correlation_15year - control.correlation_15year
                ),
            }
        )
    comparison = pd.DataFrame(comparisons)
    scores.to_csv(output / "four_vs_five_family_absolute_scores.csv", index=False)
    blocks.to_csv(output / "five_family_block_scores.csv", index=False)
    comparison.to_csv(output / "five_family_improvement_over_four_family.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "five_family_modern_hma_curves.csv", index=False)

    five_scores = scores[scores.candidate == "five_family"].copy()
    five_scores["variable"] = "temperature"
    absolute = modern.summarize_variable(five_scores, blocks)
    hma_comparison = comparison[comparison.region == "hma"]
    promotion_gates = {
        "five_family_passes_round68_absolute_regional_and_spatial_gates": absolute[
            "passes_spatial_field"
        ],
        "median_regional_rmse_change_not_negative": float(
            comparison.regional_rmse_improvement_over_four_family.median()
        ) >= 0,
        "worst_regional_rmse_regression_at_most_5pct": float(
            comparison.regional_rmse_improvement_over_four_family.min()
        ) >= -0.05,
        "more_than_half_regional_pairs_improve": float(
            np.mean(comparison.regional_rmse_improvement_over_four_family > 0)
        ) > 0.5,
        "median_hma_spatial_residual_rmse_change_not_negative": float(
            hma_comparison.spatial_residual_rmse_improvement_over_four_family.median()
        ) >= 0,
        "worst_hma_spatial_residual_regression_at_most_5pct": float(
            hma_comparison.spatial_residual_rmse_improvement_over_four_family.min()
        ) >= -0.05,
        "more_than_half_hma_products_improve_spatially": float(
            np.mean(hma_comparison.spatial_residual_rmse_improvement_over_four_family > 0)
        ) > 0.5,
        "median_15year_correlation_does_not_decrease": float(
            comparison.correlation_15year_change.median()
        ) >= 0,
    }
    accepted = bool(all(promotion_gates.values()))
    summary = {
        "status": "complete",
        "allowed_years": [[1421, 1920], [1940, 2000]],
        "locked_1921_1939_accessed": False,
        "coarse_hma_cell_count": len(cells),
        "family_definition": ["local_enkf", "lmr", "neukom", "phyda", "mode_ra"],
        "recenter_closure_max_abs": closure,
        "five_family_absolute_validation": absolute,
        "median_regional_rmse_improvement_over_four_family": float(
            comparison.regional_rmse_improvement_over_four_family.median()
        ),
        "worst_regional_rmse_improvement_over_four_family": float(
            comparison.regional_rmse_improvement_over_four_family.min()
        ),
        "positive_regional_pair_fraction": float(
            np.mean(comparison.regional_rmse_improvement_over_four_family > 0)
        ),
        "median_hma_spatial_residual_improvement_over_four_family": float(
            hma_comparison.spatial_residual_rmse_improvement_over_four_family.median()
        ),
        "worst_hma_spatial_residual_improvement_over_four_family": float(
            hma_comparison.spatial_residual_rmse_improvement_over_four_family.min()
        ),
        "positive_hma_spatial_product_fraction": float(
            np.mean(hma_comparison.spatial_residual_rmse_improvement_over_four_family > 0)
        ),
        "median_15year_correlation_change": float(
            comparison.correlation_15year_change.median()
        ),
        "promotion_gates": promotion_gates,
        "passes_five_family_promotion": accepted,
        "decision": (
            "accept_versioned_coarse_five_family_temperature_reference_candidate"
            if accepted
            else "retain_four_family_reference_and_mode_ra_as_separate_evidence"
        ),
        "production_release_authorized": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    curves = pd.DataFrame(curve_rows)
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 8), constrained_layout=True)
    for product in curves.observation.unique():
        values = curves[curves.observation == product]
        axes[0].plot(values.year, values.observed, linewidth=0.9, alpha=0.65, label=product)
    five_series = curves.groupby("year").five_family.mean()
    support_weights = weights
    four_series = modern.weighted_series(four_modern, support_weights)
    axes[0].plot(five_series.index, five_series, color="#111111", linewidth=2.2, label="Five-family")
    axes[0].plot(np.arange(1940, 2001), four_series, color="#0072B2", linewidth=1.8, label="Four-family")
    axes[0].axhline(0, color="#777777", linewidth=0.7)
    axes[0].set(title="Modern HMA temperature reference comparison", ylabel="Anomaly (degC)")
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    x = np.arange(len(hma_comparison))
    axes[1].bar(
        x - 0.18,
        hma_comparison.regional_rmse_improvement_over_four_family * 100,
        width=0.36,
        color="#0072B2",
        label="Regional",
    )
    axes[1].bar(
        x + 0.18,
        hma_comparison.spatial_residual_rmse_improvement_over_four_family * 100,
        width=0.36,
        color="#D55E00",
        label="Spatial residual",
    )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(x, hma_comparison.observation, rotation=25, ha="right")
    axes[1].set(xlabel="Observation product", ylabel="Improvement over four-family (%)")
    axes[1].legend(frameon=False)
    figure.savefig(output / "five_family_temperature_reference_validation.png", dpi=180)
    plt.close(figure)

    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Five-Family Coarse Temperature Reference",
                "",
                f"- Absolute regional/spatial gates pass: `{str(absolute['passes_spatial_field']).lower()}`.",
                f"- Median/worst regional improvement over four-family: `{summary['median_regional_rmse_improvement_over_four_family']:.2%}` / `{summary['worst_regional_rmse_improvement_over_four_family']:.2%}`.",
                f"- Positive regional pairs: `{summary['positive_regional_pair_fraction']:.1%}`.",
                f"- Median/worst HMA spatial-residual improvement: `{summary['median_hma_spatial_residual_improvement_over_four_family']:.2%}` / `{summary['worst_hma_spatial_residual_improvement_over_four_family']:.2%}`.",
                f"- Promotion passes: `{str(accepted).lower()}`.",
                "",
                f"Decision: `{summary['decision']}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged, "output": state(target)},
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
