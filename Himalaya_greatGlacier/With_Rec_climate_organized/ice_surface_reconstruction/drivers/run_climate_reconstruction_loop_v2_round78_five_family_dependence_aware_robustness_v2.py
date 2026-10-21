"""Dependence-aware leave-one-family-out validation of the five-family reference."""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DRIVER_DIR = Path(__file__).resolve().parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))
import run_climate_reconstruction_loop_v2_round30_temperature_reference_spatial_consensus_v1 as old
import run_climate_reconstruction_loop_v2_round68_mode_ra_modern_validation_v1 as modern
import run_climate_reconstruction_loop_v2_round71_mode_ra_lia_spatial_consensus_v1 as spatial


ALLOWED_YEARS = np.concatenate((np.arange(1421, 1921), np.arange(1940, 2001)))
WINDOWS = tuple((start, start + 49) for start in range(1450, 1900, 50))
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


def clean_bbox(dataset: h5py.Dataset, selection: slice, lat_slice: slice, lon_slice: slice) -> np.ndarray:
    values = np.asarray(dataset[selection, lat_slice, lon_slice], dtype=float)
    fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).reshape(-1)[0])
    values[np.isclose(values, fill) | (values < -9000)] = np.nan
    return values


def load_member_coarse(
    dataset: h5py.Dataset,
    lat_slice: slice,
    lon_slice: slice,
    fine_lat: np.ndarray,
    fine_lon: np.ndarray,
    mask: np.ndarray,
    support: pd.DataFrame,
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
) -> np.ndarray:
    paleo = clean_bbox(dataset, slice(421, 921), lat_slice, lon_slice)
    recent = clean_bbox(dataset, slice(940, 1001), lat_slice, lon_slice)
    values = np.concatenate((paleo, recent), axis=0)
    baseline = (ALLOWED_YEARS >= 1961) & (ALLOWED_YEARS <= 1990)
    values = values - np.nanmean(values[baseline], axis=0)[None]
    return spatial.aggregate_reference(
        values, fine_lat, fine_lon, mask, support, coarse_lat, coarse_lon
    )


def weighted_fraction(condition: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights[condition]) / np.sum(weights))


def pairwise_correlations(fields: np.ndarray) -> np.ndarray:
    return np.asarray(
        [float(np.corrcoef(fields[left], fields[right])[0, 1]) for left, right in combinations(range(len(fields)), 2)]
    )


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_78_five_family_dependence_aware_robustness_v2"
    round72 = loop / "round_72_mode_ra_five_family_temperature_reference_v1"
    round73 = loop / "round_73_mode_ra_hma_source_overlap_v1"
    round70 = loop / "round_70_mode_ra_lia_regional_adjudication_v2"
    public = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
    mask_path = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
    paths = {
        "five_family_candidate": round72 / "temperature_reference_five_family_coarse_allowed_v1.nc",
        "round72_summary": round72 / "summary.json",
        "round73_summary": round73 / "summary.json",
        "round70_windows": round70 / "mode_ra_lia_50year_adjudication.csv",
        "public_reference_source": public,
        "hma_mask": mask_path,
        "round30_driver": Path(old.__file__).resolve(),
        "round68_driver": Path(modern.__file__).resolve(),
        "round71_driver": Path(spatial.__file__).resolve(),
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    if not json.loads(paths["round72_summary"].read_text(encoding="utf-8"))[
        "passes_five_family_promotion"
    ]:
        raise RuntimeError("round72-candidate-did-not-pass")
    overlap = json.loads(paths["round73_summary"].read_text(encoding="utf-8"))
    if overlap["eligible_as_proxy_evidence_independent_family"]:
        raise RuntimeError("round73-overlap-caveat-not-active")

    with h5py.File(paths["five_family_candidate"], "r") as handle:
        candidate_years = np.asarray(handle["time"], dtype=int)
        coarse_lat = np.asarray(handle["lat"], dtype=float)
        coarse_lon = np.asarray(handle["lon"], dtype=float)
        four_control = np.asarray(handle["temperature_reference_four_family"], dtype=float)
        five_candidate = np.asarray(handle["temperature_reference_five_family"], dtype=float)
    if not np.array_equal(candidate_years, ALLOWED_YEARS):
        raise RuntimeError("candidate-year-axis-failed")
    if np.any((candidate_years >= 1921) & (candidate_years <= 1939)):
        raise RuntimeError("locked-period-in-candidate")
    grid_lat = np.unique(coarse_lat)
    grid_lon = np.unique(coarse_lon)
    support = pd.DataFrame(
        {"latitude": coarse_lat, "longitude": coarse_lon}
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
        ii, jj = np.where(full_mask)
        lat_slice = slice(int(ii.min()), int(ii.max()) + 1)
        lon_slice = slice(int(jj.min()), int(jj.max()) + 1)
        fine_lat = full_lat[lat_slice]
        fine_lon = full_lon[lon_slice]
        mask = full_mask[lat_slice, lon_slice]
        temp_cells = modern.build_coarse_support(fine_lat, fine_lon, mask, grid_lat, grid_lon)
        support = pd.DataFrame(
            [
                {
                    "latitude": cell["latitude"],
                    "longitude": cell["longitude"],
                    "fine_hma_cell_count": cell["fine_cell_count"],
                }
                for cell in temp_cells
            ]
        )
        if len(support) != len(coarse_lat) or not np.allclose(support.latitude, coarse_lat) or not np.allclose(support.longitude, coarse_lon):
            raise RuntimeError("reconstructed-coarse-support-conflict")
        member_fields = {
            name: load_member_coarse(
                handle[variable], lat_slice, lon_slice, fine_lat, fine_lon, mask,
                support, grid_lat, grid_lon
            )
            for name, variable in old.MEMBERS.items()
        }
        observations = {}
        for product, variable in modern.TEMPERATURE_OBSERVATIONS.items():
            recent = clean_bbox(handle[variable], slice(940, 1001), lat_slice, lon_slice)
            recent = modern.recenter(recent, np.arange(1940, 2001))
            observations[product] = spatial.aggregate_reference(
                recent, fine_lat, fine_lon, mask, support, grid_lat, grid_lon
            )

    families = {
        "local_enkf": np.mean(np.stack([member_fields["HGAO"], member_fields["O"]]), axis=0),
        "lmr": member_fields["LMR"],
        "neukom": np.mean(
            np.stack([value for name, value in member_fields.items() if name.startswith("Nk_")]), axis=0
        ),
        "phyda": member_fields["PHYDA"],
    }
    reconstructed_four = np.mean(np.stack(list(families.values())), axis=0)
    four_closure = float(np.max(np.abs(reconstructed_four - four_control)))
    mode_ra = 5.0 * five_candidate - 4.0 * four_control
    five_identity_closure = float(
        np.max(np.abs(five_candidate - (4.0 * reconstructed_four + mode_ra) / 5.0))
    )
    families["mode_ra"] = mode_ra
    family_stack = np.stack([families[name] for name in FAMILY_NAMES])
    candidates = {"full_five_family": np.mean(family_stack, axis=0)}
    for omitted, name in enumerate(FAMILY_NAMES):
        candidates[f"leave_out_{name}"] = np.mean(np.delete(family_stack, omitted, axis=0), axis=0)
    full_candidate_closure = float(
        np.max(np.abs(candidates["full_five_family"] - five_candidate))
    )

    weights = np.cos(np.deg2rad(coarse_lat)) * support.fine_hma_cell_count.to_numpy(float)
    cell_lon = coarse_lon
    modern_selection = ALLOWED_YEARS >= 1940
    region_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for candidate_name, candidate_all in candidates.items():
        candidate = candidate_all[modern_selection]
        for product, observation in observations.items():
            for region, (lower, upper) in modern.REGIONS.items():
                selected_cells = (cell_lon >= lower) & (cell_lon < upper)
                metrics = modern.pair_metrics(
                    candidate[:, selected_cells],
                    observation[:, selected_cells],
                    np.arange(1940, 2001),
                    weights[selected_cells],
                )
                region_rows.append(
                    {
                        "candidate": candidate_name,
                        "observation": product,
                        "region": region,
                        **{key: value for key, value in metrics.items() if not key.endswith("_series")},
                    }
                )
                for block_index, (start, end) in enumerate(modern.BLOCKS, start=1):
                    selected_years = (np.arange(1940, 2001) >= start) & (np.arange(1940, 2001) <= end)
                    candidate_series = metrics["candidate_series"][selected_years]
                    observation_series = metrics["observation_series"][selected_years]
                    rmse = float(np.sqrt(np.mean((candidate_series - observation_series) ** 2)))
                    zero_rmse = float(np.sqrt(np.mean(observation_series**2)))
                    block_rows.append(
                        {
                            "candidate": candidate_name,
                            "observation": product,
                            "region": region,
                            "block": block_index,
                            "start_year": start,
                            "end_year": end,
                            "regional_skill": 1.0 - rmse / zero_rmse,
                        }
                    )
    region_scores = pd.DataFrame(region_rows)
    block_scores = pd.DataFrame(block_rows)
    region_scores.to_csv(output / "dependence_aware_modern_scores.csv", index=False)
    block_scores.to_csv(output / "dependence_aware_modern_block_scores.csv", index=False)
    modern_summaries = {
        name: modern.summarize_variable(
            region_scores[region_scores.candidate == name],
            block_scores[block_scores.candidate == name],
        )
        for name in candidates
    }

    project_windows = pd.read_csv(paths["round70_windows"])
    if list(zip(project_windows.start_year, project_windows.end_year)) != list(WINDOWS):
        raise RuntimeError("project-window-axis-conflict")
    paleo_years = np.arange(1450, 1900)
    paleo_selection = (ALLOWED_YEARS >= 1450) & (ALLOWED_YEARS <= 1899)
    family_paleo = family_stack[:, paleo_selection]
    window_rows: list[dict[str, Any]] = []
    for window_index, (start, end) in enumerate(WINDOWS):
        selected = (paleo_years >= start) & (paleo_years <= end)
        family_window = np.mean(family_paleo[:, selected], axis=1)
        central = np.mean(family_window, axis=0)
        spread = np.std(family_window, axis=0, ddof=1)
        majority_cold = np.sum(family_window < 0, axis=0) >= 3
        resolved = np.abs(central) >= spread
        loo_fields = np.stack(
            [np.mean(np.delete(family_window, omitted, axis=0), axis=0) for omitted in range(5)]
        )
        all_loo_cold = np.all(loo_fields < 0, axis=0)
        correlations = pairwise_correlations(family_window)
        family_regional = np.asarray(
            [modern.weighted_series(family_window[index][None], weights)[0] for index in range(5)]
        )
        project_mean = float(project_windows.loc[window_index, "project_corrected_mean_degC"])
        window_rows.append(
            {
                "start_year": start,
                "end_year": end,
                "majority_cold_area_fraction": weighted_fraction(majority_cold, weights),
                "amplitude_resolved_area_fraction": weighted_fraction(resolved, weights),
                "all_leave_one_family_out_cold_area_fraction": weighted_fraction(all_loo_cold, weights),
                "median_pairwise_family_spatial_correlation": float(np.median(correlations)),
                "minimum_pairwise_family_spatial_correlation": float(np.min(correlations)),
                "project_corrected_mean_degC": project_mean,
                "warmest_family_regional_mean_degC": float(family_regional.max()),
                "project_warmer_than_every_family": bool(project_mean > family_regional.max()),
                **{f"{name}_regional_mean_degC": value for name, value in zip(FAMILY_NAMES, family_regional)},
            }
        )
    windows = pd.DataFrame(window_rows)
    windows.to_csv(output / "dependence_aware_lia_spatial_windows.csv", index=False)

    lofo_names = [name for name in candidates if name.startswith("leave_out_")]
    lofo_spatial_pass_count = sum(modern_summaries[name]["passes_spatial_field"] for name in lofo_names)
    gate_counts = {
        "majority_cold_area_at_least_70pct_window_count": int((windows.majority_cold_area_fraction >= 0.70).sum()),
        "amplitude_resolved_area_at_least_50pct_window_count": int((windows.amplitude_resolved_area_fraction >= 0.50).sum()),
        "all_loo_cold_area_at_least_70pct_window_count": int((windows.all_leave_one_family_out_cold_area_fraction >= 0.70).sum()),
        "median_spatial_correlation_at_least_0_30_window_count": int((windows.median_pairwise_family_spatial_correlation >= 0.30).sum()),
    }
    lia_spatial_gates = {
        name.replace("_window_count", "_at_least_7_of_9"): count >= 7
        for name, count in gate_counts.items()
    }
    project_warm_count = int(windows.project_warmer_than_every_family.sum())
    gates = {
        "four_family_reconstruction_closure_below_1e_6": four_closure <= 1e-6,
        "five_family_identity_closure_below_1e_6": max(five_identity_closure, full_candidate_closure) <= 1e-6,
        "full_five_family_passes_modern_spatial_field": modern_summaries["full_five_family"]["passes_spatial_field"],
        "all_five_leave_one_family_out_candidates_pass_modern_spatial_field": lofo_spatial_pass_count == 5,
        "all_lia_spatial_robustness_gates_pass": bool(all(lia_spatial_gates.values())),
        "project_warmer_than_every_family_in_at_least_8_of_9_windows": project_warm_count >= 8,
        "locked_1921_1939_absent": True,
    }
    accepted = bool(all(gates.values()))
    candidate_state = state(paths["five_family_candidate"])
    summary = {
        "status": "complete",
        "allowed_periods": [[1421, 1920], [1940, 2000]],
        "locked_1921_1939_accessed": False,
        "family_definition": list(FAMILY_NAMES),
        "uncertainty_interpretation": "method_family_sensitivity_envelope_not_independent_sample_uncertainty",
        "proxy_evidence_independence": False,
        "closure": {
            "four_family_max_abs_degC": four_closure,
            "five_family_identity_max_abs_degC": five_identity_closure,
            "full_candidate_max_abs_degC": full_candidate_closure,
        },
        "modern": {
            "full_and_leave_one_out": modern_summaries,
            "leave_one_out_spatial_pass_count": lofo_spatial_pass_count,
        },
        "lia_spatial": {
            "gate_counts": gate_counts,
            "gates": lia_spatial_gates,
            "median_majority_cold_area_fraction": float(windows.majority_cold_area_fraction.median()),
            "median_amplitude_resolved_area_fraction": float(windows.amplitude_resolved_area_fraction.median()),
            "median_all_loo_cold_area_fraction": float(windows.all_leave_one_family_out_cold_area_fraction.median()),
            "median_pairwise_family_spatial_correlation": float(windows.median_pairwise_family_spatial_correlation.median()),
        },
        "project_warmer_than_every_family_window_count": project_warm_count,
        "gates": gates,
        "passes_dependence_aware_robustness": accepted,
        "candidate_freeze": {
            "status": "frozen_for_final_test_readiness" if accepted else "not_frozen",
            "artifact": candidate_state,
            "locked_final_test_opened": False,
        },
        "decision": (
            "freeze_allowed_period_five_method_temperature_reference_candidate_for_final_test_readiness"
            if accepted
            else "retain_unfrozen_candidate_and_resolve_leave_one_family_out_failure"
        ),
        "production_release_authorized": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "candidate_freeze_manifest.json").write_text(
        json.dumps(summary["candidate_freeze"], indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.7), constrained_layout=True)
    candidate_order = list(candidates)
    x = np.arange(len(candidate_order))
    axes[0, 0].bar(x, [modern_summaries[name]["median_region_product_skill"] * 100 for name in candidate_order], color="#0072B2")
    axes[0, 0].axhline(5, color="#777777", linestyle="--", linewidth=0.9)
    axes[0, 0].set_xticks(x, candidate_order, rotation=35, ha="right", fontsize=8)
    axes[0, 0].set(title="Modern regional skill", ylabel="Median skill (%)")
    axes[0, 1].bar(x, [modern_summaries[name]["median_hma_spatial_residual_skill"] * 100 for name in candidate_order], color="#D55E00")
    axes[0, 1].axhline(0, color="#777777", linewidth=0.9)
    axes[0, 1].set_xticks(x, candidate_order, rotation=35, ha="right", fontsize=8)
    axes[0, 1].set(title="Modern spatial residual skill", ylabel="Median skill (%)")
    wx = np.arange(len(windows))
    labels = [f"{row.start_year}-{row.end_year}" for row in windows.itertuples()]
    axes[1, 0].plot(wx, windows.majority_cold_area_fraction * 100, marker="o", label="Majority cold")
    axes[1, 0].plot(wx, windows.amplitude_resolved_area_fraction * 100, marker="s", label="Amplitude resolved")
    axes[1, 0].plot(wx, windows.all_leave_one_family_out_cold_area_fraction * 100, marker="^", label="All LOO cold")
    axes[1, 0].set_xticks(wx, labels, rotation=35, ha="right", fontsize=8)
    axes[1, 0].set(title="LIA dependence-aware spatial support", ylabel="HMA area (%)", ylim=(0, 105))
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(wx, windows.project_corrected_mean_degC, color="#CC79A7", marker="s", label="Corrected project")
    axes[1, 1].plot(wx, windows.warmest_family_regional_mean_degC, color="#111111", marker="o", label="Warmest method family")
    axes[1, 1].axhline(0, color="#777777", linewidth=0.8)
    axes[1, 1].set_xticks(wx, labels, rotation=35, ha="right", fontsize=8)
    axes[1, 1].set(title="Project warm-bias robustness", ylabel="Anomaly (degC)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    figure.savefig(output / "five_family_dependence_aware_robustness.png", dpi=180)
    plt.close(figure)

    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Five-Family Dependence-Aware Robustness",
                "",
                f"- Modern leave-one-family-out spatial passes: `{lofo_spatial_pass_count}/5`.",
                f"- LIA gate counts: `{gate_counts}`.",
                f"- Median majority-cold / amplitude-resolved / all-LOO-cold area: `{windows.majority_cold_area_fraction.median():.1%}` / `{windows.amplitude_resolved_area_fraction.median():.1%}` / `{windows.all_leave_one_family_out_cold_area_fraction.median():.1%}`.",
                f"- Project warmer than every method family: `{project_warm_count}/9` windows.",
                f"- Dependence-aware candidate pass: `{str(accepted).lower()}`.",
                "",
                "Family spread is retained as a method-sensitivity envelope, not an independent-sample confidence interval.",
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
        json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
