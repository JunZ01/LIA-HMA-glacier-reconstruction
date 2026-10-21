"""Run Round 115 v2 on the frozen common observed HMA support."""

from __future__ import annotations

import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-spec-failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    driver = Path(__file__).resolve().parent
    base = load_module(
        "round115_v1_base",
        driver / "run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1.py",
    )
    common = base.load_common(driver)
    root = base.project_root()
    output = (
        root
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_115_temperature_low_rank_oof_v2"
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "proxy9": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "ledger": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_110_unresolved_proxy_metadata_recovery_v4/proxy_evidence_master_ledger_metadata_recovered.csv",
        "public": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "round113_gates": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_113_frozen_multiscale_validation_gates_v1/acceptance_gates.csv",
        "v1_contract": driver / "round115_temperature_low_rank_oof_v1_contract.md",
        "contract": driver / "round115_temperature_low_rank_oof_v2_contract.md",
        "v1_driver": driver / "run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1.py",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    shutil.copy2(paths["contract"], output / "ROUND_CONTRACT.md")
    before = {name: base.file_state(path) for name, path in paths.items()}
    config = {
        "experiment_id": "round_115_temperature_low_rank_oof_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "years_read": [1940, 2000],
        "locked_1921_1939_accessed": False,
        "external_reserve_opened": False,
        "technical_change_from_v1": "shared all-five-product observed HMA support",
        "legacy_control_variable": base.LEGACY_VARIABLE,
        "blocks": base.BLOCKS,
        "rank_grid": base.RANK_GRID,
        "ridge_grid": base.RIDGE_GRID,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    loaded = scipy.io.loadmat(paths["proxy9"], variable_names=["years", "observations"])
    proxy_years = np.asarray(loaded["years"], dtype=int).reshape(-1)
    proxy_all = np.asarray(loaded["observations"], dtype=float)
    if not np.array_equal(proxy_years, base.YEARS) or proxy_all.shape != (61, 297):
        raise RuntimeError("proxy9-input-contract-failed")
    ledger = pd.read_csv(paths["ledger"], low_memory=False)
    proxy_ledger = ledger[ledger.network.eq("proxy9")].copy()
    proxy_ledger["network_record_1based"] = proxy_ledger.network_record_1based.astype(int)
    proxy_ledger = proxy_ledger.sort_values("network_record_1based")
    if len(proxy_ledger) != 297 or not np.array_equal(
        proxy_ledger.network_record_1based.to_numpy(), np.arange(1, 298)
    ):
        raise RuntimeError("proxy9-ledger-join-contract-failed")
    selected = (
        proxy_ledger.documented_network_role_v4.eq("documented_temperature")
        & proxy_ledger.operator_metadata_ready_v4.astype(str).str.lower().eq("true")
    ).to_numpy()
    if int(selected.sum()) != 226:
        raise RuntimeError(f"temperature-proxy-count-failed:{int(selected.sum())}")
    proxy = proxy_all[:, selected]
    proxy_ledger.loc[selected].to_csv(output / "selected_temperature_proxy_groups.csv", index=False)

    grid_lat, grid_lon = common.state_coordinates()
    with h5py.File(paths["public"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).reshape(-1)
        time_index = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in base.YEARS])
        selected_years = source_years[time_index]
        if not np.array_equal(selected_years, base.YEARS):
            raise RuntimeError("public-year-axis-failed")
        if np.any((selected_years >= 1921) & (selected_years <= 1939)):
            raise RuntimeError("locked-period-access-attempt")
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        observations = {
            name: base.recenter(
                common.cube_to_state(
                    common.source_grid(
                        common.clean(np.asarray(handle[variable][time_index], dtype=float)),
                        source_lat,
                        source_lon,
                        grid_lat,
                        grid_lon,
                    )
                )
            )
            for name, variable in base.OBSERVATION_VARIABLES.items()
        }
        legacy = base.recenter(
            common.cube_to_state(
                common.source_grid(
                    common.clean(np.asarray(handle[base.LEGACY_VARIABLE][time_index], dtype=float)),
                    source_lat,
                    source_lon,
                    grid_lat,
                    grid_lon,
                )
            )
        )

    mask05 = np.load(paths["hma_mask"]).astype(float)
    mask2 = common.source_grid(mask05[None], source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    full_weights = common.region_weights(grid_lat, grid_lon, mask2)
    hma_mask_state = full_weights["hma"] > 0
    observation_stack = np.stack(list(observations.values()))
    consensus = np.nanmean(observation_stack, axis=0)
    observed_support = hma_mask_state & np.all(np.isfinite(consensus), axis=1)
    unsupported = hma_mask_state & ~observed_support
    if int(hma_mask_state.sum()) != 105 or int(observed_support.sum()) != 99 or int(unsupported.sum()) != 6:
        raise RuntimeError(
            f"observed-support-contract-failed:{int(hma_mask_state.sum())}:{int(observed_support.sum())}:{int(unsupported.sum())}"
        )
    if not np.all(np.isfinite(observation_stack[:, observed_support])):
        raise RuntimeError("common-observed-support-not-complete")
    if not np.all(np.isfinite(legacy[observed_support])):
        raise RuntimeError("legacy-incomplete-on-observed-support")
    weights = {name: values * observed_support for name, values in full_weights.items()}
    if any(np.sum(values > 0) < 3 for values in weights.values()):
        raise RuntimeError("regional-observed-support-insufficient")
    target = consensus[observed_support]
    area_weights = weights["hma"][observed_support]

    support_rows = []
    lon2 = np.broadcast_to(grid_lon[None], mask2.shape).T.reshape(-1)
    lat2 = np.broadcast_to(grid_lat[:, None], mask2.shape).T.reshape(-1)
    for index in np.flatnonzero(hma_mask_state):
        support_rows.append(
            {
                "state_index_0based": int(index),
                "latitude": float(lat2[index]),
                "longitude": float(lon2[index]),
                "observed_support": bool(observed_support[index]),
                "all_five_products_missing_all_years": bool(unsupported[index]),
            }
        )
    pd.DataFrame(support_rows).to_csv(output / "modern_observed_support.csv", index=False)

    oof = np.full((len(hma_mask_state), len(base.YEARS)), np.nan)
    selection_rows = []
    inner_rows = []
    for outer_fold, (start, end) in enumerate(base.BLOCKS, start=1):
        outer_test = (base.YEARS >= start) & (base.YEARS <= end)
        outer_train = ~outer_test
        rank, ridge, rows = base.select_hyperparameters(
            proxy,
            target,
            outer_train,
            observations,
            weights,
            observed_support,
            area_weights,
            common,
        )
        for row in rows:
            row["outer_fold"] = int(outer_fold)
            row["outer_period"] = f"{start}-{end}"
        inner_rows.extend(rows)
        predicted, diagnostics = base.fit_low_rank(
            proxy,
            target,
            outer_train,
            outer_test,
            area_weights,
            rank,
            ridge,
        )
        oof[np.ix_(observed_support, outer_test)] = predicted
        selection_rows.append(
            {
                "outer_fold": int(outer_fold),
                "outer_period": f"{start}-{end}",
                "selected_rank": int(rank),
                "selected_ridge": float(ridge),
                **diagnostics,
            }
        )
    if not np.all(np.isfinite(oof[observed_support])):
        raise RuntimeError("oof-stitch-incomplete")
    candidate = base.recenter(oof)
    branches = {
        "low_rank_oof_candidate": candidate,
        "equal_six_legacy": legacy,
        "zero_anomaly": np.zeros_like(legacy),
    }
    pairs, blocks, curves, spatial = base.score_all(branches, observations, weights, common)
    candidate_pairs = pairs[pairs.branch.eq("low_rank_oof_candidate")]
    candidate_blocks = blocks[blocks.branch.eq("low_rank_oof_candidate")]
    candidate_spatial = spatial[spatial.branch.eq("low_rank_oof_candidate")]
    relative = candidate_pairs.improvement_vs_equal_six.to_numpy(float)
    relative_paths = candidate_blocks.improvement_vs_equal_six.to_numpy(float)
    absolute = candidate_pairs.skill_vs_zero.to_numpy(float)
    gates = {
        "M_REL_1_median_rmse_improvement_over_equal_six_at_least_0_05": bool(np.nanmedian(relative) >= 0.05),
        "M_REL_2_worst_pair_regression_at_most_0_05": bool(np.nanmin(relative) >= -0.05 - 1e-12),
        "M_REL_3_fraction_block_region_product_paths_improving_at_least_0_75": bool(np.mean(relative_paths > 0) >= 0.75),
        "M_ABS_1_median_rmse_skill_over_zero_at_least_0_05": bool(np.nanmedian(absolute) >= 0.05),
        "M_ABS_2_worst_rmse_skill_over_zero_at_least_minus_0_05": bool(np.nanmin(absolute) >= -0.05 - 1e-12),
        "M_ABS_3_fraction_positive_regional_skill_at_least_0_75": bool(np.mean(absolute > 0) >= 0.75),
        "M_ABS_4_median_annual_regional_correlation_at_least_0_30": bool(np.nanmedian(candidate_pairs.annual_correlation) >= 0.30),
        "M_SPAT_1_median_spatial_skill_positive": bool(np.nanmedian(candidate_spatial.spatial_residual_skill) > 0),
        "M_SPAT_2_worst_spatial_skill_at_least_minus_0_05": bool(np.nanmin(candidate_spatial.spatial_residual_skill) >= -0.05 - 1e-12),
        "M_SPAT_3_fraction_positive_spatial_skill_at_least_0_75": bool(np.mean(candidate_spatial.spatial_residual_skill > 0) >= 0.75),
        "M_SPAT_4_median_spatial_pattern_correlation_at_least_0_20": bool(np.nanmedian(candidate_spatial.median_annual_spatial_pattern_correlation) >= 0.20),
    }
    metrics = {
        "median_rmse_improvement_over_equal_six": float(np.nanmedian(relative)),
        "worst_pair_rmse_improvement_over_equal_six": float(np.nanmin(relative)),
        "fraction_block_region_product_paths_improving": float(np.mean(relative_paths > 0)),
        "median_skill_over_zero": float(np.nanmedian(absolute)),
        "worst_skill_over_zero": float(np.nanmin(absolute)),
        "fraction_positive_regional_skill": float(np.mean(absolute > 0)),
        "median_annual_regional_correlation": float(np.nanmedian(candidate_pairs.annual_correlation)),
        "median_15year_regional_correlation": float(np.nanmedian(candidate_pairs.correlation_15year)),
        "median_spatial_residual_skill": float(np.nanmedian(candidate_spatial.spatial_residual_skill)),
        "worst_spatial_residual_skill": float(np.nanmin(candidate_spatial.spatial_residual_skill)),
        "fraction_positive_spatial_skill": float(np.mean(candidate_spatial.spatial_residual_skill > 0)),
        "median_spatial_pattern_correlation": float(np.nanmedian(candidate_spatial.median_annual_spatial_pattern_correlation)),
    }
    passes = bool(all(gates.values()))

    pd.DataFrame(selection_rows).to_csv(output / "outer_fold_hyperparameter_selection.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(output / "inner_hyperparameter_scores.csv", index=False)
    pairs.to_csv(output / "regional_product_scores.csv", index=False)
    blocks.to_csv(output / "block_region_product_scores.csv", index=False)
    spatial.to_csv(output / "spatial_product_scores.csv", index=False)
    curves.to_csv(output / "regional_curves.csv", index=False)
    np.savez_compressed(
        output / "temperature_low_rank_oof_fields.npz",
        years=base.YEARS,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        hma_mask=mask2,
        observed_support=observed_support,
        candidate=candidate,
        equal_six_legacy=legacy,
        **{f"observation_{name}": values for name, values in observations.items()},
    )

    hma = curves[curves.region.eq("hma")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for series, color, width in (
        ("low_rank_oof_candidate", "#0072B2", 2.2),
        ("equal_six_legacy", "#D55E00", 1.8),
    ):
        values = hma[(hma.kind == "reconstruction") & (hma.series == series)]
        axes[0].plot(values.year, values.value_degC, color=color, linewidth=width, label=series)
    obs = hma[hma.kind.eq("observation")].groupby("year", as_index=False).value_degC.mean()
    axes[0].plot(obs.year, obs.value_degC, color="#111111", linewidth=1.8, label="five-product mean")
    axes[0].axhline(0, color="#777777", linewidth=0.7)
    axes[0].set_ylabel("HMA anomaly (degC)")
    axes[0].set_title("Nested out-of-fold temperature on common observed support")
    axes[0].legend(frameon=False, ncol=3)
    plot = spatial.pivot(index="observation", columns="branch", values="spatial_residual_skill")
    plot[["low_rank_oof_candidate", "equal_six_legacy"]].plot.bar(
        ax=axes[1], color=["#0072B2", "#D55E00"], width=0.72
    )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_ylabel("Spatial residual skill vs zero")
    axes[1].set_xlabel("")
    axes[1].legend(frameon=False)
    fig.savefig(output / "temperature_low_rank_oof_diagnostics.png", dpi=220)
    plt.close(fig)

    after = {name: base.file_state(path) for name, path in paths.items()}
    unchanged = bool(before == after)
    if not unchanged:
        raise RuntimeError("source-changed-during-round115-v2")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "experiment_id": "round_115_temperature_low_rank_oof_v2",
        "status": "complete",
        "passes_all_required_modern_gates": passes,
        "decision": "advance_to_full_temperature_candidate" if passes else "close_low_rank_annual_proxy9_method",
        "locked_1921_1939_accessed": False,
        "external_reserve_opened": False,
        "source_hashes_unchanged": unchanged,
        "selected_temperature_evidence_group_count": int(selected.sum()),
        "hma_state_cell_count": int(hma_mask_state.sum()),
        "observed_support_cell_count": int(observed_support.sum()),
        "all_product_missing_cell_count": int(unsupported.sum()),
        "outer_fold_selections": selection_rows,
        "metrics": metrics,
        "gates": gates,
        "interpretation_boundary": "The six all-product-missing edge cells are not instrumentally validated; the five products share station information.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Round 115 Temperature Low-Rank OOF V2",
        "",
        f"- Decision: `{summary['decision']}`.",
        f"- Passes every required modern gate: `{str(passes).lower()}`.",
        f"- Documented-temperature evidence groups: `{int(selected.sum())}`.",
        f"- Observed support: `{int(observed_support.sum())}` of `{int(hma_mask_state.sum())}` HMA state cells; all-product-missing edge cells: `{int(unsupported.sum())}`.",
        f"- Selected ranks: `{[row['selected_rank'] for row in selection_rows]}`.",
        f"- Selected ridge penalties: `{[row['selected_ridge'] for row in selection_rows]}`.",
        f"- Median improvement over equal-six: `{metrics['median_rmse_improvement_over_equal_six']:.3%}`.",
        f"- Improving block-region-product paths: `{metrics['fraction_block_region_product_paths_improving']:.3%}`.",
        f"- Median absolute skill: `{metrics['median_skill_over_zero']:.3%}`.",
        f"- Median annual regional correlation: `{metrics['median_annual_regional_correlation']:.3f}`.",
        f"- Median spatial residual skill: `{metrics['median_spatial_residual_skill']:.3%}`.",
        f"- Median spatial pattern correlation: `{metrics['median_spatial_pattern_correlation']:.3f}`.",
        "- Locked 1921-1939 accessed: `false`.",
        "- Round 114 external reserve opened: `false`.",
        f"- Source hashes unchanged: `{str(unchanged).lower()}`.",
        "",
        "## Gates",
        "",
        *[f"- `{name}`: `{str(value).lower()}`" for name, value in gates.items()],
        "",
        "The six unsupported cells remain an explicit coverage boundary. No value was fabricated for them.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
