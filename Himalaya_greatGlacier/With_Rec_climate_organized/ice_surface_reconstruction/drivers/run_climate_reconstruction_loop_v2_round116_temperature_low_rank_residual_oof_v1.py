"""Test a nested proxy low-rank residual correction around equal-six."""

from __future__ import annotations

import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

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


def select_residual_hyperparameters(
    base,
    common,
    proxy: np.ndarray,
    residual_target: np.ndarray,
    legacy: np.ndarray,
    outer_train: np.ndarray,
    observations: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    support: np.ndarray,
    area_weights: np.ndarray,
):
    rows = []
    objectives = {}
    for rank in base.RANK_GRID:
        for ridge in base.RIDGE_GRID:
            current = []
            local_rows = []
            for inner_fold, (start, end) in enumerate(base.BLOCKS, start=1):
                valid = outer_train & (base.YEARS >= start) & (base.YEARS <= end)
                if not valid.any():
                    continue
                train = outer_train & ~valid
                predicted, _ = base.fit_low_rank(
                    proxy,
                    residual_target,
                    train,
                    valid,
                    area_weights,
                    rank,
                    ridge,
                )
                candidate = legacy[:, valid].copy()
                candidate[support] += predicted
                objective = base.regional_rmse_objective(
                    candidate,
                    valid,
                    observations,
                    weights,
                    common,
                )
                current.append(objective)
                local_rows.append(
                    {
                        "rank": int(rank),
                        "ridge": float(ridge),
                        "inner_fold": int(inner_fold),
                        "inner_period": f"{start}-{end}",
                        "inner_objective_degC": objective,
                    }
                )
            if len(current) != 4:
                raise RuntimeError(f"inner-fold-count-failed:{rank}:{ridge}:{len(current)}")
            median = float(np.nanmedian(current))
            objectives[(int(rank), float(ridge))] = median
            for row in local_rows:
                row["hyperparameter_median_objective_degC"] = median
            rows.extend(local_rows)
    selected = min(objectives, key=lambda key: (objectives[key], key[0], -key[1]))
    for row in rows:
        row["selected"] = bool((int(row["rank"]), float(row["ridge"])) == selected)
    return selected[0], selected[1], rows


def main() -> int:
    driver = Path(__file__).resolve().parent
    base = load_module(
        "round116_base",
        driver / "run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1.py",
    )
    common = base.load_common(driver)
    root = base.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    round115 = loop / "round_115_temperature_low_rank_oof_v2"
    output = loop / "round_116_temperature_low_rank_residual_oof_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "proxy9": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "ledger": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_110_unresolved_proxy_metadata_recovery_v4/proxy_evidence_master_ledger_metadata_recovered.csv",
        "round115_fields": round115 / "temperature_low_rank_oof_fields.npz",
        "round115_summary": round115 / "summary.json",
        "round115_manifest": round115 / "source_manifest.json",
        "round113_gates": loop / "round_113_frozen_multiscale_validation_gates_v1/acceptance_gates.csv",
        "contract": driver / "round116_temperature_low_rank_residual_oof_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "base_driver": driver / "run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1.py",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    prior = json.loads(paths["round115_summary"].read_text(encoding="utf-8"))
    if prior["passes_all_required_modern_gates"] or not prior["source_hashes_unchanged"]:
        raise RuntimeError("round115-boundary-contract-failed")
    if prior["locked_1921_1939_accessed"] or prior["external_reserve_opened"]:
        raise RuntimeError("round115-data-boundary-failed")
    shutil.copy2(paths["contract"], output / "ROUND_CONTRACT.md")
    before = {name: base.file_state(path) for name, path in paths.items()}
    (output / "config.json").write_text(
        json.dumps(
            {
                "experiment_id": "round_116_temperature_low_rank_residual_oof_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "years_read": [1940, 2000],
                "locked_1921_1939_accessed": False,
                "external_reserve_opened": False,
                "target": "five-product mean minus equal-six",
                "candidate": "equal-six plus nested OOF predicted residual",
                "blocks": base.BLOCKS,
                "rank_grid": base.RANK_GRID,
                "ridge_grid": base.RIDGE_GRID,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fields = np.load(paths["round115_fields"])
    years = np.asarray(fields["years"], dtype=int)
    if not np.array_equal(years, base.YEARS):
        raise RuntimeError("round115-year-closure-failed")
    support = np.asarray(fields["observed_support"], dtype=bool)
    hma_mask = np.asarray(fields["hma_mask"], dtype=bool)
    grid_lat = np.asarray(fields["grid_lat"], dtype=float)
    grid_lon = np.asarray(fields["grid_lon"], dtype=float)
    legacy = np.asarray(fields["equal_six_legacy"], dtype=float)
    observations = {
        name: np.asarray(fields[f"observation_{name}"], dtype=float)
        for name in base.OBSERVATION_VARIABLES
    }
    if int(support.sum()) != 99 or legacy.shape != (336, 61):
        raise RuntimeError("round115-field-support-closure-failed")
    weights = common.region_weights(grid_lat, grid_lon, hma_mask)
    weights = {name: values * support for name, values in weights.items()}
    area_weights = weights["hma"][support]
    consensus = np.mean(np.stack([values[support] for values in observations.values()]), axis=0)
    residual_target = consensus - legacy[support]

    loaded = scipy.io.loadmat(paths["proxy9"], variable_names=["years", "observations"])
    proxy_years = np.asarray(loaded["years"], dtype=int).reshape(-1)
    proxy_all = np.asarray(loaded["observations"], dtype=float)
    if not np.array_equal(proxy_years, base.YEARS) or proxy_all.shape != (61, 297):
        raise RuntimeError("proxy9-contract-failed")
    ledger = pd.read_csv(paths["ledger"], low_memory=False)
    proxy_ledger = ledger[ledger.network.eq("proxy9")].copy()
    proxy_ledger["network_record_1based"] = proxy_ledger.network_record_1based.astype(int)
    proxy_ledger = proxy_ledger.sort_values("network_record_1based")
    selected = (
        proxy_ledger.documented_network_role_v4.eq("documented_temperature")
        & proxy_ledger.operator_metadata_ready_v4.astype(str).str.lower().eq("true")
    ).to_numpy()
    if len(proxy_ledger) != 297 or int(selected.sum()) != 226:
        raise RuntimeError("temperature-proxy-selection-closure-failed")
    proxy = proxy_all[:, selected]
    proxy_ledger.loc[selected].to_csv(output / "selected_temperature_proxy_groups.csv", index=False)

    residual_oof = np.full_like(legacy, np.nan)
    inner_rows = []
    selection_rows = []
    for outer_fold, (start, end) in enumerate(base.BLOCKS, start=1):
        test = (base.YEARS >= start) & (base.YEARS <= end)
        train = ~test
        rank, ridge, rows = select_residual_hyperparameters(
            base,
            common,
            proxy,
            residual_target,
            legacy,
            train,
            observations,
            weights,
            support,
            area_weights,
        )
        for row in rows:
            row["outer_fold"] = int(outer_fold)
            row["outer_period"] = f"{start}-{end}"
        inner_rows.extend(rows)
        prediction, diagnostics = base.fit_low_rank(
            proxy,
            residual_target,
            train,
            test,
            area_weights,
            rank,
            ridge,
        )
        residual_oof[np.ix_(support, test)] = prediction
        selection_rows.append(
            {
                "outer_fold": int(outer_fold),
                "outer_period": f"{start}-{end}",
                "selected_rank": int(rank),
                "selected_ridge": float(ridge),
                **diagnostics,
            }
        )
    if not np.all(np.isfinite(residual_oof[support])):
        raise RuntimeError("residual-oof-stitch-failed")
    candidate = legacy.copy()
    candidate[support] += residual_oof[support]
    candidate = base.recenter(candidate)
    branches = {
        "low_rank_residual_oof_candidate": candidate,
        "equal_six_legacy": legacy,
        "zero_anomaly": np.zeros_like(legacy),
    }
    pairs, blocks, curves, spatial = base.score_all(branches, observations, weights, common)
    cp = pairs[pairs.branch.eq("low_rank_residual_oof_candidate")]
    cb = blocks[blocks.branch.eq("low_rank_residual_oof_candidate")]
    cs = spatial[spatial.branch.eq("low_rank_residual_oof_candidate")]
    relative = cp.improvement_vs_equal_six.to_numpy(float)
    paths_relative = cb.improvement_vs_equal_six.to_numpy(float)
    absolute = cp.skill_vs_zero.to_numpy(float)
    gates = {
        "M_REL_1_median_rmse_improvement_over_equal_six_at_least_0_05": bool(np.nanmedian(relative) >= 0.05),
        "M_REL_2_worst_pair_regression_at_most_0_05": bool(np.nanmin(relative) >= -0.05 - 1e-12),
        "M_REL_3_fraction_block_region_product_paths_improving_at_least_0_75": bool(np.mean(paths_relative > 0) >= 0.75),
        "M_ABS_1_median_rmse_skill_over_zero_at_least_0_05": bool(np.nanmedian(absolute) >= 0.05),
        "M_ABS_2_worst_rmse_skill_over_zero_at_least_minus_0_05": bool(np.nanmin(absolute) >= -0.05 - 1e-12),
        "M_ABS_3_fraction_positive_regional_skill_at_least_0_75": bool(np.mean(absolute > 0) >= 0.75),
        "M_ABS_4_median_annual_regional_correlation_at_least_0_30": bool(np.nanmedian(cp.annual_correlation) >= 0.30),
        "M_SPAT_1_median_spatial_skill_positive": bool(np.nanmedian(cs.spatial_residual_skill) > 0),
        "M_SPAT_2_worst_spatial_skill_at_least_minus_0_05": bool(np.nanmin(cs.spatial_residual_skill) >= -0.05 - 1e-12),
        "M_SPAT_3_fraction_positive_spatial_skill_at_least_0_75": bool(np.mean(cs.spatial_residual_skill > 0) >= 0.75),
        "M_SPAT_4_median_spatial_pattern_correlation_at_least_0_20": bool(np.nanmedian(cs.median_annual_spatial_pattern_correlation) >= 0.20),
    }
    metrics = {
        "median_rmse_improvement_over_equal_six": float(np.nanmedian(relative)),
        "worst_pair_rmse_improvement_over_equal_six": float(np.nanmin(relative)),
        "fraction_block_region_product_paths_improving": float(np.mean(paths_relative > 0)),
        "median_skill_over_zero": float(np.nanmedian(absolute)),
        "worst_skill_over_zero": float(np.nanmin(absolute)),
        "fraction_positive_regional_skill": float(np.mean(absolute > 0)),
        "median_annual_regional_correlation": float(np.nanmedian(cp.annual_correlation)),
        "median_15year_regional_correlation": float(np.nanmedian(cp.correlation_15year)),
        "median_spatial_residual_skill": float(np.nanmedian(cs.spatial_residual_skill)),
        "worst_spatial_residual_skill": float(np.nanmin(cs.spatial_residual_skill)),
        "fraction_positive_spatial_skill": float(np.mean(cs.spatial_residual_skill > 0)),
        "median_spatial_pattern_correlation": float(np.nanmedian(cs.median_annual_spatial_pattern_correlation)),
    }
    passes = bool(all(gates.values()))

    pd.DataFrame(selection_rows).to_csv(output / "outer_fold_hyperparameter_selection.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(output / "inner_hyperparameter_scores.csv", index=False)
    pairs.to_csv(output / "regional_product_scores.csv", index=False)
    blocks.to_csv(output / "block_region_product_scores.csv", index=False)
    spatial.to_csv(output / "spatial_product_scores.csv", index=False)
    curves.to_csv(output / "regional_curves.csv", index=False)
    np.savez_compressed(
        output / "temperature_low_rank_residual_oof_fields.npz",
        years=base.YEARS,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        hma_mask=hma_mask,
        observed_support=support,
        candidate=candidate,
        residual_oof=residual_oof,
        equal_six_legacy=legacy,
    )

    hma = curves[curves.region.eq("hma")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for series, color, width in (
        ("low_rank_residual_oof_candidate", "#0072B2", 2.2),
        ("equal_six_legacy", "#D55E00", 1.8),
    ):
        values = hma[(hma.kind == "reconstruction") & (hma.series == series)]
        axes[0].plot(values.year, values.value_degC, color=color, linewidth=width, label=series)
    obs = hma[hma.kind.eq("observation")].groupby("year", as_index=False).value_degC.mean()
    axes[0].plot(obs.year, obs.value_degC, color="#111111", linewidth=1.7, label="five-product mean")
    axes[0].axhline(0, color="#777777", linewidth=0.7)
    axes[0].set_ylabel("HMA anomaly (degC)")
    axes[0].set_title("Nested OOF residual correction around equal-six")
    axes[0].legend(frameon=False, ncol=3)
    plot = spatial.pivot(index="observation", columns="branch", values="spatial_residual_skill")
    plot[["low_rank_residual_oof_candidate", "equal_six_legacy"]].plot.bar(
        ax=axes[1], color=["#0072B2", "#D55E00"], width=0.72
    )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_ylabel("Spatial residual skill vs zero")
    axes[1].set_xlabel("")
    axes[1].legend(frameon=False)
    fig.savefig(output / "temperature_low_rank_residual_oof_diagnostics.png", dpi=220)
    plt.close(fig)

    after = {name: base.file_state(path) for name, path in paths.items()}
    unchanged = bool(before == after)
    if not unchanged:
        raise RuntimeError("source-changed-during-round116")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "experiment_id": "round_116_temperature_low_rank_residual_oof_v1",
        "status": "complete",
        "passes_all_required_modern_gates": passes,
        "decision": "advance_residual_method_to_full_temperature_candidate" if passes else "close_annual_low_rank_proxy_correction_family",
        "locked_1921_1939_accessed": False,
        "external_reserve_opened": False,
        "source_hashes_unchanged": unchanged,
        "selected_temperature_evidence_group_count": int(selected.sum()),
        "observed_support_cell_count": int(support.sum()),
        "outer_fold_selections": selection_rows,
        "metrics": metrics,
        "gates": gates,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Round 116 Temperature Low-Rank Residual OOF",
        "",
        f"- Decision: `{summary['decision']}`.",
        f"- Passes every required modern gate: `{str(passes).lower()}`.",
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
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
