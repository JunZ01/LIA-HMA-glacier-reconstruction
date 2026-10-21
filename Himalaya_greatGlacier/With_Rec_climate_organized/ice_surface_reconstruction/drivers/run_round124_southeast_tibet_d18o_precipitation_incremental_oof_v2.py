"""Incrementally test MiMei and Xinpu d18O in the frozen precipitation OOF model."""

from __future__ import annotations

import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        "round124_base",
        driver / "run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1.py",
    )
    residual = load_module(
        "round124_residual",
        driver / "run_climate_reconstruction_loop_v2_round116_temperature_low_rank_residual_oof_v1.py",
    )
    helpers = load_module(
        "round124_helpers",
        driver / "run_climate_reconstruction_loop_v2_round120_external_temperature_incremental_oof_v1.py",
    )
    round123 = load_module(
        "round124_round123",
        driver / "qualify_round123_southeast_tibet_d18o_v1.py",
    )
    common = base.load_common(driver)
    root = base.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    round117_dir = loop / "round_117_precipitation_documented_moisture_residual_oof_v1"
    round123_dir = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_southeast_tibet_d18o_proxy_qualification_20260818_v1"
    )
    output = loop / "round_124_d18o_pr_oof_v2"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "proxy9": root / (
            "validation_data/results/paleo_surface_validation_v0/"
            "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/"
            "proxy9_allowed_1940_2000_official_baseline_v2.mat"
        ),
        "ledger": loop / "round_110_unresolved_proxy_metadata_recovery_v4/proxy_evidence_master_ledger_metadata_recovered.csv",
        "round117_fields": round117_dir / "precipitation_documented_moisture_residual_oof_fields.npz",
        "round117_summary": round117_dir / "summary.json",
        "round123_summary": round123_dir / "summary.json",
        "mimei": round123_dir / "raw_unopened/mimei_tree_ring_d18o.txt",
        "xinpu": round123_dir / "raw_unopened/xinpu_tree_ring_d18o.txt",
        "contract": driver / "round124_southeast_tibet_d18o_precipitation_incremental_oof_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "base_driver": driver / "run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1.py",
        "residual_driver": driver / "run_climate_reconstruction_loop_v2_round116_temperature_low_rank_residual_oof_v1.py",
        "helper_driver": driver / "run_climate_reconstruction_loop_v2_round120_external_temperature_incremental_oof_v1.py",
        "round123_driver": driver / "qualify_round123_southeast_tibet_d18o_v1.py",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    prior117 = json.loads(paths["round117_summary"].read_text(encoding="utf-8"))
    prior123 = json.loads(paths["round123_summary"].read_text(encoding="utf-8"))
    if prior117["status"] != "complete" or not prior117["source_hashes_unchanged"]:
        raise RuntimeError("round117-source-contract-failed")
    if prior117["locked_1921_1939_accessed"] or prior117["paleo_reserve_opened"]:
        raise RuntimeError("round117-boundary-contract-failed")
    if prior123["status"] != "complete" or not prior123["source_hashes_unchanged"]:
        raise RuntimeError("round123-source-contract-failed")
    if prior123["locked_1921_1939_accessed"] or prior123["external_validation_reserve_opened"]:
        raise RuntimeError("round123-boundary-contract-failed")
    shutil.copy2(paths["contract"], output / "ROUND_CONTRACT.md")
    before = {name: base.file_state(path) for name, path in paths.items()}

    proxy_mat = scipy.io.loadmat(paths["proxy9"], variable_names=["years", "observations"])
    proxy_years = np.asarray(proxy_mat["years"], dtype=int).reshape(-1)
    proxy_all = np.asarray(proxy_mat["observations"], dtype=float)
    if not np.array_equal(proxy_years, base.YEARS) or proxy_all.shape != (61, 297):
        raise RuntimeError("proxy9-input-contract-failed")
    ledger = pd.read_csv(paths["ledger"], low_memory=False)
    proxy_ledger = ledger[ledger.network.eq("proxy9")].copy()
    proxy_ledger["network_record_1based"] = proxy_ledger.network_record_1based.astype(int)
    proxy_ledger = proxy_ledger.sort_values("network_record_1based")
    selected = (
        proxy_ledger.documented_network_role_v4.eq("documented_moisture")
        & proxy_ledger.operator_metadata_ready_v4.astype(str).str.lower().eq("true")
    ).to_numpy()
    if len(proxy_ledger) != 297 or int(selected.sum()) != 25:
        raise RuntimeError("documented-moisture-selection-contract-failed")
    proxy = proxy_all[:, selected]
    mimei, mimei_audit = round123.parse_candidate(paths["mimei"])
    xinpu, xinpu_audit = round123.parse_candidate(paths["xinpu"])
    if mimei_audit["locked_1921_1939_values_decoded"] or xinpu_audit["locked_1921_1939_values_decoded"]:
        raise RuntimeError("locked-candidate-value-decoded")
    pd.DataFrame({"year": base.YEARS, "mimei_d18o": mimei, "xinpu_d18o": xinpu}).to_csv(
        output / "new_d18o_proxy_values_1940_2000.csv", index=False
    )

    fields = np.load(paths["round117_fields"])
    if not np.array_equal(fields["years"], base.YEARS):
        raise RuntimeError("round117-year-closure-failed")
    grid_lat = np.asarray(fields["grid_lat"], dtype=float)
    grid_lon = np.asarray(fields["grid_lon"], dtype=float)
    hma_mask = np.asarray(fields["hma_mask"], dtype=bool)
    support = np.asarray(fields["observed_support"], dtype=bool)
    legacy = np.asarray(fields["equal_six_legacy"], dtype=float)
    observations = {
        "CRU": np.asarray(fields["observation_CRU"], dtype=float),
        "GPCC": np.asarray(fields["observation_GPCC"], dtype=float),
    }
    weights = common.region_weights(grid_lat, grid_lon, hma_mask)
    weights = {name: values * support for name, values in weights.items()}
    consensus = np.mean(np.stack([values[support] for values in observations.values()]), axis=0)
    target = consensus - legacy[support]
    area_weights = weights["hma"][support]

    matrices = {
        "plus_mimei": np.column_stack([proxy, mimei]),
        "plus_xinpu": np.column_stack([proxy, xinpu]),
        "plus_both": np.column_stack([proxy, mimei, xinpu]),
    }
    candidates: dict[str, np.ndarray] = {}
    selections: list[dict[str, Any]] = []
    inner_scores: list[dict[str, Any]] = []
    for branch, matrix in matrices.items():
        candidate, branch_selections, branch_inner = helpers.run_branch(
            branch,
            matrix,
            base,
            residual,
            common,
            target,
            legacy,
            observations,
            weights,
            support,
            area_weights,
        )
        candidates[branch] = candidate
        selections.extend(branch_selections)
        inner_scores.extend(branch_inner)

    control = np.asarray(fields["candidate"], dtype=float)
    branches = {
        **candidates,
        "round117_25_proxy_control": control,
        "equal_six_legacy": legacy,
        "zero_anomaly": np.zeros_like(legacy),
    }
    pairs, blocks, curves, spatial = base.score_all(branches, observations, weights, common)
    evaluated = ["round117_25_proxy_control", "plus_mimei", "plus_xinpu", "plus_both"]
    metrics: dict[str, dict[str, float]] = {}
    gates: dict[str, dict[str, bool]] = {}
    for branch in evaluated:
        metrics[branch], gates[branch] = helpers.branch_metrics(branch, pairs, blocks, spatial)
    baseline = metrics["round117_25_proxy_control"]
    deltas = {
        branch: {name: value - baseline[name] for name, value in current.items()}
        for branch, current in metrics.items()
        if branch != "round117_25_proxy_control"
    }
    combined_passes = bool(all(gates["plus_both"].values()))

    pd.DataFrame(selections).to_csv(output / "outer_fold_hyperparameter_selection.csv", index=False)
    pd.DataFrame(inner_scores).to_csv(output / "inner_hyperparameter_scores.csv", index=False)
    pairs.to_csv(output / "regional_product_scores.csv", index=False)
    blocks.to_csv(output / "block_region_product_scores.csv", index=False)
    spatial.to_csv(output / "spatial_product_scores.csv", index=False)
    curves.to_csv(output / "regional_curves.csv", index=False)
    np.savez_compressed(
        output / "southeast_tibet_d18o_precipitation_incremental_oof_fields.npz",
        years=base.YEARS,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        hma_mask=hma_mask,
        observed_support=support,
        plus_mimei=candidates["plus_mimei"],
        plus_xinpu=candidates["plus_xinpu"],
        plus_both=candidates["plus_both"],
        round117_25_proxy_control=control,
        equal_six_legacy=legacy,
    )

    colors = {
        "round117_25_proxy_control": "#777777",
        "plus_mimei": "#0072B2",
        "plus_xinpu": "#009E73",
        "plus_both": "#D55E00",
        "equal_six_legacy": "#CC79A7",
    }
    hma = curves[curves.region.eq("hma")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for branch in ["equal_six_legacy", *evaluated]:
        values = hma[(hma.kind == "reconstruction") & (hma.series == branch)]
        axes[0].plot(values.year, values.value_degC, color=colors[branch], linewidth=1.6, label=branch)
    obs = hma[hma.kind.eq("observation")].groupby("year", as_index=False).value_degC.mean()
    axes[0].plot(obs.year, obs.value_degC, color="#111111", linewidth=1.8, label="CRU-GPCC mean")
    axes[0].axhline(0, color="#888888", linewidth=0.7)
    axes[0].set_ylabel("HMA precipitation anomaly (mm/day)")
    axes[0].set_title("Southeast Tibet d18O incremental precipitation OOF")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    metric_names = [
        "median_rmse_improvement_over_equal_six",
        "worst_pair_rmse_improvement_over_equal_six",
        "median_spatial_residual_skill",
        "worst_spatial_residual_skill",
    ]
    x = np.arange(len(metric_names))
    width = 0.19
    for index, branch in enumerate(evaluated):
        axes[1].bar(
            x + (index - 1.5) * width,
            [metrics[branch][name] for name in metric_names],
            width=width,
            color=colors[branch],
            label=branch,
        )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(x, ["median RMSE", "worst RMSE", "median spatial", "worst spatial"])
    axes[1].set_ylabel("improvement / skill")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(output / "round124_d18o_pr_oof.png", dpi=220)
    plt.close(fig)

    after = {name: base.file_state(path) for name, path in paths.items()}
    unchanged = bool(before == after)
    if not unchanged:
        raise RuntimeError("source-changed-during-round124-v2")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "experiment_id": "round_124_d18o_pr_oof_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "decision": (
            "advance_both_d18o_precipitation_branch"
            if combined_passes
            else "do_not_promote_d18o_addition_in_current_annual_low_rank_method"
        ),
        "combined_branch_passes_all_required_modern_gates": combined_passes,
        "locked_1921_1939_accessed": False,
        "locked_1921_1939_candidate_values_decoded": 0,
        "external_validation_reserve_opened": False,
        "source_hashes_unchanged": unchanged,
        "existing_moisture_proxy_count": 25,
        "branch_proxy_counts": {name: int(matrix.shape[1]) for name, matrix in matrices.items()},
        "metrics": metrics,
        "metric_deltas_vs_round117": deltas,
        "gates": gates,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Round 124 Southeast Tibet d18O Precipitation Incremental OOF",
        "",
        f"- Decision: `{summary['decision']}`.",
        f"- Combined branch passes every frozen gate: `{str(combined_passes).lower()}`.",
        "- Locked 1921-1939 candidate values decoded: `0`.",
        "- External final-test reserve opened: `false`.",
        f"- Source hashes unchanged: `{str(unchanged).lower()}`.",
        "",
        "## Branch results",
        "",
    ]
    for branch in evaluated:
        current = metrics[branch]
        lines.extend(
            [
                f"### {branch}",
                "",
                f"- Gates passed: `{sum(gates[branch].values())}/{len(gates[branch])}`.",
                f"- Median improvement over equal-six: `{current['median_rmse_improvement_over_equal_six']:.3%}`.",
                f"- Worst pair improvement: `{current['worst_pair_rmse_improvement_over_equal_six']:.3%}`.",
                f"- Improving paths: `{current['fraction_block_region_product_paths_improving']:.3%}`.",
                f"- Median spatial residual skill: `{current['median_spatial_residual_skill']:.3%}`.",
                f"- Worst spatial residual skill: `{current['worst_spatial_residual_skill']:.3%}`.",
                "",
            ]
        )
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

