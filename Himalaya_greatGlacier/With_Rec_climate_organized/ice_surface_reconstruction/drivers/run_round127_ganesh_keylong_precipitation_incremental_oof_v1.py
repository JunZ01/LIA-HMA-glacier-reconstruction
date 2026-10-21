"""Test Ganesh and Keylong d18O in the frozen Round 117 precipitation model."""

from __future__ import annotations

import importlib.util
import json
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


def main() -> int:
    driver = Path(__file__).resolve().parent
    base = load_module("r127_base", driver / "run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1.py")
    residual = load_module("r127_residual", driver / "run_climate_reconstruction_loop_v2_round116_temperature_low_rank_residual_oof_v1.py")
    helpers = load_module("r127_helpers", driver / "run_climate_reconstruction_loop_v2_round120_external_temperature_incremental_oof_v1.py")
    r125 = load_module("r127_r125", driver / "qualify_round125_ganesh_keylong_d18o_v1.py")
    common = base.load_common(driver)
    root = base.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    r117 = loop / "round_117_precipitation_documented_moisture_residual_oof_v1"
    candidates_dir = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_ganesh_keylong_d18o_proxy_qualification_20260818_v1"
    )
    identity_dir = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_ganesh_keylong_identity_closure_20260818_v1"
    )
    output = loop / "round_127_gk_pr_oof_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "proxy9": root / (
            "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/"
            "round_03_nested_ensemble_selection/proxy9_allowed_1940_2000_official_baseline_v2.mat"
        ),
        "ledger": loop / "round_110_unresolved_proxy_metadata_recovery_v4/proxy_evidence_master_ledger_metadata_recovered.csv",
        "r117_fields": r117 / "precipitation_documented_moisture_residual_oof_fields.npz",
        "r117_summary": r117 / "summary.json",
        "r125_summary": candidates_dir / "summary.json",
        "r126_summary": identity_dir / "summary.json",
        "ganesh": candidates_dir / "raw_unopened/ganesh2018d18o-noaa.txt",
        "keylong": candidates_dir / "raw_unopened/keylong2020d18o-noaa.txt",
        "driver": Path(__file__).resolve(),
        "helper_driver": driver / "run_climate_reconstruction_loop_v2_round120_external_temperature_incremental_oof_v1.py",
        "r125_driver": driver / "qualify_round125_ganesh_keylong_d18o_v1.py",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: base.file_state(path) for name, path in paths.items()}
    for key in ("r117_summary", "r125_summary", "r126_summary"):
        prior = json.loads(paths[key].read_text(encoding="utf-8"))
        if prior["status"] != "complete" or not prior["source_hashes_unchanged"]:
            raise RuntimeError(f"prior-contract-failed:{key}")
        if prior.get("locked_1921_1939_accessed", False) or prior.get("external_validation_reserve_opened", False):
            raise RuntimeError(f"prior-boundary-failed:{key}")
    identity = json.loads(paths["r126_summary"].read_text(encoding="utf-8"))["decisions"]
    if any(value["identity_status"] != "new_sequence" for value in identity.values()):
        raise RuntimeError("candidate-identity-not-new")

    mat = scipy.io.loadmat(paths["proxy9"], variable_names=["years", "observations"])
    years = np.asarray(mat["years"], dtype=int).reshape(-1)
    proxy_all = np.asarray(mat["observations"], dtype=float)
    if not np.array_equal(years, base.YEARS) or proxy_all.shape != (61, 297):
        raise RuntimeError("proxy9-contract-failed")
    ledger = pd.read_csv(paths["ledger"], low_memory=False)
    p9 = ledger[ledger.network.eq("proxy9")].copy()
    p9["network_record_1based"] = p9.network_record_1based.astype(int)
    p9 = p9.sort_values("network_record_1based")
    selected = (
        p9.documented_network_role_v4.eq("documented_moisture")
        & p9.operator_metadata_ready_v4.astype(str).str.lower().eq("true")
    ).to_numpy()
    if len(p9) != 297 or int(selected.sum()) != 25:
        raise RuntimeError("moisture-selection-contract-failed")
    proxy = proxy_all[:, selected]
    ganesh, ga = r125.parse_noaa_candidate(paths["ganesh"], years)
    keylong, ka = r125.parse_noaa_candidate(paths["keylong"], years)
    if ga["locked_1921_1939_values_decoded"] or ka["locked_1921_1939_values_decoded"]:
        raise RuntimeError("locked-candidate-value-decoded")
    pd.DataFrame({"year": years, "ganesh_d18o": ganesh, "keylong_d18o": keylong}).to_csv(
        output / "new_proxy_values_1940_2000.csv", index=False
    )

    fields = np.load(paths["r117_fields"])
    support = np.asarray(fields["observed_support"], dtype=bool)
    hma_mask = np.asarray(fields["hma_mask"], dtype=bool)
    grid_lat = np.asarray(fields["grid_lat"], dtype=float)
    grid_lon = np.asarray(fields["grid_lon"], dtype=float)
    legacy = np.asarray(fields["equal_six_legacy"], dtype=float)
    observations = {name: np.asarray(fields[f"observation_{name}"], dtype=float) for name in ("CRU", "GPCC")}
    weights = {name: values * support for name, values in common.region_weights(grid_lat, grid_lon, hma_mask).items()}
    consensus = np.mean(np.stack([values[support] for values in observations.values()]), axis=0)
    target = consensus - legacy[support]
    area_weights = weights["hma"][support]

    matrices = {
        "plus_ganesh": np.column_stack([proxy, ganesh]),
        "plus_keylong": np.column_stack([proxy, keylong]),
        "plus_both": np.column_stack([proxy, ganesh, keylong]),
    }
    candidates = {}
    selections = []
    inner = []
    for branch, matrix in matrices.items():
        candidate, current_selections, current_inner = helpers.run_branch(
            branch, matrix, base, residual, common, target, legacy, observations, weights, support, area_weights
        )
        candidates[branch] = candidate
        selections.extend(current_selections)
        inner.extend(current_inner)
    control = np.asarray(fields["candidate"], dtype=float)
    branches = {
        **candidates,
        "round117_control": control,
        "equal_six_legacy": legacy,
        "zero_anomaly": np.zeros_like(legacy),
    }
    pairs, blocks, curves, spatial = base.score_all(branches, observations, weights, common)
    evaluated = ["round117_control", "plus_ganesh", "plus_keylong", "plus_both"]
    metrics = {}
    gates = {}
    for branch in evaluated:
        metrics[branch], gates[branch] = helpers.branch_metrics(branch, pairs, blocks, spatial)
    baseline = metrics["round117_control"]
    deltas = {
        branch: {name: value - baseline[name] for name, value in current.items()}
        for branch, current in metrics.items()
        if branch != "round117_control"
    }
    combined_passes = bool(all(gates["plus_both"].values()))
    pd.DataFrame(selections).to_csv(output / "outer_fold_selection.csv", index=False)
    pd.DataFrame(inner).to_csv(output / "inner_scores.csv", index=False)
    pairs.to_csv(output / "regional_product_scores.csv", index=False)
    blocks.to_csv(output / "block_scores.csv", index=False)
    spatial.to_csv(output / "spatial_scores.csv", index=False)
    curves.to_csv(output / "regional_curves.csv", index=False)

    hma = curves[curves.region.eq("hma")]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    colors = {"round117_control": "#777777", "plus_ganesh": "#0072B2", "plus_keylong": "#009E73", "plus_both": "#D55E00", "equal_six_legacy": "#CC79A7"}
    for branch in ["equal_six_legacy", *evaluated]:
        values = hma[(hma.kind == "reconstruction") & (hma.series == branch)]
        ax.plot(values.year, values.value_degC, color=colors[branch], linewidth=1.6, label=branch)
    obs = hma[hma.kind.eq("observation")].groupby("year", as_index=False).value_degC.mean()
    ax.plot(obs.year, obs.value_degC, color="#111111", linewidth=1.8, label="CRU-GPCC mean")
    ax.axhline(0, color="#888888", linewidth=0.7)
    ax.set_ylabel("HMA precipitation anomaly (mm/day)")
    ax.set_title("Ganesh and Keylong incremental precipitation OOF")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(output / "round127_gk_pr_oof.png", dpi=220)
    plt.close(fig)

    after = {name: base.file_state(path) for name, path in paths.items()}
    unchanged = bool(before == after)
    if not unchanged:
        raise RuntimeError("source-changed-during-round127")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2), encoding="utf-8"
    )
    summary = {
        "experiment_id": "round127_ganesh_keylong_precipitation_incremental_oof_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "decision": "advance_combined_branch" if combined_passes else "close_proxy_count_as_repair_of_current_global_low_rank_method",
        "combined_branch_passes_all_required_modern_gates": combined_passes,
        "locked_1921_1939_accessed": False,
        "locked_1921_1939_candidate_values_decoded": 0,
        "external_validation_reserve_opened": False,
        "source_hashes_unchanged": unchanged,
        "metrics": metrics,
        "metric_deltas_vs_round117": deltas,
        "gates": gates,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Round 127 Ganesh and Keylong Precipitation Incremental OOF", "", f"- Decision: `{summary['decision']}`.", f"- Combined branch passes all gates: `{str(combined_passes).lower()}`.", "- Locked candidate values decoded: `0`.", ""]
    for branch in evaluated:
        item = metrics[branch]
        lines.extend([f"## {branch}", "", f"- Gates: `{sum(gates[branch].values())}/{len(gates[branch])}`.", f"- Median RMSE improvement: `{item['median_rmse_improvement_over_equal_six']:.3%}`.", f"- Worst pair improvement: `{item['worst_pair_rmse_improvement_over_equal_six']:.3%}`.", f"- Median spatial skill: `{item['median_spatial_residual_skill']:.3%}`.", ""])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
