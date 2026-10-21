"""Run frozen EnKF operator branches for the western 1861-1900 increment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io

import attribute_temperature_west_increment_1861_1900_v2 as launch


base = launch.experiment
ROOT = base.ROOT
LOOP = base.LOOP
RESULTS = base.RESULTS
OUTPUT = RESULTS / "climate_reconstruction_temperature_west_bivariate_operator_decomposition_1861_1900_v1"
YEARS = base.YEARS
BRANCHES = ("baseline", "combined_rebuild", "temperature_only", "precipitation_only", "zero_bivariate_anomaly")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    prepare = json.loads((OUTPUT / "prepare_summary.json").read_text(encoding="utf-8"))
    if not prepare["source_hashes_unchanged"]:
        raise RuntimeError("Preparation source-integrity gate failed")
    manifest27 = LOOP / "round_27_temperature_innovation_gain_sign_decomposition_v1/source_manifest.json"
    recovered = json.loads(manifest27.read_text(encoding="utf-8"))["sources_before"]
    paths = {
        "prior": Path(recovered["prior"]["path"]),
        "indices": Path(recovered["indices"]["path"]),
        "proxy6": Path(recovered["proxy6"]["path"]),
        "proxy9": Path(recovered["proxy9"]["path"]),
        "regem6": Path(recovered["regem6"]["path"]),
        "regem9": Path(recovered["regem9"]["path"]),
        "reference_grid": LOOP / "round_21_temperature_reference_field_v1/temperature_reference_family_balanced_allowed_v1.nc",
        "hma_mask": RESULTS / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "baseline_attribution": RESULTS / "climate_reconstruction_temperature_west_increment_attribution_1861_1900_v2/annual_west_increment_closure_1861_1900.csv",
        "components6": OUTPUT / "proxy6_bivariate_operator_components.npz",
        "components9": OUTPUT / "proxy9_bivariate_operator_components.npz",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in paths.items()}
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        pools = np.asarray(handle["all_Xb_indices"], dtype=int) - 1
    target, _, _ = base.western_target(prior, paths["reference_grid"], paths["hma_mask"])
    archived = pd.read_csv(paths["baseline_attribution"]).set_index("year").loc[YEARS]
    annual_rows = []
    proxy_rows = []
    network_summaries = {}
    all_closures = []

    for network in (6, 9):
        hxb, observations, locked_reads = base.round25.read_proxy_allowed(paths[f"proxy{network}"], YEARS)
        if locked_reads:
            raise RuntimeError("Locked values read")
        residuals = np.asarray(scipy.io.loadmat(paths[f"regem{network}"])["R0v_filled"], dtype=float)
        error_variance = np.nanvar(residuals, axis=1, ddof=1)
        with np.load(paths[f"components{network}"], allow_pickle=False) as component_file:
            components = {name: np.asarray(component_file[name]) for name in component_file.files}
        selected = components["selected_indices_0based"].astype(int)
        selected_mean = np.mean(hxb[selected], axis=1, keepdims=True)
        branch_operators = {}
        branch_operators["baseline"] = hxb.copy()
        branch_operators["combined_rebuild"] = hxb.copy()
        branch_operators["temperature_only"] = hxb.copy()
        branch_operators["precipitation_only"] = hxb.copy()
        branch_operators["zero_bivariate_anomaly"] = hxb.copy()
        branch_operators["combined_rebuild"][selected] = selected_mean + components["full_anomaly"]
        branch_operators["temperature_only"][selected] = selected_mean + components["temperature_anomaly"]
        branch_operators["precipitation_only"][selected] = selected_mean + components["precipitation_anomaly"]
        branch_operators["zero_bivariate_anomaly"][selected] = selected_mean
        combined_closure = float(np.max(np.abs(branch_operators["combined_rebuild"] - hxb)))
        mean_invariance = max(float(np.max(np.abs(np.mean(operator, axis=1) - np.mean(hxb, axis=1)))) for operator in branch_operators.values())
        if combined_closure > 1e-12 or mean_invariance > 1e-12:
            raise RuntimeError(f"proxy{network} operator closure failed")

        curves = {}
        branch_contributions = {}
        branch_gains = {}
        for branch in BRANCHES:
            contribution, gains, closure = base.run_contributions(branch_operators[branch], observations, error_variance, pools, target)
            all_closures.append(closure)
            curves[branch] = np.sum(contribution, axis=0)
            branch_contributions[branch] = contribution
            branch_gains[branch] = gains
            for year_index, year in enumerate(YEARS):
                annual_rows.append({
                    "network": network,
                    "branch": branch,
                    "year": int(year),
                    "west_increment_centered_degC": float(curves[branch][year_index]),
                    "selected_bivariate_contribution_degC": float(np.sum(contribution[selected, year_index])),
                    "nonselected_contribution_degC": float(np.sum(contribution[np.setdiff1d(np.arange(len(contribution)), selected), year_index])),
                })
            for proxy_index in range(len(contribution)):
                proxy_rows.append({
                    "network": network,
                    "branch": branch,
                    "record_index_1based": proxy_index + 1,
                    "selected_bivariate": bool(proxy_index in set(selected.tolist())),
                    "trend_contribution_degC_per_century": base.trend(contribution[proxy_index]),
                    "mean_gain": float(np.mean(gains[proxy_index])),
                    "active_year_count": int(np.isfinite(observations[proxy_index]).sum()),
                })

        archived_curve = archived[f"proxy{network}_recovered_increment_degC"].to_numpy(float)
        baseline_closure = float(np.max(np.abs(curves["baseline"] - archived_curve)))
        combined_curve_closure = float(np.max(np.abs(curves["combined_rebuild"] - curves["baseline"])))
        if baseline_closure > 1e-12 or combined_curve_closure > 1e-12:
            raise RuntimeError(f"proxy{network} baseline reproduction failed")
        branch_trends = {branch: base.trend(curves[branch]) for branch in BRANCHES}
        zero = branch_trends["zero_bivariate_anomaly"]
        temperature_effect = branch_trends["temperature_only"] - zero
        precipitation_effect = branch_trends["precipitation_only"] - zero
        total_effect = branch_trends["baseline"] - zero
        nonadditivity = total_effect - temperature_effect - precipitation_effect
        innovation_invariance = max(
            float(np.max(np.abs(np.mean(branch_operators[branch], axis=1) - np.mean(hxb, axis=1))))
            for branch in BRANCHES
        )
        network_summaries[str(network)] = {
            "selected_bivariate_count": int(len(selected)),
            "combined_operator_max_abs_error": combined_closure,
            "operator_mean_invariance_max_abs_error": mean_invariance,
            "baseline_curve_max_abs_error_degC": baseline_closure,
            "combined_curve_max_abs_error_degC": combined_curve_closure,
            "innovation_invariance_max_abs_error": innovation_invariance,
            "branch_trends_degC_per_century": branch_trends,
            "effects_relative_to_zero_degC_per_century": {
                "total_bivariate_operator_effect": total_effect,
                "temperature_operator_effect": temperature_effect,
                "precipitation_operator_effect": precipitation_effect,
                "nonadditivity": nonadditivity,
            },
        }
        np.savez_compressed(
            OUTPUT / f"proxy{network}_branch_arrays.npz",
            years=YEARS,
            selected_indices_0based=selected,
            **{f"curve_{branch}": curves[branch] for branch in BRANCHES},
            **{f"contribution_{branch}": branch_contributions[branch] for branch in BRANCHES},
            **{f"gain_{branch}": branch_gains[branch] for branch in BRANCHES},
        )

    pd.DataFrame(annual_rows).to_csv(OUTPUT / "operator_branch_annual_west_increment_1861_1900.csv", index=False)
    pd.DataFrame(proxy_rows).to_csv(OUTPUT / "operator_branch_proxy_trend_contributions.csv", index=False)
    after = {name: sha256(path) for name, path in paths.items()}
    summary = {
        "status": "complete",
        "period": [1861, 1900],
        "branch_names": list(BRANCHES),
        "maximum_algebraic_closure_degC": max(all_closures),
        "networks": network_summaries,
        "input_hashes_unchanged": before == after,
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
        "oggm_run": False,
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "run_source_manifest.json").write_text(json.dumps({name: {"path": str(paths[name]), "sha256": before[name]} for name in paths}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
