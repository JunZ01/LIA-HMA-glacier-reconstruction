"""Separate fold-5 operator and 1981-2000 proxy-network composition effects."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-load-failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trend(values: np.ndarray, years: np.ndarray) -> float:
    return float(100.0 * np.polyfit(years, values, 1)[0])


def main() -> int:
    root = project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_fold5_fixed_network_v1"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    output.mkdir(parents=True)
    archive = Path("D:/Users/JUNZ/Desktop/文件整理/1 论文/work1/compared_result_all")
    paths = {
        "operators": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_blocked_cv_v1/blocked_cv_operators.mat",
        "baseline_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_controlled_enkf_early20_v1/run_controlled_enkf.py",
        "evaluation_driver": root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/climate_reconstruction_jja_temperature_network_repair_v1/run_experiment.py",
        "prior": root / "validation_data/results/paleo_surface_validation_v0/enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": archive / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy9": archive / "python_data/proxy_data_9.mat",
        "archived_regem": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_proxy9_regem_replay_v1/regem_output.mat",
        "baseline_arrays": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_controlled_enkf_early20_v1/controlled_enkf_attribution_arrays.npz",
        "hma_mask": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "instrumental_netcdf": Path("E:/3spyderProject/HMA_500/Down_scale_new/out_data/climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
    }
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    base = load_module("fixed_network_base", paths["baseline_driver"])
    evaluation = load_module("fixed_network_eval", paths["evaluation_driver"])
    with h5py.File(paths["operators"], "r") as handle:
        hxb = np.asarray(handle["hxb"], dtype=float).transpose(2, 1, 0)

    years = np.arange(1000, 2001)
    selected = (years >= 1981) & (years <= 2000)
    selected_years = years[selected]
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        samples = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["archived_regem"])["R0v_filled"], dtype=float)
    common = np.isfinite(proxy["observations"][:, selected]).all(axis=1)
    common_indices = np.flatnonzero(common)
    if len(common_indices) != 125:
        raise RuntimeError(f"unexpected-common-network-count:{len(common_indices)}")

    grid_lat, grid_lon, state_lat, _ = base._state_coordinates()
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
    mask05 = np.load(paths["hma_mask"])
    mask2 = evaluation.source_grid(mask05[None].astype(float), source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    state_mask = mask2.T.reshape(-1)
    regional_weights = np.cos(np.deg2rad(state_lat)) * state_mask
    regional_weights /= regional_weights.sum()
    target_weights = base._target_weights(grid_lat, grid_lon, 33.75, 75.75)

    branches = {}
    for operator_name, operator_index in (("archived", 0), ("fold5", 6)):
        for network_name in ("dynamic", "common125"):
            item = {key: np.asarray(value).copy() for key, value in proxy.items()}
            if operator_name == "fold5":
                item["hxb"] = proxy["hxb"] + (hxb[:, operator_index, :] - hxb[:, 0, :])
            if network_name == "common125":
                item["observations"][~common, :] = np.nan
            branch = f"{operator_name}_{network_name}"
            branches[branch] = base._run_branch(
                prior, samples, item, residuals, target_weights, regional_weights
            )

    with np.load(paths["baseline_arrays"]) as saved:
        baseline_state_closure = float(
            np.max(
                np.abs(
                    np.asarray(branches["archived_dynamic"]["analysis"])
                    - saved["proxy9_analysis"]
                )
            )
        )
    enkf_closure = max(
        float(result[key])
        for result in branches.values()
        for key in (
            "state_increment_closure_max_abs_c",
            "target_increment_closure_max_abs_c",
            "regional_increment_closure_max_abs_c",
        )
    )
    if baseline_state_closure > 1e-12 or enkf_closure > 1e-10:
        raise RuntimeError(f"enkf-closure-failed:{baseline_state_closure}:{enkf_closure}")

    regional = {
        name: np.asarray(result["regional_analysis"])[selected]
        for name, result in branches.items()
    }
    row_data = []
    for name, values in regional.items():
        row_data.append(
            {
                "branch": name,
                "operator": name.split("_")[0],
                "network": name.split("_")[1],
                "record_count": 125 if name.endswith("common125") else "year_varying",
                "mean_c": float(np.mean(values)),
                "trend_c_per_century": trend(values, selected_years),
                "std_c": float(np.std(values)),
            }
        )
    summary_table = pd.DataFrame(row_data)
    summary_table.to_csv(output / "fixed_network_branch_summary.csv", index=False)

    archived_dynamic = regional["archived_dynamic"]
    fold5_dynamic = regional["fold5_dynamic"]
    archived_common = regional["archived_common125"]
    fold5_common = regional["fold5_common125"]
    decomposition = {
        "operator_effect_dynamic_network_c_per_century": trend(
            fold5_dynamic - archived_dynamic, selected_years
        ),
        "operator_effect_common125_c_per_century": trend(
            fold5_common - archived_common, selected_years
        ),
        "network_effect_archived_operator_c_per_century": trend(
            archived_dynamic - archived_common, selected_years
        ),
        "network_effect_fold5_operator_c_per_century": trend(
            fold5_dynamic - fold5_common, selected_years
        ),
    }
    decomposition["operator_network_interaction_c_per_century"] = (
        decomposition["operator_effect_dynamic_network_c_per_century"]
        - decomposition["operator_effect_common125_c_per_century"]
    )
    decomposition_closure = abs(
        decomposition["network_effect_fold5_operator_c_per_century"]
        - decomposition["network_effect_archived_operator_c_per_century"]
        - decomposition["operator_network_interaction_c_per_century"]
    )
    if decomposition_closure > 1e-10:
        raise RuntimeError(f"decomposition-closure-failed:{decomposition_closure}")

    pd.DataFrame(
        {"year": selected_years, **regional}
    ).to_csv(output / "fixed_network_regional_series_1981_2000.csv", index=False)
    pd.DataFrame(
        {
            "proxy9_record_1based": common_indices + 1,
            "latitude": proxy["locations"][common, 0],
            "longitude": proxy["locations"][common, 1],
            "proxy_type": proxy["proxy_types"][common],
        }
    ).to_csv(output / "common125_records.csv", index=False)

    figure, axis = plt.subplots(figsize=(11.4, 6.8), constrained_layout=True)
    styles = {
        "archived_dynamic": ("#111111", "-"),
        "fold5_dynamic": ("#b2182b", "-"),
        "archived_common125": ("#555555", "--"),
        "fold5_common125": ("#007f8b", "--"),
    }
    for name, values in regional.items():
        color, linestyle = styles[name]
        axis.plot(selected_years, values - values.mean(), color=color, ls=linestyle, lw=2.0, label=name)
    axis.axhline(0.0, color="#777777", lw=0.7)
    axis.grid(alpha=0.18)
    axis.set_xlabel("Year")
    axis.set_ylabel("HMA temperature contribution (degC, 1981-2000 centered)")
    axis.set_title("Fold-5 operator effect under dynamic and fixed 125-record networks")
    axis.legend(frameon=False, ncol=2)
    figure.savefig(output / "figure_fold5_fixed_network_curves.png", dpi=220)
    plt.close(figure)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    summary = {
        "experiment_id": "climate_reconstruction_proxy9_fold5_fixed_network_v1",
        "status": "pass" if pure else "fail",
        "common_network_record_count": int(len(common_indices)),
        "baseline_state_closure_max_abs_c": baseline_state_closure,
        "enkf_closure_max_abs_c": enkf_closure,
        "decomposition_closure_c_per_century": decomposition_closure,
        "source_hashes_unchanged": pure,
        "branches": summary_table.to_dict(orient="records"),
        "effect_decomposition": decomposition,
        "interpretation_boundary": "Fixed-network control only. The common 125 records are not selected as a production proxy network.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "input_manifest_before.json").write_text(json.dumps(hashes_before, indent=2), encoding="utf-8")
    (output / "input_manifest_after.json").write_text(json.dumps(hashes_after, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy9 Fold-5 Fixed-Network Control v1\n\n"
        f"Status: **{summary['status']}**. Operator and network effects close exactly under the four-branch decomposition.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not pure:
        raise RuntimeError("source-purity-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
