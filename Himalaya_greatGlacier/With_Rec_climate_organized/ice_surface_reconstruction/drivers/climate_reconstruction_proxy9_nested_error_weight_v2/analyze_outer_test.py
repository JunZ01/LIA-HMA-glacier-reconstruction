"""Evaluate frozen nested-CV error inflation on untouched 1981-2000 data."""

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
    output = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_proxy9_nested_error_weight_v2"
    )
    output.mkdir(parents=True, exist_ok=True)
    archive = Path(
        "D:/Users/JUNZ/Desktop/文件整理/1 论文/work1/compared_result_all"
    )
    paths = {
        "nested_cv_frozen": output / "nested_cv_error_frozen.mat",
        "design_audit": output / "design_rank_audit.mat",
        "record205_diagnosis": output / "record205_fold3_design_diagnosis_v2.mat",
        "nested_cv_v1": root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_proxy9_nested_error_weight_v1/nested_cv_error.mat",
        "fold5_operators": root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_proxy9_blocked_cv_v1/blocked_cv_operators.mat",
        "fold5_attribution": root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_proxy9_blocked_cv_fold5_attribution_v1/"
        "proxy9_fold5_contribution_attribution.csv",
        "baseline_driver": root
        / "Himalaya_greatGlacier/With_Rec_climate_organized/"
        "ice_surface_reconstruction/drivers/"
        "climate_reconstruction_controlled_enkf_early20_v1/"
        "run_controlled_enkf.py",
        "evaluation_driver": root
        / "Himalaya_greatGlacier/With_Rec_climate_organized/"
        "ice_surface_reconstruction/drivers/"
        "climate_reconstruction_jja_temperature_network_repair_v1/"
        "run_experiment.py",
        "prior": root
        / "validation_data/results/paleo_surface_validation_v0/"
        "enkf_reconstruction_source_recovery_audit_v2/"
        "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": archive
        / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "proxy9": archive / "python_data/proxy_data_9.mat",
        "archived_regem": root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_proxy9_regem_replay_v1/regem_output.mat",
        "baseline_arrays": root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_controlled_enkf_early20_v1/"
        "controlled_enkf_attribution_arrays.npz",
        "hma_mask": root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_input_spatial_audit_v3/"
        "hma_mask_on_netcdf_grid.npy",
        "instrumental_netcdf": Path(
            "E:/3spyderProject/HMA_500/Down_scale_new/out_data/"
            "climate_anomaly_05deg_with_ensemble_1000-2000.nc"
        ),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing-inputs:{missing}")
    hashes_before = {name: sha256(path) for name, path in paths.items()}

    base = load_module("nested_weight_v2_base", paths["baseline_driver"])
    evaluation = load_module("nested_weight_v2_eval", paths["evaluation_driver"])
    with h5py.File(paths["nested_cv_frozen"], "r") as handle:
        eligible_nested_mse = np.asarray(
            handle["eligible_nested_mse"], dtype=float
        ).squeeze()
        nested_mse = np.asarray(handle["nested_mse"], dtype=float).squeeze()
        prediction_count = np.asarray(
            handle["prediction_count"], dtype=float
        ).squeeze()
        valid_fold_count = np.asarray(
            handle["valid_fold_count"], dtype=float
        ).squeeze()
        fallback = np.asarray(
            handle["fallback_to_archived_variance"], dtype=bool
        ).squeeze()
        rank_deficient = np.asarray(
            handle["rank_deficient"], dtype=bool
        ).transpose()
    with h5py.File(paths["nested_cv_v1"], "r") as handle:
        nested_mse_v1 = np.asarray(handle["nested_mse"], dtype=float).squeeze()
    with h5py.File(paths["fold5_operators"], "r") as handle:
        hxb = np.asarray(handle["hxb"], dtype=float).transpose(2, 1, 0)

    fallback_records = np.flatnonzero(fallback) + 1
    rank_positions = np.argwhere(rank_deficient) + 1
    if not np.array_equal(fallback_records, np.array([205])):
        raise RuntimeError(f"unexpected-fallback-records:{fallback_records.tolist()}")
    if not np.array_equal(rank_positions, np.array([[205, 3]])):
        raise RuntimeError(f"unexpected-rank-pattern:{rank_positions.tolist()}")
    eligible = ~fallback
    if np.min(valid_fold_count[eligible]) < 3:
        raise RuntimeError("eligible-record-below-three-valid-folds")
    nested_replay_closure = float(
        np.max(np.abs(eligible_nested_mse[eligible] - nested_mse_v1[eligible]))
    )
    if nested_replay_closure > 1e-12:
        raise RuntimeError(f"nested-replay-closure-failed:{nested_replay_closure}")

    years = np.arange(1000, 2001)
    selected = (years >= 1981) & (years <= 2000)
    selected_years = years[selected]
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        samples = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(
        scipy.io.loadmat(paths["archived_regem"])["R0v_filled"], dtype=float
    )
    archived_variance = np.var(residuals, axis=1, ddof=1)
    if np.any(~np.isfinite(archived_variance)) or np.any(archived_variance <= 0):
        raise RuntimeError("invalid-archived-variance")

    target_variance = archived_variance.copy()
    target_variance[eligible] = np.maximum(
        archived_variance[eligible], eligible_nested_mse[eligible]
    )
    if not np.array_equal(target_variance[fallback], archived_variance[fallback]):
        raise RuntimeError("fallback-variance-changed")
    if np.any(target_variance < archived_variance):
        raise RuntimeError("variance-reduction-not-allowed")
    scale = np.sqrt(target_variance / archived_variance)
    inflated_residuals = residuals * scale[:, None]
    realized_variance = np.var(inflated_residuals, axis=1, ddof=1)
    variance_closure = float(np.max(np.abs(realized_variance - target_variance)))
    if variance_closure > 1e-10:
        raise RuntimeError(f"variance-scaling-closure-failed:{variance_closure}")

    grid_lat, grid_lon, state_lat, _ = base._state_coordinates()
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
    mask05 = np.load(paths["hma_mask"])
    mask2 = (
        evaluation.source_grid(
            mask05[None].astype(float),
            source_lat,
            source_lon,
            grid_lat,
            grid_lon,
        )[0]
        >= 0.5
    )
    state_mask = mask2.T.reshape(-1)
    regional_weights = np.cos(np.deg2rad(state_lat)) * state_mask
    regional_weights /= regional_weights.sum()
    target_weights = base._target_weights(grid_lat, grid_lon, 33.75, 75.75)

    archived_proxy = {key: np.asarray(value).copy() for key, value in proxy.items()}
    fold5_proxy = {key: np.asarray(value).copy() for key, value in proxy.items()}
    fold5_proxy["hxb"] = proxy["hxb"] + (hxb[:, 6, :] - hxb[:, 0, :])
    branches = {
        "archived_operator_archived_R": base._run_branch(
            prior,
            samples,
            archived_proxy,
            residuals,
            target_weights,
            regional_weights,
        ),
        "archived_operator_nested_R": base._run_branch(
            prior,
            samples,
            archived_proxy,
            inflated_residuals,
            target_weights,
            regional_weights,
        ),
        "fold5_operator_archived_R": base._run_branch(
            prior,
            samples,
            fold5_proxy,
            residuals,
            target_weights,
            regional_weights,
        ),
        "fold5_operator_nested_R": base._run_branch(
            prior,
            samples,
            fold5_proxy,
            inflated_residuals,
            target_weights,
            regional_weights,
        ),
    }
    with np.load(paths["baseline_arrays"]) as saved:
        baseline_state_closure = float(
            np.max(
                np.abs(
                    np.asarray(branches["archived_operator_archived_R"]["analysis"])
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
        raise RuntimeError(
            f"enkf-closure-failed:{baseline_state_closure}:{enkf_closure}"
        )

    regional = {
        name: np.asarray(result["regional_analysis"])[selected]
        for name, result in branches.items()
    }
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        observations = {}
        for label, variable in {
            "Berkeley Earth": "tas_obs_bea_05deg",
            "CRU": "tas_obs_cru_tmp_05deg",
            "GISTEMP": "tas_obs_GIST_05deg",
            "HadCRUT5": "tas_obs_had5_05deg",
        }.items():
            values = np.asarray(handle[variable], dtype=float)
            values[values < -9000] = np.nan
            field = evaluation.anomaly(
                evaluation.source_grid(
                    values, source_lat, source_lon, grid_lat, grid_lon
                ),
                years,
                (1961, 1990),
            )
            observations[label] = evaluation.hma_mean(
                field, grid_lat, mask2
            )[selected]
    observation_mean = np.mean(np.stack(list(observations.values())), axis=0)

    observed_centered = observation_mean - observation_mean.mean()
    rows = []
    for name, values in regional.items():
        centered = values - values.mean()
        rows.append(
            {
                "branch": name,
                "trend_c_per_century": trend(values, selected_years),
                "correlation_to_four_product_mean": float(
                    np.corrcoef(centered, observed_centered)[0, 1]
                ),
                "centered_rmse_to_four_product_mean_c": float(
                    np.sqrt(np.mean((centered - observed_centered) ** 2))
                ),
                "std_c": float(np.std(values)),
            }
        )
    branch_summary = pd.DataFrame(rows)
    branch_summary.to_csv(output / "nested_error_branch_summary.csv", index=False)

    attribution = pd.read_csv(paths["fold5_attribution"])
    reliability = pd.DataFrame(
        {
            "proxy9_record_1based": np.arange(1, 298),
            "archived_variance": archived_variance,
            "raw_nested_cv_mse": nested_mse,
            "eligible_nested_cv_mse": eligible_nested_mse,
            "nested_prediction_count": prediction_count,
            "nested_valid_fold_count": valid_fold_count,
            "fallback_to_archived_variance": fallback,
            "target_variance": target_variance,
            "variance_inflation_factor": target_variance / archived_variance,
            "variance_was_inflated": target_variance > archived_variance,
        }
    ).merge(
        attribution[
            [
                "proxy9_record_1based",
                "combined_name",
                "group",
                "trend_contribution_change_c_per_century",
                "fold5_holdout_rmse",
                "fold5_holdout_correlation",
            ]
        ],
        on="proxy9_record_1based",
        how="left",
    )
    reliability.to_csv(output / "proxy9_nested_error_weights.csv", index=False)

    colors = {
        "archived_operator_archived_R": "#1f77b4",
        "archived_operator_nested_R": "#2ca02c",
        "fold5_operator_archived_R": "#d62728",
        "fold5_operator_nested_R": "#9467bd",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), constrained_layout=True)
    for name, values in regional.items():
        axes[0].plot(
            selected_years,
            values - values.mean(),
            lw=1.9,
            color=colors[name],
            label=name,
        )
    axes[0].plot(
        selected_years,
        observed_centered,
        color="#333333",
        lw=2.4,
        label="four-product mean",
    )
    axes[0].axhline(0.0, color="#777777", lw=0.7)
    axes[0].grid(alpha=0.18)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Centered HMA temperature (degC)")
    axes[0].set_title("Untouched 1981-2000 outer test")
    axes[0].legend(frameon=False, fontsize=7.3)

    finite = np.isfinite(eligible_nested_mse)
    axes[1].scatter(
        archived_variance[finite],
        eligible_nested_mse[finite],
        c=np.where(
            target_variance[finite] > archived_variance[finite],
            "#b2182b",
            "#2166ac",
        ),
        s=24,
        alpha=0.7,
    )
    limit = float(
        np.nanpercentile(
            np.r_[archived_variance[finite], eligible_nested_mse[finite]], 99
        )
    )
    axes[1].plot([0, limit], [0, limit], color="#777777", lw=0.8)
    axes[1].set_xlim(0, limit)
    axes[1].set_ylim(0, limit)
    axes[1].grid(alpha=0.18)
    axes[1].set_xlabel("Archived RegEM variance")
    axes[1].set_ylabel("Eligible nested 1901-1980 CV MSE")
    axes[1].set_title("Frozen record-wise inflation rule")
    figure.savefig(output / "figure_nested_error_outer_test.png", dpi=220)
    plt.close(figure)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    source_hashes_unchanged = hashes_before == hashes_after
    summary = {
        "experiment_id": "climate_reconstruction_proxy9_nested_error_weight_v2",
        "date": "2026-08-13",
        "status": "pass" if source_hashes_unchanged else "fail",
        "outer_test_years": [1981, 2000],
        "nested_training_years": [1901, 1980],
        "eligible_nested_weight_records": int(eligible.sum()),
        "fallback_to_archived_variance_records_1based": fallback_records.tolist(),
        "rank_deficient_record_fold_1based": rank_positions.tolist(),
        "nested_replay_closure_except_fallback": nested_replay_closure,
        "baseline_state_closure_max_abs_c": baseline_state_closure,
        "enkf_closure_max_abs_c": enkf_closure,
        "variance_scaling_closure": variance_closure,
        "source_hashes_unchanged": source_hashes_unchanged,
        "records_with_inflated_variance": int(
            np.sum(target_variance > archived_variance)
        ),
        "median_variance_inflation_factor": float(
            np.median(target_variance / archived_variance)
        ),
        "p90_variance_inflation_factor": float(
            np.quantile(target_variance / archived_variance, 0.9)
        ),
        "four_product_mean_trend_c_per_century": trend(
            observation_mean, selected_years
        ),
        "branches": branch_summary.to_dict(orient="records"),
        "interpretation_boundary": (
            "Weights were frozen from 1901-1980 only. The 1981-2000 interval "
            "is an untouched outer test. No production error model is selected "
            "automatically, and rec_ref is not used."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "input_manifest_before.json").write_text(
        json.dumps(hashes_before, indent=2), encoding="utf-8"
    )
    (output / "input_manifest_after.json").write_text(
        json.dumps(hashes_after, indent=2), encoding="utf-8"
    )
    if not source_hashes_unchanged:
        raise RuntimeError("source-purity-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
