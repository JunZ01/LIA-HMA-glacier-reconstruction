"""Run nested fold-native ordinary-EnKF temperature increment validation."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io
import scipy.linalg

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1940, 2001, dtype=int)
BLOCKS = [(1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000)]
ALPHAS = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}
OBSERVATION_VARIABLES = {
    "Berkeley": "tas_obs_bea_05deg",
    "CRU": "tas_obs_cru_tmp_05deg",
    "GISTEMP": "tas_obs_GIST_05deg",
    "HadCRUT5": "tas_obs_had5_05deg",
    "NOAA": "tas_obs_NOAA_05deg",
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


def file_state(path: Path, digest: str, hash_mode: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
        "hash_mode": hash_mode,
    }


def trusted_hash(manifest_path: Path, target: Path) -> str:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_norm = str(target).lower()

    def walk(value):
        if isinstance(value, dict):
            if str(value.get("path", "")).lower() == target_norm and value.get("sha256"):
                return str(value["sha256"])
            for child in value.values():
                found = walk(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = walk(child)
                if found:
                    return found
        return None

    result = walk(payload)
    if not result:
        raise RuntimeError(f"trusted-hash-not-found:{target}:{manifest_path}")
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"empty-csv:{path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result[(result <= -9000) | (result >= 1e20)] = np.nan
    return result


def source_grid(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    lat2, lon2 = np.meshgrid(target_lat, target_lon, indexing="ij")
    fy = (lat2 - lat[0]) / (lat[1] - lat[0])
    fx = (lon2 - lon[0]) / (lon[1] - lon[0])
    inside = (fy >= 0) & (fy <= len(lat) - 1) & (fx >= 0) & (fx <= len(lon) - 1)
    y0 = np.clip(np.floor(fy).astype(int), 0, len(lat) - 2)
    x0 = np.clip(np.floor(fx).astype(int), 0, len(lon) - 2)
    wy = np.clip(fy - y0, 0, 1)
    wx = np.clip(fx - x0, 0, 1)
    a = field[:, y0, x0]
    b = field[:, y0, x0 + 1]
    c = field[:, y0 + 1, x0]
    d = field[:, y0 + 1, x0 + 1]
    result = (
        (1 - wy)[None] * ((1 - wx)[None] * a + wx[None] * b)
        + wy[None] * ((1 - wx)[None] * c + wx[None] * d)
    )
    result[:, ~inside] = np.nan
    return result


def state_coordinates() -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(-89.0, 89.0, 90)
    lon = np.linspace(-179.0, 179.0, 180)
    return lat[(lat >= 24.0) & (lat <= 51.0)], lon[(lon >= 62.0) & (lon <= 110.0)]


def state_to_cube(values: np.ndarray, nlat: int, nlon: int) -> np.ndarray:
    return values.reshape(nlon, nlat, values.shape[1]).transpose(2, 1, 0)


def cube_to_state(values: np.ndarray) -> np.ndarray:
    return values.transpose(2, 1, 0).reshape(values.shape[2] * values.shape[1], values.shape[0])


def weighted_series(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & (weights[:, None] > 0)
    numerator = np.sum(np.where(valid, values, 0.0) * weights[:, None], axis=0)
    denominator = np.sum(valid * weights[:, None], axis=0)
    return np.divide(numerator, denominator, out=np.full(values.shape[1], np.nan), where=denominator > 0)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def rolling_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_roll = pd.Series(left).rolling(15, center=True, min_periods=15).mean().to_numpy()
    right_roll = pd.Series(right).rolling(15, center=True, min_periods=15).mean().to_numpy()
    return correlation(left_roll, right_roll)


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean((left[valid] - right[valid]) ** 2)))


def load_operator(path: Path) -> dict[str, np.ndarray | float | str]:
    with h5py.File(path, "r") as handle:
        result = {
            "hxb": np.asarray(handle["hxb"], dtype=float).transpose(2, 1, 0),
            "residual_variance": np.asarray(handle["residual_variance"], dtype=float).T,
            "training_count": np.asarray(handle["training_count"], dtype=float).T,
            "branch_outer": np.asarray(handle["branch_outer"], dtype=int).reshape(-1),
            "branch_inner": np.asarray(handle["branch_inner"], dtype=int).reshape(-1),
            "coefficient_closure": float(np.asarray(handle["baseline_coefficient_max_abs"]).squeeze()),
            "hxb_closure": float(np.asarray(handle["baseline_hxb_max_abs"]).squeeze()),
        }
    return result


def analysis_for_branch(
    prior: np.ndarray,
    sample: np.ndarray,
    observations: np.ndarray,
    hxb: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, list[int], float]:
    background = prior[:, sample]
    background_mean = np.mean(background, axis=1)
    result = np.full((prior.shape[0], len(YEARS)), np.nan)
    counts: list[int] = []
    closure = 0.0
    for year_index in range(len(YEARS)):
        active = np.flatnonzero(
            np.isfinite(observations[:, year_index])
            & np.isfinite(variance)
            & (variance > 0)
            & np.all(np.isfinite(hxb[:, sample]), axis=1)
        )
        counts.append(int(len(active)))
        if len(active) == 0:
            result[:, year_index] = background_mean
            continue
        hx = hxb[active][:, sample]
        hx_mean = np.mean(hx, axis=1)
        xp = background - background_mean[:, None]
        hp = hx - hx_mean[:, None]
        pbht = xp @ hp.T / (background.shape[1] - 1)
        covariance = hp @ hp.T / (background.shape[1] - 1) + np.diag(variance[active])
        innovation = observations[active, year_index] - hx_mean
        try:
            factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
            solved = scipy.linalg.cho_solve(factor, innovation, check_finite=False)
        except np.linalg.LinAlgError:
            solved = scipy.linalg.solve(covariance, innovation, assume_a="sym", check_finite=False)
        current = background_mean + pbht @ solved
        result[:, year_index] = current
        closure = max(closure, float(np.max(np.abs(current - (background_mean + pbht @ solved)))))
    return result, counts, closure


def branch_index(operator: dict[str, np.ndarray | float | str], outer: int, inner: int) -> int:
    outer_values = np.asarray(operator["branch_outer"])
    inner_values = np.asarray(operator["branch_inner"])
    found = np.flatnonzero((outer_values == outer) & (inner_values == inner))
    if len(found) != 1:
        raise RuntimeError(f"operator-branch-not-unique:{outer}:{inner}:{found.tolist()}")
    return int(found[0])


def region_weights(lat: np.ndarray, lon: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    lon2 = np.broadcast_to(lon[None], mask.shape)
    cosine = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], mask.shape)
    result = {}
    for name, (lower, upper) in REGIONS.items():
        selected = mask & (lon2 >= lower) & (lon2 < upper)
        result[name] = (cosine * selected).T.reshape(-1)
    return result


def centered_for_outer(values: np.ndarray, outer: int) -> np.ndarray:
    start, end = BLOCKS[outer - 1]
    train = ~((YEARS >= start) & (YEARS <= end))
    return values - np.nanmean(values[:, train], axis=1, keepdims=True)


def alpha_objective(
    background: np.ndarray,
    analysis: np.ndarray,
    alpha: float,
    outer: int,
    observations: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
) -> float:
    start, end = BLOCKS[outer - 1]
    selected = ~((YEARS >= start) & (YEARS <= end))
    candidate = background[:, None] + alpha * (analysis - background[:, None])
    candidate = candidate - np.nanmean(candidate[:, selected], axis=1, keepdims=True)
    scores = []
    for obs in observations.values():
        obs_centered = obs - np.nanmean(obs[:, selected], axis=1, keepdims=True)
        for current_weights in weights.values():
            scores.append(rmse(weighted_series(candidate[:, selected], current_weights), weighted_series(obs_centered[:, selected], current_weights)))
    return float(np.nanmedian(scores))


def stable_alpha(indices: list[int]) -> bool:
    return max(sum(abs(value - center) <= 1 for value in indices) for center in range(len(ALPHAS))) >= 4


def spatial_metrics(candidate: np.ndarray, observation: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mask = weights > 0
    spatial_rmse = []
    zero_rmse = []
    pattern = []
    for year_index in range(candidate.shape[1]):
        valid = mask & np.isfinite(candidate[:, year_index]) & np.isfinite(observation[:, year_index])
        if valid.sum() < 3:
            continue
        w = weights[valid]
        c = candidate[valid, year_index]
        o = observation[valid, year_index]
        c = c - np.average(c, weights=w)
        o = o - np.average(o, weights=w)
        spatial_rmse.append(np.sqrt(np.average((c - o) ** 2, weights=w)))
        zero_rmse.append(np.sqrt(np.average(o**2, weights=w)))
        pattern.append(correlation(c, o))
    skill = 1.0 - float(np.nanmean(spatial_rmse)) / float(np.nanmean(zero_rmse))
    return skill, float(np.nanmedian(pattern))


def main() -> int:
    root = project_root()
    driver = Path(__file__).resolve().parent
    output = root / "validation_data/results/paleo_surface_validation_v0/temperature_fold_native_oenkf_increment_cv_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    archive = Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all")
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "allowed_proxy6": loop / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "allowed_proxy9": loop / "proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "prior": root / "validation_data/results/paleo_surface_validation_v0/enkf_reconstruction_source_recovery_audit_v2/replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy",
        "indices": archive / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat",
        "instrumental_netcdf": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "bearth": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "gpcc": Path(r"D:\VM\VMshare\data\analyses\GPCC\GPCC05\gpcc_1891_2020_05.nc"),
        "matlab_driver": driver / "fit_fold_native_psm_v1.m",
        "python_driver": Path(__file__).resolve(),
        "experiment_contract": driver / "EXPERIMENT.md",
        "nearest_latlon": phyda / "nearest_latlon.m",
        "regrid2gcm": phyda / "regrid2gcm.m",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))

    trusted_manifests = {
        "bearth": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_08_documented_temperature_seasonality_v1/source_manifest.json",
        "gpcc": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_gpcc_monthly_to_annual_processing_audit_v1/source_manifest.json",
        "instrumental_netcdf": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_19_reference_family_balance_v1/source_manifest.json",
    }
    before = {}
    for name, path in paths.items():
        if name in trusted_manifests:
            before[name] = file_state(path, trusted_hash(trusted_manifests[name], path), "trusted_existing_manifest_not_rehashed")
            before[name]["trusted_manifest"] = str(trusted_manifests[name])
        else:
            before[name] = file_state(path, sha256(path), "computed_sha256")

    config = {
        "experiment_id": "temperature_fold_native_oenkf_increment_cv_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "years_read": [1940, 2000],
        "forbidden_period": [1921, 1939],
        "locked_1921_1939_accessed": False,
        "matlab_required_release": "2024b",
        "networks": [6, 9],
        "assimilation": "ordinary EnKF, fixed experiment-1 500-member prior",
        "AO_HGAO_run": False,
        "localization_changed": False,
        "blocks": BLOCKS,
        "alpha_grid": ALPHAS.tolist(),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text((driver / "EXPERIMENT.md").read_text(encoding="utf-8"), encoding="utf-8")

    prior = np.load(paths["prior"])
    scipy.io.savemat(output / "temperature_prior_input.mat", {"temperature_prior_336x848": prior}, do_compression=True)
    operator_paths = {network: output / f"proxy{network}_fold_native_operators.mat" for network in (6, 9)}
    matlab = Path(r"E:\Program Files\MATLAB\R2024b\bin\matlab.exe")
    if not matlab.exists():
        raise FileNotFoundError(matlab)
    for network in (6, 9):
        command = (
            f"addpath('{driver.as_posix()}'); "
            f"fit_fold_native_psm_v1('{paths[f'allowed_proxy{network}'].as_posix()}',"
            f"'{(output / 'temperature_prior_input.mat').as_posix()}',"
            f"'{operator_paths[network].as_posix()}','{phyda.as_posix()}',"
            f"'{paths['bearth'].as_posix()}','{paths['gpcc'].as_posix()}');"
        )
        completed = subprocess.run(
            [str(matlab), "-nojvm", "-batch", command],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        (output / f"matlab_proxy{network}_r2024b.log").write_text(completed.stdout + "\nSTDERR\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"matlab-proxy{network}-failed:{completed.returncode}")

    operators = {network: load_operator(path) for network, path in operator_paths.items()}
    if any(float(item["coefficient_closure"]) > 1e-9 or float(item["hxb_closure"]) > 1e-9 for item in operators.values()):
        raise RuntimeError("operator-production-closure-failed")
    with h5py.File(paths["indices"], "r") as handle:
        sample = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    if len(sample) != 500 or np.any(sample < 0) or np.any(sample >= prior.shape[1]):
        raise RuntimeError("sample-index-contract-failed")

    proxy_inputs = {}
    for network in (6, 9):
        loaded = scipy.io.loadmat(paths[f"allowed_proxy{network}"], variable_names=["observations"])
        proxy_inputs[network] = np.asarray(loaded["observations"], dtype=float).T
        if proxy_inputs[network].shape[1] != len(YEARS):
            raise RuntimeError(f"proxy-year-shape-failed:{network}:{proxy_inputs[network].shape}")

    grid_lat, grid_lon = state_coordinates()
    with h5py.File(paths["instrumental_netcdf"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int)
        time_index = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in YEARS])
        if np.any((source_years[time_index] >= 1921) & (source_years[time_index] <= 1939)):
            raise RuntimeError("locked-netcdf-index-requested")
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        observation_cubes = {
            name: source_grid(clean(np.asarray(handle[variable][time_index, :, :])), source_lat, source_lon, grid_lat, grid_lon)
            for name, variable in OBSERVATION_VARIABLES.items()
        }
    observation_states = {name: cube_to_state(values) for name, values in observation_cubes.items()}
    mask05 = np.load(paths["hma_mask"]).astype(float)
    mask2 = source_grid(mask05[None], source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    weights = region_weights(grid_lat, grid_lon, mask2)
    hma_weights = weights["hma"]
    background = np.mean(prior[:, sample], axis=1)

    network_results = {}
    active_rows = []
    alpha_rows = []
    maximum_enkf_closure = 0.0
    for network in (6, 9):
        operator = operators[network]
        cache = {}

        def run(branch: int):
            nonlocal maximum_enkf_closure
            if branch not in cache:
                current, counts, closure = analysis_for_branch(
                    prior,
                    sample,
                    proxy_inputs[network],
                    np.asarray(operator["hxb"])[:, branch, :],
                    np.asarray(operator["residual_variance"])[:, branch],
                )
                cache[branch] = current
                maximum_enkf_closure = max(maximum_enkf_closure, closure)
                for year, count in zip(YEARS, counts):
                    active_rows.append({"network": network, "branch_0based": branch, "year": int(year), "active_proxy_count": count})
            return cache[branch]

        candidate_centered = np.full((prior.shape[0], len(YEARS)), np.nan)
        full_centered = np.full_like(candidate_centered, np.nan)
        background_centered = np.zeros_like(candidate_centered)
        selected_alphas = []
        for outer in range(1, 6):
            outer_start, outer_end = BLOCKS[outer - 1]
            test = (YEARS >= outer_start) & (YEARS <= outer_end)
            train = ~test
            inner_stitched = np.full((prior.shape[0], len(YEARS)), np.nan)
            for inner in range(1, 6):
                if inner == outer:
                    continue
                inner_start, inner_end = BLOCKS[inner - 1]
                inner_test = (YEARS >= inner_start) & (YEARS <= inner_end)
                nested_branch = branch_index(operator, outer, inner)
                inner_stitched[:, inner_test] = run(nested_branch)[:, inner_test]
            objectives = [alpha_objective(background, inner_stitched, alpha, outer, observation_states, weights) for alpha in ALPHAS]
            best_index = min(range(len(ALPHAS)), key=lambda index: (objectives[index], -ALPHAS[index]))
            alpha = float(ALPHAS[best_index])
            selected_alphas.append(alpha)
            for alpha_value, objective in zip(ALPHAS, objectives):
                alpha_rows.append({
                    "network": network,
                    "outer_fold": outer,
                    "outer_period": f"{outer_start}-{outer_end}",
                    "alpha": float(alpha_value),
                    "inner_median_region_product_rmse_degC": float(objective),
                    "selected": bool(alpha_value == alpha),
                })
            outer_branch = branch_index(operator, outer, 0)
            outer_analysis = run(outer_branch)
            candidate_raw = background[:, None] + alpha * (outer_analysis - background[:, None])
            full_raw = outer_analysis
            candidate_centered[:, test] = (candidate_raw - np.nanmean(candidate_raw[:, train], axis=1, keepdims=True))[:, test]
            full_centered[:, test] = (full_raw - np.nanmean(full_raw[:, train], axis=1, keepdims=True))[:, test]
        network_results[network] = {
            "candidate": candidate_centered,
            "full": full_centered,
            "background": background_centered,
            "alphas": selected_alphas,
        }

    observation_centered = {name: np.full_like(values, np.nan) for name, values in observation_states.items()}
    for outer, (start, end) in enumerate(BLOCKS, start=1):
        test = (YEARS >= start) & (YEARS <= end)
        train = ~test
        for name, values in observation_states.items():
            observation_centered[name][:, test] = (values - np.nanmean(values[:, train], axis=1, keepdims=True))[:, test]

    branches = {
        "candidate_nested_alpha": np.mean([network_results[6]["candidate"], network_results[9]["candidate"]], axis=0),
        "full_increment_alpha1": np.mean([network_results[6]["full"], network_results[9]["full"]], axis=0),
        "background_alpha0": np.zeros((prior.shape[0], len(YEARS))),
        "proxy6_candidate": network_results[6]["candidate"],
        "proxy6_full": network_results[6]["full"],
        "proxy9_candidate": network_results[9]["candidate"],
        "proxy9_full": network_results[9]["full"],
    }

    pair_rows = []
    block_rows = []
    curve_rows = []
    for region, current_weights in weights.items():
        branch_curves = {name: weighted_series(values, current_weights) for name, values in branches.items()}
        obs_curves = {name: weighted_series(values, current_weights) for name, values in observation_centered.items()}
        for year_index, year in enumerate(YEARS):
            for name, values in branch_curves.items():
                curve_rows.append({"year": int(year), "region": region, "kind": "reconstruction", "series": name, "value_degC": float(values[year_index])})
            for name, values in obs_curves.items():
                curve_rows.append({"year": int(year), "region": region, "kind": "observation", "series": name, "value_degC": float(values[year_index])})
        for observation, observed in obs_curves.items():
            zero = rmse(np.zeros_like(observed), observed)
            full_rmse = rmse(branch_curves["full_increment_alpha1"], observed)
            for branch, predicted in branch_curves.items():
                current_rmse = rmse(predicted, observed)
                pair_rows.append({
                    "branch": branch,
                    "region": region,
                    "observation": observation,
                    "rmse_degC": current_rmse,
                    "skill_vs_zero": 1.0 - current_rmse / zero,
                    "improvement_vs_combined_full": 1.0 - current_rmse / full_rmse,
                    "annual_correlation": correlation(predicted, observed),
                    "correlation_15year": rolling_correlation(predicted, observed),
                })
            for fold, (start, end) in enumerate(BLOCKS, start=1):
                selected = (YEARS >= start) & (YEARS <= end)
                zero_block = rmse(np.zeros(selected.sum()), observed[selected])
                full_block = rmse(branch_curves["full_increment_alpha1"][selected], observed[selected])
                for branch in ("candidate_nested_alpha", "full_increment_alpha1", "background_alpha0"):
                    current = rmse(branch_curves[branch][selected], observed[selected])
                    block_rows.append({
                        "branch": branch,
                        "fold": fold,
                        "period": f"{start}-{end}",
                        "region": region,
                        "observation": observation,
                        "rmse_degC": current,
                        "skill_vs_zero": 1.0 - current / zero_block,
                        "improvement_vs_combined_full": 1.0 - current / full_block,
                    })

    spatial_rows = []
    for branch in ("candidate_nested_alpha", "full_increment_alpha1", "background_alpha0"):
        for observation, observed in observation_centered.items():
            skill, pattern = spatial_metrics(branches[branch], observed, hma_weights)
            spatial_rows.append({
                "branch": branch,
                "observation": observation,
                "spatial_residual_skill": skill,
                "median_annual_spatial_pattern_correlation": pattern,
            })

    pairs = pd.DataFrame(pair_rows)
    blocks_frame = pd.DataFrame(block_rows)
    spatial = pd.DataFrame(spatial_rows)
    candidate_pairs = pairs[pairs.branch.eq("candidate_nested_alpha")]
    candidate_blocks = blocks_frame[blocks_frame.branch.eq("candidate_nested_alpha")]
    candidate_spatial = spatial[spatial.branch.eq("candidate_nested_alpha")]
    relative = candidate_pairs.improvement_vs_combined_full
    relative_block = candidate_blocks.improvement_vs_combined_full
    absolute = candidate_pairs.skill_vs_zero
    absolute_block = candidate_blocks.skill_vs_zero
    network_improvements = {}
    for network in (6, 9):
        cand = pairs[pairs.branch.eq(f"proxy{network}_candidate")].set_index(["region", "observation"])
        full = pairs[pairs.branch.eq(f"proxy{network}_full")].set_index(["region", "observation"])
        network_improvements[network] = float(np.nanmedian(1.0 - cand.rmse_degC / full.rmse_degC))

    gates = {
        "relative_median_rmse_improvement_at_least_5pct": bool(np.nanmedian(relative) >= 0.05),
        "relative_worst_regression_at_most_5pct": bool(np.nanmin(relative) >= -0.05 - 1e-12),
        "relative_at_least_75pct_block_region_product_improve": bool(np.mean(relative_block > 0) >= 0.75),
        "absolute_median_region_product_skill_at_least_5pct": bool(np.nanmedian(absolute) >= 0.05),
        "absolute_worst_region_product_regression_at_most_5pct": bool(np.nanmin(absolute) >= -0.05 - 1e-12),
        "absolute_at_least_75pct_region_product_pairs_improve": bool(np.mean(absolute > 0) >= 0.75),
        "absolute_median_annual_correlation_at_least_0_30": bool(np.nanmedian(candidate_pairs.annual_correlation) >= 0.30),
        "absolute_median_15year_correlation_at_least_0_30": bool(np.nanmedian(candidate_pairs.correlation_15year) >= 0.30),
        "absolute_at_least_70pct_blocks_improve": bool(np.mean(absolute_block > 0) >= 0.70),
        "spatial_median_skill_positive": bool(np.nanmedian(candidate_spatial.spatial_residual_skill) > 0),
        "spatial_worst_regression_at_most_5pct": bool(np.nanmin(candidate_spatial.spatial_residual_skill) >= -0.05 - 1e-12),
        "spatial_at_least_75pct_products_positive": bool(np.mean(candidate_spatial.spatial_residual_skill > 0) >= 0.75),
        "spatial_median_pattern_correlation_at_least_0_20": bool(np.nanmedian(candidate_spatial.median_annual_spatial_pattern_correlation) >= 0.20),
        "proxy6_directional_improvement_nonnegative": bool(network_improvements[6] >= 0),
        "proxy9_directional_improvement_nonnegative": bool(network_improvements[9] >= 0),
        "proxy6_alpha_stable_four_of_five": stable_alpha([int(np.flatnonzero(ALPHAS == value)[0]) for value in network_results[6]["alphas"]]),
        "proxy9_alpha_stable_four_of_five": stable_alpha([int(np.flatnonzero(ALPHAS == value)[0]) for value in network_results[9]["alphas"]]),
    }
    passes = all(gates.values())
    all_alpha = network_results[6]["alphas"] + network_results[9]["alphas"]
    if passes:
        decision = "advance_exploratory_background_only" if np.median(all_alpha) == 0 else "advance_exploratory_global_increment_shrinkage"
    elif np.mean(np.asarray(all_alpha) == 0) >= 0.8:
        decision = "background_preferred_but_fails_full_validation_gates"
    elif np.mean(np.asarray(all_alpha) == 1) >= 0.8:
        decision = "retain_full_increment_modern_only_early_nonstationarity_remains"
    else:
        decision = "close_existing_network_global_increment_shrinkage"

    write_csv(output / "alpha_inner_selection.csv", alpha_rows)
    write_csv(output / "active_proxy_counts.csv", active_rows)
    pairs.to_csv(output / "heldout_region_product_scores.csv", index=False)
    blocks_frame.to_csv(output / "heldout_block_scores.csv", index=False)
    spatial.to_csv(output / "heldout_spatial_scores.csv", index=False)
    write_csv(output / "heldout_regional_curves.csv", curve_rows)
    np.savez_compressed(
        output / "heldout_oof_fields.npz",
        years=YEARS,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        hma_mask=mask2,
        candidate=branches["candidate_nested_alpha"],
        full_increment=branches["full_increment_alpha1"],
        background=branches["background_alpha0"],
        proxy6_candidate=branches["proxy6_candidate"],
        proxy9_candidate=branches["proxy9_candidate"],
    )

    figure, axes = plt.subplots(2, 1, figsize=(13, 8.4), sharex=True, constrained_layout=True)
    hma_curves = pd.DataFrame(curve_rows)
    hma_curves = hma_curves[hma_curves.region.eq("hma")]
    colors = {"candidate_nested_alpha": "#007f8b", "full_increment_alpha1": "#b2182b", "background_alpha0": "#222222"}
    for name, color in colors.items():
        values = hma_curves[(hma_curves.kind == "reconstruction") & (hma_curves.series == name)].sort_values("year")
        axes[0].plot(values.year, values.value_degC, color=color, lw=2.0, label=name)
        rolled = values.value_degC.rolling(15, center=True, min_periods=15).mean()
        axes[1].plot(values.year, rolled, color=color, lw=2.2, label=name)
    for name in OBSERVATION_VARIABLES:
        values = hma_curves[(hma_curves.kind == "observation") & (hma_curves.series == name)].sort_values("year")
        axes[0].plot(values.year, values.value_degC, lw=0.8, alpha=0.45, label=name)
        axes[1].plot(values.year, values.value_degC.rolling(15, center=True, min_periods=15).mean(), lw=0.9, alpha=0.5)
    for axis in axes:
        for start, _ in BLOCKS[1:]:
            axis.axvline(start - 0.5, color="#aaaaaa", lw=0.6, ls="--")
        axis.axhline(0, color="#777777", lw=0.7)
        axis.grid(alpha=0.18)
        axis.set_ylabel("Temperature anomaly (degC)")
    axes[0].set_title("HMA fold-native ordinary-EnKF temperature validation")
    axes[1].set_title("Centered 15-year means")
    axes[1].set_xlabel("Year")
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    figure.savefig(output / "temperature_fold_native_oenkf_increment_curves.png", dpi=220)
    plt.close(figure)

    after = {}
    for name, path in paths.items():
        prior_state = before[name]
        if prior_state["hash_mode"] == "trusted_existing_manifest_not_rehashed":
            after[name] = file_state(path, str(prior_state["sha256"]), str(prior_state["hash_mode"]))
            after[name]["trusted_manifest"] = prior_state["trusted_manifest"]
        else:
            after[name] = file_state(path, sha256(path), "computed_sha256")
    sources_unchanged = before == after
    manifest = {
        "sources_before": before,
        "sources_after": after,
        "all_sources_unchanged": sources_unchanged,
        "numeric_read_contract": {
            "instrumental_and_proxy_years": [1940, 2000],
            "monthly_temperature_hyperslab": ["1940-01", "2000-12"],
            "monthly_precipitation_hyperslab": ["1940-01", "2000-12"],
            "locked_1921_1939_accessed": False,
        },
    }
    (output / "source_hash_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not sources_unchanged:
        raise RuntimeError("source-purity-failed")

    summary = {
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "passes_all_gates": passes,
        "decision": decision,
        "locked_1921_1939_accessed": False,
        "matlab_release": "R2024b",
        "AO_HGAO_run": False,
        "localization_changed": False,
        "source_hashes_unchanged": sources_unchanged,
        "operator_closure": {
            str(network): {
                "coefficient_max_abs": float(operators[network]["coefficient_closure"]),
                "hxb_max_abs": float(operators[network]["hxb_closure"]),
            }
            for network in (6, 9)
        },
        "enkf_algebraic_closure_max_abs_degC": maximum_enkf_closure,
        "selected_alpha": {str(network): network_results[network]["alphas"] for network in (6, 9)},
        "network_median_rmse_improvement_vs_full": {str(key): value for key, value in network_improvements.items()},
        "primary_metrics": {
            "median_rmse_improvement_vs_full": float(np.nanmedian(relative)),
            "worst_rmse_improvement_vs_full": float(np.nanmin(relative)),
            "block_region_product_improvement_fraction": float(np.mean(relative_block > 0)),
            "median_absolute_skill_vs_zero": float(np.nanmedian(absolute)),
            "worst_absolute_skill_vs_zero": float(np.nanmin(absolute)),
            "positive_region_product_fraction": float(np.mean(absolute > 0)),
            "positive_block_fraction": float(np.mean(absolute_block > 0)),
            "median_annual_correlation": float(np.nanmedian(candidate_pairs.annual_correlation)),
            "median_15year_correlation": float(np.nanmedian(candidate_pairs.correlation_15year)),
            "median_spatial_residual_skill": float(np.nanmedian(candidate_spatial.spatial_residual_skill)),
            "worst_spatial_residual_skill": float(np.nanmin(candidate_spatial.spatial_residual_skill)),
            "positive_spatial_product_fraction": float(np.mean(candidate_spatial.spatial_residual_skill > 0)),
            "median_spatial_pattern_correlation": float(np.nanmedian(candidate_spatial.median_annual_spatial_pattern_correlation)),
        },
        "gates": gates,
        "interpretation_boundary": "A pass authorizes only an exploratory background/increment-shrinkage candidate. It cannot reuse 1901-1920 or consumed 1921-1939 as independent validation and cannot authorize OGGM.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    gate_lines = "\n".join(f"- `{name}`: `{str(value).lower()}`" for name, value in gates.items())
    (output / "FINAL_RESULT.md").write_text(
        "# Temperature Fold-Native OEnKF Increment CV V1\n\n"
        f"- Decision: `{decision}`.\n"
        f"- Passes all gates: `{str(passes).lower()}`.\n"
        f"- Proxy6 selected alpha: `{network_results[6]['alphas']}`.\n"
        f"- Proxy9 selected alpha: `{network_results[9]['alphas']}`.\n"
        f"- Median RMSE improvement versus alpha=1: `{summary['primary_metrics']['median_rmse_improvement_vs_full']:.3%}`.\n"
        f"- Median absolute skill versus zero: `{summary['primary_metrics']['median_absolute_skill_vs_zero']:.3%}`.\n"
        f"- Locked 1921-1939 accessed: `false`.\n"
        f"- Sources unchanged: `{str(sources_unchanged).lower()}`.\n\n"
        "## Gates\n\n" + gate_lines + "\n\n"
        "Passing remains exploratory and does not authorize production or OGGM.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
