"""Run a new controlled proxy6/proxy9 EnKF reconstruction and attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.linalg


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_char(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(value)) for value in values if int(value) != 0).strip()


def _state_coordinates() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_lat = np.linspace(-89.0, 89.0, 90)
    all_lon = np.linspace(-179.0, 179.0, 180)
    grid_lat = all_lat[(all_lat >= 24.0) & (all_lat <= 51.0)]
    grid_lon = all_lon[(all_lon >= 62.0) & (all_lon <= 110.0)]
    state_lat = np.tile(grid_lat, len(grid_lon))
    state_lon = np.repeat(grid_lon, len(grid_lat))
    return grid_lat, grid_lon, state_lat, state_lon


def _target_weights(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> np.ndarray:
    lat_hi = int(np.searchsorted(grid_lat, target_lat))
    lon_hi = int(np.searchsorted(grid_lon, target_lon))
    lat_lo = lat_hi - 1
    lon_lo = lon_hi - 1
    fy = (target_lat - grid_lat[lat_lo]) / (grid_lat[lat_hi] - grid_lat[lat_lo])
    fx = (target_lon - grid_lon[lon_lo]) / (grid_lon[lon_hi] - grid_lon[lon_lo])
    weights = np.zeros(len(grid_lat) * len(grid_lon), dtype=float)
    weights[lon_lo * len(grid_lat) + lat_lo] = (1.0 - fx) * (1.0 - fy)
    weights[lon_lo * len(grid_lat) + lat_hi] = (1.0 - fx) * fy
    weights[lon_hi * len(grid_lat) + lat_lo] = fx * (1.0 - fy)
    weights[lon_hi * len(grid_lat) + lat_hi] = fx * fy
    return weights


def _haversine_km(lat: np.ndarray, lon: np.ndarray, target_lat: float, target_lon: float) -> np.ndarray:
    lat1 = np.deg2rad(lat)
    lat2 = np.deg2rad(target_lat)
    dlat = lat1 - lat2
    dlon = np.deg2rad(lon - target_lon)
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(value))


def _load_proxy(path: Path) -> dict[str, np.ndarray]:
    years = np.arange(1000, 2001)
    with h5py.File(path, "r") as archive:
        references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        count = len(references)
        hxb = np.empty((count, 848), dtype=float)
        observations = np.full((count, len(years)), np.nan)
        locations = np.full((count, 2), np.nan)
        coefficients = np.full((count, 3), np.nan)
        proxy_types = np.empty(count, dtype="U32")
        psm_methods = np.empty(count, dtype="U32")
        for index, reference in enumerate(references):
            proxy = archive[reference]
            hxb[index] = np.asarray(proxy["HXb"], dtype=float).reshape(-1, order="F")
            ages = np.asarray(proxy["age_yng"], dtype=float).reshape(-1, order="F")
            values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            by_year = {int(age): values[row] for row, age in enumerate(ages) if np.isfinite(age)}
            observations[index] = [by_year.get(int(year), np.nan) for year in years]
            locations[index] = np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")[:2]
            coef = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            coefficients[index, : min(3, len(coef))] = coef[:3]
            proxy_types[index] = _decode_char(proxy["ptype"])
            psm_methods[index] = _decode_char(proxy["psm_mthd"])
    return {
        "hxb": hxb,
        "observations": observations,
        "locations": locations,
        "coefficients": coefficients,
        "proxy_types": proxy_types,
        "psm_methods": psm_methods,
    }


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=window).mean().to_numpy()


def _trend(years: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(values)
    x = years[finite].astype(float)
    x -= np.mean(x)
    return float(100.0 * np.polyfit(x, values[finite], 1)[0])


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0.0 or np.std(right[valid]) == 0.0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _lag_scan(series: np.ndarray, reference: np.ndarray, years: np.ndarray, period: tuple[int, int], window: int, bounds: tuple[int, int]) -> pd.DataFrame:
    left = dict(zip(years.tolist(), _rolling(series, window).tolist()))
    right = dict(zip(years.tolist(), _rolling(reference, window).tolist()))
    rows = []
    for lag in range(bounds[0], bounds[1] + 1):
        pairs = [(left[year], right[year + lag]) for year in range(period[0], period[1] + 1) if period[0] <= year + lag <= period[1] and np.isfinite(left[year]) and np.isfinite(right[year + lag])]
        x, y = zip(*pairs) if pairs else ([], [])
        rows.append({"lag_years": lag, "pair_count": len(pairs), "correlation": _correlation(np.asarray(x), np.asarray(y))})
    return pd.DataFrame(rows)


def _anomaly(values: np.ndarray, years: np.ndarray, period: tuple[int, int]) -> np.ndarray:
    selected = (years >= period[0]) & (years <= period[1])
    return values - np.mean(values[..., selected], axis=-1, keepdims=True)


def _phase_projection(values: np.ndarray, reference: np.ndarray, window: int) -> float:
    smooth_values = _rolling(values, window)
    smooth_reference = _rolling(reference, window)
    valid = np.isfinite(smooth_values) & np.isfinite(smooth_reference)
    x = smooth_values[valid] - np.mean(smooth_values[valid])
    y = smooth_reference[valid] - np.mean(smooth_reference[valid])
    denominator = float(y @ y)
    return float(x @ y / denominator) if denominator > 0.0 else float("nan")


def _run_branch(
    prior: np.ndarray,
    sample_indices: np.ndarray,
    proxy: dict[str, np.ndarray],
    residuals: np.ndarray,
    target_weights: np.ndarray,
    regional_weights: np.ndarray,
) -> dict[str, np.ndarray | float]:
    background = prior[:, sample_indices]
    hxb = proxy["hxb"][:, sample_indices]
    observations = proxy["observations"]
    members = background.shape[1]
    background_mean = np.mean(background, axis=1)
    background_prime = background - background_mean[:, None]
    hxb_mean = np.mean(hxb, axis=1)
    hxb_prime = hxb - hxb_mean[:, None]
    pbht = background_prime @ hxb_prime.T / (members - 1)
    hpbh = hxb_prime @ hxb_prime.T / (members - 1)
    error_variance = np.var(residuals, axis=1, ddof=1)
    analysis = np.repeat(background_mean[:, None], observations.shape[1], axis=1)
    target_contributions = np.zeros_like(observations)
    regional_contributions = np.zeros_like(observations)
    solved_weights = np.zeros_like(observations)
    innovations = np.full_like(observations, np.nan)
    target_pbht = target_weights @ pbht
    regional_pbht = regional_weights @ pbht
    groups: dict[tuple[int, ...], list[int]] = {}
    for year_index in range(observations.shape[1]):
        active = tuple(np.flatnonzero(np.isfinite(observations[:, year_index])))
        groups.setdefault(active, []).append(year_index)
    state_closure = 0.0
    for active_tuple, year_indices_list in groups.items():
        if not active_tuple:
            continue
        active = np.asarray(active_tuple, dtype=int)
        year_indices = np.asarray(year_indices_list, dtype=int)
        covariance = hpbh[np.ix_(active, active)] + np.diag(error_variance[active])
        innovation = observations[np.ix_(active, year_indices)] - hxb_mean[active, None]
        try:
            factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
            weights = scipy.linalg.cho_solve(factor, innovation, check_finite=False)
        except np.linalg.LinAlgError:
            weights = scipy.linalg.solve(covariance, innovation, assume_a="sym", check_finite=False)
        state_terms = pbht[:, active, None] * weights[None, :, :]
        increment = np.sum(state_terms, axis=1)
        analysis[:, year_indices] += increment
        state_closure = max(state_closure, float(np.max(np.abs(increment - pbht[:, active] @ weights))))
        target_contributions[np.ix_(active, year_indices)] = target_pbht[active, None] * weights
        regional_contributions[np.ix_(active, year_indices)] = regional_pbht[active, None] * weights
        solved_weights[np.ix_(active, year_indices)] = weights
        innovations[np.ix_(active, year_indices)] = innovation
    target_background = float(target_weights @ background_mean)
    regional_background = float(regional_weights @ background_mean)
    target_analysis = target_weights @ analysis
    regional_analysis = regional_weights @ analysis
    target_closure = float(np.max(np.abs(target_analysis - (target_background + np.sum(target_contributions, axis=0)))))
    regional_closure = float(np.max(np.abs(regional_analysis - (regional_background + np.sum(regional_contributions, axis=0)))))
    return {
        "analysis": analysis,
        "target_analysis": target_analysis,
        "regional_analysis": regional_analysis,
        "target_background": target_background,
        "regional_background": regional_background,
        "target_contributions": target_contributions,
        "regional_contributions": regional_contributions,
        "weights": solved_weights,
        "innovations": innovations,
        "state_increment_closure_max_abs_c": state_closure,
        "target_increment_closure_max_abs_c": target_closure,
        "regional_increment_closure_max_abs_c": regional_closure,
    }


def _ranking(
    branch: str,
    years: np.ndarray,
    result: dict[str, np.ndarray | float],
    proxy: dict[str, np.ndarray],
    reference: np.ndarray,
    target_lat: float,
    target_lon: float,
    period: tuple[int, int],
    window: int,
) -> pd.DataFrame:
    selected = (years >= period[0]) & (years <= period[1])
    xyears = years[selected]
    contributions = np.asarray(result["target_contributions"])[:, selected]
    reference_selected = reference[selected]
    distances = _haversine_km(proxy["locations"][:, 0], proxy["locations"][:, 1], target_lat, target_lon)
    rows = []
    for index, values in enumerate(contributions):
        rows.append({
            "branch": branch,
            "proxy_index_1based": index + 1,
            "latitude": float(proxy["locations"][index, 0]),
            "longitude": float(proxy["locations"][index, 1]),
            "distance_to_3584_km": float(distances[index]),
            "proxy_type": str(proxy["proxy_types"][index]),
            "psm_method": str(proxy["psm_methods"][index]),
            "active_year_count_1901_1950": int(np.isfinite(np.asarray(result["innovations"])[index, selected]).sum()),
            "mean_innovation_1901_1950": float(np.nanmean(np.asarray(result["innovations"])[index, selected])) if np.isfinite(np.asarray(result["innovations"])[index, selected]).any() else float("nan"),
            "target_trend_contribution_c_per_century": _trend(xyears, values),
            "rolling15_reference_projection": _phase_projection(values, reference_selected, window),
            "target_increment_std_c": float(np.std(values)),
            "b0": float(proxy["coefficients"][index, 0]),
            "b1": float(proxy["coefficients"][index, 1]),
            "b2": float(proxy["coefficients"][index, 2]),
        })
    return pd.DataFrame(rows)


def _metrics(name: str, years: np.ndarray, values: np.ndarray, reference: np.ndarray, period: tuple[int, int], window: int, lag_bounds: tuple[int, int]) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    selected = (years >= period[0]) & (years <= period[1])
    x = values[selected]
    y = reference[selected]
    smooth_x = _rolling(x, window)
    smooth_y = _rolling(y, window)
    lag = _lag_scan(values, reference, years, period, window, lag_bounds)
    best = lag.loc[lag.correlation.idxmax()]
    return ({
        "series": name,
        "trend_c_per_century": _trend(years[selected], x),
        "reference_trend_c_per_century": _trend(years[selected], y),
        "annual_correlation": _correlation(x, y),
        "rolling15_correlation": _correlation(smooth_x, smooth_y),
        "best_lag_years": int(best.lag_years),
        "best_lag_correlation": float(best.correlation),
        "best_lag_pair_count": int(best.pair_count),
    }, lag.assign(series=name))


def _plot_curves(years: np.ndarray, series: dict[str, np.ndarray], period: tuple[int, int], plot_period: tuple[int, int], window: int, output: Path) -> None:
    selected = (years >= plot_period[0]) & (years <= plot_period[1])
    center = (years >= period[0]) & (years <= period[1])
    colors = {"rec_ref": "#2166ac", "controlled_proxy6": "#7b3294", "controlled_proxy9": "#008837", "controlled_mean": "#111111"}
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), sharex=True, constrained_layout=True)
    for name, values in series.items():
        centered = values - np.mean(values[center])
        axes[0].plot(years[selected], centered[selected], lw=2.1 if name in {"rec_ref", "controlled_mean"} else 1.5, color=colors[name], label=name)
        smooth = _rolling(centered, window)
        axes[1].plot(years[selected], smooth[selected], lw=2.3, color=colors[name], label=name)
    for axis in axes:
        axis.axvspan(period[0], period[1], color="#d9d9d9", alpha=0.25)
        axis.axhline(0.0, color="#777777", lw=0.7)
        axis.grid(alpha=0.2)
        axis.set_ylabel("Temperature (degC)")
        axis.legend(frameon=False, ncol=2)
    axes[0].set_title("3584 controlled experiment-1 annual curves (centered on 1901-1950)")
    axes[1].set_title(f"Centered {window}-year mean")
    axes[1].set_xlabel("Year")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def _plot_top_contributions(years: np.ndarray, rankings: pd.DataFrame, results: dict[str, dict[str, np.ndarray | float]], period: tuple[int, int], window: int, output: Path) -> None:
    selected = (years >= period[0] - window) & (years <= period[1] + window)
    figure, axes = plt.subplots(2, 1, figsize=(13.0, 9.0), sharex=True, constrained_layout=True)
    for axis, branch in zip(axes, ("proxy6", "proxy9")):
        branch_table = rankings[rankings.branch.eq(branch)].copy()
        chosen = branch_table.reindex(branch_table.target_trend_contribution_c_per_century.abs().sort_values(ascending=False).index).head(8)
        contributions = np.asarray(results[branch]["target_contributions"])
        for _, row in chosen.iterrows():
            index = int(row.proxy_index_1based) - 1
            values = contributions[index]
            centered = values - np.mean(values[(years >= period[0]) & (years <= period[1])])
            label = f"#{index + 1} {row.proxy_type} {row.target_trend_contribution_c_per_century:+.3f} C/century"
            axis.plot(years[selected], _rolling(centered, window)[selected], lw=1.4, label=label)
        axis.axvspan(period[0], period[1], color="#d9d9d9", alpha=0.25)
        axis.axhline(0.0, color="#666666", lw=0.7)
        axis.grid(alpha=0.2)
        axis.set_ylabel("Target increment (degC)")
        axis.set_title(f"{branch}: largest absolute 1901-1950 trend contributions")
        axis.legend(frameon=False, ncol=2, fontsize=8)
    axes[1].set_xlabel("Year")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    args = parser.parse_args()
    root = _project_root()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    archive_root = Path(config["archive_root"])
    source = config["sources"]
    paths = {
        "prior": root / source["prior"],
        "indices": archive_root / source["indices"],
        "proxy6": archive_root / source["proxy6"],
        "proxy9": archive_root / source["proxy9"],
        "proxy6_regem": root / source["proxy6_regem"],
        "proxy9_regem": root / source["proxy9_regem"],
        "proxy6_matlab_replay": root / source["proxy6_matlab_replay"],
        "proxy9_matlab_replay": root / source["proxy9_matlab_replay"],
        "reference_netcdf": Path(source["reference_netcdf"]),
    }
    output = root / config["result_root"]
    output.mkdir(parents=True, exist_ok=True)
    hashes_before = {name: _sha256(path) for name, path in paths.items()}

    years = np.arange(1000, 2001)
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as archive:
        sample_indices = np.asarray(archive["all_Xb_indices"], dtype=int)[0] - 1
    proxies = {"proxy6": _load_proxy(paths["proxy6"]), "proxy9": _load_proxy(paths["proxy9"])}
    residuals = {
        "proxy6": np.asarray(scipy.io.loadmat(paths["proxy6_regem"])["R0v_filled"], dtype=float),
        "proxy9": np.asarray(scipy.io.loadmat(paths["proxy9_regem"])["R0v_filled"], dtype=float),
    }
    grid_lat, grid_lon, state_lat, state_lon = _state_coordinates()
    target_lat = float(config["target"]["latitude"])
    target_lon = float(config["target"]["longitude"])
    target_weights = _target_weights(grid_lat, grid_lon, target_lat, target_lon)
    regional_weights = np.cos(np.deg2rad(state_lat))
    regional_weights /= regional_weights.sum()

    results = {
        branch: _run_branch(prior, sample_indices, proxies[branch], residuals[branch], target_weights, regional_weights)
        for branch in ("proxy6", "proxy9")
    }
    matlab = {
        "proxy6": scipy.io.loadmat(paths["proxy6_matlab_replay"]),
        "proxy9": scipy.io.loadmat(paths["proxy9_matlab_replay"]),
    }
    python_matlab = {
        branch: float(np.max(np.abs(np.asarray(results[branch]["analysis"]) - np.asarray(matlab[branch]["Xa_m"], dtype=float))))
        for branch in ("proxy6", "proxy9")
    }

    with h5py.File(paths["reference_netcdf"], "r") as dataset:
        latitudes = np.asarray(dataset["lat"], dtype=float)
        longitudes = np.asarray(dataset["lon"], dtype=float)
        lat_index = int(np.flatnonzero(np.isclose(latitudes, target_lat, atol=1e-10))[0])
        lon_index = int(np.flatnonzero(np.isclose(longitudes, target_lon, atol=1e-10))[0])
        reference = np.asarray(
            dataset[source["reference_variable"]][:, lat_index, lon_index], dtype=float
        )
    reference_period = tuple(config["periods"]["reference"])
    early_period = tuple(config["periods"]["early_20c"])
    plot_period = tuple(config["periods"]["plot"])
    window = int(config["phase"]["lowpass_window_years"])
    lag_bounds = tuple(config["phase"]["lag_years"])
    target_anomaly = {branch: _anomaly(np.asarray(results[branch]["target_analysis"]), years, reference_period) for branch in results}
    controlled_mean = 0.5 * (target_anomaly["proxy6"] + target_anomaly["proxy9"])
    series = {"rec_ref": reference, "controlled_proxy6": target_anomaly["proxy6"], "controlled_proxy9": target_anomaly["proxy9"], "controlled_mean": controlled_mean}

    ranking_frames = [
        _ranking(branch, years, results[branch], proxies[branch], reference, target_lat, target_lon, early_period, window)
        for branch in ("proxy6", "proxy9")
    ]
    rankings = pd.concat(ranking_frames, ignore_index=True)
    rankings.to_csv(output / "proxy_contribution_ranking_1901_1950.csv", index=False)

    metric_rows = []
    lag_frames = []
    for name in ("controlled_proxy6", "controlled_proxy9", "controlled_mean"):
        metric, lag = _metrics(name, years, series[name], reference, early_period, window, lag_bounds)
        metric_rows.append(metric)
        lag_frames.append(lag)
    metrics = pd.DataFrame(metric_rows)
    lag_table = pd.concat(lag_frames, ignore_index=True)
    metrics.to_csv(output / "target_3584_phase_metrics.csv", index=False)
    lag_table.to_csv(output / "target_3584_lag_scan.csv", index=False)

    contribution_closure = {}
    for branch in ("proxy6", "proxy9"):
        selected = (years >= early_period[0]) & (years <= early_period[1])
        analysis_trend = _trend(years[selected], target_anomaly[branch][selected])
        contribution_trend = float(rankings.loc[rankings.branch.eq(branch), "target_trend_contribution_c_per_century"].sum())
        analysis_phase = _phase_projection(target_anomaly[branch][selected], reference[selected], window)
        contribution_phase = float(rankings.loc[rankings.branch.eq(branch), "rolling15_reference_projection"].sum())
        contribution_closure[branch] = {
            "analysis_trend_c_per_century": analysis_trend,
            "sum_proxy_trend_c_per_century": contribution_trend,
            "trend_contribution_closure_c_per_century": abs(analysis_trend - contribution_trend),
            "analysis_rolling15_reference_projection": analysis_phase,
            "sum_proxy_rolling15_reference_projection": contribution_phase,
            "phase_projection_closure": abs(analysis_phase - contribution_phase),
        }

    _plot_curves(years, series, early_period, plot_period, window, output / "target_3584_controlled_curves.png")
    _plot_top_contributions(years, rankings, results, early_period, window, output / "target_3584_top_proxy_contribution_curves.png")
    np.savez_compressed(
        output / "controlled_enkf_attribution_arrays.npz",
        years=years,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        state_lat=state_lat,
        state_lon=state_lon,
        target_weights=target_weights,
        regional_weights=regional_weights,
        proxy6_analysis=results["proxy6"]["analysis"],
        proxy9_analysis=results["proxy9"]["analysis"],
        proxy6_target_contributions=results["proxy6"]["target_contributions"],
        proxy9_target_contributions=results["proxy9"]["target_contributions"],
        proxy6_regional_contributions=results["proxy6"]["regional_contributions"],
        proxy9_regional_contributions=results["proxy9"]["regional_contributions"],
        proxy6_weights=results["proxy6"]["weights"],
        proxy9_weights=results["proxy9"]["weights"],
        proxy6_innovations=results["proxy6"]["innovations"],
        proxy9_innovations=results["proxy9"]["innovations"],
        rec_ref_target=reference,
    )

    gates = config["gates"]
    gate_values: dict[str, float] = {}
    for branch in ("proxy6", "proxy9"):
        gate_values[f"{branch}_python_matlab_state_max_abs_c"] = python_matlab[branch]
        for key in ("state_increment_closure_max_abs_c", "target_increment_closure_max_abs_c", "regional_increment_closure_max_abs_c"):
            gate_values[f"{branch}_{key}"] = float(results[branch][key])
        gate_values[f"{branch}_trend_contribution_closure_c_per_century"] = contribution_closure[branch]["trend_contribution_closure_c_per_century"]
        gate_values[f"{branch}_phase_projection_closure"] = contribution_closure[branch]["phase_projection_closure"]
    passed = all(
        value <= float(gates[next(name for name in gates if name in key)])
        for key, value in gate_values.items()
    )
    hashes_after = {name: _sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    passed = passed and pure

    summary = {
        "experiment_id": config["experiment_id"],
        "status": "pass" if passed else "fail",
        "scientific_scope": config["scope"],
        "interpretation": "New internally closed fixed-prior reconstruction; not an exact attribution of the historical proxy9 archive and not a final climate product.",
        "runtime": {"python": sys.executable, "python_version": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__},
        "closure": {"gate_values": gate_values, "gates": gates, "contribution_closure": contribution_closure},
        "target_3584_metrics": metrics.to_dict(orient="records"),
        "top_negative_trend_contributors": rankings.nsmallest(10, "target_trend_contribution_c_per_century").to_dict(orient="records"),
        "top_positive_trend_contributors": rankings.nlargest(10, "target_trend_contribution_c_per_century").to_dict(orient="records"),
        "source_purity": {"unchanged": pure, "hashes_before": hashes_before, "hashes_after": hashes_after},
        "files": {"curves": "target_3584_controlled_curves.png", "contribution_curves": "target_3584_top_proxy_contribution_curves.png", "ranking": "proxy_contribution_ranking_1901_1950.csv", "arrays": "controlled_enkf_attribution_arrays.npz"},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    final = f"""# Controlled EnKF Early-20th-Century Result

Status: **{summary['status']}**.

This is a new fixed-prior experiment-1 reconstruction. It is internally
decomposable and does not claim to reproduce the historical proxy9 archive.
No OGGM run or glacier tuning was performed; `rec_ref` is comparison-only.

## 3584, 1901-1950

{metrics.to_markdown(index=False)}

## Closure

All source files unchanged: **{pure}**.

Maximum Python-to-MATLAB state differences are
`{python_matlab['proxy6']:.3e}` degC for proxy6 and
`{python_matlab['proxy9']:.3e}` degC for proxy9. Per-proxy increments close
at state, target, cosine-weighted HMA region, trend, and linear low-frequency
projection according to `summary.json`.

## Boundary

The proxy ranking is an exact simultaneous-update algebraic decomposition only
within this new controlled fixed-prior experiment. Because proxy observations
share covariance, individual terms are not independent physical causal effects.
This is not yet a 50-experiment reconstruction product and must not be used as
OGGM forcing without a separately approved expansion and validation.
"""
    (output / "FINAL_RESULT.md").write_text(final, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "closure": gate_values, "metrics": summary["target_3584_metrics"]}, indent=2, ensure_ascii=True))
    if not passed:
        raise RuntimeError("controlled-enkf-hard-gate-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
