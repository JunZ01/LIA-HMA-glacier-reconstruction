"""Separate precipitation EnKF update trends into innovation and gain behavior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


EXPERIMENT_ID = "climate_reconstruction_precipitation_fixed_support_innovation_gain_decomposition_1901_1920_v1"
YEARS = np.arange(1901, 1921, dtype=int)
LOCKED_PERIOD = (1921, 1939)
REGIONS = {
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_90_100": (90.0, 100.0),
}
REGION_LABELS = {"west_lt80": "<80E", "central_80_90": "80-90E", "east_90_100": "90-100E"}


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest(path),
    }


def vector(handle: h5py.File, name: str) -> np.ndarray:
    return np.asarray(handle[name], dtype=float).reshape(-1)


def trend(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if int(valid.sum()) < 3:
        return float("nan")
    return float(np.polyfit(YEARS[valid], values[valid], 1)[0] * 100.0)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def bracket(values: np.ndarray, target: float) -> tuple[int, int, float]:
    if target <= values[0]:
        return 0, 0, 0.0
    if target >= values[-1]:
        last = len(values) - 1
        return last, last, 0.0
    upper = int(np.searchsorted(values, target, side="right"))
    lower = upper - 1
    fraction = float((target - values[lower]) / (values[upper] - values[lower]))
    return lower, upper, fraction


def support_weight_vector(
    support: pd.DataFrame,
    native_lat: np.ndarray,
    native_lon: np.ndarray,
    lower_lon: float,
    upper_lon: float,
) -> tuple[np.ndarray, int]:
    selected = support[
        support["joint_complete"].astype(bool)
        & (support["cru_station_month_fraction_1901_1920"] > 0)
        & (support["gpcc_direct_gauge_month_fraction_1901_1920"] > 0)
        & (support["longitude"] >= lower_lon)
        & (support["longitude"] < upper_lon)
    ]
    weights = np.zeros(len(native_lat) * len(native_lon), dtype=float)
    for row in selected.itertuples(index=False):
        lat0, lat1, fy = bracket(native_lat, float(row.latitude))
        lon0, lon1, fx = bracket(native_lon, float(row.longitude))
        area = float(np.cos(np.deg2rad(row.latitude)))
        combinations = (
            (lat0, lon0, (1.0 - fy) * (1.0 - fx)),
            (lat1, lon0, fy * (1.0 - fx)),
            (lat0, lon1, (1.0 - fy) * fx),
            (lat1, lon1, fy * fx),
        )
        for ilat, ilon, fraction in combinations:
            weights[ilat + len(native_lat) * ilon] += area * fraction
    if len(selected) == 0 or weights.sum() <= 0:
        raise RuntimeError(f"empty-support-region:{lower_lon}:{upper_lon}")
    weights /= weights.sum()
    return weights, int(len(selected))


def main() -> int:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "prior": result_root / "climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "replay_input": result_root / "climate_reconstruction_precipitation_enkf_exp1_replay_v1/replay_input.mat",
        "attribution": result_root / "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/attribution_output.mat",
        "native_posterior": result_root / "climate_reconstruction_precipitation_other_positive_b2_longitude_block_ablation_v1/longitude_block_ablation.mat",
        "support_grid": result_root / "climate_reconstruction_precipitation_reference_gauge_support_1901_1920_v1/hma_grid_reference_support_1901_1920.csv",
        "fixed_support_curves": result_root / "climate_reconstruction_precipitation_reference_support_by_longitude_1901_1920_v2/longitude_support_curves_v1_reproduced.csv",
        "fixed_support_summary": result_root / "climate_reconstruction_precipitation_reference_support_by_longitude_1901_1920_v2/summary.json",
        "locked_contract": result_root / "climate_reconstruction_precipitation_instrumental_baseline_v1/LOCKED_FINAL_PERIOD.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    if np.any((YEARS >= LOCKED_PERIOD[0]) & (YEARS <= LOCKED_PERIOD[1])):
        raise RuntimeError("locked-period-requested")
    locked = json.loads(sources["locked_contract"].read_text(encoding="utf-8"))
    support_summary = json.loads(sources["fixed_support_summary"].read_text(encoding="utf-8"))
    if locked["status"] != "locked_not_evaluated" or support_summary["locked_1921_1939_accessed"]:
        raise RuntimeError("locked-period-contract-failed")
    before = {name: source_state(path) for name, path in sources.items()}

    replay = loadmat(
        sources["replay_input"],
        variable_names=["sample_indices", "HXb0", "y0", "R0v_filled"],
    )
    samples = np.asarray(replay["sample_indices"], dtype=int).reshape(-1) - 1
    hxb = np.asarray(replay["HXb0"], dtype=float)
    observations = np.asarray(replay["y0"], dtype=float)
    residuals = np.asarray(replay["R0v_filled"], dtype=float)
    if hxb.shape != (202, 500) or observations.shape != (202, 1001) or residuals.shape[0] != 202:
        raise RuntimeError(f"replay-shape-contract-failed:{hxb.shape}:{observations.shape}:{residuals.shape}")
    target_columns = YEARS - 1000
    target_observations = observations[:, target_columns].T
    if not np.all(np.isfinite(target_observations)):
        raise RuntimeError("active-proxy-network-is-not-constant-and-complete")

    with h5py.File(sources["prior"], "r") as handle:
        prior_members = np.asarray(handle["precipitation_prior_336x848_mm_day"], dtype=float)
        native_lat = vector(handle, "lat")
        native_lon = vector(handle, "lon")
    xb = prior_members[samples, :].T
    xb_mean = np.mean(xb, axis=1)
    xp = xb - xb_mean[:, None]
    hxb_mean = np.mean(hxb, axis=1)
    hxp = hxb - hxb_mean[:, None]
    innovations = target_observations - hxb_mean[None, :]
    nens = xb.shape[1]
    cross_covariance = (xp @ hxp.T) / (nens - 1)
    proxy_covariance = (hxp @ hxp.T) / (nens - 1)
    rdiag = np.nanvar(residuals, axis=1, ddof=1)
    if not np.all(np.isfinite(rdiag)) or np.any(rdiag <= 0):
        raise RuntimeError("R-diagonal-contract-failed")
    gain = np.linalg.solve(proxy_covariance + np.diag(rdiag), cross_covariance.T).T

    with h5py.File(sources["attribution"], "r") as handle:
        archived_active_count = vector(handle, "active_proxy_count")
        archived_gain_weight = np.asarray(handle["gain_weight"], dtype=float)
        archived_innovation = np.asarray(handle["innovation"], dtype=float)
        archived_total_increment = vector(handle, "total_increment")
        archived_state_weights = vector(handle, "state_weights")
    hma_gain_weight = archived_state_weights @ gain
    hma_gain_closure = float(np.max(np.abs(archived_gain_weight - hma_gain_weight[None, :])))
    innovation_closure = float(np.max(np.abs(archived_innovation - innovations)))
    hma_update = innovations @ hma_gain_weight
    hma_update_closure = float(np.max(np.abs(hma_update - archived_total_increment)))
    archived_gain_yearly_variation = float(np.max(np.abs(archived_gain_weight - archived_gain_weight[0, :])))
    if (
        not np.all(archived_active_count == 202)
        or max(hma_gain_closure, innovation_closure, hma_update_closure) > 1e-11
        or archived_gain_yearly_variation != 0.0
    ):
        raise RuntimeError("archived-innovation-gain-closure-failed")

    with h5py.File(sources["native_posterior"], "r") as handle:
        posterior_state = np.asarray(handle["posterior_mean"][0, :, :], dtype=float)
        native_years = vector(handle, "years").astype(int)
    if posterior_state.shape != (20, 336) or not np.array_equal(native_years, YEARS):
        raise RuntimeError("native-posterior-contract-failed")

    support = pd.read_csv(sources["support_grid"])
    fixed_curves = pd.read_csv(sources["fixed_support_curves"])
    if not np.array_equal(fixed_curves["year"].to_numpy(dtype=int), YEARS):
        raise RuntimeError("fixed-support-year-contract-failed")
    annual = pd.DataFrame({"year": YEARS})
    score_rows = []
    contribution_rows = []
    for region, (lower_lon, upper_lon) in REGIONS.items():
        weights, count = support_weight_vector(support, native_lat, native_lon, lower_lon, upper_lon)
        expected_count = int(next(row["gridcell_count"] for row in support_summary["eligible_supported_regions"] if row["longitude_region"] == region))
        if count != expected_count:
            raise RuntimeError(f"support-count-closure-failed:{region}:{count}:{expected_count}")
        effective_gain = weights @ gain
        update = innovations @ effective_gain
        prior = float(weights @ xb_mean)
        posterior = posterior_state @ weights
        posterior_update_closure = float(np.max(np.abs(posterior - prior - update)))
        if posterior_update_closure > 1e-11:
            raise RuntimeError(f"regional-posterior-update-closure-failed:{region}:{posterior_update_closure}")

        archived = fixed_curves[f"{region}_dual_any_rec_result_mm_day"].to_numpy(dtype=float)
        update_change = update - update[0]
        posterior_change = posterior - posterior[0]
        archived_change = archived - archived[0]
        prior_change = np.zeros_like(update)
        annual[f"{region}_archived_05deg_change_from_1901_mm_day"] = archived_change
        annual[f"{region}_native_posterior_change_from_1901_mm_day"] = posterior_change
        annual[f"{region}_innovation_update_change_from_1901_mm_day"] = update_change
        annual[f"{region}_gain_change_component_mm_day"] = np.zeros_like(update)
        annual[f"{region}_prior_change_from_1901_mm_day"] = prior_change

        contribution = innovations * effective_gain[None, :]
        contribution_trends = np.asarray([trend(contribution[:, proxy]) for proxy in range(contribution.shape[1])])
        for proxy_index, value in enumerate(contribution_trends, start=1):
            contribution_rows.append({
                "longitude_region": region,
                "proxy_index_matlab": proxy_index,
                "effective_gain_weight": float(effective_gain[proxy_index - 1]),
                "innovation_contribution_trend_mm_day_per_century": float(value),
            })
        score_rows.append({
            "longitude_region": region,
            "dual_any_gridcell_count": count,
            "active_proxy_count_each_year": 202,
            "effective_gain_yearly_max_abs_variation": 0.0,
            "innovation_update_trend_mm_day_per_century": trend(update),
            "gain_change_trend_component_mm_day_per_century": 0.0,
            "interaction_trend_component_mm_day_per_century": 0.0,
            "native_posterior_trend_mm_day_per_century": trend(posterior),
            "archived_05deg_fixed_support_trend_mm_day_per_century": trend(archived),
            "native_vs_archived_centered_correlation": correlation(posterior_change, archived_change),
            "native_vs_archived_centered_rmse_mm_day": float(np.sqrt(np.mean(np.square(posterior_change - archived_change)))),
            "native_minus_archived_trend_mm_day_per_century": trend(posterior) - trend(archived),
            "posterior_update_closure_max_abs_error_mm_day": posterior_update_closure,
            "proxy_contribution_trend_sum_closure_mm_day_per_century": float(np.sum(contribution_trends) - trend(update)),
            "temporal_update_variation_attributed_to_innovations_fraction": 1.0,
        })
    scores = pd.DataFrame(score_rows)
    proxy_contributions = pd.DataFrame(contribution_rows)

    output.mkdir(parents=True)
    annual.to_csv(output / "fixed_support_innovation_gain_curves_1901_1920.csv", index=False)
    scores.to_csv(output / "fixed_support_innovation_gain_scores.csv", index=False)
    proxy_contributions.to_csv(output / "regional_proxy_innovation_contribution_trends.csv", index=False)

    colors = {"archived": "#D55E00", "native": "#009E73", "prior": "#0072B2", "gain": "#CC79A7"}
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), sharex=True, constrained_layout=True)
    for axis, region in zip(axes.reshape(-1)[:3], REGIONS):
        axis.plot(YEARS, annual[f"{region}_archived_05deg_change_from_1901_mm_day"], color=colors["archived"], linewidth=2.2, label="Archived 0.5-deg reconstruction")
        axis.plot(YEARS, annual[f"{region}_innovation_update_change_from_1901_mm_day"], color=colors["native"], linewidth=1.5, linestyle="--", label="Innovation through fixed gain")
        axis.plot(YEARS, annual[f"{region}_prior_change_from_1901_mm_day"], color=colors["prior"], linewidth=1.2, linestyle=":", label="Prior")
        axis.plot(YEARS, annual[f"{region}_gain_change_component_mm_day"], color=colors["gain"], linewidth=1.0, linestyle="-.", label="Gain change")
        count = int(scores.loc[scores["longitude_region"] == region, "dual_any_gridcell_count"].iloc[0])
        axis.set_title(f"{REGION_LABELS[region]} fixed dual support (n={count})")
        axis.set_ylabel("Change from 1901 (mm/day)")
        axis.axhline(0, color="#B0B0B0", linewidth=0.7)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Year")
    axes[0, 0].legend(frameon=False, ncol=2)

    axis = axes[1, 1]
    x = np.arange(len(scores))
    width = 0.25
    axis.bar(x - width, scores["archived_05deg_fixed_support_trend_mm_day_per_century"], width, color=colors["archived"], label="Archived 0.5-deg")
    axis.bar(x, scores["innovation_update_trend_mm_day_per_century"], width, color=colors["native"], label="Innovation")
    axis.bar(x + width, scores["gain_change_trend_component_mm_day_per_century"], width, color=colors["gain"], label="Gain change")
    axis.set_xticks(x, [REGION_LABELS[name] for name in scores["longitude_region"]])
    axis.set_ylabel("Trend (mm/day/century)")
    axis.set_title("Innovation versus gain-change trend")
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.savefig(output / "fixed_support_innovation_gain_decomposition_curves.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "active_proxy_count_each_year": archived_active_count.astype(int).tolist(),
        "gain_stationarity": {
            "archived_hma_gain_yearly_max_abs_variation": archived_gain_yearly_variation,
            "full_gain_matrix_is_time_invariant_by_design": True,
        },
        "closure": {
            "python_vs_matlab_hma_gain_weight_max_abs_error": hma_gain_closure,
            "python_vs_matlab_innovation_max_abs_error": innovation_closure,
            "python_vs_matlab_hma_update_max_abs_error_mm_day": hma_update_closure,
            "maximum_regional_posterior_update_error_mm_day": float(scores["posterior_update_closure_max_abs_error_mm_day"].max()),
            "maximum_proxy_contribution_trend_sum_error_mm_day_per_century": float(scores["proxy_contribution_trend_sum_closure_mm_day_per_century"].abs().max()),
        },
        "scores": scores.to_dict(orient="records"),
        "classification": "innovation_variation_drives_all_1901_1920_update_trends_gain_is_exactly_stationary",
        "interpretation": "All 202 proxies are active in every target year and the full EnKF gain matrix is time invariant. Therefore the western negative update and central/eastern weak positive updates are generated by changing proxy innovations passed through one fixed covariance/gain operator, not by year-to-year gain changes.",
        "interpretation_boundary": "The support-informed native-grid operator uses fixed bilinear area weights to connect 0.5-degree supported cells to the 2-degree EnKF state. The innovation-versus-gain conclusion is independent of this particular fixed linear spatial operator, but exact 0.5-degree amplitudes remain those in the archived output.",
    }
    after = {name: source_state(path) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "driver_sha256": digest(Path(__file__))}, indent=2),
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-innovation-gain-audit")
    lines = [
        "# Fixed-support Innovation/Gain Decomposition",
        "",
        "All 202 proxies are active in all 20 years. The full EnKF gain is exactly stationary; only innovations vary.",
        "",
    ]
    for row in scores.to_dict(orient="records"):
        lines.append(
            f"- {row['longitude_region']} n={row['dual_any_gridcell_count']}: innovation/gain-change/native-posterior trends "
            f"`{row['innovation_update_trend_mm_day_per_century']:+.3f}` / "
            f"`{row['gain_change_trend_component_mm_day_per_century']:+.3f}` / "
            f"`{row['native_posterior_trend_mm_day_per_century']:+.3f} mm/day/century`."
        )
    lines.extend([
        "",
        "Conclusion: 1901-1920 update trends come entirely from proxy innovations passed through a fixed gain. The next useful diagnosis is innovation structure by physically defined proxy groups, not gain tuning.",
    ])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
