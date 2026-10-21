"""Test 1901-1920 annual-OOF precipitation sensitivity to each selected proxy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_precipitation_nested_seasonal_tree_psm_cv_v1 as psm
import run_precipitation_annual_oof_nested_blocked_field_validation_v1 as blocked
import run_precipitation_annual_oof_early_candidate_v1 as early


EXPERIMENT_ID = "climate_reconstruction_precipitation_annual_oof_early_leave_one_out_v1"
YEARS = np.arange(1901, 1921, dtype=int)
MODERN_YEARS = np.arange(1941, 2001, dtype=int)


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path, hashed: bool = False) -> dict[str, object]:
    stat = path.stat()
    result = {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    if hashed:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def trend(values: np.ndarray) -> float:
    return float(np.polyfit(YEARS, values, 1)[0] * 100.0)


def main() -> int:
    project = root()
    result_root = project / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    loop = result_root / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    archive = Path(r"D:\Users\JUNZ\Desktop") / "".join(chr(v) for v in (25991, 20214, 25972, 29702)) / ("1 " + "".join(chr(v) for v in (35770, 25991))) / "work1/compared_result_all"
    sources = {
        "classification": result_root / "climate_reconstruction_precipitation_annual_oof_operator_sensitivity_v1/annual_oof_proxy_classification.csv",
        "allowed_proxy6": loop / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "proxy6": archive / "python_data/proxy_data_6.mat",
        "metadata": result_root / "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "replay": result_root / "climate_reconstruction_precipitation_enkf_exp1_replay_v1/replay_input.mat",
        "prior": result_root / "climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "berkeley_temperature": psm.BERKELEY_TEMPERATURE,
        "gpcc_precipitation": psm.GPCC_PRECIPITATION,
        "facts": project / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: state(path, name in {"classification", "allowed_proxy6", "metadata", "facts", "driver"}) for name, path in sources.items()}

    classification = pd.read_csv(sources["classification"])
    selected = classification.loc[classification["annual_oof_admissible"], "proxy_index"].to_numpy(int)
    if len(selected) != 15:
        raise RuntimeError(f"selected-count-conflict:{len(selected)}")

    allowed = scipy.io.loadmat(sources["allowed_proxy6"])
    allowed_years = np.asarray(allowed["years"]).reshape(-1).astype(int)
    modern_rows = np.asarray([int(np.flatnonzero(allowed_years == year)[0]) for year in MODERN_YEARS])
    modern_obs = np.asarray(allowed["observations"], dtype=float)[modern_rows, :201]
    sample = np.asarray(allowed["closure_indices"]).reshape(-1).astype(int) - 1
    archived_hxb = np.asarray(allowed["closure_hxb"], dtype=float)
    replay = scipy.io.loadmat(sources["replay"], variable_names=["R0v_filled"])
    error_variance = np.nanvar(np.asarray(replay["R0v_filled"], dtype=float), axis=1, ddof=1)
    with h5py.File(sources["prior"], "r") as handle:
        prior = np.asarray(handle["precipitation_prior_336x848_mm_day"], dtype=float).T[:, sample]
    prior_mean = prior.mean(axis=1)

    metadata = pd.read_csv(sources["metadata"])
    tree = metadata[(metadata["proxy_type"] == "tree") & (metadata["coefficient_count"] == 3)].sort_values("proxy_index")
    latitudes = tree["latitude"].to_numpy(float)
    longitudes = tree["longitude"].to_numpy(float)
    bea_monthly, _, _ = psm.read_berkeley_points(latitudes, longitudes)
    gpcc_monthly, _, _ = psm.read_netcdf_points(psm.GPCC_PRECIPITATION, "precip", 1891, latitudes, longitudes)
    bea_t = psm.center_reference(psm.aggregate_temperature(bea_monthly, psm.SEASONS["annual"]))
    bea_p = psm.center_reference(psm.aggregate_precipitation(gpcc_monthly, psm.SEASONS["annual"]))

    mt = np.full((201, 500), np.nan)
    mp = np.full((201, 500), np.nan)
    with h5py.File(sources["proxy6"], "r") as handle:
        references = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column in range(201):
            proxy = handle[references[column]]
            mt[column] = np.asarray(proxy["M_t"]).reshape(-1, order="F")[sample]
            mp[column] = np.asarray(proxy["M_p"]).reshape(-1, order="F")[sample]

    rebuilt = np.repeat(archived_hxb.mean(axis=1)[:, None], 500, axis=1)
    coefficients: dict[int, np.ndarray] = {}
    full_training = np.ones(len(MODERN_YEARS), dtype=bool)
    for proxy_index in selected:
        column = proxy_index - 1
        coef = blocked.fit(modern_obs[:, column], bea_t[:, column], bea_p[:, column], full_training)
        if coef is None:
            raise RuntimeError(f"selected-full-fit-failed:{proxy_index}")
        coefficients[int(proxy_index)] = coef
        rebuilt[column] = coef[0] + coef[1] * mt[column] + coef[2] * mp[column]
    rebuilt[201] = archived_hxb[201]

    early_obs, locked_rows_skipped = early.early_observations(sources["proxy6"])
    grid_lat, grid_lon = blocked.state_coordinates()
    lon2 = np.broadcast_to(grid_lon[None], (len(grid_lat), len(grid_lon)))
    masks = {name: (lon2 >= lo) & (lon2 < hi) for name, (lo, hi) in blocked.REGIONS.items()}
    branches: list[tuple[str, int | None]] = [("all_15", None)] + [(f"drop_{idx}", int(idx)) for idx in selected]
    curve_rows: list[dict[str, object]] = []
    trend_rows: list[dict[str, object]] = []

    for branch, dropped in branches:
        hxb = rebuilt.copy()
        if dropped is not None:
            column = dropped - 1
            hxb[column] = archived_hxb[column].mean()
        analysis = np.full((prior.shape[0], len(YEARS)), np.nan)
        for row in range(len(YEARS)):
            active = np.flatnonzero(np.isfinite(early_obs[row]))
            analysis[:, row] = blocked.analysis_mean(prior, hxb[active], early_obs[row, active], error_variance[active])
        cube = blocked.to_cube(analysis - prior_mean[:, None], len(grid_lat), len(grid_lon))
        for region, mask in masks.items():
            values = blocked.area_series(cube, grid_lat, mask)
            value_trend = trend(values)
            trend_rows.append({"branch": branch, "dropped_proxy_index": dropped, "region": region, "trend_mm_day_per_century": value_trend})
            curve_rows.extend({"branch": branch, "dropped_proxy_index": dropped, "region": region, "year": int(year), "value_mm_day": float(value)} for year, value in zip(YEARS, values))

    trends = pd.DataFrame(trend_rows)
    curves = pd.DataFrame(curve_rows)
    baseline = trends[trends["branch"] == "all_15"].set_index("region")["trend_mm_day_per_century"]
    drops = trends[trends["branch"] != "all_15"].copy()
    drops["trend_change_from_all_15"] = drops.apply(lambda row: row["trend_mm_day_per_century"] - baseline[row["region"]], axis=1)
    drops["sign_matches_all_15"] = drops.apply(lambda row: np.sign(row["trend_mm_day_per_century"]) == np.sign(baseline[row["region"]]), axis=1)

    output.mkdir(parents=True)
    curves.to_csv(output / "leave_one_out_regional_curves_1901_1920.csv", index=False)
    trends.to_csv(output / "leave_one_out_regional_trends_1901_1920.csv", index=False)
    drops.to_csv(output / "leave_one_out_influence.csv", index=False)
    pd.DataFrame([{"proxy_index": idx, "b0": coefficients[idx][0], "b1_temperature": coefficients[idx][1], "b2_precipitation": coefficients[idx][2]} for idx in selected]).to_csv(output / "selected_proxy_coefficients.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), constrained_layout=True)
    for axis, region in zip(axes.ravel(), blocked.REGIONS):
        subset = curves[curves["region"] == region]
        for _, group in subset[subset["branch"] != "all_15"].groupby("branch"):
            axis.plot(group["year"], group["value_mm_day"], color="#999999", alpha=0.35, linewidth=0.8)
        main = subset[subset["branch"] == "all_15"]
        axis.plot(main["year"], main["value_mm_day"], color="#0072B2", linewidth=2.4, label="All 15")
        axis.axhline(0, color="#BBBBBB", linewidth=0.7)
        axis.set(title=region, ylabel="Precipitation update (mm/day)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    fig.savefig(output / "leave_one_out_regional_curves_1901_1920.png", dpi=220)
    plt.close(fig)

    region_summary = {}
    for region in blocked.REGIONS:
        values = drops.loc[drops["region"] == region, "trend_mm_day_per_century"].to_numpy(float)
        changes = drops.loc[drops["region"] == region, "trend_change_from_all_15"].to_numpy(float)
        region_summary[region] = {
            "all_15_trend_mm_day_per_century": float(baseline[region]),
            "leave_one_out_min": float(values.min()),
            "leave_one_out_max": float(values.max()),
            "leave_one_out_sign_stability_fraction": float(np.mean(np.sign(values) == np.sign(baseline[region]))),
            "largest_absolute_change": float(np.max(np.abs(changes))),
            "most_influential_proxy_index": int(drops.loc[(drops["region"] == region) & (np.abs(drops["trend_change_from_all_15"]) == np.max(np.abs(changes))), "dropped_proxy_index"].iloc[0]),
        }
    after = {name: state(path, name in {"classification", "allowed_proxy6", "metadata", "facts", "driver"}) for name, path in sources.items()}
    unchanged = all(before[name] == after[name] for name in before)
    hma = region_summary["hma"]
    robust = hma["leave_one_out_sign_stability_fraction"] == 1.0 and hma["largest_absolute_change"] < abs(hma["all_15_trend_mm_day_per_century"])
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "period": [1901, 1920],
        "locked_1921_1939_values_read_or_exported": False,
        "locked_value_rows_explicitly_skipped": int(locked_rows_skipped),
        "selected_proxy_count": int(len(selected)),
        "region_summary": region_summary,
        "hma_drying_leave_one_out_robust": bool(robust),
        "decision": "retain_candidate_structure_pending_independent_evidence" if robust else "candidate_is_proxy_sensitive_revisit_operator_selection",
        "all_declared_sources_unchanged": bool(unchanged),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Annual-OOF Early Candidate Leave-One-Out Test",
        "",
        f"- Selected proxies: `{len(selected)}`; branches: `{len(branches)}`.",
        f"- HMA all-15 trend: `{hma['all_15_trend_mm_day_per_century']:.6f}` mm/day/century.",
        f"- HMA leave-one-out range: `{hma['leave_one_out_min']:.6f}` to `{hma['leave_one_out_max']:.6f}`.",
        f"- HMA trend-sign stability: `{hma['leave_one_out_sign_stability_fraction']:.1%}`.",
        f"- Most influential proxy: `{hma['most_influential_proxy_index']}`; largest trend change `{hma['largest_absolute_change']:.6f}`.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not unchanged:
        raise RuntimeError("declared-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
