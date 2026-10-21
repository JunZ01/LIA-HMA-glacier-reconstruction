"""Build the authorized 1901-1920 annual-OOF precipitation candidate."""

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


EXPERIMENT_ID = "climate_reconstruction_precipitation_annual_oof_early_candidate_v1"
EARLY_YEARS = np.arange(1901, 1921, dtype=int)
MODERN_YEARS = np.arange(1941, 2001, dtype=int)
REGIONS = blocked.REGIONS


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path, hashed: bool = False) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    if hashed:
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(chunk)
        result["sha256"] = value.hexdigest()
    return result


def read_flat_value(dataset: h5py.Dataset, flat_index: int) -> float:
    if dataset.ndim == 1:
        return float(dataset[flat_index])
    coordinate = np.unravel_index(flat_index, dataset.shape, order="F")
    return float(dataset[coordinate])


def early_observations(proxy_path: Path) -> tuple[np.ndarray, int]:
    observations = np.full((len(EARLY_YEARS), 202), np.nan)
    locked_skipped = 0
    with h5py.File(proxy_path, "r") as handle:
        references = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for column, reference in enumerate(references):
            proxy = handle[reference]
            ages = np.asarray(proxy["age_yng"]).reshape(-1, order="F")
            locked_skipped += int(np.sum((ages >= 1921) & (ages <= 1939)))
            positions = {int(age): index for index, age in enumerate(ages) if np.isfinite(age) and int(age) in EARLY_YEARS}
            for row, year in enumerate(EARLY_YEARS):
                if year in positions:
                    observations[row, column] = read_flat_value(proxy["data"], positions[year])
    return observations, locked_skipped


def zscore(values: np.ndarray) -> np.ndarray:
    scale = float(np.nanstd(values))
    return (values - np.nanmean(values)) / scale


def trend(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    return float(np.polyfit(EARLY_YEARS[valid], values[valid], 1)[0] * 100)


def main() -> int:
    project = root()
    result_root = project / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    loop = result_root / "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection"
    archive = Path(r"D:\Users\JUNZ\Desktop") / "".join(chr(v) for v in (25991, 20214, 25972, 29702)) / ("1 " + "".join(chr(v) for v in (35770, 25991))) / "work1/compared_result_all"
    sources = {
        "modern_gate": result_root / "climate_reconstruction_precipitation_annual_oof_nested_blocked_field_validation_v2/summary.json",
        "classification": result_root / "climate_reconstruction_precipitation_annual_oof_operator_sensitivity_v1/annual_oof_proxy_classification.csv",
        "allowed_proxy6": loop / "proxy6_allowed_1940_2000_official_baseline_v2.mat",
        "proxy6": archive / "python_data/proxy_data_6.mat",
        "metadata": result_root / "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/proxy6_metadata.csv",
        "replay": result_root / "climate_reconstruction_precipitation_enkf_exp1_replay_v1/replay_input.mat",
        "prior": result_root / "climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "berkeley_temperature": psm.BERKELEY_TEMPERATURE,
        "gpcc_precipitation": psm.GPCC_PRECIPITATION,
        "production_curves": result_root / "climate_reconstruction_precipitation_full_hma_six_member_regional_decomposition_1901_1920_v1/full_hma_and_dual_support_member_curves.csv",
        "mada_curves": result_root / "climate_reconstruction_improvement_loop_v2/round_102_tree_ring_hydroclimate_evidence_v4/allowed_period_standardized_curves.csv",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    modern_gate = json.loads(sources["modern_gate"].read_text(encoding="utf-8"))
    if modern_gate["decision"] != "advance_to_1901_1920_candidate_reconstruction" or not modern_gate["candidate_gate"]["passes_gate"]:
        raise RuntimeError("modern-gate-did-not-authorize-early-candidate")
    before = {name: state(path, hashed=name in {"modern_gate", "classification", "allowed_proxy6", "metadata", "facts", "driver"}) for name, path in sources.items()}

    classification = pd.read_csv(sources["classification"])
    selected = classification.loc[classification["annual_oof_admissible"], "proxy_index"].to_numpy(int)
    if len(selected) != 15:
        raise RuntimeError(f"selected-count-conflict:{len(selected)}")
    allowed = scipy.io.loadmat(sources["allowed_proxy6"])
    years = np.asarray(allowed["years"]).reshape(-1).astype(int)
    observations_modern_all = np.asarray(allowed["observations"], dtype=float)
    modern_rows = np.asarray([int(np.flatnonzero(years == year)[0]) for year in MODERN_YEARS])
    observations_modern = observations_modern_all[modern_rows, :201]
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

    hxb = np.repeat(archived_hxb.mean(axis=1)[:, None], 500, axis=1)
    full_training = np.ones(len(MODERN_YEARS), dtype=bool)
    for proxy_index in selected:
        column = proxy_index - 1
        coefficients = blocked.fit(observations_modern[:, column], bea_t[:, column], bea_p[:, column], full_training)
        if coefficients is None:
            raise RuntimeError(f"selected-full-fit-failed:{proxy_index}")
        hxb[column] = coefficients[0] + coefficients[1] * mt[column] + coefficients[2] * mp[column]
    hxb[201] = archived_hxb[201]

    observations_early, locked_rows_skipped = early_observations(sources["proxy6"])
    analysis = np.full((prior.shape[0], len(EARLY_YEARS)), np.nan)
    active_counts = []
    for row in range(len(EARLY_YEARS)):
        active = np.flatnonzero(np.isfinite(observations_early[row]))
        active_counts.append(int(len(active)))
        analysis[:, row] = blocked.analysis_mean(prior, hxb[active], observations_early[row, active], error_variance[active])
    increment = analysis - prior_mean[:, None]

    grid_lat, grid_lon = blocked.state_coordinates()
    cube = blocked.to_cube(increment, len(grid_lat), len(grid_lon))
    lon2 = np.broadcast_to(grid_lon[None], (len(grid_lat), len(grid_lon)))
    region_masks = {name: (lon2 >= lower) & (lon2 < upper) for name, (lower, upper) in REGIONS.items()}
    candidate_curves = {region: blocked.area_series(cube, grid_lat, mask) for region, mask in region_masks.items()}

    production = pd.read_csv(sources["production_curves"])
    prod = production[(production["mask"] == "full_hma") & (production["field"] == "ensemble") & production["year"].between(1901, 1920)]
    production_curves = {region: prod[prod["region"] == region].sort_values("year")["precipitation_anomaly_mm_day"].to_numpy(float) for region in REGIONS}
    mada = pd.read_csv(sources["mada_curves"])
    mada_hma = mada[(mada["period"] == "early_1901_1920") & (mada["evidence"] == "MADA_JJA_PDSI") & (mada["region"] == "hma") & (mada["series"] == "evidence")].sort_values("year")["z_value"].to_numpy(float)
    candidate_z = zscore(candidate_curves["hma"])
    production_z = zscore(production_curves["hma"])
    candidate_mada_r = float(np.corrcoef(candidate_z, mada_hma)[0, 1])
    production_mada_r = float(np.corrcoef(production_z, mada_hma)[0, 1])

    curve_rows = []
    trend_rows = []
    for region in REGIONS:
        for series, values in (("annual_oof_candidate", candidate_curves[region]), ("production_equal_six", production_curves[region])):
            curve_rows.extend({"year": int(year), "region": region, "series": series, "value_mm_day": float(value)} for year, value in zip(EARLY_YEARS, values))
            trend_rows.append({"region": region, "series": series, "trend_mm_day_per_century": trend(values)})
    output.mkdir(parents=True)
    pd.DataFrame(curve_rows).to_csv(output / "candidate_and_production_regional_curves_1901_1920.csv", index=False)
    pd.DataFrame(trend_rows).to_csv(output / "candidate_and_production_regional_trends_1901_1920.csv", index=False)
    np.savez_compressed(output / "annual_oof_candidate_native_2deg_1901_1920.npz", years=EARLY_YEARS, lat=grid_lat, lon=grid_lon, increment_mm_day=cube, selected_proxy_indices=selected)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), constrained_layout=True)
    for axis, region in zip(axes.ravel(), REGIONS):
        axis.plot(EARLY_YEARS, production_curves[region], color="#D55E00", linewidth=2, label="Production equal-six")
        axis.plot(EARLY_YEARS, candidate_curves[region], color="#009E73", linewidth=2.2, label="Annual OOF candidate")
        axis.axhline(0, color="#BBBBBB", linewidth=0.7)
        axis.set(title=region, ylabel="Precipitation update/anomaly (mm/day)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    fig.savefig(output / "annual_oof_candidate_regional_curves_1901_1920.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    axis.plot(EARLY_YEARS, production_z, color="#D55E00", linewidth=2, label=f"Production, r(MADA)={production_mada_r:.2f}")
    axis.plot(EARLY_YEARS, candidate_z, color="#009E73", linewidth=2.2, label=f"Annual OOF candidate, r(MADA)={candidate_mada_r:.2f}")
    axis.plot(EARLY_YEARS, mada_hma, color="#0072B2", linewidth=2, label="MADA JJA PDSI")
    axis.axhline(0, color="#BBBBBB", linewidth=0.7)
    axis.set(xlabel="Year", ylabel="Standardized anomaly", title="1901-1920 HMA phase diagnostic")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    fig.savefig(output / "annual_oof_candidate_vs_mada_1901_1920.png", dpi=220)
    plt.close(fig)

    candidate_trend = trend(candidate_curves["hma"])
    production_trend = trend(production_curves["hma"])
    phase_improvement = candidate_mada_r - production_mada_r
    direction_resolved = bool(np.sign(trend(candidate_z)) == np.sign(trend(mada_hma)))
    decision = "open_multi_member_annual_oof_reconstruction_preflight" if direction_resolved and phase_improvement >= 0.2 else "retain_as_single_branch_diagnostic_only"
    after = {name: state(path, hashed=name in {"modern_gate", "classification", "allowed_proxy6", "metadata", "facts", "driver"}) for name, path in sources.items()}
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "period": [1901, 1920],
        "locked_1921_1939_values_read_or_exported": False,
        "locked_value_rows_explicitly_skipped": int(locked_rows_skipped),
        "modern_gate_passed": True,
        "selected_proxy_count": int(len(selected)),
        "active_proxy_count_min": int(min(active_counts)),
        "active_proxy_count_max": int(max(active_counts)),
        "hma": {
            "candidate_trend_mm_day_per_century": candidate_trend,
            "production_equal_six_trend_mm_day_per_century": production_trend,
            "candidate_mada_correlation": candidate_mada_r,
            "production_mada_correlation": production_mada_r,
            "candidate_minus_production_mada_correlation": phase_improvement,
            "candidate_trend_sign_matches_mada": direction_resolved,
        },
        "decision": decision,
        "all_declared_sources_unchanged": before == after,
        "interpretation_boundary": "The candidate passed the modern field gate before early construction. It is one proxy6 standard-EnKF branch and is not yet a replacement for the six-member production field.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text("# Annual OOF Early Candidate\n\nFreeze the 15-proxy annual OOF rule and fit 1941-2000 PSMs before reading only 1901-1920 proxy values. Explicitly skip 1921-1939 values. Keep prior, RegEM errors, and glacier-ice operator fixed. Do not modify production or run OGGM.\n", encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Annual OOF Early Candidate\n\n"
        f"- HMA trend: candidate `{candidate_trend:+.3f}` versus production `{production_trend:+.3f} mm/day/century`.\n"
        f"- MADA correlation: candidate `{candidate_mada_r:+.3f}` versus production `{production_mada_r:+.3f}`.\n"
        f"- Candidate trend sign matches MADA: `{direction_resolved}`. Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-early-candidate")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
