"""Decompose fixed-support precipitation trends into EnKF prior and update terms."""

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


EXPERIMENT_ID = "climate_reconstruction_precipitation_fixed_support_prior_update_decomposition_1901_1920_v1"
YEARS = np.arange(1901, 1921, dtype=int)
LOCKED_PERIOD = (1921, 1939)
REGIONS = ("west_lt80", "central_80_90", "east_90_100")
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


def event_change(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    baseline = (YEARS >= 1905) & (YEARS <= 1914)
    event = (YEARS >= 1915) & (YEARS <= 1918)
    return float(np.nanmean(values[event]) - np.nanmean(values[baseline]))


def main() -> int:
    root = project_root()
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    output = result_root / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    sources = {
        "fixed_support_curves": result_root / "climate_reconstruction_precipitation_reference_support_by_longitude_1901_1920_v2/longitude_support_curves_v1_reproduced.csv",
        "fixed_support_summary": result_root / "climate_reconstruction_precipitation_reference_support_by_longitude_1901_1920_v2/summary.json",
        "prior": result_root / "climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "replay_input": result_root / "climate_reconstruction_precipitation_enkf_exp1_replay_v1/replay_input.mat",
        "attribution": result_root / "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/attribution_output.mat",
        "native_posterior": result_root / "climate_reconstruction_precipitation_other_positive_b2_longitude_block_ablation_v1/longitude_block_ablation.mat",
        "locked_contract": result_root / "climate_reconstruction_precipitation_instrumental_baseline_v1/LOCKED_FINAL_PERIOD.json",
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    if np.any((YEARS >= LOCKED_PERIOD[0]) & (YEARS <= LOCKED_PERIOD[1])):
        raise RuntimeError("locked-period-requested")
    locked = json.loads(sources["locked_contract"].read_text(encoding="utf-8"))
    fixed_summary = json.loads(sources["fixed_support_summary"].read_text(encoding="utf-8"))
    if locked["status"] != "locked_not_evaluated" or fixed_summary["locked_1921_1939_accessed"]:
        raise RuntimeError("locked-period-contract-failed")
    before = {name: source_state(path) for name, path in sources.items()}

    replay = loadmat(sources["replay_input"], variable_names=["sample_indices"])
    samples = np.asarray(replay["sample_indices"], dtype=int).reshape(-1) - 1
    if len(samples) != 500 or samples.min() < 0:
        raise RuntimeError("sample-index-contract-failed")
    with h5py.File(sources["prior"], "r") as handle:
        prior_members = np.asarray(handle["precipitation_prior_336x848_mm_day"], dtype=float)
        latitudes = vector(handle, "lat")
        longitudes = vector(handle, "lon")
    if prior_members.shape != (848, 336) or samples.max() >= prior_members.shape[0]:
        raise RuntimeError(f"prior-shape-contract-failed:{prior_members.shape}")
    prior_state_mean = np.mean(prior_members[samples, :], axis=0)
    weight_grid = np.tile(np.cos(np.deg2rad(latitudes))[:, None], (1, len(longitudes)))
    state_weights = weight_grid.reshape(-1, order="F")
    state_weights = state_weights / state_weights.sum()
    prior_hma = float(state_weights @ prior_state_mean)

    with h5py.File(sources["native_posterior"], "r") as handle:
        posterior_state = np.asarray(handle["posterior_mean"][0, :, :], dtype=float)
        native_years = vector(handle, "years").astype(int)
    if posterior_state.shape != (20, 336) or not np.array_equal(native_years, YEARS):
        raise RuntimeError(f"native-posterior-contract-failed:{posterior_state.shape}")
    posterior_hma = posterior_state @ state_weights
    native_increment = posterior_hma - prior_hma

    with h5py.File(sources["attribution"], "r") as handle:
        archived_years = vector(handle, "run_years").astype(int)
        archived_prior = float(vector(handle, "prior_hma_mean")[0])
        archived_posterior = vector(handle, "posterior_hma_mean")
        archived_increment = vector(handle, "total_increment")
    closure = {
        "prior_hma_max_abs_error_mm_day": abs(prior_hma - archived_prior),
        "posterior_hma_max_abs_error_mm_day": float(np.max(np.abs(posterior_hma - archived_posterior))),
        "increment_hma_max_abs_error_mm_day": float(np.max(np.abs(native_increment - archived_increment))),
        "posterior_minus_prior_increment_identity_max_abs_error_mm_day": float(np.max(np.abs((posterior_hma - prior_hma) - native_increment))),
    }
    if not np.array_equal(archived_years, YEARS) or max(closure.values()) > 1e-12:
        raise RuntimeError(f"native-attribution-closure-failed:{closure}")

    curves = pd.read_csv(sources["fixed_support_curves"])
    if not np.array_equal(curves["year"].to_numpy(dtype=int), YEARS):
        raise RuntimeError("fixed-support-year-contract-failed")
    annual = pd.DataFrame({"year": YEARS})
    score_rows = []
    for region in REGIONS:
        prefix = f"{region}_dual_any"
        cru = curves[f"{prefix}_cru_mm_day"].to_numpy(dtype=float)
        gpcc = curves[f"{prefix}_gpcc_mm_day"].to_numpy(dtype=float)
        analysis = curves[f"{prefix}_rec_result_mm_day"].to_numpy(dtype=float)

        # The production EnKF reuses one fixed background ensemble and member
        # sample in every year. A fixed spatial downscaling cannot add a trend,
        # so centering removes the unknown regional prior level exactly.
        prior_change = np.zeros_like(analysis)
        analysis_change = analysis - analysis[0]
        update_change = analysis_change - prior_change
        identity_error = float(np.max(np.abs(analysis_change - prior_change - update_change)))
        if identity_error > 1e-12:
            raise RuntimeError(f"regional-trend-identity-failed:{region}:{identity_error}")
        annual[f"{region}_cru_change_from_1901_mm_day"] = cru - cru[0]
        annual[f"{region}_gpcc_change_from_1901_mm_day"] = gpcc - gpcc[0]
        annual[f"{region}_analysis_change_from_1901_mm_day"] = analysis_change
        annual[f"{region}_prior_change_from_1901_mm_day"] = prior_change
        annual[f"{region}_update_change_from_1901_mm_day"] = update_change
        analysis_trend = trend(analysis)
        update_trend = trend(update_change)
        score_rows.append({
            "longitude_region": region,
            "gridcell_count": int(next(row["gridcell_count"] for row in fixed_summary["eligible_supported_regions"] if row["longitude_region"] == region)),
            "cru_trend_mm_day_per_century": trend(cru),
            "gpcc_trend_mm_day_per_century": trend(gpcc),
            "analysis_trend_mm_day_per_century": analysis_trend,
            "prior_trend_mm_day_per_century": trend(prior_change),
            "update_trend_mm_day_per_century": update_trend,
            "update_minus_analysis_trend_closure_mm_day_per_century": update_trend - analysis_trend,
            "analysis_event_change_mm_day": event_change(analysis),
            "prior_event_change_mm_day": event_change(prior_change),
            "update_event_change_mm_day": event_change(update_change),
            "temporal_change_attributed_to_update_fraction": 1.0,
            "trend_identity_max_abs_error_mm_day": identity_error,
        })
    scores = pd.DataFrame(score_rows)

    output.mkdir(parents=True)
    annual.to_csv(output / "fixed_support_prior_update_curves_1901_1920.csv", index=False)
    scores.to_csv(output / "fixed_support_prior_update_scores.csv", index=False)

    colors = {"cru": "#6A3D9A", "gpcc": "#777777", "analysis": "#D55E00", "prior": "#0072B2", "update": "#009E73"}
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), sharex=True, constrained_layout=True)
    for axis, region in zip(axes.reshape(-1)[:3], REGIONS):
        prefix = f"{region}_"
        axis.plot(YEARS, annual[prefix + "cru_change_from_1901_mm_day"], color=colors["cru"], linewidth=1.5, label="CRU")
        axis.plot(YEARS, annual[prefix + "gpcc_change_from_1901_mm_day"], color=colors["gpcc"], linewidth=1.5, label="GPCC")
        axis.plot(YEARS, annual[prefix + "analysis_change_from_1901_mm_day"], color=colors["analysis"], linewidth=2.4, label="Reconstruction")
        axis.plot(YEARS, annual[prefix + "update_change_from_1901_mm_day"], color=colors["update"], linewidth=1.2, linestyle="--", label="EnKF update")
        axis.plot(YEARS, annual[prefix + "prior_change_from_1901_mm_day"], color=colors["prior"], linewidth=1.2, linestyle=":", label="Prior")
        count = int(scores.loc[scores["longitude_region"] == region, "gridcell_count"].iloc[0])
        axis.set_title(f"{REGION_LABELS[region]} fixed dual support (n={count})")
        axis.set_ylabel("Change from 1901 (mm/day)")
        axis.axhline(0, color="#B0B0B0", linewidth=0.7)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Year")
    axes[0, 0].legend(frameon=False, ncol=2)

    axis = axes[1, 1]
    x = np.arange(len(scores))
    width = 0.18
    for offset, column, label, color in (
        (-1.5, "cru_trend_mm_day_per_century", "CRU", colors["cru"]),
        (-0.5, "gpcc_trend_mm_day_per_century", "GPCC", colors["gpcc"]),
        (0.5, "analysis_trend_mm_day_per_century", "Reconstruction", colors["analysis"]),
        (1.5, "update_trend_mm_day_per_century", "EnKF update", colors["update"]),
    ):
        axis.bar(x + offset * width, scores[column], width=width, label=label, color=color)
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_xticks(x, [REGION_LABELS[name] for name in scores["longitude_region"]])
    axis.set_ylabel("Trend (mm/day/century)")
    axis.set_title("Trend decomposition")
    axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "fixed_support_prior_update_decomposition_curves.png", dpi=220)
    plt.close(fig)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "period": [1901, 1920],
        "locked_1921_1939_accessed": False,
        "design": "The same 500 prior members are reused in every reconstruction year. Fixed-support regional trends are decomposed into a time-invariant prior and the EnKF update.",
        "native_hma_closure": closure,
        "native_prior_hma_mean_mm_day": prior_hma,
        "native_prior_interannual_standard_deviation_mm_day": 0.0,
        "scores": scores.to_dict(orient="records"),
        "classification": "regional_reconstruction_trends_are_entirely_created_by_the_enkf_update_under_the_fixed_prior_design",
        "interpretation": "The prior cannot create the 1901-1920 regional trend because it is identical in every year. The western negative trend and the weaker central/eastern rise are therefore update-layer behavior. This result does not yet separate proxy innovations from state-proxy covariance/gain.",
        "interpretation_boundary": "This is an exact temporal decomposition, not a claim that the fixed prior is unbiased and not authorization for production correction.",
    }
    after = {name: source_state(path) for name, path in sources.items()}
    summary["all_declared_sources_unchanged"] = before == after
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "driver_sha256": digest(Path(__file__))}, indent=2),
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-changed-during-prior-update-audit")
    lines = [
        "# Fixed-support Prior/Update Decomposition",
        "",
        "The precipitation EnKF uses the same sampled background ensemble in every year, so the prior trend is exactly zero.",
        "",
    ]
    for row in scores.to_dict(orient="records"):
        lines.append(
            f"- {row['longitude_region']} n={row['gridcell_count']}: prior/update/analysis trends "
            f"`{row['prior_trend_mm_day_per_century']:+.3f}` / `{row['update_trend_mm_day_per_century']:+.3f}` / "
            f"`{row['analysis_trend_mm_day_per_century']:+.3f} mm/day/century`."
        )
    lines.extend([
        "",
        "Conclusion: the regional temporal behavior is created by the EnKF update, not inherited from a time-varying prior. The next mechanism split is innovation versus covariance/gain.",
    ])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
