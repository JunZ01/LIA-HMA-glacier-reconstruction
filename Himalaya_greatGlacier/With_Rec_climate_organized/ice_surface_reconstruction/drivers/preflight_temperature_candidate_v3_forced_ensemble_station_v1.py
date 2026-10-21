"""Test whether three forced backgrounds form a viable station uncertainty ensemble."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
INPUT_DIR = RESULTS / (
    "climate_reconstruction_improvement_loop_v2/"
    "temperature_forced_background_member_station_diagnostic_v1"
)
INPUT_CSV = INPUT_DIR / "station_year_background_members_allowed.csv"
OUTPUT = RESULTS / (
    "climate_reconstruction_temperature_candidate_v3_"
    "forced_ensemble_station_preflight_v1"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()

PERIODS = {
    "1880-1900": (1880, 1900),
    "1901-1920": (1901, 1920),
    "1940-1950": (1940, 1950),
}
MEMBERS = [
    "fixed_increment_mode_sim_degC",
    "fixed_increment_mpi_degC",
    "fixed_increment_miroc_degC",
]
ORIGINAL = "project_six_member_degC"
TARGETS = ["qcf_station_degC", "qcu_station_degC"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empirical_crps(members: np.ndarray, observed: np.ndarray) -> np.ndarray:
    first = np.mean(np.abs(members - observed[:, None]), axis=1)
    pairwise = np.abs(members[:, :, None] - members[:, None, :])
    return first - 0.5 * np.mean(pairwise, axis=(1, 2))


def station_metrics(group: pd.DataFrame, target: str, members: list[str]) -> dict[str, float] | None:
    observed = group[target].to_numpy(float)
    original = group[ORIGINAL].to_numpy(float)
    ensemble = group[members].to_numpy(float)
    valid = np.isfinite(observed) & np.isfinite(original) & np.all(np.isfinite(ensemble), axis=1)
    if int(valid.sum()) < 5:
        return None
    observed = observed[valid]
    original = original[valid]
    ensemble = ensemble[valid]
    mean = np.mean(ensemble, axis=1)
    lower = np.min(ensemble, axis=1)
    upper = np.max(ensemble, axis=1)
    covered = (observed >= lower) & (observed <= upper)
    original_error = original - observed
    mean_error = mean - observed
    return {
        "year_count": int(len(observed)),
        "coverage": float(np.mean(covered)),
        "covered_years": int(np.sum(covered)),
        "original_rmse": float(np.sqrt(np.mean(original_error**2))),
        "ensemble_mean_rmse": float(np.sqrt(np.mean(mean_error**2))),
        "original_mae": float(np.mean(np.abs(original_error))),
        "ensemble_mean_mae": float(np.mean(np.abs(mean_error))),
        "ensemble_crps": float(np.mean(empirical_crps(ensemble, observed))),
        "original_bias": float(np.mean(original_error)),
        "ensemble_mean_bias": float(np.mean(mean_error)),
        "ensemble_spread": float(np.mean(np.std(ensemble, axis=1, ddof=1))),
    }


def aggregate(
    frame: pd.DataFrame,
    target: str,
    period: str,
    region: str,
    members: list[str],
) -> dict[str, object] | None:
    rows: list[dict[str, float]] = []
    for _, group in frame.groupby("station_id"):
        metrics = station_metrics(group, target, members)
        if metrics is not None:
            rows.append(metrics)
    if not rows:
        return None
    station = pd.DataFrame(rows)
    result: dict[str, object] = {
        "target": target.replace("_station_degC", ""),
        "period": period,
        "region": region,
        "station_count": int(len(station)),
        "station_year_count": int(station["year_count"].sum()),
        "pooled_envelope_coverage": float(
            station["covered_years"].sum() / station["year_count"].sum()
        ),
        "median_station_envelope_coverage": float(station["coverage"].median()),
    }
    for column in [
        "original_rmse",
        "ensemble_mean_rmse",
        "original_mae",
        "ensemble_mean_mae",
        "ensemble_crps",
        "original_bias",
        "ensemble_mean_bias",
        "ensemble_spread",
    ]:
        result[f"equal_station_mean_{column}_degC"] = float(station[column].mean())
    result["rmse_improvement"] = float(
        1.0
        - result["equal_station_mean_ensemble_mean_rmse_degC"]
        / result["equal_station_mean_original_rmse_degC"]
    )
    result["mae_improvement"] = float(
        1.0
        - result["equal_station_mean_ensemble_mean_mae_degC"]
        / result["equal_station_mean_original_mae_degC"]
    )
    result["crps_improvement_vs_original_mae"] = float(
        1.0
        - result["equal_station_mean_ensemble_crps_degC"]
        / result["equal_station_mean_original_mae_degC"]
    )
    return result


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "EXPERIMENT_CONTRACT.md"
    contract.write_text(
        "# Candidate v3 forced-ensemble station preflight\n\n"
        "Treat the existing ModE-Sim, MPI-ESM1-2-LR, and MIROC-ES2L fixed-increment "
        "station branches as an equal-probability uncertainty ensemble. Fit no model "
        "weight and read no climate or station source beyond the existing allowed-year "
        "derived table. qcf is primary and qcu is sensitivity only. Evaluate 1880-1900, "
        "1901-1920, and 1940-1950; reject any locked 1921-1939 row before reading values. "
        "The branch advances only if all qcf HMA periods have at least 2/3 pooled envelope "
        "coverage, nonnegative ensemble-mean RMSE and CRPS improvements, no eligible "
        "region-period RMSE regression worse than 5%, and no leave-one-model-out HMA "
        "period RMSE regression worse than 10%. This preflight cannot authorize EnKF, "
        "production, or OGGM use.\n",
        encoding="utf-8",
    )
    sources = {"input": INPUT_CSV, "facts": FACTS, "driver": DRIVER, "contract": contract}
    before = {name: sha256(path) for name, path in sources.items()}

    frame = pd.read_csv(INPUT_CSV)
    locked = frame["year"].between(1921, 1939)
    if bool(locked.any()):
        raise RuntimeError("Locked-period row present in allowed input")
    required = {"station_id", "region_group", "year", ORIGINAL, *TARGETS, *MEMBERS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    rows: list[dict[str, object]] = []
    loo_rows: list[dict[str, object]] = []
    for period, (start, end) in PERIODS.items():
        period_frame = frame[frame["year"].between(start, end)]
        for target in TARGETS:
            for region in ["hma", "west_lt80", "central_80_90", "east_ge90"]:
                subset = period_frame if region == "hma" else period_frame[period_frame["region_group"].eq(region)]
                result = aggregate(subset, target, period, region, MEMBERS)
                if result is not None:
                    rows.append(result)
            if target == "qcf_station_degC":
                for omitted in MEMBERS:
                    retained = [member for member in MEMBERS if member != omitted]
                    result = aggregate(period_frame, target, period, "hma", retained)
                    if result is not None:
                        result["omitted_member"] = omitted.removeprefix("fixed_increment_").removesuffix("_degC")
                        loo_rows.append(result)

    metrics = pd.DataFrame(rows)
    loo = pd.DataFrame(loo_rows)
    metrics.to_csv(OUTPUT / "period_region_ensemble_metrics.csv", index=False)
    loo.to_csv(OUTPUT / "leave_one_model_out_metrics.csv", index=False)

    primary_hma = metrics[(metrics["target"].eq("qcf")) & (metrics["region"].eq("hma"))]
    primary_region = metrics[
        (metrics["target"].eq("qcf"))
        & (~metrics["region"].eq("hma"))
        & (metrics["station_count"] >= 2)
    ]
    gates = {
        "all_three_periods_available": bool(len(primary_hma) == 3),
        "all_hma_periods_pooled_coverage_at_least_two_thirds": bool(
            len(primary_hma) == 3
            and np.all(primary_hma["pooled_envelope_coverage"] >= 2.0 / 3.0)
        ),
        "all_hma_periods_ensemble_mean_rmse_nonnegative_improvement": bool(
            len(primary_hma) == 3 and np.all(primary_hma["rmse_improvement"] >= 0.0)
        ),
        "all_hma_periods_crps_nonnegative_improvement": bool(
            len(primary_hma) == 3
            and np.all(primary_hma["crps_improvement_vs_original_mae"] >= 0.0)
        ),
        "no_eligible_region_period_rmse_regression_worse_than_5pct": bool(
            len(primary_region) > 0 and np.all(primary_region["rmse_improvement"] >= -0.05)
        ),
        "no_leave_one_model_hma_period_rmse_regression_worse_than_10pct": bool(
            len(loo) == 9 and np.all(loo["rmse_improvement"] >= -0.10)
        ),
    }
    passes = bool(all(gates.values()))

    periods = list(PERIODS)
    plot = primary_hma.set_index("period").reindex(periods)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    axes[0].bar(periods, plot["pooled_envelope_coverage"], color="#0072B2")
    axes[0].axhline(2.0 / 3.0, color="#B2182B", linestyle="--", linewidth=1.2)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Station coverage")
    axes[0].set_ylabel("Fraction inside 3-model envelope")
    axes[1].bar(periods, 100 * plot["rmse_improvement"], color="#009E73")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Ensemble-mean RMSE")
    axes[1].set_ylabel("Improvement over original (%)")
    axes[2].bar(periods, 100 * plot["crps_improvement_vs_original_mae"], color="#CC79A7")
    axes[2].axhline(0, color="#333333", linewidth=0.8)
    axes[2].set_title("Probabilistic CRPS")
    axes[2].set_ylabel("Improvement over original (%)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    figure = OUTPUT / "forced_ensemble_station_preflight.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    summary = {
        "status": "complete",
        "candidate": "equal_probability_mode_sim_mpi_miroc_fixed_increment_ensemble",
        "fitted_weights": False,
        "periods": periods,
        "primary_target": "GHCN-M qcf",
        "sensitivity_target": "GHCN-M qcu",
        "gates": gates,
        "all_gates_pass": passes,
        "advance_to_paleo_ensemble_preflight": passes,
        "decision": (
            "advance_forced_ensemble_to_independent_paleo_preflight"
            if passes
            else "close_forced_ensemble_candidate_v3_before_enkf"
        ),
        "locked_1921_1939_rows_read": 0,
        "enkf_run": False,
        "production_changed": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# Candidate v3 Forced-Ensemble Station Preflight",
        "",
        f"Decision: `{summary['decision']}`.",
        "",
    ]
    for row in primary_hma.itertuples(index=False):
        report.append(
            f"- {row.period}: {row.station_count} stations, envelope coverage "
            f"`{100 * row.pooled_envelope_coverage:.1f}%`, ensemble-mean RMSE change "
            f"`{100 * row.rmse_improvement:+.1f}%`, CRPS change "
            f"`{100 * row.crps_improvement_vs_original_mae:+.1f}%`."
        )
    report.extend(
        [
            "",
            f"- All gates pass: `{str(passes).lower()}`.",
            "- No model weight was fitted; qcu is sensitivity only.",
            "- No 1921-1939 row, EnKF, production field, or OGGM run was used.",
        ]
    )
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    after = {name: sha256(path) for name, path in sources.items()}
    unchanged = before == after
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {"before": before, "after": after, "source_hashes_unchanged": unchanged},
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("Source file changed")
    print(json.dumps(summary, indent=2))
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
