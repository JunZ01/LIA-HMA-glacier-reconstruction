"""Test whether scalar amplification can repair the LIA temperature increment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MEMBERS = ("ensemble", "AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest}


def rmse(values: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((values - target) ** 2)))


def fit_scale(increment: np.ndarray, required: np.ndarray, nonnegative: bool) -> float:
    denominator = float(increment @ increment)
    value = float(increment @ required / denominator) if denominator > 0 else 0.0
    return max(0.0, value) if nonnegative else value


def analyze(name: str, background: np.ndarray, increment: np.ndarray, reference: np.ndarray) -> tuple[dict[str, object], pd.DataFrame]:
    required = reference - background
    current = background + increment
    current_rmse = rmse(current, reference)
    constrained = fit_scale(increment, required, True)
    unconstrained = fit_scale(increment, required, False)
    constrained_rmse = rmse(background + constrained * increment, reference)
    unconstrained_rmse = rmse(background + unconstrained * increment, reference)
    loo_rows = []
    loo_prediction = np.empty(len(reference), dtype=float)
    loo_scales = np.empty(len(reference), dtype=float)
    for heldout in range(len(reference)):
        train = np.arange(len(reference)) != heldout
        scale = fit_scale(increment[train], required[train], True)
        prediction = background[heldout] + scale * increment[heldout]
        loo_prediction[heldout] = prediction
        loo_scales[heldout] = scale
        loo_rows.append({
            "series": name,
            "heldout_window_index": heldout,
            "selected_nonnegative_scale": scale,
            "heldout_prediction_degC": float(prediction),
            "heldout_reference_degC": float(reference[heldout]),
            "heldout_error_degC": float(prediction - reference[heldout]),
        })
    opposite = (increment * required) < 0
    improvement = 100.0 * (current_rmse - constrained_rmse) / current_rmse if current_rmse > 0 else 0.0
    return ({
        "series": name,
        "opposite_sign_window_count": int(opposite.sum()),
        "window_count": int(len(reference)),
        "current_scale": 1.0,
        "current_rmse_degC": current_rmse,
        "background_only_rmse_degC": rmse(background, reference),
        "constrained_nonnegative_scale": constrained,
        "constrained_rmse_degC": constrained_rmse,
        "constrained_rmse_improvement_percent": improvement,
        "unconstrained_scale": unconstrained,
        "unconstrained_rmse_degC": unconstrained_rmse,
        "loo_nonnegative_scale_min": float(np.min(loo_scales)),
        "loo_nonnegative_scale_max": float(np.max(loo_scales)),
        "loo_nonnegative_scale_median": float(np.median(loo_scales)),
        "loo_prediction_rmse_degC": rmse(loo_prediction, reference),
    }, pd.DataFrame(loo_rows))


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    source = loop / "round_25_temperature_prior_update_decomposition_v1/lia_50year_prior_update_decomposition.csv"
    output = loop / "round_26_temperature_increment_scale_feasibility_v1"
    paths = {"source": source, "contract": output / "ROUND_CONTRACT.md", "driver": Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    windows = pd.read_csv(source)
    if windows[["start_year", "end_year"]].values.tolist() != [[year, year + 49] for year in range(1400, 1900, 50)]:
        raise RuntimeError("window-contract-failed")
    reference = windows.reference_central_degC.to_numpy(float)
    rows: list[dict[str, object]] = []
    loo_tables: list[pd.DataFrame] = []
    components: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for member in MEMBERS:
        if member == "ensemble":
            background = windows.background_degC.to_numpy(float)
            increment = windows.increment_degC.to_numpy(float)
        else:
            background = windows[f"{member}_background_degC"].to_numpy(float)
            increment = windows[f"{member}_increment_degC"].to_numpy(float)
        components[member] = (background, increment)
        result, loo = analyze(member, background, increment, reference)
        rows.append(result)
        loo_tables.append(loo)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "increment_scale_feasibility.csv", index=False)
    loo_table = pd.concat(loo_tables, ignore_index=True)
    loo_table["start_year"] = loo_table.heldout_window_index.map(windows.start_year.to_dict())
    loo_table["end_year"] = loo_table.heldout_window_index.map(windows.end_year.to_dict())
    loo_table.to_csv(output / "leave_one_window_out_scale_predictions.csv", index=False)

    ensemble = metrics[metrics.series.eq("ensemble")].iloc[0]
    close_scalar = bool(
        ensemble.opposite_sign_window_count >= 5
        or ensemble.constrained_rmse_improvement_percent < 5.0
    )
    all_members_close = bool(
        ((metrics.opposite_sign_window_count >= 5) | (metrics.constrained_rmse_improvement_percent < 5.0)).all()
    )
    decision = "close_scalar_gain_scaling_and_test_covariance_operator_sign_structure" if close_scalar and all_members_close else "retain_scalar_gain_scaling_for_independent_validation"
    summary = {
        "status": "complete",
        "locked_1921_1939_values_read": False,
        "ensemble_opposite_sign_windows": int(ensemble.opposite_sign_window_count),
        "ensemble_constrained_scale": float(ensemble.constrained_nonnegative_scale),
        "ensemble_constrained_rmse_improvement_percent": float(ensemble.constrained_rmse_improvement_percent),
        "ensemble_unconstrained_scale": float(ensemble.unconstrained_scale),
        "ensemble_loo_scale_range": [float(ensemble.loo_nonnegative_scale_min), float(ensemble.loo_nonnegative_scale_max)],
        "all_seven_series_close_scalar_route": all_members_close,
        "decision": decision,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    centers = (windows.start_year + windows.end_year) / 2
    background, increment = components["ensemble"]
    required = reference - background
    constrained = float(ensemble.constrained_nonnegative_scale)
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True, constrained_layout=True)
    axes[0].bar(centers - 7, increment, width=14, color="#d55e00", label="Archived increment")
    axes[0].bar(centers + 7, required, width=14, color="#0072b2", label="Required update")
    axes[0].axhline(0, color="#444444", lw=0.8)
    axes[0].set(ylabel="Update anomaly (degC)", title="Archived versus required LIA update sign")
    axes[0].legend(frameon=False)
    axes[1].plot(centers, reference, marker="s", color="#006d77", lw=2, label="Reference")
    axes[1].plot(centers, background + increment, marker="o", color="#d55e00", lw=1.8, label="Scale 1")
    axes[1].plot(centers, background + constrained * increment, marker="^", color="#5e3c99", lw=1.8, label=f"Best nonnegative scale {constrained:.3f}")
    axes[1].axhline(0, color="#444444", lw=0.8)
    axes[1].set(xlabel="50-year window center", ylabel="Temperature anomaly (degC)", title="Scalar-gain feasibility")
    axes[1].legend(frameon=False)
    figure.savefig(output / "temperature_increment_scale_feasibility.png", dpi=190)
    plt.close(figure)

    (output / "FINAL_RESULT.md").write_text(
        "# Temperature Increment-Scale Feasibility\n\n"
        f"- Ensemble increment has the wrong sign in `{int(ensemble.opposite_sign_window_count)}/10` LIA windows.\n"
        f"- Best nonnegative scalar: `{float(ensemble.constrained_nonnegative_scale):.4f}`.\n"
        f"- RMSE change versus archived scale 1: `{float(ensemble.constrained_rmse_improvement_percent):+.2f}%`.\n"
        f"- Unconstrained scalar: `{float(ensemble.unconstrained_scale):+.4f}`; a negative value reverses rather than amplifies the update.\n"
        f"- All seven ensemble/member series close the scalar route: `{str(all_members_close).lower()}`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="utf-8",
    )

    after = {name: state(path) for name, path in paths.items()}
    pure = before == after
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": pure}, indent=2), encoding="utf-8")
    if not pure:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
