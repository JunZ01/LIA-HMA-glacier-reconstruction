"""Recompute controlled proxy9 EnKF posteriors after record holdouts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


def _load_baseline_driver(path: Path):
    spec = importlib.util.spec_from_file_location("controlled_enkf_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot-load-baseline-driver:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _subset_proxy(proxy: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[keep] for name, values in proxy.items()}


def _case_definitions(config: dict) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = [
        {"case": "no_removal", "kind": "baseline", "removed_1based": []}
    ]
    for index in config["single_holdouts_1based"]:
        cases.append(
            {
                "case": f"without_{int(index):03d}",
                "kind": "single",
                "removed_1based": [int(index)],
            }
        )
    for name, indices in config["group_holdouts_1based"].items():
        cases.append(
            {
                "case": f"without_{name}",
                "kind": "group",
                "removed_1based": [int(index) for index in indices],
            }
        )
    return cases


def _plot_cases(
    years: np.ndarray,
    target_curves: dict[str, np.ndarray],
    regional_curves: dict[str, np.ndarray],
    reference: np.ndarray,
    case_kinds: dict[str, str],
    kind: str,
    period: tuple[int, int],
    plot_period: tuple[int, int],
    window: int,
    output: Path,
) -> None:
    names = [name for name, value in case_kinds.items() if value == kind]
    selected = (years >= plot_period[0]) & (years <= plot_period[1])
    center = (years >= period[0]) & (years <= period[1])
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, max(len(names), 1)))
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 9.0), sharex=True, constrained_layout=True)
    baseline_target = target_curves["no_removal"] - np.mean(target_curves["no_removal"][center])
    baseline_region = regional_curves["no_removal"] - np.mean(regional_curves["no_removal"][center])
    reference_centered = reference - np.mean(reference[center])
    axes[0, 0].plot(years[selected], reference_centered[selected], color="#2166ac", lw=1.5, label="rec_ref")
    axes[1, 0].plot(years[selected], pd.Series(reference_centered).rolling(window, center=True, min_periods=window).mean()[selected], color="#2166ac", lw=2.3, label="rec_ref")
    for axis, baseline in ((axes[0, 0], baseline_target), (axes[0, 1], baseline_region)):
        axis.plot(years[selected], baseline[selected], color="#111111", lw=2.2, label="no_removal")
    for axis, baseline in ((axes[1, 0], baseline_target), (axes[1, 1], baseline_region)):
        smooth = pd.Series(baseline).rolling(window, center=True, min_periods=window).mean().to_numpy()
        axis.plot(years[selected], smooth[selected], color="#111111", lw=2.6, label="no_removal")
    for color, name in zip(colors, names):
        target = target_curves[name] - np.mean(target_curves[name][center])
        region = regional_curves[name] - np.mean(regional_curves[name][center])
        for axis, values in ((axes[0, 0], target), (axes[0, 1], region)):
            axis.plot(years[selected], values[selected], color=color, lw=1.0, alpha=0.78, label=name)
        for axis, values in ((axes[1, 0], target), (axes[1, 1], region)):
            smooth = pd.Series(values).rolling(window, center=True, min_periods=window).mean().to_numpy()
            axis.plot(years[selected], smooth[selected], color=color, lw=2.0, label=name)
    titles = (
        "3584 annual anomaly",
        "Cosine-weighted HMA annual anomaly",
        f"3584 centered {window}-year mean",
        f"HMA centered {window}-year mean",
    )
    for axis, title in zip(axes.flat, titles):
        axis.axvspan(period[0], period[1], color="#d9d9d9", alpha=0.22)
        axis.axhline(0.0, color="#777777", lw=0.7)
        axis.grid(alpha=0.2)
        axis.set_title(title)
        axis.set_ylabel("Temperature (degC)")
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    figure.suptitle(f"Proxy9 {kind} holdout recomputations", fontsize=14)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    root = _project_root()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = config["sources"]
    archive_root = Path(config["archive_root"])
    paths = {
        "baseline_driver": root / source["baseline_driver"],
        "baseline_arrays": root / source["baseline_arrays"],
        "prior": root / source["prior"],
        "indices": archive_root / source["indices"],
        "proxy9": archive_root / source["proxy9"],
        "proxy9_regem": root / source["proxy9_regem"],
        "reference_netcdf": Path(source["reference_netcdf"]),
    }
    output = root / config["result_root"]
    output.mkdir(parents=True, exist_ok=True)
    hashes_before = {name: _sha256(path) for name, path in paths.items()}

    baseline = _load_baseline_driver(paths["baseline_driver"])
    years = np.arange(1000, 2001)
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as archive:
        sample_indices = np.asarray(archive["all_Xb_indices"], dtype=int)[0] - 1
    proxy = baseline._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["proxy9_regem"])["R0v_filled"], dtype=float)
    if residuals.shape[0] != proxy["observations"].shape[0]:
        raise RuntimeError("proxy-regem-row-count-mismatch")

    grid_lat, grid_lon, state_lat, _ = baseline._state_coordinates()
    target_lat = float(config["target"]["latitude"])
    target_lon = float(config["target"]["longitude"])
    target_weights = baseline._target_weights(grid_lat, grid_lon, target_lat, target_lon)
    regional_weights = np.cos(np.deg2rad(state_lat))
    regional_weights /= regional_weights.sum()

    baseline_result = baseline._run_branch(
        prior, sample_indices, proxy, residuals, target_weights, regional_weights
    )
    with np.load(paths["baseline_arrays"]) as saved:
        saved_analysis = np.asarray(saved["proxy9_analysis"], dtype=float)
    state_difference = float(
        np.max(np.abs(np.asarray(baseline_result["analysis"]) - saved_analysis))
    )
    preflight = {
        "status": "pass"
        if state_difference <= float(config["gates"]["no_removal_state_max_abs_c"])
        else "fail",
        "no_removal_state_max_abs_c": state_difference,
        "gate_c": float(config["gates"]["no_removal_state_max_abs_c"]),
    }
    (output / "preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(preflight, indent=2, ensure_ascii=True))
    if preflight["status"] != "pass":
        raise RuntimeError("no-removal-baseline-closure-failed")
    if args.preflight_only:
        return 0

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
    early = (years >= early_period[0]) & (years <= early_period[1])

    cases = _case_definitions(config)
    definitions = []
    metrics = []
    target_curves: dict[str, np.ndarray] = {}
    regional_curves: dict[str, np.ndarray] = {}
    case_kinds: dict[str, str] = {}
    closures: dict[str, dict[str, float]] = {}

    for case in cases:
        name = str(case["case"])
        kind = str(case["kind"])
        removed = [int(index) for index in case["removed_1based"]]
        if name == "no_removal":
            result = baseline_result
        else:
            removed_zero = np.asarray(removed, dtype=int) - 1
            if np.any(removed_zero < 0) or np.any(removed_zero >= proxy["observations"].shape[0]):
                raise RuntimeError(f"holdout-index-out-of-range:{name}")
            keep = np.ones(proxy["observations"].shape[0], dtype=bool)
            keep[removed_zero] = False
            result = baseline._run_branch(
                prior,
                sample_indices,
                _subset_proxy(proxy, keep),
                residuals[keep],
                target_weights,
                regional_weights,
            )
        target = baseline._anomaly(
            np.asarray(result["target_analysis"]), years, reference_period
        )
        region = baseline._anomaly(
            np.asarray(result["regional_analysis"]), years, reference_period
        )
        target_curves[name] = target
        regional_curves[name] = region
        case_kinds[name] = kind
        closures[name] = {
            key: float(result[key])
            for key in (
                "state_increment_closure_max_abs_c",
                "target_increment_closure_max_abs_c",
                "regional_increment_closure_max_abs_c",
            )
        }
        target_metric, _ = baseline._metrics(
            name, years, target, reference, early_period, window, lag_bounds
        )
        metrics.append(
            {
                "case": name,
                "kind": kind,
                "removed_count": len(removed),
                "removed_1based": ";".join(str(index) for index in removed),
                "target_trend_c_per_century": target_metric["trend_c_per_century"],
                "target_rolling15_correlation": target_metric["rolling15_correlation"],
                "target_best_lag_years": target_metric["best_lag_years"],
                "target_best_lag_correlation": target_metric["best_lag_correlation"],
                "target_phase_projection": baseline._phase_projection(
                    target[early], reference[early], window
                ),
                "hma_trend_c_per_century": baseline._trend(years[early], region[early]),
            }
        )
        definitions.append(
            {
                "case": name,
                "kind": kind,
                "removed_count": len(removed),
                "removed_1based": ";".join(str(index) for index in removed),
            }
        )
        print(f"completed {name}: removed={removed}")

    metric_table = pd.DataFrame(metrics)
    baseline_row = metric_table.loc[metric_table.case.eq("no_removal")].iloc[0]
    for column in (
        "target_trend_c_per_century",
        "target_rolling15_correlation",
        "target_phase_projection",
        "hma_trend_c_per_century",
    ):
        metric_table[f"delta_{column}"] = metric_table[column] - float(baseline_row[column])
    for name in target_curves:
        metric_table.loc[metric_table.case.eq(name), "target_max_abs_change_c"] = float(
            np.max(np.abs(target_curves[name] - target_curves["no_removal"]))
        )
        metric_table.loc[metric_table.case.eq(name), "hma_max_abs_change_c"] = float(
            np.max(np.abs(regional_curves[name] - regional_curves["no_removal"]))
        )

    pd.DataFrame(definitions).to_csv(output / "holdout_case_definitions.csv", index=False)
    metric_table.to_csv(output / "holdout_metrics_1901_1950.csv", index=False)
    pd.DataFrame({"year": years, **target_curves}).to_csv(
        output / "target_3584_holdout_curves.csv", index=False
    )
    pd.DataFrame({"year": years, **regional_curves}).to_csv(
        output / "hma_holdout_curves.csv", index=False
    )
    np.savez_compressed(
        output / "proxy9_holdout_curves.npz",
        years=years,
        rec_ref_target=reference,
        **{f"target_{name}": values for name, values in target_curves.items()},
        **{f"hma_{name}": values for name, values in regional_curves.items()},
    )

    for kind in ("single", "group"):
        _plot_cases(
            years,
            target_curves,
            regional_curves,
            reference,
            case_kinds,
            kind,
            early_period,
            plot_period,
            window,
            output / f"proxy9_{kind}_holdout_curves.png",
        )

    gates = config["gates"]
    closure_pass = all(
        values[key] <= float(gates[key])
        for values in closures.values()
        for key in values
    )
    hashes_after = {name: _sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    passed = preflight["status"] == "pass" and closure_pass and pure

    ranked_trend = metric_table.loc[~metric_table.case.eq("no_removal")].copy()
    ranked_trend["abs_delta_trend"] = ranked_trend[
        "delta_target_trend_c_per_century"
    ].abs()
    ranked_phase = metric_table.loc[~metric_table.case.eq("no_removal")].copy()
    ranked_phase["abs_delta_phase"] = ranked_phase["delta_target_phase_projection"].abs()
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "pass" if passed else "fail",
        "scientific_scope": config["scope"],
        "interpretation": "Each holdout is a complete simultaneous EnKF recomputation; rec_ref is comparison-only.",
        "runtime": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
        },
        "preflight": preflight,
        "closure": closures,
        "source_purity": {
            "unchanged": pure,
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
        },
        "largest_target_trend_changes": ranked_trend.nlargest(
            5, "abs_delta_trend"
        ).drop(columns="abs_delta_trend").to_dict(orient="records"),
        "largest_target_phase_changes": ranked_phase.nlargest(
            5, "abs_delta_phase"
        ).drop(columns="abs_delta_phase").to_dict(orient="records"),
        "files": {
            "metrics": "holdout_metrics_1901_1950.csv",
            "target_curves": "target_3584_holdout_curves.csv",
            "hma_curves": "hma_holdout_curves.csv",
            "single_plot": "proxy9_single_holdout_curves.png",
            "group_plot": "proxy9_group_holdout_curves.png",
            "arrays": "proxy9_holdout_curves.npz",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    display_columns = [
        "case",
        "target_trend_c_per_century",
        "delta_target_trend_c_per_century",
        "target_rolling15_correlation",
        "target_best_lag_years",
        "target_phase_projection",
        "delta_target_phase_projection",
        "hma_trend_c_per_century",
    ]
    final = f"""# Proxy9 Holdout Early-20th-Century Result

Status: **{summary['status']}**.

Every case is a complete controlled proxy9 EnKF recomputation after deleting
the named observation/operator/residual rows. No saved contribution curve was
subtracted. The no-removal state reproduces the controlled baseline with a
maximum absolute difference of `{state_difference:.3e}` degC.

## 3584 and HMA, 1901-1950

{metric_table[display_columns].to_markdown(index=False)}

## Boundary

All source files unchanged: **{pure}**. `rec_ref` is comparison-only. These
results diagnose sensitivity within one fixed-prior experiment and are not a
final reconstruction or OGGM forcing product.
"""
    (output / "FINAL_RESULT.md").write_text(final, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "preflight": preflight,
                "largest_target_trend_changes": summary["largest_target_trend_changes"],
                "largest_target_phase_changes": summary["largest_target_phase_changes"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    if not passed:
        raise RuntimeError("proxy9-holdout-hard-gate-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
