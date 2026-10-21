"""Evaluate documented temperature seasons against annual and intercept controls."""

from __future__ import annotations

import csv
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

BLOCKS = [(1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000)]


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def season_code(value: object) -> str:
    text = str(value).strip()
    normalized = text.replace("[", "").replace("]", "").replace(",", " ")
    numbers = [item for item in normalized.split() if item.isdigit()]
    if text == "Summer" or numbers == ["6", "7", "8"]:
        return "JJA"
    if numbers == ["4", "5", "6"]:
        return "AMJ"
    if numbers == ["4"]:
        return "APR"
    if numbers == [str(month) for month in range(1, 13)]:
        return "ANN"
    raise RuntimeError(f"unsupported-documented-season:{text}")


def fit_rmse(response: np.ndarray, predictor: np.ndarray, years: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intercept = np.full(5, np.nan)
    model = np.full(5, np.nan)
    for fold, (start, end) in enumerate(BLOCKS):
        test = (years >= start) & (years <= end)
        train = ~test
        valid_train = train & np.isfinite(response) & np.isfinite(predictor)
        valid_test = test & np.isfinite(response) & np.isfinite(predictor)
        if valid_train.sum() < 20 or valid_test.sum() < 3:
            continue
        mean = float(np.mean(response[valid_train]))
        coefficient = np.linalg.lstsq(
            np.column_stack([np.ones(valid_train.sum()), predictor[valid_train]]),
            response[valid_train], rcond=None,
        )[0]
        intercept[fold] = float(np.sqrt(np.mean((response[valid_test] - mean) ** 2)))
        prediction = coefficient[0] + coefficient[1] * predictor[valid_test]
        model[fold] = float(np.sqrt(np.mean((response[valid_test] - prediction) ** 2)))
    return intercept, model


def bootstrap(values: np.ndarray, seed: int, draws: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    result = np.full(draws, np.nan)
    for draw in range(draws):
        sample = rng.integers(0, values.shape[0], size=values.shape[0])
        result[draw] = np.nanmedian(values[sample])
    return float(np.nanpercentile(result, 2.5)), float(np.nanpercentile(result, 97.5))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    project = root()
    drivers = Path(__file__).resolve().parent
    base = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = base / "round_08_documented_temperature_seasonality_v1"
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths = {
        "season_export": output / "temperature_seasons_v1.mat",
        "inventory": base / "round_07_proxy_network_suitability_inventory_v2/proxy9_suitability_inventory.csv",
        "proxy9": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "berkeley": phyda / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc",
        "loader": phyda / "load_obs_S.m", "mon2ann": phyda / "mon2ann.m",
        "contract": output / "ROUND_CONTRACT.md", "facts": project / "LIA_PROJECT_FACTS.md",
        "matlab_driver": drivers / "export_climate_reconstruction_loop_v2_round08_temperature_seasons_v1.m",
        "analysis_driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    inventory = pd.read_csv(paths["inventory"])
    temperature_records = inventory[inventory["network_role"] == "documented_temperature"].sort_values("record_index_1based").drop_duplicates("identity_key")
    if len(temperature_records) != 214:
        raise RuntimeError("temperature-identity-count-failed")
    proxy = scipy.io.loadmat(paths["proxy9"], variable_names=["observations"])
    observations = np.asarray(proxy["observations"], dtype=float)
    with h5py.File(paths["season_export"], "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel()
        seasonal = {
            "ANN": np.asarray(handle["annual"], dtype=float).T,
            "JJA": np.asarray(handle["jja"], dtype=float).T,
            "AMJ": np.asarray(handle["amj"], dtype=float).T,
            "APR": np.asarray(handle["april"], dtype=float).T,
        }
    if not np.array_equal(years, np.arange(1940, 2001)) or observations.shape != (61, 297):
        raise RuntimeError("array-contract-failed")
    rows = []
    annual_all, documented_all, intercept_all = [], [], []
    for record in temperature_records.itertuples(index=False):
        index = int(record.record_index_1based) - 1
        code = season_code(record.documented_seasonality)
        response = observations[:, index]
        intercept_rmse, annual_rmse = fit_rmse(response, seasonal["ANN"][:, index], years)
        _, documented_rmse = fit_rmse(response, seasonal[code][:, index], years)
        annual_all.append(annual_rmse)
        documented_all.append(documented_rmse)
        intercept_all.append(intercept_rmse)
        improvement_annual = 1.0 - documented_rmse / annual_rmse
        improvement_intercept = 1.0 - documented_rmse / intercept_rmse
        rows.append({
            "record_index_1based": int(record.record_index_1based), "identity_key": record.identity_key,
            "documented_seasonality": record.documented_seasonality, "season_code": code,
            "finite_fold_count": int(np.isfinite(documented_rmse).sum()),
            "intercept_median_rmse": float(np.nanmedian(intercept_rmse)),
            "annual_temperature_median_rmse": float(np.nanmedian(annual_rmse)),
            "documented_season_median_rmse": float(np.nanmedian(documented_rmse)),
            "median_improvement_over_annual": float(np.nanmedian(improvement_annual)),
            "median_improvement_over_intercept": float(np.nanmedian(improvement_intercept)),
        })
    write_csv(output / "record_temperature_seasonality_skill.csv", rows)
    annual_all = np.asarray(annual_all)
    documented_all = np.asarray(documented_all)
    intercept_all = np.asarray(intercept_all)
    improvement_annual = 1.0 - documented_all / annual_all
    improvement_intercept = 1.0 - documented_all / intercept_all
    lower_annual, upper_annual = bootstrap(improvement_annual, seed=801)
    lower_intercept, upper_intercept = bootstrap(improvement_intercept, seed=802)
    record_improvement = np.nanmedian(improvement_annual, axis=1)
    improved_fraction = float(np.mean(record_improvement > 0))

    group_rows = []
    codes = np.asarray([row["season_code"] for row in rows])
    for code in ("JJA", "ANN", "AMJ", "APR"):
        selected = codes == code
        if not np.any(selected):
            continue
        group_rows.append({
            "season_code": code, "identity_count": int(selected.sum()),
            "median_improvement_over_annual": float(np.nanmedian(improvement_annual[selected])),
            "median_improvement_over_intercept": float(np.nanmedian(improvement_intercept[selected])),
            "improved_identity_fraction": float(np.mean(np.nanmedian(improvement_annual[selected], axis=1) > 0)),
        })
    write_csv(output / "season_group_skill.csv", group_rows)
    median_annual = float(np.nanmedian(improvement_annual))
    median_intercept = float(np.nanmedian(improvement_intercept))
    gates = {
        "median_improvement_over_annual_at_least_1pct": bool(median_annual >= 0.01),
        "bootstrap_lower_over_annual_positive": bool(lower_annual > 0),
        "improved_identity_fraction_over_50pct": bool(improved_fraction > 0.50),
        "median_improvement_over_intercept_at_least_1pct": bool(median_intercept >= 0.01),
        "bootstrap_lower_over_intercept_positive": bool(lower_intercept > 0),
    }
    passes = bool(all(gates.values()))
    summary = {
        "status": "complete", "locked_1921_1939_accessed": False,
        "temperature_identity_count": int(len(rows)),
        "season_counts": {row["season_code"]: row["identity_count"] for row in group_rows},
        "documented_season_vs_annual": {
            "median_relative_rmse_improvement": median_annual,
            "bootstrap_95_lower": lower_annual, "bootstrap_95_upper": upper_annual,
            "improved_identity_fraction": improved_fraction,
        },
        "documented_season_vs_intercept": {
            "median_relative_rmse_improvement": median_intercept,
            "bootstrap_95_lower": lower_intercept, "bootstrap_95_upper": upper_intercept,
        },
        "gates": gates, "passes_documented_temperature_seasonality": passes,
        "decision": "permit_separate_temperature_operator_test" if passes else "close_documented_seasonality_as_network_wide_temperature_repair",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].boxplot([improvement_annual[codes == code][np.isfinite(improvement_annual[codes == code])] * 100 for code in ("JJA", "ANN", "AMJ", "APR")], labels=["JJA", "ANN", "AMJ", "APR"], showfliers=False)
    axes[0].axhline(0, color="#444444", linewidth=1); axes[0].axhline(1, color="#9b2226", linestyle="--", linewidth=1.4)
    axes[0].set(title="Documented season vs annual", ylabel="Relative RMSE improvement (%)")
    axes[1].hist(record_improvement * 100, bins=35, color="#005f73", edgecolor="white")
    axes[1].axvline(0, color="#9b2226", linewidth=1.4)
    axes[1].set(title="Identity-level median improvement", xlabel="Relative RMSE improvement (%)", ylabel="Identities")
    fig.suptitle("Documented temperature seasonality", fontsize=14)
    fig.savefig(output / "documented_temperature_seasonality.png", dpi=180)
    plt.close(fig)
    (output / "FINAL_RESULT.md").write_text(
        "# Documented Temperature Seasonality\n\n"
        f"- Identities: `{len(rows)}`; seasons: `{json.dumps(summary['season_counts'], sort_keys=True)}`.\n"
        f"- Versus annual: median improvement `{median_annual:.3%}`, 95% CI `[{lower_annual:.3%}, {upper_annual:.3%}]`, improved identities `{improved_fraction:.3%}`.\n"
        f"- Versus intercept: median improvement `{median_intercept:.3%}`, 95% CI `[{lower_intercept:.3%}, {upper_intercept:.3%}]`.\n"
        f"- All gates pass: `{str(passes).lower()}`.\n\nDecision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    after = {name: state(path) for name, path in paths.items()}
    (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": before == after, "bootstrap_draws": 5000}, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
