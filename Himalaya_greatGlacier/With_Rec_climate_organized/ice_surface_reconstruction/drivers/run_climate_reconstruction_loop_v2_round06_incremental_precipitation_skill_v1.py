"""Test incremental proxy prediction skill from the Round 05 precipitation term."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLOCKS = [(1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000)]
BRANCHES = (
    "gpcc_fit_gpcc_eval",
    "cru_fit_cru_eval",
    "gpcc_fit_cru_eval",
    "cru_fit_gpcc_eval",
)


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return {
        "path": str(path), "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns), "sha256": value.hexdigest(),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_record_median(values: np.ndarray, seed: int = 0, draws: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    nrecord = values.shape[0]
    result = np.full(draws, np.nan)
    for draw in range(draws):
        sample = rng.integers(0, nrecord, size=nrecord)
        result[draw] = np.nanmedian(values[sample])
    return float(np.nanpercentile(result, 2.5)), float(np.nanpercentile(result, 97.5))


def main() -> int:
    project = root()
    round05 = project / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/round_05_cross_product_psm_stability_v1"
    )
    output = project / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/round_06_incremental_precipitation_skill_v1"
    )
    paths = {
        "round05_mat": round05 / "cross_product_psm_v1.mat",
        "round05_contract": round05 / "ROUND_CONTRACT.md",
        "proxy9": project / (
            "validation_data/results/paleo_surface_validation_v0/"
            "climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/"
            "proxy9_allowed_1940_2000_official_baseline_v2.mat"
        ),
        "contract": output / "ROUND_CONTRACT.md",
        "facts": project / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    proxy = scipy.io.loadmat(paths["proxy9"], variable_names=["observations", "model_type"])
    observations = np.asarray(proxy["observations"], dtype=float)
    model_type = np.asarray(proxy["model_type"], dtype=int).ravel()
    with h5py.File(paths["round05_mat"], "r") as handle:
        years = np.asarray(handle["years"], dtype=int).ravel()
        temperature = np.asarray(handle["temperature"], dtype=float).T
        holdout = np.asarray(handle["holdout_rmse"], dtype=float).transpose(3, 2, 1, 0)
    if not np.array_equal(years, np.arange(1940, 2001)) or observations.shape != (61, 297):
        raise RuntimeError("input-contract-failed")
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("locked-period-accessed")
    sensitive = model_type == 2
    indices = np.flatnonzero(sensitive)
    if len(indices) != 290:
        raise RuntimeError("proxy-count-failed")

    intercept_rmse = np.full((len(indices), 5), np.nan)
    temperature_rmse = np.full_like(intercept_rmse, np.nan)
    test_count = np.zeros_like(intercept_rmse, dtype=int)
    for local, record in enumerate(indices):
        response = observations[:, record]
        predictor = temperature[:, record]
        for fold, (start, end) in enumerate(BLOCKS):
            test = (years >= start) & (years <= end)
            train = ~test
            valid_train = train & np.isfinite(response) & np.isfinite(predictor)
            valid_test = test & np.isfinite(response) & np.isfinite(predictor)
            if valid_train.sum() < 20 or valid_test.sum() < 3:
                continue
            intercept = float(np.mean(response[valid_train]))
            design = np.column_stack([np.ones(valid_train.sum()), predictor[valid_train]])
            coefficient = np.linalg.lstsq(design, response[valid_train], rcond=None)[0]
            intercept_prediction = np.full(valid_test.sum(), intercept)
            temperature_prediction = coefficient[0] + coefficient[1] * predictor[valid_test]
            intercept_rmse[local, fold] = float(np.sqrt(np.mean((response[valid_test] - intercept_prediction) ** 2)))
            temperature_rmse[local, fold] = float(np.sqrt(np.mean((response[valid_test] - temperature_prediction) ** 2)))
            test_count[local, fold] = int(valid_test.sum())

    candidate = {
        "gpcc_fit_gpcc_eval": holdout[sensitive, :, 0, 0],
        "cru_fit_cru_eval": holdout[sensitive, :, 1, 1],
        "gpcc_fit_cru_eval": holdout[sensitive, :, 0, 1],
        "cru_fit_gpcc_eval": holdout[sensitive, :, 1, 0],
    }
    improvement = {
        name: 1.0 - values / temperature_rmse for name, values in candidate.items()
    }

    branch_rows: list[dict[str, object]] = []
    gates: dict[str, bool] = {}
    for branch_index, name in enumerate(BRANCHES):
        values = improvement[name]
        valid = np.isfinite(values)
        lower, upper = bootstrap_record_median(values, seed=100 + branch_index)
        median_improvement = float(np.nanmedian(values))
        branch_rows.append({
            "branch": name,
            "paired_proxy_fold_count": int(valid.sum()),
            "temperature_only_median_rmse": float(np.nanmedian(temperature_rmse[valid])),
            "bivariate_median_rmse": float(np.nanmedian(candidate[name][valid])),
            "median_relative_rmse_improvement": median_improvement,
            "improved_proxy_fold_fraction": float(np.mean(values[valid] > 0)),
            "bootstrap_95_lower": lower,
            "bootstrap_95_upper": upper,
        })
        gates[f"{name}_median_improvement_at_least_1pct"] = bool(median_improvement >= 0.01)
        gates[f"{name}_bootstrap_lower_positive"] = bool(lower > 0)
    write_csv(output / "branch_incremental_skill.csv", branch_rows)

    record_rows: list[dict[str, object]] = []
    robust = np.zeros(len(indices), dtype=bool)
    for local, record in enumerate(indices):
        medians = {name: float(np.nanmedian(improvement[name][local])) for name in BRANCHES}
        robust[local] = all(value > 0 for value in medians.values())
        record_rows.append({
            "record_index_1based": int(record + 1),
            "intercept_only_median_rmse": float(np.nanmedian(intercept_rmse[local])),
            "temperature_only_median_rmse": float(np.nanmedian(temperature_rmse[local])),
            **{f"{name}_median_improvement": value for name, value in medians.items()},
            "positive_all_four_paths": bool(robust[local]),
        })
    write_csv(output / "record_incremental_skill.csv", record_rows)
    robust_fraction = float(np.mean(robust))
    gates["positive_all_four_record_fraction_at_least_50pct"] = bool(robust_fraction >= 0.50)

    temp_vs_intercept = 1.0 - temperature_rmse / intercept_rmse
    passes = bool(all(gates.values()))
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "record_count": int(len(indices)),
        "paired_proxy_fold_count": int(np.isfinite(temperature_rmse).sum()),
        "temperature_only_vs_intercept": {
            "median_relative_rmse_improvement": float(np.nanmedian(temp_vs_intercept)),
            "improved_proxy_fold_fraction": float(np.mean(temp_vs_intercept[np.isfinite(temp_vs_intercept)] > 0)),
        },
        "branches": {row["branch"]: {key: value for key, value in row.items() if key != "branch"} for row in branch_rows},
        "positive_all_four_record_count": int(robust.sum()),
        "positive_all_four_record_fraction": robust_fraction,
        "gates": gates,
        "passes_incremental_precipitation_skill": passes,
        "decision": "permit_separate_operator_test" if passes else "close_current_annual_bivariate_precipitation_operator_promotion",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    data = [improvement[name][np.isfinite(improvement[name])] * 100 for name in BRANCHES]
    axes[0].boxplot(data, tick_labels=["G/G", "C/C", "G/C", "C/G"], showfliers=False)
    axes[0].axhline(1.0, color="#9b2226", linestyle="--", linewidth=1.5)
    axes[0].axhline(0.0, color="#444444", linewidth=1)
    axes[0].set(title="Increment over temperature-only", ylabel="Relative RMSE improvement (%)", xlabel="Fit/evaluation product")
    record_matrix = np.column_stack([
        [row[f"{name}_median_improvement"] * 100 for row in record_rows] for name in BRANCHES
    ])
    axes[1].hist(np.nanmin(record_matrix, axis=1), bins=30, color="#005f73", edgecolor="white")
    axes[1].axvline(0.0, color="#9b2226", linewidth=1.5)
    axes[1].set(title="Weakest path per record", xlabel="Minimum median improvement (%)", ylabel="Proxy records")
    fig.suptitle("Incremental precipitation skill in proxy9 PSMs", fontsize=14)
    fig.savefig(output / "incremental_precipitation_skill.png", dpi=180)
    plt.close(fig)

    report = [
        "# Incremental Precipitation Skill", "",
        f"- Temperature-only versus intercept median improvement: `{summary['temperature_only_vs_intercept']['median_relative_rmse_improvement']:.3%}`.",
    ]
    for row in branch_rows:
        report.append(
            f"- `{row['branch']}`: median improvement `{row['median_relative_rmse_improvement']:.3%}`, "
            f"95% bootstrap CI `[{row['bootstrap_95_lower']:.3%}, {row['bootstrap_95_upper']:.3%}]`, "
            f"improved proxy-fold fraction `{row['improved_proxy_fold_fraction']:.3%}`."
        )
    report.extend([
        f"- Positive on all four paths: `{robust.sum()}/{len(robust)}` (`{robust_fraction:.3%}`).",
        f"- All predeclared gates pass: `{str(passes).lower()}`.", "",
        f"Decision: `{summary['decision']}`.", "",
        "No climate source was reread, no EnKF or OGGM model was run, and 1921-1939 was not accessed.",
    ])
    (output / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    (output / "source_manifest.json").write_text(json.dumps({
        "sources_before": before, "sources_after": after, "source_states_unchanged": before == after,
        "bootstrap_draws": 5000, "bootstrap_unit": "proxy_record", "bootstrap_seed_base": 100,
    }, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
