"""Round 02: bounded cross-fitted screening of legacy member composition."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


EXPERIMENT_ID = "climate_reconstruction_improvement_loop_v1/round_02_network_operator_repair"
MEMBERS = ["AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9"]
FIT_REGIONS = ["west_lt80", "central_80_90", "east_ge90"]
EVAL_REGIONS = ["hma", *FIT_REGIONS]
BLOCKS = [(1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000)]
FIXED_WEIGHTS = {
    "equal_six_control": np.full(6, 1.0 / 6.0),
    "proxy9_mean": np.asarray([0.0, 1 / 3, 0.0, 1 / 3, 0.0, 1 / 3]),
    "ao_network_mean": np.asarray([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
}


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


def state(path: Path) -> dict[str, object]:
    item = path.stat()
    return {"path": str(path), "size_bytes": int(item.st_size), "mtime_ns": int(item.st_mtime_ns)}


def fit_weights(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = np.isfinite(target) & np.all(np.isfinite(matrix), axis=1)
    x = matrix[valid]
    y = target[valid]
    if x.shape[0] < 30:
        raise RuntimeError("insufficient-training-values")
    result = minimize(
        lambda weights: float(np.mean(np.square(x @ weights - y))),
        np.full(len(MEMBERS), 1.0 / len(MEMBERS)),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(MEMBERS),
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"weight-fit-failed:{result.message}")
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    return weights / weights.sum()


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) == 0 or np.std(right[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0" / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    round01 = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_01_blocked_cv_baseline"
    curves_path = round01 / "regional_annual_curves.csv"
    summary_path = round01 / "summary.json"
    lock_contract = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_instrumental_baseline_v1/LOCKED_FINAL_PERIOD.json"
    facts = root / "LIA_PROJECT_FACTS.md"
    driver = Path(__file__).resolve()
    sources = [curves_path, summary_path, lock_contract, facts]
    if missing := [str(path) for path in sources if not path.exists()]:
        raise FileNotFoundError("missing-round02-inputs:" + ";".join(missing))
    locked = json.loads(lock_contract.read_text(encoding="utf-8"))
    if locked.get("status") != "locked_not_evaluated":
        raise RuntimeError("locked-period-contract-failed")
    before = {str(path): state(path) for path in sources}
    hashes = {str(path): digest(path) for path in sources}

    frame = pd.read_csv(curves_path)
    if int(frame["year"].min()) != 1940 or int(frame["year"].max()) != 2000:
        raise RuntimeError("unexpected-round01-year-domain")
    if frame["year"].between(1921, 1939).any():
        raise RuntimeError("locked-period-present")
    years = np.arange(1940, 2001, dtype=int)
    weight_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    fitted_curves: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for variable in ("tas", "pr"):
        fitted_curves[variable] = {branch: {} for branch in [*FIXED_WEIGHTS, "stacked_nonnegative"]}
        recon = {}
        obs = {}
        for region in EVAL_REGIONS:
            recon[region] = np.column_stack([
                frame[(frame["variable"] == variable) & (frame["region"] == region) & (frame["series_type"] == "reconstruction") & (frame["series"] == member)].sort_values("year")["value"].to_numpy()
                for member in MEMBERS
            ])
            labels = frame[(frame["variable"] == variable) & (frame["region"] == region) & (frame["series_type"] == "observation")]["series"].unique().tolist()
            obs[region] = {
                label: frame[(frame["variable"] == variable) & (frame["region"] == region) & (frame["series_type"] == "observation") & (frame["series"] == label)].sort_values("year")["value"].to_numpy()
                for label in labels
            }
        crossfit = {branch: {region: np.full(len(years), np.nan) for region in EVAL_REGIONS} for branch in [*FIXED_WEIGHTS, "stacked_nonnegative"]}

        for fold, (start, end) in enumerate(BLOCKS, start=1):
            test = (years >= start) & (years <= end)
            train = ~test
            train_x = np.concatenate([recon[region][train] for region in FIT_REGIONS], axis=0)
            train_y = np.concatenate([
                np.nanmean(np.column_stack(list(obs[region].values())), axis=1)[train]
                for region in FIT_REGIONS
            ])
            stacked = fit_weights(train_x, train_y)
            branch_weights = {**FIXED_WEIGHTS, "stacked_nonnegative": stacked}
            for branch, weights in branch_weights.items():
                for member, value in zip(MEMBERS, weights):
                    weight_rows.append({"variable": variable, "fold": fold, "test_start": start, "test_end": end, "branch": branch, "member": member, "weight": float(value)})
                for region in EVAL_REGIONS:
                    crossfit[branch][region][test] = recon[region][test] @ weights

        for branch in crossfit:
            for region in EVAL_REGIONS:
                values = crossfit[branch][region]
                if np.any(~np.isfinite(values)):
                    raise RuntimeError(f"crossfit-coverage-failed:{variable}:{branch}:{region}")
                fitted_curves[variable][branch][region] = values
                for year, value in zip(years, values):
                    curve_rows.append({"year": int(year), "variable": variable, "branch": branch, "region": region, "value": float(value)})
                for observation, observed in obs[region].items():
                    valid = np.isfinite(values) & np.isfinite(observed)
                    delta = values[valid] - observed[valid]
                    score_rows.append({
                        "variable": variable,
                        "branch": branch,
                        "region": region,
                        "observation": observation,
                        "valid_year_count": int(valid.sum()),
                        "bias": float(np.mean(delta)),
                        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
                        "correlation": correlation(values, observed),
                    })

        control = {(row["region"], row["observation"]): row for row in score_rows if row["variable"] == variable and row["branch"] == "equal_six_control"}
        for branch in [*FIXED_WEIGHTS, "stacked_nonnegative"]:
            selected = [row for row in score_rows if row["variable"] == variable and row["branch"] == branch]
            ratios = [row["rmse"] / control[(row["region"], row["observation"])]["rmse"] - 1.0 for row in selected]
            median_rmse = float(np.median([row["rmse"] for row in selected]))
            control_median = float(np.median([row["rmse"] for row in control.values()]))
            improvement = 1.0 - median_rmse / control_median
            aggregate_rows.append({
                "variable": variable,
                "branch": branch,
                "median_region_reference_rmse": median_rmse,
                "control_median_rmse": control_median,
                "relative_improvement": float(improvement),
                "maximum_pair_relative_regression": float(max(ratios)),
                "passes_5pct_primary": bool(improvement >= 0.05),
                "passes_5pct_no_regression": bool(max(ratios) <= 0.05 + 1e-12),
                "passes_screen": bool(improvement >= 0.05 and max(ratios) <= 0.05 + 1e-12),
            })

    passed = [row for row in aggregate_rows if row["branch"] != "equal_six_control" and row["passes_screen"]]
    decision = "advance_best_screened_composition_to_full_operator_oof" if passed else "close_legacy_composition_reweighting_screen"
    selected = max(passed, key=lambda row: row["relative_improvement"]) if passed else None

    output.mkdir(parents=True)
    write_csv(output / "outer_fold_weights.csv", weight_rows)
    write_csv(output / "crossfit_region_reference_scores.csv", score_rows)
    write_csv(output / "composition_screen_summary.csv", aggregate_rows)
    write_csv(output / "crossfit_regional_curves.csv", curve_rows)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    styles = {"equal_six_control": ("#111111", 2.6), "proxy9_mean": ("#0072B2", 1.7), "ao_network_mean": ("#009E73", 1.7), "stacked_nonnegative": ("#D55E00", 2.0)}
    for axis, variable in zip(axes, ("tas", "pr")):
        for branch, (color, width) in styles.items():
            axis.plot(years, fitted_curves[variable][branch]["hma"], color=color, linewidth=width, label=branch)
        obs_lines = frame[(frame["variable"] == variable) & (frame["region"] == "hma") & (frame["series_type"] == "observation")]
        for index, label in enumerate(obs_lines["series"].unique()):
            values = obs_lines[obs_lines["series"] == label].sort_values("year")["value"].to_numpy()
            axis.plot(years, values, color=plt.cm.Greys(0.4 + 0.1 * index), linestyle="--", linewidth=1.0, alpha=0.8)
        axis.axhline(0, color="#AAAAAA", linewidth=0.7)
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel("degC" if variable == "tas" else "mm/day")
        axis.set_title("Temperature composition screen" if variable == "tas" else "Precipitation composition screen")
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    axes[1].set_xlabel("Year")
    fig.savefig(output / "figure_hma_composition_crossfit_1940_2000.png", dpi=220)
    plt.close(fig)

    after = {str(path): state(path) for path in sources}
    if before != after:
        raise RuntimeError("source-changed-during-round02")
    config = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outer_blocks": BLOCKS,
        "control": "equal_six_control",
        "variants": ["proxy9_mean", "ao_network_mean", "stacked_nonnegative"],
        "primary_gate": "relative median region-reference RMSE improvement >= 5 percent",
        "no_regression_gate": "maximum region-reference RMSE regression <= 5 percent",
        "interpretation_boundary": "Weights are cross-fitted, but the legacy base members were calibrated over 1940-2000. Passing this screen authorizes full operator-level OOF confirmation only, not production promotion.",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "sha256": hashes, "driver_sha256": digest(driver)}, indent=2), encoding="utf-8")
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "decision": decision,
        "selected_for_operator_oof": selected,
        "aggregate": aggregate_rows,
        "all_declared_sources_unchanged": before == after,
        "interpretation_boundary": config["interpretation_boundary"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# Round 02 Contract\n\nDecision: determine whether a bounded method/network composition can beat the equal six-member control strongly enough to justify a full operator-level OOF rerun.\n",
        encoding="utf-8",
    )
    result_line = f"Selected: `{selected['variable']} / {selected['branch']}`." if selected else "No variant passed both gates; close this composition screen."
    (output / "FINAL_RESULT.md").write_text(
        "# Round 02 Composition Screen\n\n"
        f"- Decision: `{decision}`\n"
        f"- {result_line}\n"
        "- Locked 1921-1939 values accessed: `false`.\n"
        "- A passing screen is not a production acceptance; it only authorizes full operator-level OOF confirmation.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
