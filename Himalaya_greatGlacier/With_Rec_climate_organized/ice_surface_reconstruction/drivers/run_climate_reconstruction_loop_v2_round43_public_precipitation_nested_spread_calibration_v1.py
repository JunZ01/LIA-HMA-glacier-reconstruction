"""Nested global spread calibration for the public precipitation ensemble."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1940, 2001)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
PRODUCTS = ("GPCC", "CRU")
REGIONS = ("hma", "west_lt80", "central_80_90", "east_ge90")
MEMBERS = ("HGAO", "LMR", "O")


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists(): return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    stat = path.stat(); return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def crps(members: np.ndarray, observation: float) -> float:
    return float(np.mean(np.abs(members - observation)) - 0.5 * np.mean(np.abs(members[:, None] - members[None, :])))


def inflate(members: np.ndarray, scalar: float) -> np.ndarray:
    center = np.mean(members, axis=1, keepdims=True); return center + scalar * (members - center)


def main() -> int:
    project = root(); loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"; output = loop / "round_43_public_precipitation_nested_spread_calibration_v1"
    paths = {"round42_series": loop / "round_42_public_precipitation_probabilistic_envelope_v1/regional_member_observation_series.csv", "round42_summary": loop / "round_42_public_precipitation_probabilistic_envelope_v1/summary.json", "contract": output / "ROUND_CONTRACT.md", "driver": Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists(): raise FileNotFoundError(path)
    if (output / "summary.json").exists(): raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}; series = pd.read_csv(paths["round42_series"])
    if set(series.region) != set(REGIONS) or np.any((series.year >= 1921) & (series.year <= 1939)): raise RuntimeError("series-contract-failed")
    by_region = {region: series[series.region == region].sort_values("year").reset_index(drop=True) for region in REGIONS}
    hma = by_region["hma"]; hma_members = hma[list(MEMBERS)].to_numpy(float)
    rows: list[dict[str, object]] = []; scalar_rows: list[dict[str, object]] = []
    for fold, (start, end) in enumerate(BLOCKS, start=1):
        test = (YEARS >= start) & (YEARS <= end); train = ~test
        for fit_product in PRODUCTS:
            fit_observed = hma[fit_product].to_numpy(float)
            def objective(scalar: float) -> float:
                ensemble = inflate(hma_members[train], scalar)
                return float(np.mean([crps(ensemble[index], value) for index, value in enumerate(fit_observed[train])]))
            optimized = minimize_scalar(objective, bounds=(0.0, 10.0), method="bounded", options={"xatol": 1e-8})
            if not optimized.success: raise RuntimeError(f"spread-fit-failed:{fold}:{fit_product}")
            scalar = float(optimized.x); scalar_rows.append({"fold": fold, "fit_product": fit_product, "spread_scalar": scalar, "training_crps": float(optimized.fun)})
            for eval_product in PRODUCTS:
                for region in REGIONS:
                    frame = by_region[region]; members = frame[list(MEMBERS)].to_numpy(float); observed = frame[eval_product].to_numpy(float); calibrated = inflate(members[test], scalar); raw = members[test]; climatology = observed[train]
                    calibrated_crps = np.asarray([crps(calibrated[index], value) for index, value in enumerate(observed[test])]); raw_crps = np.asarray([crps(raw[index], value) for index, value in enumerate(observed[test])]); climatology_crps = np.asarray([crps(climatology, value) for value in observed[test]])
                    coverage = (observed[test] >= np.min(calibrated, axis=1)) & (observed[test] <= np.max(calibrated, axis=1))
                    rows.append({"fold": fold, "start_year": start, "end_year": end, "fit_product": fit_product, "eval_product": eval_product, "region": region, "spread_scalar": scalar, "calibrated_crps": float(np.mean(calibrated_crps)), "raw_crps": float(np.mean(raw_crps)), "climatology_crps": float(np.mean(climatology_crps)), "crps_skill_over_climatology": float(1 - np.mean(calibrated_crps) / np.mean(climatology_crps)), "crps_improvement_over_raw": float(1 - np.mean(calibrated_crps) / np.mean(raw_crps)), "member_range_coverage": float(np.mean(coverage)), "covered_count": int(np.sum(coverage)), "test_count": int(test.sum())})
    scores = pd.DataFrame(rows); scalars = pd.DataFrame(scalar_rows); scores.to_csv(output / "nested_spread_scores.csv", index=False); scalars.to_csv(output / "spread_scalars.csv", index=False)
    path_rows: list[dict[str, object]] = []; aggregate_coverages: dict[str, float] = {}; hma_positive: dict[str, int] = {}
    for fit_product in PRODUCTS:
        for eval_product in PRODUCTS:
            key = f"{fit_product}_fit_{eval_product}_eval"; selected = scores[(scores.fit_product == fit_product) & (scores.eval_product == eval_product) & (scores.region == "hma")]
            coverage = float(selected.covered_count.sum() / selected.test_count.sum()); aggregate_coverages[key] = coverage; hma_positive[key] = int(np.sum(selected.crps_skill_over_climatology > 0))
            path_rows.append({"fit_product": fit_product, "eval_product": eval_product, "median_hma_crps_skill_over_climatology": float(np.median(selected.crps_skill_over_climatology)), "median_hma_crps_improvement_over_raw": float(np.median(selected.crps_improvement_over_raw)), "positive_hma_fold_count": hma_positive[key], "aggregate_hma_member_range_coverage": coverage})
    paths_frame = pd.DataFrame(path_rows); paths_frame.to_csv(output / "path_summary.csv", index=False)
    scalar_cv = {product: float(scalars[scalars.fit_product == product].spread_scalar.std(ddof=1) / scalars[scalars.fit_product == product].spread_scalar.mean()) for product in PRODUCTS}
    all_scores = scores.crps_skill_over_climatology.to_numpy(float)
    gates = {"all_four_hma_median_crps_skills_at_least_5pct": bool((paths_frame.median_hma_crps_skill_over_climatology >= 0.05).all()), "all_four_hma_median_improvements_over_raw_positive": bool((paths_frame.median_hma_crps_improvement_over_raw > 0).all()), "all_four_hma_paths_positive_at_least_4_of_5_blocks": all(value >= 4 for value in hma_positive.values()), "maximum_region_block_climatology_regression_at_most_10pct": float(np.max(-all_scores)) <= 0.10, "improving_region_block_fraction_over_60pct": float(np.mean(all_scores > 0)) > 0.60, "all_four_hma_coverages_between_35_and_65pct": all(0.35 <= value <= 0.65 for value in aggregate_coverages.values()), "both_fit_product_scalar_cv_at_most_0_25": all(value <= 0.25 for value in scalar_cv.values())}
    passes = bool(all(gates.values())); summary = {"status": "complete", "locked_1921_1939_accessed": False, "spread_scalar_medians": {product: float(scalars[scalars.fit_product == product].spread_scalar.median()) for product in PRODUCTS}, "spread_scalar_cv": scalar_cv, "path_results": paths_frame.to_dict(orient="records"), "maximum_region_block_regression": float(np.max(-all_scores)), "improving_region_block_fraction": float(np.mean(all_scores > 0)), "gates": gates, "passes_nested_spread_calibration": passes, "decision": "construct_allowed_period_calibrated_regional_uncertainty_envelope" if passes else "close_public_precipitation_probabilistic_calibration"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for product, color in [("GPCC", "#0072b2"), ("CRU", "#d55e00")]: axes[0].plot(scalars[scalars.fit_product == product].fold, scalars[scalars.fit_product == product].spread_scalar, marker="o", label=product, color=color)
    axes[0].set(xlabel="Held-out block", ylabel="Spread scalar", title="Nested HMA spread calibration"); axes[0].legend(frameon=False)
    labels = [f"{row.fit_product}->{row.eval_product}" for row in paths_frame.itertuples(index=False)]; axes[1].bar(labels, paths_frame.median_hma_crps_skill_over_climatology * 100, color=["#0072b2" if value >= 5 else "#d55e00" for value in paths_frame.median_hma_crps_skill_over_climatology * 100]); axes[1].axhline(0, color="#555555"); axes[1].axhline(5, color="#009e73", linestyle="--"); axes[1].set(ylabel="Median HMA CRPS skill (%)", title="Held-out probabilistic skill"); axes[1].tick_params(axis="x", rotation=20); figure.savefig(output / "nested_spread_calibration.png", dpi=180); plt.close(figure)
    lines = ["# Nested Public-Precipitation Spread Calibration", "", f"- Median scalars: `{summary['spread_scalar_medians']}`; CV: `{scalar_cv}`."]
    for row in paths_frame.itertuples(index=False): lines.append(f"- {row.fit_product}->{row.eval_product}: climatology skill `{row.median_hma_crps_skill_over_climatology:.2%}`, raw improvement `{row.median_hma_crps_improvement_over_raw:.2%}`, positive `{row.positive_hma_fold_count}/5`, coverage `{row.aggregate_hma_member_range_coverage:.1%}`.")
    lines.extend(["", f"All gates pass: `{str(passes).lower()}`.", f"Decision: `{summary['decision']}`."]); (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}; unchanged = before == after; (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged: raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
