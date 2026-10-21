"""Validate the raw three-member public precipitation ensemble probabilistically."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1940, 2001)
BLOCKS = ((1940, 1951), (1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))
MEMBERS = {"HGAO": "pr_rec_ref_HGAO_c05d", "LMR": "pr_rec_ref_LMR_c05d", "O": "pr_rec_ref_O_c05d"}
OBSERVATIONS = {"CRU": "pr_obs_cru_pre_05deg", "GPCC": "pr_obs_gpcp_05deg"}
REGIONS = {"hma": (-np.inf, np.inf), "west_lt80": (-np.inf, 80.0), "central_80_90": (80.0, 90.0), "east_ge90": (90.0, np.inf)}


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists(): return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    stat = path.stat(); return {"path": str(path), "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def clean(dataset: h5py.Dataset, selection: slice) -> np.ndarray:
    values = np.asarray(dataset[selection], dtype=float); fill = float(np.asarray(dataset.attrs.get("_FillValue", -9999.0)).squeeze()); values[np.isclose(values, fill) | (values < -9000)] = np.nan; return values


def recenter(values: np.ndarray) -> np.ndarray:
    baseline = (YEARS >= 1961) & (YEARS <= 1990); return values - np.nanmean(values[baseline], axis=0)[None]


def weighted_series(values: np.ndarray, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask; valid = np.isfinite(values); numerator = np.nansum(values * weights[None], axis=(1, 2)); denominator = np.sum(valid * weights[None], axis=(1, 2)); return numerator / np.where(denominator == 0, np.nan, denominator)


def crps_ensemble(members: np.ndarray, observation: float) -> float:
    members = members[np.isfinite(members)]
    return float(np.mean(np.abs(members - observation)) - 0.5 * np.mean(np.abs(members[:, None] - members[None, :])))


def main() -> int:
    project = root(); loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"; output = loop / "round_42_public_precipitation_probabilistic_envelope_v1"
    paths = {"source": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"), "mask": project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy", "round31_summary": loop / "round_31_precipitation_fixed_reference_candidate_skill_v1/summary.json", "contract": output / "ROUND_CONTRACT.md", "driver": Path(__file__).resolve()}
    for path in paths.values():
        if not path.exists(): raise FileNotFoundError(path)
    if (output / "summary.json").exists(): raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}; mask = np.load(paths["mask"]).astype(bool)
    with h5py.File(paths["source"], "r") as handle:
        all_years = np.asarray(handle["time"], dtype=int).ravel(); indices = np.flatnonzero((all_years >= 1940) & (all_years <= 2000)); selection = slice(int(indices[0]), int(indices[-1]) + 1)
        if not np.array_equal(all_years[indices], YEARS) or np.any((all_years[indices] >= 1921) & (all_years[indices] <= 1939)): raise RuntimeError("year-contract-failed")
        latitudes = np.asarray(handle["lat"], dtype=float).ravel(); longitudes = np.asarray(handle["lon"], dtype=float).ravel()
        members = {name: recenter(clean(handle[variable], selection)) for name, variable in MEMBERS.items()}; observations = {name: recenter(clean(handle[variable], selection)) for name, variable in OBSERVATIONS.items()}
    longitude_grid = np.broadcast_to(longitudes[None], mask.shape); region_masks = {name: mask & (longitude_grid >= lower) & (longitude_grid < upper) for name, (lower, upper) in REGIONS.items()}
    block_rows: list[dict[str, object]] = []; coverage_rows: list[dict[str, object]] = []; series_rows: list[dict[str, object]] = []
    for region_name, region_mask in region_masks.items():
        common = region_mask.copy()
        for values in [*members.values(), *observations.values()]: common &= np.all(np.isfinite(values), axis=0)
        if common.sum() == 0: raise RuntimeError(f"empty-common-support:{region_name}")
        member_series = np.column_stack([weighted_series(members[name], common, latitudes) for name in MEMBERS]); observation_series = {name: weighted_series(values, common, latitudes) for name, values in observations.items()}
        for index, year in enumerate(YEARS):
            series_rows.append({"year": int(year), "region": region_name, **{name: float(member_series[index, column]) for column, name in enumerate(MEMBERS)}, **{name: float(values[index]) for name, values in observation_series.items()}})
        for observation_name, observed in observation_series.items():
            ranks = np.sum(member_series < observed[:, None], axis=1); covered = (observed >= np.min(member_series, axis=1)) & (observed <= np.max(member_series, axis=1))
            coverage_rows.append({"observation": observation_name, "region": region_name, "member_range_coverage": float(np.mean(covered)), "rank_0_fraction": float(np.mean(ranks == 0)), "rank_1_fraction": float(np.mean(ranks == 1)), "rank_2_fraction": float(np.mean(ranks == 2)), "rank_3_fraction": float(np.mean(ranks == 3)), "mean_member_spread": float(np.mean(np.std(member_series, axis=1, ddof=1))), "mean_absolute_ensemble_mean_error": float(np.mean(np.abs(np.mean(member_series, axis=1) - observed)))})
            for fold, (start, end) in enumerate(BLOCKS, start=1):
                test = (YEARS >= start) & (YEARS <= end); train = ~test; climatology = observed[train]
                ensemble_crps = np.asarray([crps_ensemble(member_series[index], observed[index]) for index in np.flatnonzero(test)]); climatology_crps = np.asarray([crps_ensemble(climatology, observed[index]) for index in np.flatnonzero(test)])
                skill = float(1 - np.mean(ensemble_crps) / np.mean(climatology_crps))
                block_rows.append({"observation": observation_name, "region": region_name, "fold": fold, "start_year": start, "end_year": end, "ensemble_mean_crps": float(np.mean(ensemble_crps)), "climatology_mean_crps": float(np.mean(climatology_crps)), "crps_skill": skill, "member_range_coverage": float(np.mean(covered[test]))})
    blocks = pd.DataFrame(block_rows); coverage = pd.DataFrame(coverage_rows); series = pd.DataFrame(series_rows); blocks.to_csv(output / "probabilistic_block_scores.csv", index=False); coverage.to_csv(output / "probabilistic_full_period_coverage.csv", index=False); series.to_csv(output / "regional_member_observation_series.csv", index=False)
    improvements = blocks.crps_skill.to_numpy(float); hma = blocks[blocks.region == "hma"]
    hma_positive_counts = {observation: int(np.sum(hma[hma.observation == observation].crps_skill > 0)) for observation in OBSERVATIONS}
    coverage_pairs = {f"{row.observation}_{row.region}": float(row.member_range_coverage) for row in coverage.itertuples(index=False)}
    gates = {"median_block_crps_skill_at_least_5pct": float(np.median(improvements)) >= 0.05, "maximum_block_regression_at_most_10pct": float(np.max(-improvements)) <= 0.10, "improving_block_fraction_over_60pct": float(np.mean(improvements > 0)) > 0.60, "both_observations_hma_positive_at_least_4_of_5_blocks": all(value >= 4 for value in hma_positive_counts.values()), "all_observation_region_coverages_between_35_and_65pct": bool(((coverage.member_range_coverage >= 0.35) & (coverage.member_range_coverage <= 0.65)).all())}
    passes = bool(all(gates.values())); summary = {"status": "complete", "locked_1921_1939_accessed": False, "median_block_crps_skill": float(np.median(improvements)), "maximum_block_regression": float(np.max(-improvements)), "improving_block_fraction": float(np.mean(improvements > 0)), "hma_positive_fold_counts": hma_positive_counts, "coverage_by_observation_region": coverage_pairs, "gates": gates, "passes_raw_probabilistic_envelope": passes, "decision": "construct_allowed_period_probabilistic_reference_envelope" if passes else "close_raw_public_precipitation_probabilistic_envelope"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    hma_series = series[series.region == "hma"]; figure, axes = plt.subplots(2, 1, figsize=(12, 7.5), constrained_layout=True)
    axes[0].fill_between(hma_series.year, hma_series[list(MEMBERS)].min(axis=1), hma_series[list(MEMBERS)].max(axis=1), color="#9ecae1", alpha=0.5, label="Member range"); axes[0].plot(hma_series.year, hma_series[list(MEMBERS)].mean(axis=1), color="#08519c", label="Member mean"); axes[0].plot(hma_series.year, hma_series.GPCC, color="#009e73", label="GPCC"); axes[0].plot(hma_series.year, hma_series.CRU, color="#d55e00", label="CRU"); axes[0].set(ylabel="HMA precipitation anomaly (mm/day)", title="Raw three-member regional envelope"); axes[0].legend(frameon=False, ncol=4)
    for observation, color in [("GPCC", "#009e73"), ("CRU", "#d55e00")]: axes[1].plot(blocks[(blocks.region == "hma") & (blocks.observation == observation)].fold, blocks[(blocks.region == "hma") & (blocks.observation == observation)].crps_skill * 100, marker="o", label=observation, color=color)
    axes[1].axhline(0, color="#555555"); axes[1].axhline(5, color="#08519c", linestyle="--"); axes[1].set(xlabel="Held-out block", ylabel="CRPS skill (%)", title="HMA probabilistic skill against training climatology"); axes[1].legend(frameon=False); figure.savefig(output / "public_precipitation_probabilistic_envelope.png", dpi=180); plt.close(figure)
    lines = ["# Public Precipitation Probabilistic Envelope", "", f"- Median block CRPS skill: `{summary['median_block_crps_skill']:.2%}`; improving blocks: `{summary['improving_block_fraction']:.1%}`.", f"- HMA positive folds: `{hma_positive_counts}`.", f"- All coverage gates pass: `{str(gates['all_observation_region_coverages_between_35_and_65pct']).lower()}`.", "", f"All gates pass: `{str(passes).lower()}`.", f"Decision: `{summary['decision']}`."]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = {name: state(path) for name, path in paths.items()}; unchanged = before == after; (output / "source_manifest.json").write_text(json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2), encoding="utf-8")
    if not unchanged: raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
