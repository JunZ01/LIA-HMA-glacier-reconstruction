"""Validate the frozen 50:50 spatial blend on explicitly declared common support."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import preflight_temperature_pmip_equal_blend_spatial_field_v1 as base
import preflight_temperature_pmip_equal_blend_spatial_field_v3 as retry


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_pmip_equal_blend_spatial_common_support_v1"
REFERENCE = base.REFERENCE
CATALOG = base.CATALOG
MODE_SIM = base.MODE_SIM
REGIONAL_WINDOWS = base.REGIONAL_WINDOWS
ASSIGNMENT = base.ASSIGNMENT
SUPPORT = base.SUPPORT
FACTS = base.FACTS
DRIVER = Path(__file__).resolve()
PRIOR = base.regional_base.SOURCE_RECOVERY / "replay_proxy6_enkf_exp1_matlab_v1/matlab_prior_336x848.npy"
INDICES = base.regional_base.ARCHIVE / "fixed_proxy_experiments_69/tas/indices/all_Xb_indices.mat"
PROXY6 = base.regional_base.ARCHIVE / "python_data/proxy_data_6.mat"
PROXY9 = base.regional_base.ARCHIVE / "python_data/proxy_data_9.mat"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def domain_mean(values: np.ndarray, point_mask: np.ndarray, point_lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(point_lat[point_mask]))
    selected = values[:, point_mask]
    return np.sum(selected * weights[None], axis=1) / np.sum(weights)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Fixed 50:50 Spatial Common-Support Contract\n\n"
        "Retain the frozen 0.5*background + 0.5*equal-three-model-mean formula and eight 50-year "
        "windows from 1500-1899. Define common support only by finite availability in every input "
        "and window, before scoring. Require >=99.9% area-weighted HMA coverage; zero excluded "
        "GTN-G cells; maximum change in accepted HMA aggregate windows <=0.005 degC; GTN-G aggregate "
        "closure <=1e-6 degC; positive HMA spatiotemporal RMSE improvement; >=67% area-weighted "
        "common-support cells improving; >=12/15 GTN-G regions improving; and no more than 10% "
        "degradation in 13-08, 14-03, 15-01, or 15-02. No amplitude is fitted, no annual production "
        "field is built, and no 1901-1939 value is read.\n",
        encoding="utf-8",
    )
    paths = {
        "reference": REFERENCE, "catalog": CATALOG, "mode_sim": MODE_SIM,
        "regional_windows": REGIONAL_WINDOWS, "assignment": ASSIGNMENT, "support": SUPPORT,
        "prior": PRIOR, "indices": INDICES, "proxy6": PROXY6, "proxy9": PROXY9,
        "facts": FACTS, "driver": DRIVER, "contract": contract,
    }
    before = {name: sha256(path) for name, path in paths.items()}

    with Dataset(REFERENCE) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        hma_mask = np.asarray(dataset.variables["hma_mask"][:], dtype=bool)
        variable = dataset.variables["tas_reference_family_balanced"]
        reference = np.stack([
            np.nanmean(np.asarray(variable[np.flatnonzero((years >= start) & (years <= end))], dtype=float), axis=0)[hma_mask]
            for start, end in base.WINDOWS
        ])
    background = base.background_windows(lat, lon, hma_mask)
    model_fields = {"ModE-Sim": base.mode_sim_windows(lat, lon, hma_mask)}
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    request_audit = []
    base.remote_segment_values = retry.chunked_remote_segment_values
    for model in ["MPI-ESM1-2-LR", "MIROC-ES2L"]:
        model_fields[model], audit = base.remote_model_windows(
            model, [item for item in catalog if item["model"] == model], lat, lon, hma_mask
        )
        request_audit.extend(audit)
    forced = np.mean(np.stack([model_fields[model] for model in base.MODELS]), axis=0)
    candidate = 0.5 * background + 0.5 * forced

    finite = np.all(np.isfinite(reference), axis=0)
    for values in [background, forced, candidate, *model_fields.values()]:
        finite &= np.all(np.isfinite(values), axis=0)
    points = base.target_points(lat, lon, hma_mask)
    point_lat = points[:, 0]
    weights = np.cos(np.deg2rad(point_lat))
    coverage_fraction = float(np.sum(weights[finite]) / np.sum(weights))

    assignment = pd.read_csv(ASSIGNMENT)
    support = pd.read_csv(SUPPORT).sort_values("o2region")
    domain_masks = {"hma": np.ones(int(hma_mask.sum()), dtype=bool)}
    excluded_gtng_count = 0
    for region in support.itertuples(index=False):
        grid_mask = np.zeros_like(hma_mask)
        cells = assignment[assignment.o2region.eq(region.o2region)]
        grid_mask[cells.grid_row.to_numpy(int), cells.grid_col.to_numpy(int)] = True
        point_mask = grid_mask[hma_mask]
        domain_masks[region.o2region] = point_mask
        excluded_gtng_count += int(np.sum(point_mask & ~finite))

    accepted = pd.read_csv(REGIONAL_WINDOWS)
    aggregate_rows = []
    max_gtng_closure = 0.0
    max_hma_change = 0.0
    for domain, raw_mask in domain_masks.items():
        point_mask = raw_mask & finite
        series = {
            "reference": domain_mean(reference, point_mask, point_lat),
            "background": domain_mean(background, point_mask, point_lat),
            "candidate": domain_mean(candidate, point_mask, point_lat),
        }
        for index, (start, end) in enumerate(base.WINDOWS):
            row = accepted[
                accepted.domain.eq(domain) & accepted.start_year.eq(start) & accepted.end_year.eq(end)
            ].iloc[0]
            differences = {
                "reference": float(series["reference"][index] - row.reference_degC),
                "background": float(series["background"][index] - row.background_degC),
                "candidate": float(series["candidate"][index] - row.candidate_degC),
            }
            maximum = max(abs(value) for value in differences.values())
            if domain == "hma":
                max_hma_change = max(max_hma_change, maximum)
            else:
                max_gtng_closure = max(max_gtng_closure, maximum)
            aggregate_rows.append({
                "domain": domain, "start_year": start, "end_year": end,
                **{f"{name}_difference_degC": value for name, value in differences.items()},
            })

    score_rows = []
    for domain, raw_mask in domain_masks.items():
        point_mask = raw_mask & finite
        bg_rmse = float(np.sqrt(np.mean((background[:, point_mask] - reference[:, point_mask]) ** 2)))
        candidate_rmse = float(np.sqrt(np.mean((candidate[:, point_mask] - reference[:, point_mask]) ** 2)))
        score_rows.append({
            "domain": domain,
            "common_support_cell_count": int(point_mask.sum()),
            "background_spatiotemporal_rmse_degC": bg_rmse,
            "candidate_spatiotemporal_rmse_degC": candidate_rmse,
            "improvement_vs_background": float(1 - candidate_rmse / bg_rmse),
        })
    scores = pd.DataFrame(score_rows)
    regional_scores = scores[scores.domain.ne("hma")]
    hma_score = scores[scores.domain.eq("hma")].iloc[0]
    bg_cell_rmse = np.sqrt(np.mean((background[:, finite] - reference[:, finite]) ** 2, axis=0))
    candidate_cell_rmse = np.sqrt(np.mean((candidate[:, finite] - reference[:, finite]) ** 2, axis=0))
    cell_improvement = 1 - candidate_cell_rmse / bg_cell_rmse
    finite_weights = weights[finite]
    improving_fraction = float(np.sum(finite_weights[cell_improvement > 0]) / np.sum(finite_weights))
    key = regional_scores[regional_scores.domain.isin(["13-08", "14-03", "15-01", "15-02"])]
    gates = {
        "area_weighted_hma_coverage_at_least_0p999": bool(coverage_fraction >= 0.999),
        "excluded_gtng_cell_count_is_zero": bool(excluded_gtng_count == 0),
        "hma_aggregate_change_at_most_0p005_degC": bool(max_hma_change <= 0.005),
        "gtng_aggregate_closure_at_most_1e_6_degC": bool(max_gtng_closure <= 1e-6),
        "hma_spatiotemporal_rmse_improves": bool(hma_score.improvement_vs_background > 0),
        "area_weighted_improving_cell_fraction_at_least_0p67": bool(improving_fraction >= 0.67),
        "at_least_12_of_15_gtng_regions_improve": bool((regional_scores.improvement_vs_background > 0).sum() >= 12),
        "key_regions_do_not_degrade_more_than_0p10": bool((key.improvement_vs_background >= -0.10).all()),
    }
    pass_all = bool(all(gates.values()))

    excluded = pd.DataFrame(points[~finite], columns=["latitude", "longitude"])
    excluded.to_csv(OUTPUT / "excluded_hma_cells.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(OUTPUT / "common_support_aggregate_closure.csv", index=False)
    scores.to_csv(OUTPUT / "common_support_spatial_scores.csv", index=False)
    pd.DataFrame(request_audit).to_csv(OUTPUT / "remote_request_audit.csv", index=False)
    np.savez_compressed(
        OUTPUT / "equal_blend_common_support_50year_fields.npz",
        windows=np.asarray(base.WINDOWS), lat=lat, lon=lon, hma_mask=hma_mask,
        common_support=finite, reference=reference, background=background,
        forced_mean=forced, candidate=candidate,
    )

    full_maps = []
    for values in [
        np.sqrt(np.mean((background - reference) ** 2, axis=0)),
        np.sqrt(np.mean((candidate - reference) ** 2, axis=0)),
    ]:
        field = np.full(hma_mask.shape, np.nan)
        field[hma_mask] = values
        full_maps.append(field)
    improvement_values = np.full(int(hma_mask.sum()), np.nan)
    improvement_values[finite] = cell_improvement * 100
    improvement_map = np.full(hma_mask.shape, np.nan)
    improvement_map[hma_mask] = improvement_values
    full_maps.append(improvement_map)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    specs = [
        (full_maps[0], "Current background RMSE", "viridis", None, "degC"),
        (full_maps[1], "50:50 candidate RMSE", "viridis", None, "degC"),
        (full_maps[2], "Cell-wise RMSE improvement", "RdBu_r", (-100, 100), "%"),
    ]
    for axis, (field, title, cmap, limits, label) in zip(axes, specs, strict=True):
        kwargs = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
        image = axis.pcolormesh(lon, lat, field, shading="auto", cmap=cmap, **kwargs)
        axis.set(title=title, xlabel="Longitude", ylabel="Latitude")
        fig.colorbar(image, ax=axis, label=label, shrink=0.82)
    figure = OUTPUT / "temperature_pmip_equal_blend_spatial_common_support.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in paths.items()}
    unchanged = before == after
    decision = "authorize_provisional_annual_spatial_candidate_build" if pass_all else "reject_fixed_spatial_common_support_blend"
    summary = {
        "status": "complete", "locked_1901_1939_values_read": 0,
        "source_hashes_unchanged": unchanged,
        "hma_cell_count": int(hma_mask.sum()), "common_support_cell_count": int(finite.sum()),
        "area_weighted_coverage_fraction": coverage_fraction,
        "excluded_gtng_cell_count": excluded_gtng_count,
        "max_hma_aggregate_change_degC": max_hma_change,
        "max_gtng_aggregate_closure_degC": max_gtng_closure,
        "hma_spatiotemporal_rmse_improvement": float(hma_score.improvement_vs_background),
        "area_weighted_improving_cell_fraction": improving_fraction,
        "improving_gtng_region_count": int((regional_scores.improvement_vs_background > 0).sum()),
        "worst_gtng_region": str(regional_scores.loc[regional_scores.improvement_vs_background.idxmin(), "domain"]),
        "worst_gtng_region_improvement": float(regional_scores.improvement_vs_background.min()),
        "gates": gates, "passes_all_gates": pass_all, "decision": decision,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2), encoding="utf-8"
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Fixed 50:50 Spatial Common-Support Result\n\n"
        f"- Common support: `{int(finite.sum())}/{int(hma_mask.sum())}` cells "
        f"(`{100 * coverage_fraction:.4f}%` area weighted).\n"
        f"- Excluded GTN-G cells: `{excluded_gtng_count}`.\n"
        f"- HMA spatiotemporal RMSE improvement: `{100 * float(hma_score.improvement_vs_background):+.1f}%`.\n"
        f"- Area-weighted improving cells: `{100 * improving_fraction:.1f}%`.\n"
        f"- Improving GTN-G regions: `{int((regional_scores.improvement_vs_background > 0).sum())}/15`.\n"
        f"- All frozen gates pass: `{str(pass_all).lower()}`.\n\nDecision: `{decision}`.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("A source file changed")
    print(json.dumps(summary, indent=2))
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
