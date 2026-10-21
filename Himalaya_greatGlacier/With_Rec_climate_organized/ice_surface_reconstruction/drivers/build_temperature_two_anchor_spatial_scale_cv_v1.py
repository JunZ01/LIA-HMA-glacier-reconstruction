"""Cross-validate the spatial scale of a frozen two-anchor LIA bias field."""

from __future__ import annotations

import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_temperature_regional_reference_field_nested_cv_v1 as tools
import build_temperature_two_anchor_mean_bias_field_v1 as prior
import validate_temperature_external_low_frequency_v1 as base


OUTPUT = base.RESULTS / "climate_reconstruction_temperature_two_anchor_spatial_scale_cv_v1"
ASSIGNMENT = base.RESULTS / "climate_reconstruction_hma_gtng_subregion_inventory_v1/hma_grid_gtng15_assignment.csv"
SUPPORT = base.RESULTS / "climate_reconstruction_hma_gtng_subregion_inventory_v1/gtng15_region_support_inventory.csv"
SCALES_KM = [400, 600, 800, 1000, 1200, 1400]
WEST = (35.2349, 81.1144)
EAST = (29.58, 100.65)


def distance_km(lat: np.ndarray, lon: np.ndarray, point: tuple[float, float]) -> np.ndarray:
    """Great-circle distance from a lat/lon grid to one point."""
    plat, plon = np.deg2rad(point)
    glat = np.deg2rad(lat[:, None])
    glon = np.deg2rad(lon[None, :])
    a = np.sin((glat - plat) / 2) ** 2 + np.cos(glat) * np.cos(plat) * np.sin((glon - plon) / 2) ** 2
    return 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1 - a)))


def point_distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(distance_km(np.asarray([a[0]]), np.asarray([a[1]]), b)[0, 0])


def rbf_field(lat: np.ndarray, lon: np.ndarray, scale: float, amplitudes: np.ndarray) -> np.ndarray:
    """Two Gaussian RBFs whose values close exactly at both anchor points."""
    cross = np.exp(-0.5 * (point_distance_km(WEST, EAST) / scale) ** 2)
    coefficients = np.linalg.solve(np.asarray([[1.0, cross], [cross, 1.0]]), amplitudes)
    kernels = np.stack([
        np.exp(-0.5 * (distance_km(lat, lon, WEST) / scale) ** 2),
        np.exp(-0.5 * (distance_km(lat, lon, EAST) / scale) ** 2),
    ])
    return np.sum(coefficients[:, None, None] * kernels, axis=0)


def make_candidate(project: np.ndarray, spatial: np.ndarray, hma_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.zeros(len(tools.YEARS), dtype=float)
    weight[(tools.YEARS >= 1500) & (tools.YEARS <= 1899)] = 1.0
    ramp = (tools.YEARS >= 1900) & (tools.YEARS < 1940)
    weight[ramp] = (1940 - tools.YEARS[ramp]) / 40.0
    correction = weight[:, None, None] * spatial[None, :, :]
    correction[:, ~hma_mask] = 0.0
    return project + correction, correction


def score(candidate: np.ndarray, reference: np.ndarray, masks: dict[str, np.ndarray], regions: list[str], lat: np.ndarray) -> float:
    c = tools.regional_series(candidate, masks, lat)
    r = tools.regional_series(reference, masks, lat)
    return tools.rmse(tools.window_vector(c, regions), tools.window_vector(r, regions))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    paths = {
        "public": tools.PUBLIC,
        "hma_mask": tools.HMA_MASK,
        "assignment": ASSIGNMENT,
        "support": SUPPORT,
        "chongce": tools.LAKE_ICE / "chongce_zenodo_5766235.xlsx",
        "haizi": tools.RECENT / "41598_2024_79096_MOESM1_ESM.xlsx",
        "jionglacuo": tools.RECENT / "Temperature reconstruction.xlsx",
        "jiang": tools.LAKE_ICE / "jiang_co_S2095927323002943_mmc4.xlsx",
        "contract": contract,
    }
    before = {name: base.sha256(path) for name, path in paths.items()}
    hma_mask = np.load(tools.HMA_MASK).astype(bool)
    assignment = pd.read_csv(ASSIGNMENT)
    support = pd.read_csv(SUPPORT).sort_values("o2region")
    regions = support.o2region.tolist()
    masks = {}
    for row in support.itertuples(index=False):
        mask = np.zeros_like(hma_mask)
        cells = assignment[assignment.o2region.eq(row.o2region)]
        mask[cells.grid_row.to_numpy(int), cells.grid_col.to_numpy(int)] = True
        masks[row.o2region] = mask

    with h5py.File(tools.PUBLIC, "r") as handle:
        lat = np.asarray(handle["lat"], dtype=float)
        lon = np.asarray(handle["lon"], dtype=float)
        project = tools.allowed_field(handle["tas_ensemble_mean_rec_result"])
        families = {
            "local_enkf": tools.finite_mean([tools.allowed_field(handle["tas_rec_ref_HGAO_c05d"]), tools.allowed_field(handle["tas_rec_ref_O_c05d"])]),
            "lmr": tools.allowed_field(handle["tas_rec_ref_LMR_c05d"]),
            "neukom": tools.finite_mean([tools.allowed_field(handle[name]) for name in ["tas_rec_ref_Nk_AM_c05d", "tas_rec_ref_Nk_CCA_c05d", "tas_rec_ref_Nk_CPS_c05d", "tas_rec_ref_Nk_DA_c05d", "tas_rec_ref_Nk_GraphEM_c05d", "tas_rec_ref_Nk_PCR_c05d"]]),
            "phyda": tools.allowed_field(handle["tas_rec_ref_PHYDA_c05d"]),
        }

    chongce = tools.read_xlsx(paths["chongce"], 1, 3, 6, 1524, 2012)
    haizi = tools.read_xlsx(paths["haizi"], 1, 3, 2, 1115, 2017)
    west_series = tools.local_box(project, lat, lon, (80.75, 81.75, 34.75, 35.75))
    east_series = tools.local_box(project, lat, lon, (99.75, 101.25, 28.75, 30.25))
    amplitudes = np.asarray([
        -prior.aligned_window_bias(west_series, chongce, 1940, 2000),
        -prior.aligned_window_bias(east_series, haizi, 1959, 2000),
    ])

    candidates = {}
    corrections = {}
    spatial_fields = {}
    for scale in SCALES_KM:
        spatial = rbf_field(lat, lon, scale, amplitudes)
        spatial_fields[scale] = spatial
        candidates[scale], corrections[scale] = make_candidate(project, spatial, hma_mask)

    project_scores = {name: score(project, field, masks, regions, lat) for name, field in families.items()}
    outer_rows = []
    selected_scales = []
    for heldout in tools.FAMILY_ORDER:
        training = [name for name in tools.FAMILY_ORDER if name != heldout]
        training_scores = []
        for scale in SCALES_KM:
            normalized = [score(candidates[scale], families[name], masks, regions, lat) / project_scores[name] for name in training]
            training_scores.append((float(np.median(normalized)), scale))
        _, selected = min(training_scores, key=lambda item: (item[0], item[1]))
        selected_scales.append(selected)
        candidate_rmse = score(candidates[selected], families[heldout], masks, regions, lat)
        outer_rows.append({
            "held_out_family": heldout,
            "selected_scale_km": selected,
            "project_rmse": project_scores[heldout],
            "candidate_rmse": candidate_rmse,
            "improvement": 1 - candidate_rmse / project_scores[heldout],
        })
    outer = pd.DataFrame(outer_rows)
    outer.to_csv(OUTPUT / "outer_family_validation.csv", index=False)

    grid_rows = []
    for scale in SCALES_KM:
        row = {"scale_km": scale}
        normalized = []
        for name in tools.FAMILY_ORDER:
            current = score(candidates[scale], families[name], masks, regions, lat)
            row[f"{name}_improvement"] = 1 - current / project_scores[name]
            normalized.append(current / project_scores[name])
        row["median_improvement"] = 1 - float(np.median(normalized))
        grid_rows.append(row)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUTPUT / "scale_grid_scores.csv", index=False)
    final_scale = int(grid.sort_values(["median_improvement", "scale_km"], ascending=[False, True]).iloc[0].scale_km)
    candidate = candidates[final_scale]
    correction = corrections[final_scale]
    spatial = spatial_fields[final_scale]

    jion = tools.read_xlsx(paths["jionglacuo"], 1, 2, 1, 1210, 2023)
    jiang = tools.read_xlsx(paths["jiang"], 2, 7, 12, 86, 2013.1)
    jion_metric = tools.regular_record_metric(
        tools.local_box(project, lat, lon, (93.5, 95.5, 30, 32)),
        tools.local_box(candidate, lat, lon, (93.5, 95.5, 30, 32)), jion, 1970, 2000)
    jiang_metric = tools.irregular_metric(
        tools.local_box(project, lat, lon, (90.25, 91.25, 31, 32)),
        tools.local_box(candidate, lat, lon, (90.25, 91.25, 31, 32)), jiang, 1940, 2000, 1550)
    pd.DataFrame([
        {"record": "Jionglacuo_JJAS", **jion_metric},
        {"record": "Jiang_summer", **jiang_metric},
    ]).to_csv(OUTPUT / "heldout_external_validation.csv", index=False)

    outer_improvements = outer.improvement.to_numpy(float)
    modern = (tools.YEARS >= 1940) & (tools.YEARS <= 2000)
    modern_closure = float(np.nanmax(np.abs(candidate[modern] - project[modern])))
    hma_spatial = np.where(hma_mask, spatial, np.nan)
    adjacent = np.concatenate([np.abs(np.diff(hma_spatial, axis=0)).ravel(), np.abs(np.diff(hma_spatial, axis=1)).ravel()])
    max_step = float(np.nanmax(adjacent))
    max_abs = float(np.nanmax(np.abs(hma_spatial)))
    gates = {
        "all_heldout_reference_families_improve": bool(np.all(outer_improvements > 0)),
        "median_heldout_reference_improvement_at_least_20pct": bool(np.median(outer_improvements) >= 0.20),
        "jionglacuo_improves_at_least_10pct": bool(jion_metric["improvement"] >= 0.10),
        "jiang_improves_at_least_10pct": bool(jiang_metric["improvement"] >= 0.10),
        "modern_closure_at_most_1e12": bool(modern_closure <= 1e-12),
        "maximum_adjacent_step_at_most_0p10C": bool(max_step <= 0.10),
        "maximum_absolute_correction_at_most_1p5C": bool(max_abs <= 1.5),
    }
    passes = bool(all(gates.values()))
    candidate_path = OUTPUT / "temperature_two_anchor_spatial_scale_candidate_1500_2000.nc"
    selected = tools.YEARS >= 1500
    with h5py.File(candidate_path, "w") as out:
        out.create_dataset("time", data=tools.YEARS[selected])
        out.create_dataset("lat", data=lat)
        out.create_dataset("lon", data=lon)
        out.create_dataset("tas_candidate", data=candidate[selected].astype("f4"), compression="gzip", compression_opts=4)
        out.create_dataset("tas_correction", data=correction[selected].astype("f4"), compression="gzip", compression_opts=4)
        out.attrs["selected_scale_km"] = final_scale
        out.attrs["status"] = "provisional_validated_diagnostic" if passes else "diagnostic_only"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    im = axes[0].imshow(hma_spatial, origin="upper", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    axes[0].set_title(f"Two-anchor correction, {final_scale} km")
    fig.colorbar(im, ax=axes[0], label="degC")
    axes[1].plot(grid.scale_km, 100 * grid.median_improvement, marker="o")
    axes[1].axhline(20, color="#111", ls="--")
    axes[1].set(title="Reference-family scale selection", xlabel="Scale (km)", ylabel="Median improvement (%)")
    axes[2].bar(["Jionglacuo", "Jiang"], [100 * jion_metric["improvement"], 100 * jiang_metric["improvement"]])
    axes[2].axhline(10, color="#111", ls="--")
    axes[2].set(title="Held-out external validation", ylabel="Improvement (%)")
    fig.savefig(OUTPUT / "two_anchor_spatial_scale_validation.png", dpi=220)
    plt.close(fig)

    after = {name: base.sha256(path) for name, path in paths.items()}
    manifest = {"inputs": {name: {"path": str(paths[name]), "sha256_before": before[name], "sha256_after": after[name]} for name in paths}, "all_input_hashes_unchanged": before == after}
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = {
        "status": "complete", "anchor_corrections_degC": {"Chongce": float(amplitudes[0]), "Haizi": float(amplitudes[1])},
        "selected_scale_km": final_scale, "outer_selected_scales_km": selected_scales,
        "median_heldout_reference_improvement": float(np.median(outer_improvements)),
        "jionglacuo_improvement": float(jion_metric["improvement"]), "jiang_improvement": float(jiang_metric["improvement"]),
        "modern_closure_degC": modern_closure, "maximum_adjacent_step_degC": max_step, "maximum_absolute_correction_degC": max_abs,
        "gates": gates, "passes_provisional_gates": passes,
        "candidate_status": "provisional_validated_diagnostic" if passes else "diagnostic_only",
        "candidate_file": str(candidate_path), "production_release_authorized": False,
        "locked_1921_1939_accessed": False, "phase_1901_1920_used_for_fitting_or_validation": False,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Two-Anchor Spatial-Scale Cross-Validation\n\n"
        f"- Selected scale: `{final_scale} km`; outer selections: `{selected_scales}`.\n"
        f"- Median held-out reference improvement: `{100*np.median(outer_improvements):.2f}%`.\n"
        f"- Jionglacuo/Jiang improvements: `{100*jion_metric['improvement']:.2f}%` / `{100*jiang_metric['improvement']:.2f}%`.\n"
        f"- Provisional gates pass: `{str(passes).lower()}`.\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
