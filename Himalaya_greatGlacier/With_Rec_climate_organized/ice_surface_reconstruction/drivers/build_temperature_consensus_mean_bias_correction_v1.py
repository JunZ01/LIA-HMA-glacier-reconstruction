"""Build a phase-preserving LIA mean-bias correction from reference consensus."""

from __future__ import annotations

import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import build_temperature_regional_reference_field_nested_cv_v1 as tools
import validate_temperature_external_low_frequency_v1 as base


OUTPUT = base.RESULTS / "climate_reconstruction_temperature_consensus_mean_bias_correction_v1"
ASSIGNMENT = base.RESULTS / "climate_reconstruction_hma_gtng_subregion_inventory_v1/hma_grid_gtng15_assignment.csv"
SUPPORT = base.RESULTS / "climate_reconstruction_hma_gtng_subregion_inventory_v1/gtng15_region_support_inventory.csv"
SMOOTH_SIGMA_CELLS = 2.0


def smooth_inside(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(field)
    numerator = gaussian_filter(np.where(valid, field, 0.0), SMOOTH_SIGMA_CELLS, mode="nearest")
    denominator = gaussian_filter(valid.astype(float), SMOOTH_SIGMA_CELLS, mode="nearest")
    result = np.divide(numerator, denominator, out=np.zeros_like(field), where=denominator > 1e-8)
    result[~mask] = 0.0
    return result


def correction_from(fields: list[np.ndarray], project: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lia = (tools.YEARS >= 1500) & (tools.YEARS <= 1899)
    differences = np.stack([np.nanmean(field[lia] - project[lia], axis=0) for field in fields])
    return smooth_inside(np.nanmedian(differences, axis=0), mask)


def apply_correction(project: np.ndarray, spatial: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.zeros(len(tools.YEARS), dtype=float)
    weight[(tools.YEARS >= 1500) & (tools.YEARS <= 1899)] = 1.0
    ramp = (tools.YEARS >= 1900) & (tools.YEARS < 1940)
    weight[ramp] = (1940 - tools.YEARS[ramp]) / 40.0
    correction = weight[:, None, None] * spatial[None, :, :]
    correction[:, ~mask] = 0.0
    return project + correction, correction


def regional_rmse(field: np.ndarray, reference: np.ndarray, masks: dict[str, np.ndarray], regions: list[str], lat: np.ndarray) -> float:
    a = tools.regional_series(field, masks, lat)
    b = tools.regional_series(reference, masks, lat)
    return tools.rmse(tools.window_vector(a, regions), tools.window_vector(b, regions))


def annual_anomaly_metric(project: np.ndarray, candidate: np.ndarray, external: pd.DataFrame, modern_start: int, modern_end: int) -> dict[str, float]:
    frame = pd.DataFrame({"year": tools.YEARS, "project": project, "candidate": candidate})
    ext = external.copy(); ext["year"] = ext.year.astype(int)
    merged = frame.merge(ext, on="year")
    for column in ["project", "candidate", "external"]:
        modern = merged.loc[merged.year.between(modern_start, modern_end), column].dropna()
        merged[f"{column}_a"] = merged[column] - modern.mean()
    rows = []
    for start in range(1550, 1900, 50):
        sub = merged.loc[merged.year.between(start, start + 49)]
        if sub.external_a.notna().sum() >= 35:
            rows.append([sub.project_a.mean(), sub.candidate_a.mean(), sub.external_a.mean()])
    values = np.asarray(rows, dtype=float)
    p_error = values[:, 0] - values[:, 2]
    c_error = values[:, 1] - values[:, 2]
    p = float(np.sqrt(np.mean(p_error ** 2)))
    c = float(np.sqrt(np.mean(c_error ** 2)))
    return {
        "project_rmse": p, "candidate_rmse": c, "improvement": 1 - c / p,
        "project_bias": float(np.mean(p_error)), "candidate_bias": float(np.mean(c_error)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    paths = {
        "public": tools.PUBLIC, "hma_mask": tools.HMA_MASK, "assignment": ASSIGNMENT, "support": SUPPORT,
        "chongce": tools.LAKE_ICE / "chongce_zenodo_5766235.xlsx",
        "haizi": tools.RECENT / "41598_2024_79096_MOESM1_ESM.xlsx",
        "jionglacuo": tools.RECENT / "Temperature reconstruction.xlsx",
        "jiang": tools.LAKE_ICE / "jiang_co_S2095927323002943_mmc4.xlsx", "contract": contract,
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

    outer_rows = []
    for heldout in tools.FAMILY_ORDER:
        training = [families[name] for name in tools.FAMILY_ORDER if name != heldout]
        spatial = correction_from(training, project, hma_mask)
        candidate, _ = apply_correction(project, spatial, hma_mask)
        p = regional_rmse(project, families[heldout], masks, regions, lat)
        c = regional_rmse(candidate, families[heldout], masks, regions, lat)
        outer_rows.append({"held_out_family": heldout, "project_rmse": p, "candidate_rmse": c, "improvement": 1 - c / p})
    outer = pd.DataFrame(outer_rows)
    outer.to_csv(OUTPUT / "outer_family_validation.csv", index=False)

    spatial = correction_from([families[name] for name in tools.FAMILY_ORDER], project, hma_mask)
    candidate, correction = apply_correction(project, spatial, hma_mask)
    chongce = tools.read_xlsx(paths["chongce"], 1, 3, 6, 1524, 2012)
    haizi = tools.read_xlsx(paths["haizi"], 1, 3, 2, 1115, 2017)
    jion = tools.read_xlsx(paths["jionglacuo"], 1, 2, 1, 1210, 2023)
    jiang = tools.read_xlsx(paths["jiang"], 2, 7, 12, 86, 2013.1)
    local_project = {
        "Chongce": tools.local_box(project, lat, lon, (80.75, 81.75, 34.75, 35.75)),
        "Haizi": tools.local_box(project, lat, lon, (99.75, 101.25, 28.75, 30.25)),
        "Jionglacuo": tools.local_box(project, lat, lon, (93.5, 95.5, 30, 32)),
        "Jiang": tools.local_box(project, lat, lon, (90.25, 91.25, 31, 32)),
    }
    local_candidate = {
        "Chongce": tools.local_box(candidate, lat, lon, (80.75, 81.75, 34.75, 35.75)),
        "Haizi": tools.local_box(candidate, lat, lon, (99.75, 101.25, 28.75, 30.25)),
        "Jionglacuo": tools.local_box(candidate, lat, lon, (93.5, 95.5, 30, 32)),
        "Jiang": tools.local_box(candidate, lat, lon, (90.25, 91.25, 31, 32)),
    }
    external_rows = [
        {"record": "Chongce_annual", **annual_anomaly_metric(local_project["Chongce"], local_candidate["Chongce"], chongce, 1940, 2000)},
        {"record": "Haizi_annual", **annual_anomaly_metric(local_project["Haizi"], local_candidate["Haizi"], haizi, 1959, 2000)},
        {"record": "Jionglacuo_JJAS", **tools.regular_record_metric(local_project["Jionglacuo"], local_candidate["Jionglacuo"], jion, 1970, 2000)},
        {"record": "Jiang_summer", **tools.irregular_metric(local_project["Jiang"], local_candidate["Jiang"], jiang, 1940, 2000, 1550)},
    ]
    external = pd.DataFrame(external_rows)
    external.to_csv(OUTPUT / "independent_record_audit.csv", index=False)

    modern = (tools.YEARS >= 1940) & (tools.YEARS <= 2000)
    modern_closure = float(np.nanmax(np.abs(candidate[modern] - project[modern])))
    hma_spatial = np.where(hma_mask, spatial, np.nan)
    adjacent = np.concatenate([np.abs(np.diff(hma_spatial, axis=0)).ravel(), np.abs(np.diff(hma_spatial, axis=1)).ravel()])
    max_step = float(np.nanmax(adjacent))
    max_abs = float(np.nanmax(np.abs(hma_spatial)))
    hma_mean = float(tools.weighted_series(spatial[None, :, :], hma_mask, lat)[0])
    improvements = outer.improvement.to_numpy(float)
    gates = {
        "all_heldout_reference_families_improve": bool(np.all(improvements > 0)),
        "median_heldout_reference_improvement_at_least_20pct": bool(np.median(improvements) >= 0.20),
        "modern_closure_at_most_1e12": bool(modern_closure <= 1e-12),
        "maximum_adjacent_step_at_most_0p10C": bool(max_step <= 0.10),
        "maximum_absolute_correction_at_most_1p5C": bool(max_abs <= 1.5),
    }
    passes = bool(all(gates.values()))
    outpath = OUTPUT / "temperature_consensus_mean_bias_candidate_1500_2000.nc"
    selected = tools.YEARS >= 1500
    with h5py.File(outpath, "w") as out:
        out.create_dataset("time", data=tools.YEARS[selected]); out.create_dataset("lat", data=lat); out.create_dataset("lon", data=lon)
        out.create_dataset("tas_candidate", data=candidate[selected].astype("f4"), compression="gzip", compression_opts=4)
        out.create_dataset("tas_correction", data=correction[selected].astype("f4"), compression="gzip", compression_opts=4)
        out.attrs["status"] = "provisional_reference_validated_diagnostic" if passes else "diagnostic_only"
        out.attrs["phase_preserved_1500_1899"] = True

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    im = axes[0].imshow(hma_spatial, origin="upper", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    axes[0].set_title("1500-1899 mean-only correction")
    fig.colorbar(im, ax=axes[0], label="degC")
    axes[1].bar(outer.held_out_family, 100 * outer.improvement); axes[1].axhline(20, color="#111", ls="--")
    axes[1].set(title="Held-out reference validation", ylabel="Improvement (%)")
    axes[2].bar(external.record, 100 * external.improvement); axes[2].axhline(0, color="#111")
    axes[2].tick_params(axis="x", rotation=25); axes[2].set(title="Independent-record audit", ylabel="Improvement (%)")
    fig.savefig(OUTPUT / "consensus_mean_bias_validation.png", dpi=220); plt.close(fig)

    after = {name: base.sha256(path) for name, path in paths.items()}
    (OUTPUT / "source_manifest.json").write_text(json.dumps({"inputs": {name: {"path": str(paths[name]), "sha256_before": before[name], "sha256_after": after[name]} for name in paths}, "all_input_hashes_unchanged": before == after}, indent=2), encoding="utf-8")
    result = {
        "status": "complete", "hma_mean_correction_degC": hma_mean,
        "median_heldout_reference_improvement": float(np.median(improvements)),
        "external_improvements": {row["record"]: float(row["improvement"]) for row in external_rows},
        "modern_closure_degC": modern_closure, "maximum_adjacent_step_degC": max_step, "maximum_absolute_correction_degC": max_abs,
        "gates": gates, "passes_reference_consensus_gates": passes,
        "candidate_status": "provisional_reference_validated_diagnostic" if passes else "diagnostic_only",
        "candidate_file": str(outpath), "production_release_authorized": False,
        "locked_1921_1939_accessed": False, "phase_1901_1920_used_for_fitting_or_validation": False,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Reference-Consensus LIA Mean-Bias Correction\n\n"
        f"- HMA mean correction: `{hma_mean:+.3f} degC`.\n"
        f"- Median held-out reference improvement: `{100*np.median(improvements):.2f}%`.\n"
        f"- Independent improvements: `{json.dumps(result['external_improvements'])}`.\n"
        f"- Reference-consensus gates pass: `{str(passes).lower()}`.\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
