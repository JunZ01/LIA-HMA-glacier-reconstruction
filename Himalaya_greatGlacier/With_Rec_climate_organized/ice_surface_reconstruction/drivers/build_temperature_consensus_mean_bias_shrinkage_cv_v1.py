"""Nested-CV shrinkage for the phase-preserving LIA mean-bias field."""

from __future__ import annotations

import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_temperature_consensus_mean_bias_correction_v1 as bias
import build_temperature_regional_reference_field_nested_cv_v1 as tools
import validate_temperature_external_low_frequency_v1 as base


OUTPUT = base.RESULTS / "climate_reconstruction_temperature_consensus_mean_bias_shrinkage_cv_v1"
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def candidate_for(project: np.ndarray, fields: list[np.ndarray], lam: float, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spatial = lam * bias.correction_from(fields, project, mask)
    candidate, correction = bias.apply_correction(project, spatial, mask)
    return candidate, correction, spatial


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    paths = {
        "public": tools.PUBLIC, "hma_mask": tools.HMA_MASK, "assignment": bias.ASSIGNMENT, "support": bias.SUPPORT,
        "chongce": tools.LAKE_ICE / "chongce_zenodo_5766235.xlsx",
        "haizi": tools.RECENT / "41598_2024_79096_MOESM1_ESM.xlsx",
        "jionglacuo": tools.RECENT / "Temperature reconstruction.xlsx",
        "jiang": tools.LAKE_ICE / "jiang_co_S2095927323002943_mmc4.xlsx", "contract": contract,
    }
    before = {name: base.sha256(path) for name, path in paths.items()}
    hma_mask = np.load(tools.HMA_MASK).astype(bool)
    assignment = pd.read_csv(bias.ASSIGNMENT)
    support = pd.read_csv(bias.SUPPORT).sort_values("o2region")
    regions = support.o2region.tolist()
    masks = {}
    for row in support.itertuples(index=False):
        mask = np.zeros_like(hma_mask)
        cells = assignment[assignment.o2region.eq(row.o2region)]
        mask[cells.grid_row.to_numpy(int), cells.grid_col.to_numpy(int)] = True
        masks[row.o2region] = mask

    with h5py.File(tools.PUBLIC, "r") as handle:
        lat = np.asarray(handle["lat"], dtype=float); lon = np.asarray(handle["lon"], dtype=float)
        project = tools.allowed_field(handle["tas_ensemble_mean_rec_result"])
        families = {
            "local_enkf": tools.finite_mean([tools.allowed_field(handle["tas_rec_ref_HGAO_c05d"]), tools.allowed_field(handle["tas_rec_ref_O_c05d"])]),
            "lmr": tools.allowed_field(handle["tas_rec_ref_LMR_c05d"]),
            "neukom": tools.finite_mean([tools.allowed_field(handle[name]) for name in ["tas_rec_ref_Nk_AM_c05d", "tas_rec_ref_Nk_CCA_c05d", "tas_rec_ref_Nk_CPS_c05d", "tas_rec_ref_Nk_DA_c05d", "tas_rec_ref_Nk_GraphEM_c05d", "tas_rec_ref_Nk_PCR_c05d"]]),
            "phyda": tools.allowed_field(handle["tas_rec_ref_PHYDA_c05d"]),
        }
    project_scores = {name: bias.regional_rmse(project, field, masks, regions, lat) for name, field in families.items()}

    outer_rows = []
    selected_lambdas = []
    inner_rows = []
    for outer in tools.FAMILY_ORDER:
        training = [name for name in tools.FAMILY_ORDER if name != outer]
        lambda_scores = []
        for lam in LAMBDAS:
            fold_ratios = []
            for inner in training:
                fit_names = [name for name in training if name != inner]
                candidate, _, _ = candidate_for(project, [families[name] for name in fit_names], lam, hma_mask)
                current = bias.regional_rmse(candidate, families[inner], masks, regions, lat)
                ratio = current / project_scores[inner]
                fold_ratios.append(ratio)
                inner_rows.append({"outer_family": outer, "inner_family": inner, "lambda": lam, "rmse_ratio": ratio})
            lambda_scores.append((float(np.median(fold_ratios)), lam))
        _, selected_lambda = min(lambda_scores, key=lambda item: (item[0], item[1]))
        selected_lambdas.append(selected_lambda)
        candidate, _, _ = candidate_for(project, [families[name] for name in training], selected_lambda, hma_mask)
        current = bias.regional_rmse(candidate, families[outer], masks, regions, lat)
        outer_rows.append({
            "held_out_family": outer, "selected_lambda": selected_lambda,
            "project_rmse": project_scores[outer], "candidate_rmse": current,
            "improvement": 1 - current / project_scores[outer],
        })
    pd.DataFrame(inner_rows).to_csv(OUTPUT / "inner_family_scores.csv", index=False)
    outer = pd.DataFrame(outer_rows); outer.to_csv(OUTPUT / "outer_family_validation.csv", index=False)
    final_lambda = float(sorted(selected_lambdas)[1])
    candidate, correction, spatial = candidate_for(project, [families[name] for name in tools.FAMILY_ORDER], final_lambda, hma_mask)

    chongce = tools.read_xlsx(paths["chongce"], 1, 3, 6, 1524, 2012)
    haizi = tools.read_xlsx(paths["haizi"], 1, 3, 2, 1115, 2017)
    jion = tools.read_xlsx(paths["jionglacuo"], 1, 2, 1, 1210, 2023)
    jiang = tools.read_xlsx(paths["jiang"], 2, 7, 12, 86, 2013.1)
    boxes = {
        "Chongce": (80.75, 81.75, 34.75, 35.75), "Haizi": (99.75, 101.25, 28.75, 30.25),
        "Jionglacuo": (93.5, 95.5, 30, 32), "Jiang": (90.25, 91.25, 31, 32),
    }
    lp = {name: tools.local_box(project, lat, lon, box) for name, box in boxes.items()}
    lc = {name: tools.local_box(candidate, lat, lon, box) for name, box in boxes.items()}
    external_rows = [
        {"record": "Chongce_annual", **bias.annual_anomaly_metric(lp["Chongce"], lc["Chongce"], chongce, 1940, 2000)},
        {"record": "Haizi_annual", **bias.annual_anomaly_metric(lp["Haizi"], lc["Haizi"], haizi, 1959, 2000)},
        {"record": "Jionglacuo_JJAS", **tools.regular_record_metric(lp["Jionglacuo"], lc["Jionglacuo"], jion, 1970, 2000)},
        {"record": "Jiang_summer", **tools.irregular_metric(lp["Jiang"], lc["Jiang"], jiang, 1940, 2000, 1550)},
    ]
    external = pd.DataFrame(external_rows); external.to_csv(OUTPUT / "independent_record_audit.csv", index=False)

    improvements = outer.improvement.to_numpy(float)
    external_improvements = external.improvement.to_numpy(float)
    modern = (tools.YEARS >= 1940) & (tools.YEARS <= 2000)
    modern_closure = float(np.nanmax(np.abs(candidate[modern] - project[modern])))
    hma_spatial = np.where(hma_mask, spatial, np.nan)
    adjacent = np.concatenate([np.abs(np.diff(hma_spatial, axis=0)).ravel(), np.abs(np.diff(hma_spatial, axis=1)).ravel()])
    max_step = float(np.nanmax(adjacent)); max_abs = float(np.nanmax(np.abs(hma_spatial)))
    hma_mean = float(tools.weighted_series(spatial[None, :, :], hma_mask, lat)[0])
    gates = {
        "all_outer_reference_families_improve": bool(np.all(improvements > 0)),
        "median_outer_reference_improvement_at_least_20pct": bool(np.median(improvements) >= 0.20),
        "at_least_three_of_four_independent_records_improve": bool(np.sum(external_improvements > 0) >= 3),
        "modern_closure_at_most_1e12": bool(modern_closure <= 1e-12),
        "maximum_adjacent_step_at_most_0p10C": bool(max_step <= 0.10),
        "maximum_absolute_correction_at_most_1p5C": bool(max_abs <= 1.5),
    }
    passes = bool(all(gates.values()))
    outpath = OUTPUT / "temperature_consensus_mean_bias_shrunk_candidate_1500_2000.nc"
    selected = tools.YEARS >= 1500
    with h5py.File(outpath, "w") as out:
        out.create_dataset("time", data=tools.YEARS[selected]); out.create_dataset("lat", data=lat); out.create_dataset("lon", data=lon)
        out.create_dataset("tas_candidate", data=candidate[selected].astype("f4"), compression="gzip", compression_opts=4)
        out.create_dataset("tas_correction", data=correction[selected].astype("f4"), compression="gzip", compression_opts=4)
        out.attrs["selected_lambda"] = final_lambda
        out.attrs["status"] = "provisional_phase_preserving_warm_bias_correction" if passes else "diagnostic_only"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    im = axes[0].imshow(hma_spatial, origin="upper", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    axes[0].set_title(f"Mean-bias correction, lambda={final_lambda:g}"); fig.colorbar(im, ax=axes[0], label="degC")
    axes[1].bar(outer.held_out_family, 100 * outer.improvement); axes[1].axhline(20, color="#111", ls="--")
    axes[1].set(title="Nested outer validation", ylabel="Improvement (%)")
    axes[2].bar(external.record, 100 * external.improvement); axes[2].axhline(0, color="#111")
    axes[2].tick_params(axis="x", rotation=25); axes[2].set(title="Independent audit", ylabel="Improvement (%)")
    fig.savefig(OUTPUT / "consensus_mean_bias_shrinkage_validation.png", dpi=220); plt.close(fig)

    after = {name: base.sha256(path) for name, path in paths.items()}
    (OUTPUT / "source_manifest.json").write_text(json.dumps({"inputs": {name: {"path": str(paths[name]), "sha256_before": before[name], "sha256_after": after[name]} for name in paths}, "all_input_hashes_unchanged": before == after}, indent=2), encoding="utf-8")
    result = {
        "status": "complete", "selected_lambda": final_lambda, "outer_selected_lambdas": selected_lambdas,
        "hma_mean_correction_degC": hma_mean, "median_outer_reference_improvement": float(np.median(improvements)),
        "external_improvements": {row["record"]: float(row["improvement"]) for row in external_rows},
        "modern_closure_degC": modern_closure, "maximum_adjacent_step_degC": max_step, "maximum_absolute_correction_degC": max_abs,
        "gates": gates, "passes_provisional_gates": passes,
        "candidate_status": "provisional_phase_preserving_warm_bias_correction" if passes else "diagnostic_only",
        "candidate_file": str(outpath), "production_release_authorized": False,
        "haizi_conflict_retained": True, "locked_1921_1939_accessed": False,
        "phase_1901_1920_used_for_fitting_or_validation": False, "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Consensus Mean-Bias Shrinkage Cross-Validation\n\n"
        f"- Selected lambda: `{final_lambda:g}`; HMA mean correction: `{hma_mean:+.3f} degC`.\n"
        f"- Median outer reference improvement: `{100*np.median(improvements):.2f}%`.\n"
        f"- Independent improvements: `{json.dumps(result['external_improvements'])}`.\n"
        f"- Provisional gates pass: `{str(passes).lower()}`.\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
