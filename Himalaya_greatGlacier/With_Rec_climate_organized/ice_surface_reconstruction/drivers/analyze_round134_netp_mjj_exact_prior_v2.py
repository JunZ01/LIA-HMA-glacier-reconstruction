"""Corrected QA for the exact NETP May-July precipitation prior."""

from __future__ import annotations

import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_v1(path: Path):
    spec = importlib.util.spec_from_file_location("round134_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_134_netp_mjj_exact_prior_v2"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round134_netp_mjj_exact_prior_v2_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    v1_path = driver / "analyze_round134_netp_mjj_exact_prior_v1.py"
    v1 = load_v1(v1_path)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "v1_analysis": v1_path,
        "analysis": Path(__file__).resolve(),
        "annual_prior": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "round129": loop / "round_129_new_d18o_exact_seasonal_prior_coupling_v3/exact_seasonal_prior_fields.mat",
        "round134_v1": loop / "round_134_netp_mjj_exact_prior_v1/netp_mjj_exact_prior_fields.mat",
        "round134_v1_stop": loop / "round_134_netp_mjj_exact_prior_v1/STOPPED.md",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: v1.fingerprint(path) for name, path in paths.items()}

    with h5py.File(paths["round134_v1"], "r") as handle:
        annual = v1.field(handle, "annual_precipitation")
        mjj = v1.field(handle, "mjj_precipitation")
        lat = np.asarray(handle["lat"], dtype=float).ravel()
        lon = np.asarray(handle["lon"], dtype=float).ravel()
        years = np.asarray(handle["member_years"], dtype=int).ravel()
        closure_max = float(np.asarray(handle["annual_closure_max_abs"]).ravel()[0])
        closure_rmse = float(np.asarray(handle["annual_closure_rmse"]).ravel()[0])
        closure_changed = int(np.asarray(handle["annual_closure_changed_count"]).ravel()[0])
    with h5py.File(paths["round129"], "r") as handle:
        annual129 = v1.field(handle, "annual_precipitation")
        jja129 = v1.field(handle, "jja_precipitation")
        lat129 = np.asarray(handle["lat"], dtype=float).ravel()
        lon129 = np.asarray(handle["lon"], dtype=float).ravel()
        years129 = np.asarray(handle["member_years"], dtype=int).ravel()

    std_mjj = np.std(mjj, axis=1, ddof=1)
    std_annual = np.std(annual, axis=1, ddof=1)
    std_jja = np.std(jja129, axis=1, ddof=1)
    zero_mjj = std_mjj == 0
    zero_annual = std_annual == 0
    zero_jja = std_jja == 0
    state_lat = np.tile(lat, len(lon))
    state_lon = np.repeat(lon, len(lat))
    netp = (state_lat >= 35) & (state_lat <= 42) & (state_lon >= 92) & (state_lon <= 108)
    weights = np.cos(np.deg2rad(state_lat[netp]))
    annual_netp = np.average(annual[netp], axis=0, weights=weights)
    mjj_netp = np.average(mjj[netp], axis=0, weights=weights)
    jja_netp = np.average(jja129[netp], axis=0, weights=weights)
    zero_rows = [
        {"cell": int(cell), "latitude": float(state_lat[cell]), "longitude": float(state_lon[cell])}
        for cell in np.flatnonzero(zero_mjj)
    ]
    diagnostics = {
        "annual_closure_max_abs": closure_max,
        "annual_closure_rmse": closure_rmse,
        "annual_closure_changed_count": closure_changed,
        "annual_round129_max_abs": float(np.max(np.abs(annual - annual129))),
        "coordinate_and_year_match": bool(np.array_equal(lat, lat129) and np.array_equal(lon, lon129) and np.array_equal(years, years129) and np.array_equal(years, np.arange(1001, 1849))),
        "mjj_zero_variance_cell_count": int(zero_mjj.sum()),
        "annual_zero_variance_cell_count": int(zero_annual.sum()),
        "jja_zero_variance_cell_count": int(zero_jja.sum()),
        "zero_variance_masks_equal": bool(np.array_equal(zero_mjj, zero_annual) and np.array_equal(zero_mjj, zero_jja)),
        "variable_cell_count": int((~zero_mjj).sum()),
        "zero_variance_cells": zero_rows,
        "netp_cell_count": int(netp.sum()),
        "netp_zero_variance_cell_count": int((zero_mjj & netp).sum()),
        "netp_minimum_mjj_standard_deviation": float(std_mjj[netp].min()),
        "mjj_all_finite": bool(np.isfinite(mjj).all()),
        "mjj_equals_annual": bool(np.array_equal(mjj, annual)),
        "mjj_equals_jja": bool(np.array_equal(mjj, jja129)),
        "netp_mjj_vs_annual_correlation": v1.correlation(mjj_netp, annual_netp),
        "netp_mjj_vs_jja_correlation": v1.correlation(mjj_netp, jja_netp),
        "netp_mjj_standard_deviation_mm_day": float(np.std(mjj_netp, ddof=1)),
    }
    gates = {
        "exact_annual_closure": closure_max == 0 and closure_rmse == 0 and closure_changed == 0 and diagnostics["annual_round129_max_abs"] == 0,
        "coordinate_and_year_match": diagnostics["coordinate_and_year_match"],
        "mjj_shape_and_finite": mjj.shape == (336, 848) and diagnostics["mjj_all_finite"],
        "production_zero_variance_mask_closure": diagnostics["zero_variance_masks_equal"] and diagnostics["variable_cell_count"] == 333 and diagnostics["mjj_zero_variance_cell_count"] == 3,
        "all_netp_cells_variable": diagnostics["netp_cell_count"] == 32 and diagnostics["netp_zero_variance_cell_count"] == 0 and diagnostics["netp_minimum_mjj_standard_deviation"] > 0,
        "mjj_distinct_from_annual_and_jja": not diagnostics["mjj_equals_annual"] and not diagnostics["mjj_equals_jja"],
    }
    if not all(gates.values()):
        raise RuntimeError(f"round134-v2-gate-failed:{gates}")

    curves = pd.DataFrame({
        "year": years,
        "annual_netp_mm_day": annual_netp,
        "mjj_netp_mm_day": mjj_netp,
        "jja_netp_mm_day": jja_netp,
        "annual_netp_31yr": v1.rolling(annual_netp),
        "mjj_netp_31yr": v1.rolling(mjj_netp),
        "jja_netp_31yr": v1.rolling(jja_netp),
    })
    curves.to_csv(output / "netp_exact_prior_curves.csv", index=False)
    pd.DataFrame(zero_rows).to_csv(output / "production_constant_cells.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.5), constrained_layout=True, sharex=True)
    series = (("annual", annual_netp, "#111111"), ("MJJ", mjj_netp, "#0072B2"), ("JJA", jja_netp, "#D55E00"))
    for label, values, color in series:
        axes[0].plot(years, (values - values.mean()) / values.std(ddof=1), color=color, linewidth=0.7, alpha=0.75, label=label)
        smooth = v1.rolling(values)
        axes[1].plot(years, (smooth - np.nanmean(smooth)) / np.nanstd(smooth, ddof=1), color=color, linewidth=1.25, label=label)
    axes[0].set(title="NETP exact precipitation priors", ylabel="annual standardized anomaly")
    axes[1].set(title="NETP exact precipitation priors, 31-year mean", xlabel="member year", ylabel="standardized anomaly")
    for axis in axes:
        axis.axhline(0, color="#888888", linewidth=0.6)
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=3)
    fig.savefig(output / "round134_netp_exact_prior_curves.png", dpi=220)
    plt.close(fig)

    after = {name: v1.fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": "round_134_netp_mjj_exact_prior_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "v1_status": "stopped_incorrect_all_cell_variance_gate",
        "v2_correction": "require_exact_production_zero_variance_mask_and_variable_NETP",
        "member_years": [1001, 1848],
        "locked_1921_1939_accessed": False,
        "proxy_or_instrumental_values_read": False,
        "external_validation_reserve_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "gates": gates,
        "passes_all_gates": bool(all(gates.values())),
        "diagnostics": diagnostics,
        "source_fingerprints_unchanged": unchanged,
        "decision": "advance_to_netp_mjj_reconstruction_contract",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 134 NETP MJJ Exact Prior V2\n\n"
        "- V1 stopped because its all-cell variance gate was inconsistent with the production representation.\n"
        f"- Exact annual closure: `{gates['exact_annual_closure']}`.\n"
        f"- Production zero-variance mask closure: `{gates['production_zero_variance_mask_closure']}`.\n"
        f"- NETP variable cells: `{diagnostics['netp_cell_count'] - diagnostics['netp_zero_variance_cell_count']}/32`.\n"
        f"- MJJ versus annual correlation: `{diagnostics['netp_mjj_vs_annual_correlation']:.3f}`.\n"
        f"- MJJ versus JJA correlation: `{diagnostics['netp_mjj_vs_jja_correlation']:.3f}`.\n"
        "- Decision: `advance_to_netp_mjj_reconstruction_contract`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
