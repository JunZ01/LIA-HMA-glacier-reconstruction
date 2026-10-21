"""Verify and summarize the exact NETP May-July precipitation prior."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": stat.st_size, "sha256": digest.hexdigest()}


def field(handle: h5py.File, name: str) -> np.ndarray:
    raw = np.asarray(handle[name], dtype=float)
    result = raw.T if raw.shape == (848, 336) else raw
    if result.shape != (336, 848):
        raise RuntimeError(f"unexpected-field-shape:{name}:{result.shape}")
    return result


def rolling(values: np.ndarray, window: int = 31) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=window).mean().to_numpy()


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_134_netp_mjj_exact_prior_v1"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": driver / "round134_netp_mjj_exact_prior_v1_contract.md",
        "exporter": driver / "export_round134_netp_mjj_exact_prior_v1.m",
        "analysis": Path(__file__).resolve(),
        "annual_prior": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
        "round129": loop / "round_129_new_d18o_exact_seasonal_prior_coupling_v3/exact_seasonal_prior_fields.mat",
        "round134": output / "netp_mjj_exact_prior_fields.mat",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: fingerprint(path) for name, path in paths.items()}

    with h5py.File(paths["round134"], "r") as handle:
        annual = field(handle, "annual_precipitation")
        mjj = field(handle, "mjj_precipitation")
        lat = np.asarray(handle["lat"], dtype=float).ravel()
        lon = np.asarray(handle["lon"], dtype=float).ravel()
        years = np.asarray(handle["member_years"], dtype=int).ravel()
        closure_max = float(np.asarray(handle["annual_closure_max_abs"]).ravel()[0])
        closure_rmse = float(np.asarray(handle["annual_closure_rmse"]).ravel()[0])
        closure_changed = int(np.asarray(handle["annual_closure_changed_count"]).ravel()[0])
    with h5py.File(paths["round129"], "r") as handle:
        annual129 = field(handle, "annual_precipitation")
        jja129 = field(handle, "jja_precipitation")
        lat129 = np.asarray(handle["lat"], dtype=float).ravel()
        lon129 = np.asarray(handle["lon"], dtype=float).ravel()
        years129 = np.asarray(handle["member_years"], dtype=int).ravel()

    annual_round129_max_abs = float(np.max(np.abs(annual - annual129)))
    coordinate_year_match = bool(
        np.array_equal(lat, lat129)
        and np.array_equal(lon, lon129)
        and np.array_equal(years, years129)
        and np.array_equal(years, np.arange(1001, 1849))
    )
    state_lat = np.tile(lat, len(lon))
    state_lon = np.repeat(lon, len(lat))
    netp = (state_lat >= 35) & (state_lat <= 42) & (state_lon >= 92) & (state_lon <= 108)
    weights = np.cos(np.deg2rad(state_lat[netp]))
    annual_netp = np.average(annual[netp], axis=0, weights=weights)
    mjj_netp = np.average(mjj[netp], axis=0, weights=weights)
    jja_netp = np.average(jja129[netp], axis=0, weights=weights)

    diagnostics = {
        "annual_closure_max_abs": closure_max,
        "annual_closure_rmse": closure_rmse,
        "annual_closure_changed_count": closure_changed,
        "annual_round129_max_abs": annual_round129_max_abs,
        "coordinate_and_year_match": coordinate_year_match,
        "netp_cell_count": int(netp.sum()),
        "mjj_all_finite": bool(np.isfinite(mjj).all()),
        "mjj_minimum_cell_standard_deviation": float(np.std(mjj, axis=1, ddof=1).min()),
        "mjj_equals_annual": bool(np.array_equal(mjj, annual)),
        "mjj_equals_jja": bool(np.array_equal(mjj, jja129)),
        "netp_mjj_vs_annual_correlation": correlation(mjj_netp, annual_netp),
        "netp_mjj_vs_jja_correlation": correlation(mjj_netp, jja_netp),
        "netp_mjj_standard_deviation_mm_day": float(np.std(mjj_netp, ddof=1)),
    }
    gates = {
        "exact_annual_closure": closure_max == 0 and closure_rmse == 0 and closure_changed == 0 and annual_round129_max_abs == 0,
        "coordinate_and_year_match": coordinate_year_match,
        "mjj_shape_finite_and_variable": bool(np.isfinite(mjj).all() and diagnostics["mjj_minimum_cell_standard_deviation"] > 0),
        "mjj_distinct_from_annual_and_jja": not diagnostics["mjj_equals_annual"] and not diagnostics["mjj_equals_jja"],
        "netp_mask_has_32_cells": int(netp.sum()) == 32,
    }
    if not all(gates.values()):
        raise RuntimeError(f"round134-prior-gate-failed:{gates}")

    curves = pd.DataFrame({
        "year": years,
        "annual_netp_mm_day": annual_netp,
        "mjj_netp_mm_day": mjj_netp,
        "jja_netp_mm_day": jja_netp,
        "annual_netp_31yr": rolling(annual_netp),
        "mjj_netp_31yr": rolling(mjj_netp),
        "jja_netp_31yr": rolling(jja_netp),
    })
    curves.to_csv(output / "netp_exact_prior_curves.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.5), constrained_layout=True, sharex=True)
    series = (("annual_netp_mm_day", "annual", "#111111"), ("mjj_netp_mm_day", "MJJ", "#0072B2"), ("jja_netp_mm_day", "JJA", "#D55E00"))
    for column, label, color in series:
        values = curves[column].to_numpy()
        axes[0].plot(years, (values - values.mean()) / values.std(ddof=1), color=color, linewidth=0.7, alpha=0.75, label=label)
        smooth = curves[column.replace("_mm_day", "_31yr")].to_numpy()
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

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    summary = {
        "experiment_id": "round_134_netp_mjj_exact_prior_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
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
        "decision": "advance_to_netp_mjj_reconstruction_contract" if all(gates.values()) else "stop_round134",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 134 NETP MJJ Exact Prior\n\n"
        f"- Exact annual closure: `{gates['exact_annual_closure']}`.\n"
        f"- NETP cells: `{diagnostics['netp_cell_count']}`.\n"
        f"- MJJ versus annual correlation: `{diagnostics['netp_mjj_vs_annual_correlation']:.3f}`.\n"
        f"- MJJ versus JJA correlation: `{diagnostics['netp_mjj_vs_jja_correlation']:.3f}`.\n"
        f"- Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
