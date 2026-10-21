"""Audit proxy9 records 203/241 and run bounded combination holdouts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.io


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("controlled_enkf", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("baseline-driver-load-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trend(years: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(values)
    x = years[finite].astype(float)
    x -= x.mean()
    return float(100.0 * np.polyfit(x, values[finite], 1)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = project_root()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    src = cfg["sources"]
    archive = Path(cfg["archive_root"])
    paths = {
        "baseline_driver": root / src["baseline_driver"],
        "baseline_arrays": root / src["baseline_arrays"],
        "prior": root / src["prior"],
        "indices": archive / src["indices"],
        "proxy9": archive / src["proxy9"],
        "proxy9_regem": root / src["proxy9_regem"],
        "reference_netcdf": Path(src["reference_netcdf"]),
        "processed_metadata_csv": Path(cfg["processed_metadata_csv"]),
    }
    out = root / cfg["result_root"]
    out.mkdir(parents=True, exist_ok=True)
    hashes_before = {key: sha256(path) for key, path in paths.items()}
    base = load_module(paths["baseline_driver"])
    years = np.arange(1000, 2001)
    prior = np.load(paths["prior"])
    with h5py.File(paths["indices"], "r") as handle:
        indices = np.asarray(handle["all_Xb_indices"], dtype=int)[0] - 1
    proxy = base._load_proxy(paths["proxy9"])
    residuals = np.asarray(scipy.io.loadmat(paths["proxy9_regem"])["R0v_filled"], dtype=float)
    grid_lat, grid_lon, state_lat, _ = base._state_coordinates()
    target_lat = float(cfg["target"]["latitude"])
    target_lon = float(cfg["target"]["longitude"])
    target_weights = base._target_weights(grid_lat, grid_lon, target_lat, target_lon)
    regional_weights = np.cos(np.deg2rad(state_lat))
    regional_weights /= regional_weights.sum()
    baseline = base._run_branch(prior, indices, proxy, residuals, target_weights, regional_weights)
    with np.load(paths["baseline_arrays"]) as saved:
        baseline_difference = float(np.max(np.abs(np.asarray(baseline["analysis"]) - saved["proxy9_analysis"])))
    preflight = {"baseline_state_max_abs_c": baseline_difference, "status": "pass" if baseline_difference <= cfg["gates"]["baseline_state_max_abs_c"] else "fail"}
    (out / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    print(json.dumps(preflight, indent=2))
    if preflight["status"] != "pass":
        raise RuntimeError("baseline-closure-failed")
    if args.preflight_only:
        return 0

    metadata = pd.read_csv(paths["processed_metadata_csv"])
    observations = np.asarray(proxy["observations"])
    audit_rows = []
    pair_rows = []
    for record in cfg["focus_records_1based"]:
        idx = int(record) - 1
        values = observations[idx]
        finite = np.isfinite(values)
        finite_years = years[finite]
        row = metadata.loc[metadata.ProxyID.eq(record)].iloc[0]
        audit_rows.append({
            "proxy_index_1based": record,
            "latitude": proxy["locations"][idx, 0],
            "longitude": proxy["locations"][idx, 1],
            "proxy_type": proxy["proxy_types"][idx],
            "psm_method": proxy["psm_methods"][idx],
            "first_finite_year": int(finite_years.min()),
            "last_finite_year": int(finite_years.max()),
            "finite_year_count": int(finite.sum()),
            "continuous_annual": bool(len(finite_years) == finite_years[-1] - finite_years[0] + 1),
            "processed_correlation": row.Correlation,
            "processed_p_value": row.PValue,
            "b0": proxy["coefficients"][idx, 0],
            "b_temperature": proxy["coefficients"][idx, 1],
            "b_precipitation": proxy["coefficients"][idx, 2],
            "regem_residual_variance": float(np.var(residuals[idx], ddof=1)),
            "raw_trend_1901_1950_per_century": trend(years[(years >= 1901) & (years <= 1950)], values[(years >= 1901) & (years <= 1950)]),
        })
        distances = base._haversine_km(proxy["locations"][:, 0], proxy["locations"][:, 1], proxy["locations"][idx, 0], proxy["locations"][idx, 1])
        for other in np.flatnonzero(distances <= 250.0):
            if other == idx:
                continue
            valid = np.isfinite(values) & np.isfinite(observations[other])
            if valid.sum() < 30:
                continue
            pair_rows.append({
                "focus_proxy": record,
                "other_proxy": other + 1,
                "distance_km": distances[other],
                "overlap_years": int(valid.sum()),
                "observation_correlation": float(np.corrcoef(values[valid], observations[other, valid])[0, 1]),
                "other_first_year": int(years[np.isfinite(observations[other])].min()),
                "other_last_year": int(years[np.isfinite(observations[other])].max()),
            })
    pd.DataFrame(audit_rows).to_csv(out / "focus_record_metadata_audit.csv", index=False)
    pd.DataFrame(pair_rows).sort_values(["focus_proxy", "observation_correlation"], ascending=[True, False]).to_csv(out / "focus_record_nearby_correlations.csv", index=False)

    with h5py.File(paths["reference_netcdf"], "r") as handle:
        lat = np.asarray(handle["lat"])
        lon = np.asarray(handle["lon"])
        reference = np.asarray(handle[src["reference_variable"]][:, np.flatnonzero(np.isclose(lat, target_lat))[0], np.flatnonzero(np.isclose(lon, target_lon))[0]], dtype=float)
    ref_period = tuple(cfg["periods"]["reference"])
    early_period = tuple(cfg["periods"]["early_20c"])
    window = int(cfg["phase"]["lowpass_window_years"])
    lag_bounds = tuple(cfg["phase"]["lag_years"])
    early = (years >= early_period[0]) & (years <= early_period[1])
    target_curves = {}
    hma_curves = {}
    metrics = []
    closures = {}
    for name, removed in cfg["cases"].items():
        if not removed:
            result = baseline
        else:
            keep = np.ones(observations.shape[0], dtype=bool)
            keep[np.asarray(removed, dtype=int) - 1] = False
            subset = {key: np.asarray(value)[keep] for key, value in proxy.items()}
            result = base._run_branch(prior, indices, subset, residuals[keep], target_weights, regional_weights)
        target = base._anomaly(np.asarray(result["target_analysis"]), years, ref_period)
        hma = base._anomaly(np.asarray(result["regional_analysis"]), years, ref_period)
        target_curves[name] = target
        hma_curves[name] = hma
        m, _ = base._metrics(name, years, target, reference, early_period, window, lag_bounds)
        metrics.append({"case": name, "removed_1based": ";".join(map(str, removed)), "target_trend_c_per_century": m["trend_c_per_century"], "rolling15_correlation": m["rolling15_correlation"], "best_lag_years": m["best_lag_years"], "phase_projection": base._phase_projection(target[early], reference[early], window), "hma_trend_c_per_century": trend(years[early], hma[early])})
        closures[name] = max(float(result[key]) for key in ("state_increment_closure_max_abs_c", "target_increment_closure_max_abs_c", "regional_increment_closure_max_abs_c"))
        print("completed", name, removed)
    table = pd.DataFrame(metrics)
    baseline_row = table.loc[table.case.eq("no_removal")].iloc[0]
    for col in ("target_trend_c_per_century", "rolling15_correlation", "phase_projection", "hma_trend_c_per_century"):
        table[f"delta_{col}"] = table[col] - baseline_row[col]
    table.to_csv(out / "combination_holdout_metrics.csv", index=False)
    pd.DataFrame({"year": years, **target_curves}).to_csv(out / "target_3584_combination_curves.csv", index=False)
    pd.DataFrame({"year": years, **hma_curves}).to_csv(out / "hma_combination_curves.csv", index=False)
    selected = (years >= cfg["periods"]["plot"][0]) & (years <= cfg["periods"]["plot"][1])
    center = early
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, .9, len(target_curves)))
    for color, name in zip(colors, target_curves):
        for axis, values in ((axes[0], target_curves[name]), (axes[1], hma_curves[name])):
            centered = values - values[center].mean()
            smooth = pd.Series(centered).rolling(window, center=True, min_periods=window).mean().to_numpy()
            axis.plot(years[selected], smooth[selected], lw=2.4 if name == "no_removal" else 1.8, color="#111111" if name == "no_removal" else color, label=name)
    for axis, title in zip(axes, ("3584 centered 15-year curves", "Cosine-weighted HMA centered 15-year curves")):
        axis.axvspan(1901, 1950, color="#d9d9d9", alpha=.25)
        axis.axhline(0, color="#777777", lw=.7)
        axis.grid(alpha=.2)
        axis.set_title(title)
        axis.set_ylabel("Temperature (degC)")
        axis.legend(frameon=False, ncol=2, fontsize=8)
    axes[1].set_xlabel("Year")
    fig.savefig(out / "proxy9_key_record_combination_curves.png", dpi=220)
    plt.close(fig)
    hashes_after = {key: sha256(path) for key, path in paths.items()}
    pure = hashes_before == hashes_after
    passed = pure and max(closures.values()) <= cfg["gates"]["closure_max_abs_c"]
    summary = {"experiment_id": cfg["experiment_id"], "status": "pass" if passed else "fail", "preflight": preflight, "source_purity_unchanged": pure, "maximum_closure_c": max(closures.values()), "metrics": table.to_dict(orient="records"), "metadata_audit": audit_rows, "interpretation_boundary": "Processed files contain no public site identifier or citation; site identity cannot be proven from coordinates alone. Holdouts diagnose influence, not record validity."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    (out / "FINAL_RESULT.md").write_text(f"# Proxy9 Key-Record Audit\n\nStatus: **{summary['status']}**.\n\nNo-removal closure: `{baseline_difference:.3e}` degC. All source hashes unchanged: **{pure}**.\n\n## Combination holdouts\n\n{table.to_markdown(index=False)}\n\n## Metadata boundary\n\nThe local processed archive verifies coordinates, annual coverage, PSM coefficients, calibration correlation and p-value, but it contains no public site identifier or citation. Coordinates alone are insufficient to prove a source identity.\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "metrics": summary["metrics"], "metadata": audit_rows}, indent=2))
    if not passed:
        raise RuntimeError("audit-hard-gate-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
