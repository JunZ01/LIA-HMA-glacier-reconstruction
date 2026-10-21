from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SITE_LAT = 27.89
SITE_LON = 86.37
YEARS = np.arange(1960, 2001, dtype=int)
PRODUCTS = ("CRU_TMIN", "CRU_TMEAN", "BEA_TMEAN")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def read_cru_tmean_djf(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    source_start_year = 1901
    first_year = 1959
    last_year = 2000
    start = (first_year - source_start_year) * 12
    stop = (last_year + 1 - source_start_year) * 12
    with netcdf_file(path, "r", mmap=True) as dataset:
        latitudes = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        longitudes = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
        iy = int(np.argmin(np.abs(latitudes - SITE_LAT)))
        ix = int(np.argmin(np.abs(longitudes - SITE_LON)))
        values = np.asarray(dataset.variables["tmp"].data[start:stop, iy, ix], dtype=float).copy()
        fill = float(dataset.variables["tmp"]._attributes.get("_FillValue", -99999.0))
        values[np.isclose(values, fill) | (values <= -9000) | (values >= 1e20)] = np.nan
        grid_lat = float(latitudes[iy])
        grid_lon = float(longitudes[ix])
    periods = pd.period_range(f"{first_year}-01", f"{last_year}-12", freq="M")
    if len(periods) != len(values):
        raise RuntimeError("cru-tmean-month-count-changed")
    monthly = {(int(period.year), int(period.month)): float(value) for period, value in zip(periods, values)}
    result = np.full(len(YEARS), np.nan)
    for index, year in enumerate(YEARS):
        keys = ((int(year) - 1, 12), (int(year), 1), (int(year), 2))
        winter = np.asarray([monthly.get(key, np.nan) for key in keys], dtype=float)
        if np.isfinite(winter).all():
            result[index] = float(np.mean(winter))
    if not np.isfinite(result).all():
        raise RuntimeError("incomplete-cru-tmean-target")
    return result, {
        "dataset": "CRU TS4.04 monthly mean temperature",
        "variable": "tmp",
        "source_slice_read": [1959, 2000],
        "grid_latitude": grid_lat,
        "grid_longitude": grid_lon,
        "season": "previous_December_current_January_February",
    }


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_239_npl065_tmin_vs_tmean_failure_attribution_v1"
    output.mkdir(parents=True, exist_ok=False)
    round238 = loop / "round_238_npl065_documented_winter_tmin_modern_oof_v4"
    phyda = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round239_npl065_tmin_vs_tmean_failure_attribution_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "round238_summary": round238 / "summary.json",
        "round238_curves": round238 / "modern_oof_curves.csv",
        "cru_tmean": phyda / "input-recon/cruts.2004151855.v4.04/tmp/cru_ts4.04.1901.2019.tmp.dat.nc",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}

    summary238 = json.loads(paths["round238_summary"].read_text(encoding="ascii"))
    if summary238.get("decision") != "close_npl065_documented_winter_temperature_modern_oof":
        raise RuntimeError("round238-decision-changed")
    curves = pd.read_csv(paths["round238_curves"])
    if not np.array_equal(curves.year.to_numpy(int), YEARS):
        raise RuntimeError("round238-year-axis-changed")
    proxy = curves.chronology.to_numpy(float)
    targets = {
        "CRU_TMIN": curves.cru_djf_tmin_observed.to_numpy(float),
        "BEA_TMEAN": curves.bea_djf_tmean_observed.to_numpy(float),
    }
    targets["CRU_TMEAN"], cru_tmean_check = read_cru_tmean_djf(paths["cru_tmean"])

    r238 = load_module(
        "round239_round238",
        drivers / "run_round238_npl065_documented_winter_tmin_modern_oof_v4.py",
    )
    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for product in PRODUCTS:
        metrics, prediction, folds = r238.blocked_oof(proxy, targets[product])
        predictions[product] = prediction
        metric_rows.append({"product": product, **metrics})
        fold_rows.extend({"product": product, **row} for row in folds)
    metrics = pd.DataFrame(metric_rows).set_index("product")

    archived = {row["product"]: row for row in summary238["metrics"]}
    closures = {}
    for product in ("CRU_TMIN", "BEA_TMEAN"):
        differences = [
            abs(float(metrics.at[product, key]) - float(archived[product][key]))
            for key in (
                "oof_correlation",
                "oof_skill_vs_training_fold_climatology",
                "oof_rmse_degC",
                "early_split_proxy_target_correlation",
                "late_split_proxy_target_correlation",
            )
        ]
        closures[product] = float(max(differences))
    if max(closures.values()) > 1e-12:
        raise RuntimeError(f"round238-metric-closure-failed:{closures}")

    target_correlations = {
        "CRU_TMIN_vs_CRU_TMEAN": r238.correlation(targets["CRU_TMIN"], targets["CRU_TMEAN"]),
        "CRU_TMIN_vs_BEA_TMEAN": r238.correlation(targets["CRU_TMIN"], targets["BEA_TMEAN"]),
        "CRU_TMEAN_vs_BEA_TMEAN": r238.correlation(targets["CRU_TMEAN"], targets["BEA_TMEAN"]),
    }
    prediction_correlations = {
        "CRU_TMIN_vs_CRU_TMEAN": r238.correlation(predictions["CRU_TMIN"], predictions["CRU_TMEAN"]),
        "CRU_TMIN_vs_BEA_TMEAN": r238.correlation(predictions["CRU_TMIN"], predictions["BEA_TMEAN"]),
        "CRU_TMEAN_vs_BEA_TMEAN": r238.correlation(predictions["CRU_TMEAN"], predictions["BEA_TMEAN"]),
    }
    passes = {product: bool(metrics.at[product, "oof_correlation"] >= 0.30) for product in PRODUCTS}
    if passes == {"CRU_TMIN": True, "CRU_TMEAN": False, "BEA_TMEAN": False}:
        decision = "npl065_signal_is_tmin_specific_not_mean_temperature_product_failure"
    elif passes == {"CRU_TMIN": True, "CRU_TMEAN": True, "BEA_TMEAN": False}:
        decision = "mean_temperature_product_representation_difference_requires_same_version_test"
    elif all(passes.values()):
        decision = "round238_berkeley_failure_not_reproduced"
    else:
        decision = "mixed_tmin_tmean_failure_attribution"

    curve_output = pd.DataFrame({"year": YEARS, "chronology": proxy})
    for product in PRODUCTS:
        curve_output[f"{product}_observed"] = targets[product]
        curve_output[f"{product}_oof"] = predictions[product]
    curve_output.to_csv(output / "three_target_oof_curves.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "three_target_oof_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "three_target_fold_metrics.csv", index=False)

    figure, axes = plt.subplots(3, 1, figsize=(11.5, 8.2), sharex=True, constrained_layout=True)
    colors = {"CRU_TMIN": "#006D77", "CRU_TMEAN": "#D17A22", "BEA_TMEAN": "#C44E52"}
    labels = {"CRU_TMIN": "CRU TS4.01 DJF Tmin", "CRU_TMEAN": "CRU TS4.04 DJF Tmean", "BEA_TMEAN": "Berkeley DJF Tmean"}
    for axis, product in zip(axes, PRODUCTS):
        row = metrics.loc[product]
        axis.plot(YEARS, targets[product], color=colors[product], linewidth=1.5, label="Observed")
        axis.plot(YEARS, predictions[product], color="#222222", linewidth=1.25, linestyle="--", label=f"OOF, r={row.oof_correlation:.3f}")
        axis.set_ylabel("degC")
        axis.set_title(labels[product])
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncol=2, fontsize=9)
    axes[-1].set_xlabel("Ring year")
    figure.savefig(output / "npl065_tmin_vs_tmean_oof_curves.png", dpi=220)
    plt.close(figure)

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "round238_metric_closure_max_abs": closures,
        "cru_tmean_check": cru_tmean_check,
        "metrics": metric_rows,
        "target_correlations": target_correlations,
        "oof_prediction_correlations": prediction_correlations,
        "correlation_threshold_diagnostics": passes,
        "decision": decision,
        "round238_admission_gate_changed": False,
        "locked_1921_1939_numeric_values_accessed": False,
        "paleo_proxy_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="ascii",
    )
    lines = [
        "# Round 239 NPL065 Tmin versus Tmean attribution",
        "",
    ]
    for product in PRODUCTS:
        lines.append(
            f"- {product}: OOF r=`{metrics.at[product, 'oof_correlation']:.3f}`, skill=`{metrics.at[product, 'oof_skill_vs_training_fold_climatology']:.3f}`."
        )
    lines.extend([
        "",
        f"Decision: `{decision}`.",
        "",
        "Round 238's frozen admission gate was not changed.",
    ])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
