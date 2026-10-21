from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import netcdf_file


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = Path(__file__).resolve().parent
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "round_232_haizi_documented_target_blocked_oof_v1"
CONTRACT = DRIVERS / "round232_haizi_documented_target_blocked_oof_v1_contract.md"
ROUND183_DRIVER = DRIVERS / "run_round183_haizi_903y_chronology_modern_oof_v2.py"
R118_DRIVER = DRIVERS / "run_climate_reconstruction_loop_v2_round118_documented_moisture_seasonal_incremental_skill_v1.py"
WORKBOOK = LOOP / "round_181_haizi_903y_temperature_supplement_v2/raw_unopened/41598_2024_79096_MOESM1_ESM.xlsx"
ROUND183_SUMMARY = LOOP / "round_183_haizi_903y_chronology_modern_oof_v2/summary.json"
ROUND214_SUMMARY = LOOP / "round_214_haizi_annual_temperature_data_recovery_v1/summary.json"
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
PHYDA = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master")
BERKELEY = PHYDA / "input-recon/Land_and_Ocean_LatLong1_1850_2016.nc"
CRU = PHYDA / "input-recon/cruts.2004151855.v4.04/tmp/cru_ts4.04.1901.2019.tmp.dat.nc"
EXPECTED_WORKBOOK_SHA256 = "35302c0ba1912b015ee8d8b1a5c05df9aa82dd44d4467f7b07278e723611d3eb"
DAOCHENG_LAT = 29.03
DAOCHENG_LON = 100.18
MATRIX_YEARS = np.arange(1940, 2001, dtype=int)
TARGET_YEARS = np.arange(1941, 2001, dtype=int)
BLOCKS = ((1941, 1952), (1953, 1964), (1965, 1976), (1977, 1988), (1989, 2000))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}


def previous_october_current_september(monthly: np.ndarray) -> np.ndarray:
    if monthly.shape != (61 * 12, 1):
        raise RuntimeError(f"Unexpected monthly shape: {monthly.shape}")
    cube = monthly[:, 0].reshape(61, 12)
    result = np.full(60, np.nan, dtype=float)
    for current_index in range(1, 61):
        values = np.concatenate((cube[current_index - 1, 9:12], cube[current_index, 0:9]))
        if np.all(np.isfinite(values)):
            result[current_index - 1] = float(np.mean(values))
    if not np.isfinite(result).all():
        raise RuntimeError("Incomplete previous-October to current-September target")
    return result


def nearest_grid_coordinates() -> dict[str, dict[str, float]]:
    output = {}
    with h5py.File(BERKELEY, "r") as handle:
        latitudes = np.asarray(handle["latitude"], dtype=float)
        longitudes = np.asarray(handle["longitude"], dtype=float)
        output["BEA"] = {
            "latitude": float(latitudes[np.argmin(np.abs(latitudes - DAOCHENG_LAT))]),
            "longitude": float(longitudes[np.argmin(np.abs(longitudes - DAOCHENG_LON))]),
        }
    with netcdf_file(CRU, "r", mmap=True) as handle:
        latitudes = np.array(handle.variables["lat"][:], dtype=float, copy=True)
        longitudes = np.array(handle.variables["lon"][:], dtype=float, copy=True)
        output["CRU"] = {
            "latitude": float(latitudes[np.argmin(np.abs(latitudes - DAOCHENG_LAT))]),
            "longitude": float(longitudes[np.argmin(np.abs(longitudes - DAOCHENG_LON))]),
        }
    return output


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3 or np.std(left[finite]) <= 1e-12 or np.std(right[finite]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = {
        "facts": FACTS,
        "contract": CONTRACT,
        "driver": Path(__file__).resolve(),
        "round183_driver": ROUND183_DRIVER,
        "r118_driver": R118_DRIVER,
        "workbook": WORKBOOK,
        "round183_summary": ROUND183_SUMMARY,
        "round214_summary": ROUND214_SUMMARY,
        "berkeley": BERKELEY,
        "cru": CRU,
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: state(path) for name, path in sources.items()}
    if before["workbook"]["sha256"] != EXPECTED_WORKBOOK_SHA256:
        raise RuntimeError("Official workbook hash changed")

    round183 = load_module("round183_for_round232", ROUND183_DRIVER)
    r118 = load_module("round118_for_round232", R118_DRIVER)
    old = json.loads(ROUND183_SUMMARY.read_text(encoding="utf-8"))
    recovered = json.loads(ROUND214_SUMMARY.read_text(encoding="utf-8"))
    if recovered["instrumental_target"] != "Daocheng station, 1959-2017":
        raise RuntimeError("Round 214 instrumental target changed")
    if recovered["target"] != "previous-October to current-September mean temperature":
        raise RuntimeError("Round 214 seasonal target changed")
    if old["exact_alias_count"] != 0 or old["near_alias_count"] != 0:
        raise RuntimeError("Round 183 alias gate is not clean")

    chronology, parse = round183.parse_modern_column(WORKBOOK, "Sheet1", True, "B")
    chronology_matrix = chronology.reindex(MATRIX_YEARS).to_numpy(float)
    if not np.isfinite(chronology_matrix).all():
        raise RuntimeError("Haizi chronology has a modern gap")
    proxy = chronology.reindex(TARGET_YEARS).to_numpy(float)

    latitudes = np.array([DAOCHENG_LAT], dtype=float)
    longitudes = np.array([DAOCHENG_LON], dtype=float)
    monthly = {
        "BEA": r118.read_berkeley_points(BERKELEY, latitudes, longitudes),
        "CRU": r118.read_netcdf3_points(CRU, "tmp", 1901, latitudes, longitudes),
    }
    targets = {name: previous_october_current_september(values) for name, values in monthly.items()}

    fold_rows = []
    summary_rows = []
    predictions = {}
    for product, target in targets.items():
        prediction = np.full(60, np.nan, dtype=float)
        climatology = np.full(60, np.nan, dtype=float)
        for fold_index, (start, end) in enumerate(BLOCKS, start=1):
            test = (TARGET_YEARS >= start) & (TARGET_YEARS <= end)
            train = ~test
            design = np.column_stack((np.ones(int(train.sum())), proxy[train]))
            coefficients = np.linalg.lstsq(design, target[train], rcond=None)[0]
            prediction[test] = coefficients[0] + coefficients[1] * proxy[test]
            climatology[test] = float(np.mean(target[train]))
            fold_rows.append(
                {
                    "product": product,
                    "fold": fold_index,
                    "test_start": start,
                    "test_end": end,
                    "fit_intercept": float(coefficients[0]),
                    "fit_slope": float(coefficients[1]),
                    "test_correlation": correlation(target[test], prediction[test]),
                    "test_rmse_degC": float(np.sqrt(np.mean((target[test] - prediction[test]) ** 2))),
                }
            )
        predictions[product] = prediction
        product_folds = pd.DataFrame([row for row in fold_rows if row["product"] == product])
        candidate_sse = float(np.sum((target - prediction) ** 2))
        baseline_sse = float(np.sum((target - climatology) ** 2))
        early = TARGET_YEARS <= 1970
        late = TARGET_YEARS >= 1971
        summary_rows.append(
            {
                "product": product,
                "oof_correlation": correlation(target, prediction),
                "oof_skill_vs_training_fold_climatology": 1.0 - candidate_sse / baseline_sse,
                "oof_rmse_degC": float(np.sqrt(candidate_sse / len(TARGET_YEARS))),
                "positive_fold_correlation_count": int((product_folds.test_correlation > 0).sum()),
                "positive_fit_slope_count": int((product_folds.fit_slope > 0).sum()),
                "early_split_proxy_target_correlation": correlation(proxy[early], target[early]),
                "late_split_proxy_target_correlation": correlation(proxy[late], target[late]),
            }
        )
    oof = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)
    prediction_correlation = correlation(predictions["BEA"], predictions["CRU"])
    gates = {
        "both_oof_correlations_ge_0_30": bool((oof.oof_correlation >= 0.30).all()),
        "both_oof_skills_positive": bool((oof.oof_skill_vs_training_fold_climatology > 0).all()),
        "both_products_four_of_five_fold_correlations_positive": bool((oof.positive_fold_correlation_count >= 4).all()),
        "all_ten_fit_slopes_positive": bool((oof.positive_fit_slope_count == 5).all()),
        "both_products_early_late_split_correlations_positive": bool(
            ((oof.early_split_proxy_target_correlation > 0) & (oof.late_split_proxy_target_correlation > 0)).all()
        ),
        "bea_cru_oof_prediction_correlation_ge_0_80": bool(prediction_correlation >= 0.80),
    }
    decision = (
        "advance_haizi_documented_target_to_new_psm_design"
        if all(gates.values())
        else "close_haizi_documented_target_modern_oof"
    )

    folds.to_csv(OUT / "blocked_oof_fold_metrics.csv", index=False)
    oof.to_csv(OUT / "blocked_oof_summary.csv", index=False)
    curves = pd.DataFrame({"year": TARGET_YEARS, "chronology": proxy})
    for product in ("BEA", "CRU"):
        curves[f"{product}_documented_target"] = targets[product]
        curves[f"{product}_oof_prediction"] = predictions[product]
    curves.to_csv(OUT / "modern_oof_curves_1941_2000.csv", index=False)

    after = {name: state(path) for name, path in sources.items()}
    if before != after:
        raise RuntimeError("A source fingerprint changed")
    result = {
        "round": 232,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "Haizi Juniperus saltuaria source-native STD chronology",
        "documented_target": {
            "station": "Daocheng",
            "latitude": DAOCHENG_LAT,
            "longitude": DAOCHENG_LON,
            "season": "previous October through current September mean temperature",
            "evaluation_years": [1941, 2000],
            "nearest_product_cells": nearest_grid_coordinates(),
        },
        "superseded_round183_target": {
            "latitude": 29.41,
            "longitude": 100.13,
            "season": "calendar annual mean",
            "old_oof": old["oof"],
        },
        "chronology_parse": parse,
        "blocks": [list(block) for block in BLOCKS],
        "oof": summary_rows,
        "bea_cru_oof_prediction_correlation": prediction_correlation,
        "gates": gates,
        "decision": decision,
        "exact_alias_count_from_round183": old["exact_alias_count"],
        "near_alias_count_from_round183": old["near_alias_count"],
        "paleo_trajectory_exported_or_inspected": False,
        "locked_1921_1939_numeric_values_decoded": False,
        "jionglacuo_accessed": False,
        "enkf_run": False,
        "production_changed": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": True,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Round 232 Result",
        "",
        f"Decision: `{decision}`.",
        "",
        "The source-native chronology was evaluated against the documented Daocheng",
        "previous-October to current-September target using five blocked folds.",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['product']}: OOF r=`{row['oof_correlation']:.3f}`, "
            f"skill=`{row['oof_skill_vs_training_fold_climatology']:.3f}`, "
            f"positive folds=`{row['positive_fold_correlation_count']}/5`."
        )
    lines.extend([
        "",
        "No paleo trajectory, sealed-period value, Jionglacuo value, production",
        "climate, EnKF, or OGGM state was accessed or changed.",
    ])
    (OUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
