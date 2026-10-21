"""Prepare locked-safe historical observations for Round 130."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


YEARS = np.arange(1000, 1921, dtype=int)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_xinpu(path: Path) -> tuple[np.ndarray, dict]:
    values = {}
    locked_decoded = 0
    all_years = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                year = int(round(float(parts[0].strip())))
            except ValueError:
                continue
            all_years.append(year)
            if 1921 <= year <= 1939:
                continue
            if year <= 1920 and year in set(YEARS.tolist()):
                values[year] = float(parts[2].strip())
    result = np.asarray([values.get(int(year), np.nan) for year in YEARS])
    return result, {"first_year": min(all_years), "last_year": max(all_years), "active_count_1000_1920": int(np.isfinite(result).sum()), "locked_values_decoded": locked_decoded}


def parse_keylong(path: Path) -> tuple[np.ndarray, dict]:
    values = {}
    locked_decoded = 0
    all_years = []
    wanted = set(YEARS.tolist())
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                year = int(round(float(parts[0])))
            except ValueError:
                continue
            all_years.append(year)
            if 1921 <= year <= 1939:
                continue
            if year in wanted:
                values[year] = float(parts[1])
    result = np.asarray([values.get(int(year), np.nan) for year in YEARS])
    return result, {"first_year": min(all_years), "last_year": max(all_years), "active_count_1000_1920": int(np.isfinite(result).sum()), "locked_values_decoded": locked_decoded}


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    base = load_module("round130_base", driver / "run_round129_new_d18o_seasonal_prior_coupling_v2.py")
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    r129 = loop / "round_129_new_d18o_exact_seasonal_prior_coupling_v3"
    r123 = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_southeast_tibet_d18o_proxy_qualification_20260818_v1"
    r125 = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_ganesh_keylong_d18o_proxy_qualification_20260818_v1"
    output = loop / "round_130_xinpu_keylong_frozen_enkf_addition_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": driver / "round130_xinpu_keylong_frozen_enkf_addition_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "xinpu": r123 / "raw_unopened/xinpu_tree_ring_d18o.txt",
        "keylong": r125 / "raw_unopened/keylong2020d18o-noaa.txt",
        "r129_summary": r129 / "summary.json",
        "r129_coefficients": r129 / "full_calibration_coefficients.csv",
        "r129_metrics": r129 / "operator_coupling_metrics.csv",
        "exact_seasonal_prior": r129 / "exact_seasonal_prior_fields.mat",
        "replay_input": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_enkf_exp1_replay_v1/replay_input.mat",
        "annual_prior": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_prior_recovery_preflight_v1/precipitation_prior_336x848_mm_day.mat",
    }
    if missing := [str(path) for path in paths.values() if not path.exists()]:
        raise FileNotFoundError("missing:" + ";".join(missing))
    shutil.copy2(paths["contract"], output / "ROUND_CONTRACT.md")
    before = {name: base.fingerprint(path) for name, path in paths.items()}
    summary = json.loads(paths["r129_summary"].read_text(encoding="utf-8"))
    passed = {item["site"] for item in summary["decisions"] if item["passes_all_gates"]}
    if passed != {"xinpu", "keylong"} or summary["locked_1921_1939_accessed"]:
        raise RuntimeError(f"round129-decision-contract-failed:{passed}")
    xinpu, xinpu_audit = parse_xinpu(paths["xinpu"])
    keylong, keylong_audit = parse_keylong(paths["keylong"])
    if xinpu_audit["locked_values_decoded"] or keylong_audit["locked_values_decoded"]:
        raise RuntimeError("locked-value-decoded")
    coefficients = pd.read_csv(paths["r129_coefficients"])
    metrics = pd.read_csv(paths["r129_metrics"])
    payload = {"years": YEARS.astype(float), "y_new": np.vstack((xinpu, keylong))}
    for site in ("xinpu", "keylong"):
        for product, short in (("BEA_GPCC", "bea"), ("CRU", "cru")):
            row = coefficients[(coefficients.site == site) & (coefficients["product"] == product)].iloc[0]
            metric = metrics[(metrics.site == site) & (metrics.calibration_product == product)].iloc[0]
            payload[f"{site}_coeff_{short}"] = row[["intercept", "temperature_coefficient", "precipitation_coefficient"]].to_numpy(float)
            payload[f"{site}_rvar_{short}"] = np.asarray([[float(metric.oof_rmse) ** 2]])
    savemat(output / "round130_input.mat", payload, do_compression=True)
    after = {name: base.fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-changed-during-round130-preparation")
    preparation = {
        "status": "complete",
        "years": [1000, 1920],
        "locked_1921_1939_accessed": False,
        "locked_1921_1939_values_decoded": 0,
        "xinpu": xinpu_audit,
        "keylong": keylong_audit,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "preparation_summary.json").write_text(json.dumps(preparation, indent=2), encoding="utf-8")
    (output / "preparation_source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    print(json.dumps(preparation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
