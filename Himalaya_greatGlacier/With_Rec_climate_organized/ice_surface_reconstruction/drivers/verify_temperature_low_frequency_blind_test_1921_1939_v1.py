"""Independent closure checks for the completed 1921-1939 blind test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from PIL import Image


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_temperature_low_frequency_blind_test_1921_1939_v1"
)
NC = OUTPUT / "temperature_low_frequency_blind_test_1921_1939_v1.nc"
METRICS = OUTPUT / "blind_test_domain_metrics.csv"
SUMMARY = OUTPUT / "summary.json"
FIGURE = OUTPUT / "temperature_low_frequency_blind_test_1921_1939.png"
REFERENCE = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\"
    r"climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
EXPECTED_REFERENCE_SHA = "450850a6654e9080e9586df06d227205b564b2e91e5ba1f95b1610b723664d90"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_mean(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    return np.nansum(values * weights[None], axis=(1, 2)) / np.sum(
        np.isfinite(values) * weights[None], axis=(1, 2)
    )


def main() -> int:
    with Dataset(NC) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        common = np.asarray(dataset.variables["common_support_mask"][:], dtype=bool)
        reference = np.asarray(dataset.variables["tas_reference_family_balanced"][:], dtype=float)
        original = np.asarray(dataset.variables["tas_project_six_member_original"][:], dtype=float)
        correction = np.asarray(dataset.variables["tas_low_frequency_correction"][:], dtype=float)
        candidate = np.asarray(dataset.variables["tas_project_six_member_corrected"][:], dtype=float)
    closure = float(np.nanmax(np.abs(candidate - original - correction)))
    finite_common = bool(
        np.all(np.isfinite(reference[:, common]))
        and np.all(np.isfinite(original[:, common]))
        and np.all(np.isfinite(correction[:, common]))
        and np.all(np.isfinite(candidate[:, common]))
    )
    original_hma = weighted_mean(original, common, lat)
    candidate_hma = weighted_mean(candidate, common, lat)
    reference_hma = weighted_mean(reference, common, lat)
    low_original = pd.Series(original_hma).rolling(11, center=True, min_periods=11).mean().to_numpy()
    low_candidate = pd.Series(candidate_hma).rolling(11, center=True, min_periods=11).mean().to_numpy()
    low_reference = pd.Series(reference_hma).rolling(11, center=True, min_periods=11).mean().to_numpy()
    valid = np.isfinite(low_reference)
    rmse_original = float(np.sqrt(np.mean((low_original[valid] - low_reference[valid]) ** 2)))
    rmse_candidate = float(np.sqrt(np.mean((low_candidate[valid] - low_reference[valid]) ** 2)))
    improvement = float(1 - rmse_candidate / rmse_original)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    metrics = pd.read_csv(METRICS)
    hma = metrics[metrics.domain.eq("hma")].iloc[0]
    with Image.open(FIGURE) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        figure_size = list(image.size)
    checks = {
        "years_are_exact_1921_1939": bool(np.array_equal(years, np.arange(1921, 1940))),
        "common_support_is_1580_cells": bool(int(common.sum()) == 1580),
        "all_fields_finite_on_common_support": finite_common,
        "stored_candidate_formula_closure_at_most_3e_7_degC": bool(closure <= 3e-7),
        "independent_hma_original_rmse_matches": bool(abs(rmse_original - float(hma.original_low11_rmse_degC)) <= 2e-7),
        "independent_hma_candidate_rmse_matches": bool(abs(rmse_candidate - float(hma.candidate_low11_rmse_degC)) <= 2e-7),
        "independent_hma_improvement_matches": bool(abs(improvement - float(hma.low11_rmse_improvement_fraction)) <= 2e-7),
        "summary_decision_is_rejection": summary["decision"] == "reject_correction_bridge_v1",
        "reference_sha256_matches_registered": sha256(REFERENCE) == EXPECTED_REFERENCE_SHA,
        "figure_is_nonblank": bool(pixels.std() > 1.0 and len(np.unique(pixels.reshape(-1, 3), axis=0)) > 100),
    }
    result = {
        "status": "complete",
        "checks": checks,
        "passes_all_checks": bool(all(checks.values())),
        "candidate_formula_closure_max_degC": closure,
        "independent_hma_original_low11_rmse_degC": rmse_original,
        "independent_hma_candidate_low11_rmse_degC": rmse_candidate,
        "independent_hma_improvement_fraction": improvement,
        "reference_sha256": EXPECTED_REFERENCE_SHA,
        "result_netcdf_sha256": sha256(NC),
        "figure_size_pixels": figure_size,
        "figure_rgb_standard_deviation": float(pixels.std()),
    }
    (OUTPUT / "VERIFICATION.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passes_all_checks"]:
        raise RuntimeError("blind-test-verification-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
