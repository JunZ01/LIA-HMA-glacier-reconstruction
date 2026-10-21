from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import run_temperature_candidate_b2_stage_c_real_proxy_v1 as b2_reader
import temperature_candidate_b2_stage_c_core_v1 as core
from preflight_temperature_candidate_c_intercept_synthetic_v1 import estimate_intercepts
from preflight_temperature_candidate_d2_resolution_v1 import make_blocks
from preflight_temperature_candidate_d_decadal_v1 import block_observations


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
STATIC_PACKAGE = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/candidate_b2_stage_c_static_input_v2.npz"
BASE_CONTRACT = LOOP / "temp_candidate_d2_resolution_prereg_v1/CANDIDATE_D2_CONTRACT.md"
INTERPOLATION_CONTRACT = LOOP / "temp_candidate_d2_resolution_prereg_v1/INTERPOLATION_CONTRACT.md"
SYNTHETIC_PREFLIGHT = LOOP / "temp_candidate_d2_resolution_synthetic_v1/summary.json"
INTERPOLATION_PREFLIGHT = LOOP / "temp_candidate_d2_interpolation_v1/summary.json"
DEFAULT_OUT = LOOP / "temp_candidate_d2_real_v1"
AUTHORIZATION_TEXT = "允许执行 Candidate D2 20年真实代理 EnKF"
AUTHORIZATION_SCHEMA = "candidate-d2-20y-real-proxy-authorization-v1"
EXPECTED_STATIC_SHA256 = "855228E67E87E00AED7CFE9CB9BA7E14DDAFAEB26CC5E3473796E7A1A17BD3A8"
ALLOWED_YEARS = np.r_[np.arange(1500, 1921), np.arange(1940, 2001)].astype(np.int32)


def validate_authorization(receipt_path: Path) -> dict[str, object]:
    if not receipt_path.exists():
        raise PermissionError("Candidate D2 real-proxy authorization receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema": AUTHORIZATION_SCHEMA,
        "authorized": True,
        "authorization_text": AUTHORIZATION_TEXT,
        "static_package_sha256": EXPECTED_STATIC_SHA256,
        "driver_sha256": core.sha256(DRIVER),
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise PermissionError(f"authorization-receipt-mismatch:{key}")
    if not str(receipt.get("authorized_at", "")).strip():
        raise PermissionError("authorization-receipt-missing-time")
    return receipt


def interpolate_field(
    years: np.ndarray,
    blocks: list[tuple[int, int, np.ndarray]],
    block_increment: np.ndarray,
) -> np.ndarray:
    centers = np.asarray([(start + end) / 2.0 for start, end, _ in blocks])
    output = np.full((len(years), block_increment.shape[1]), np.nan, dtype=float)
    for start, end in ((1500, 1920), (1940, 2000)):
        annual = (years >= start) & (years <= end)
        nodes = (centers >= start) & (centers <= end)
        for cell in range(block_increment.shape[1]):
            output[annual, cell] = np.interp(
                years[annual].astype(float), centers[nodes], block_increment[nodes, cell]
            )
    return output


def main(receipt_path: Path, output: Path) -> None:
    receipt = validate_authorization(receipt_path)
    if core.sha256(STATIC_PACKAGE) != EXPECTED_STATIC_SHA256:
        raise RuntimeError("static-package-hash-mismatch")
    if output.exists():
        raise FileExistsError(output)
    synthetic = json.loads(SYNTHETIC_PREFLIGHT.read_text(encoding="utf-8"))
    interpolation = json.loads(INTERPOLATION_PREFLIGHT.read_text(encoding="utf-8"))
    if not synthetic.get("resolution_ladder_pass") or synthetic.get("selected_resolution_years") != 20:
        raise RuntimeError("synthetic-resolution-preflight-not-passed")
    if not interpolation.get("interpolation_validation_pass"):
        raise RuntimeError("interpolation-preflight-not-passed")

    with np.load(STATIC_PACKAGE, allow_pickle=False) as package:
        years = np.asarray(package["years"], dtype=np.int32)
        if not np.array_equal(years, ALLOWED_YEARS):
            raise RuntimeError("allowed-year-axis")
        if np.any((years >= 1921) & (years <= 1939)):
            raise RuntimeError("sealed-year-present")
        state_mean = np.asarray(package["balanced_state_forced_mean"], dtype=float)
        state_residual = np.asarray(package["pooled_state_residual_library"], dtype=float)
        operator_mean = np.asarray(package["proxy6_balanced_operator_forced_mean"], dtype=float)
        operator_residual = np.asarray(package["proxy6_pooled_operator_residual_library"], dtype=float)
        variance = np.asarray(package["proxy6_residual_variance"], dtype=float)
        observations, read_audit, proxy_source = b2_reader.read_allowed_proxy_observations(6, package, years)
        corrected, nuisance_intercepts, intercept_counts = estimate_intercepts(
            years, observations, operator_mean, minimum_count=30
        )
        identified = np.isfinite(nuisance_intercepts)
        if int(identified.sum()) == 0:
            raise RuntimeError("no-identifiable-proxy-intercepts")

        blocks = make_blocks(years, 20)
        block_centers = np.asarray([(start + end) / 2.0 for start, end, _ in blocks])
        block_obs, block_operator, block_counts = block_observations(corrected, operator_mean, blocks)
        block_state_mean = np.stack([state_mean[indices].mean(axis=0) for _, _, indices in blocks])
        block_increment = np.zeros_like(block_state_mean)
        block_active_count = np.zeros(len(blocks), dtype=np.int32)
        for block_index, row in enumerate(block_obs):
            active = np.isfinite(row)
            block_active_count[block_index] = int(active.sum())
            if not active.any():
                continue
            gain = core.kalman_gain(
                state_residual,
                operator_residual[:, active],
                variance[active],
            )
            innovation = row[active] - block_operator[block_index, active]
            block_increment[block_index] = innovation @ gain.T
        annual_increment = interpolate_field(years, blocks, block_increment)
        candidate = state_mean + annual_increment
        if not np.all(np.isfinite(candidate)):
            raise RuntimeError("nonfinite-candidate")

        arrays = {
            "years": years,
            "candidate_d2_temperature_anomaly": candidate.astype(np.float32),
            "balanced_background": state_mean.astype(np.float32),
            "annual_interpolated_increment": annual_increment.astype(np.float32),
            "block_centers": block_centers,
            "block_state_mean": block_state_mean.astype(np.float32),
            "block_increment": block_increment.astype(np.float32),
            "block_active_proxy_count": block_active_count,
            "block_proxy_annual_counts": block_counts.astype(np.int16),
            "proxy6_nuisance_intercepts": nuisance_intercepts,
            "proxy6_intercept_estimation_counts": intercept_counts.astype(np.int32),
            "proxy6_identified_intercept": identified,
            "proxy6_record_indices_0based": np.asarray(package["proxy6_record_indices_0based"], dtype=np.int32),
            "retained_state_indices_0based": np.asarray(package["retained_state_indices_0based"], dtype=np.int32),
            "retained_state_latitude": np.asarray(package["retained_state_latitude"], dtype=float),
            "retained_state_longitude": np.asarray(package["retained_state_longitude"], dtype=float),
            "gtng_region_codes": np.asarray(package["gtng_region_codes"]).astype("U5"),
            "gtng_region_weights": np.asarray(package["gtng_region_weights"], dtype=float),
            "hma_weights": np.asarray(package["hma_weights"], dtype=float),
        }

    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "execution_contract": ROOT / "LIA_EXECUTION_CONTRACT.md",
        "static_package": STATIC_PACKAGE,
        "base_contract": BASE_CONTRACT,
        "interpolation_contract": INTERPOLATION_CONTRACT,
        "synthetic_preflight": SYNTHETIC_PREFLIGHT,
        "interpolation_preflight": INTERPOLATION_PREFLIGHT,
        "authorization_receipt": receipt_path,
        "driver": DRIVER,
        "core": Path(core.__file__).resolve(),
        "safe_reader": Path(b2_reader.__file__).resolve(),
        "proxy6_archive": proxy_source,
    }
    before = {name: core.sha256(path) for name, path in sources.items()}
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=output.parent))
    try:
        result_path = work / "candidate_d2_real_output_v1.npz"
        np.savez_compressed(result_path, **arrays)
        result_hash = core.sha256(result_path)
        after = {name: core.sha256(path) for name, path in sources.items()}
        if before != after:
            raise RuntimeError("source-hash-changed")
        summary = {
            "experiment": output.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "authorization": receipt,
            "selected_resolution_years": 20,
            "proxy_network": "proxy6_single_network",
            "proxy_intercept_estimation_period": [1500, 1848],
            "identified_proxy_count": int(identified.sum()),
            "unidentified_proxy_count": int((~identified).sum()),
            "block_count": len(blocks),
            "block_active_proxy_count_min": int(block_active_count.min()),
            "block_active_proxy_count_max": int(block_active_count.max()),
            "proxy_read_audit": read_audit,
            "sealed_1921_1939_values_used": False,
            "prohibited_source_accessed": False,
            "production_modified": False,
            "oggm_run": False,
            "output": str(output / result_path.name),
            "output_sha256": result_hash,
            "next_action": "execute frozen Candidate D2 observational evaluation",
        }
        (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        manifest = {name: {"path": str(path), "sha256": before[name]} for name, path in sources.items()}
        manifest["output"] = {"path": str(output / result_path.name), "sha256": result_hash}
        (work / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(work, output)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    arguments = parser.parse_args()
    main(arguments.authorization_receipt, arguments.output)
