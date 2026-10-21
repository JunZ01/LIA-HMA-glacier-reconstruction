from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


DRIVER = Path(__file__).resolve()
DRIVER_DIR = DRIVER.parent
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

import run_temperature_four_prior_strategy_c_replay_1901_1920_v1 as replay
import temperature_candidate_b2_stage_c_core_v1 as core


ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
STATIC_PACKAGE = LOOP / (
    "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/"
    "candidate_b2_stage_c_static_input_v2.npz"
)
BASE_CONTRACT = LOOP / (
    "temperature_balanced_pooled_candidate_b2_stage_c_preregistration_v1/"
    "CANDIDATE_B2_STAGE_C_CONTRACT.md"
)
CENTERING_CONTRACT = DRIVER_DIR / "temperature_candidate_b2_stage_c_observation_centering_v1_contract.md"
DEFAULT_OUT = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_real_proxy_v1"
NETWORKS = core.NETWORKS
AUTHORIZATION_TEXT = "允许执行 Candidate B2 Stage C 真实代理 EnKF"
ALLOWED_YEARS = np.r_[np.arange(1500, 1921), np.arange(1940, 2001)].astype(np.int32)


def validate_authorization(receipt_path: Path) -> dict[str, object]:
    if not receipt_path.exists():
        raise PermissionError("Stage C real-proxy EnKF authorization receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema": "candidate-b2-stage-c-authorization-v1",
        "authorized": True,
        "authorization_text": AUTHORIZATION_TEXT,
        "static_package_sha256": core.EXPECTED_STATIC_PACKAGE_SHA256,
        "driver_sha256": core.sha256(DRIVER),
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise PermissionError(f"authorization-receipt-mismatch:{key}")
    if not str(receipt.get("authorized_at", "")).strip():
        raise PermissionError("authorization-receipt-missing-time")
    return receipt


def read_allowed_proxy_observations(
    network: int,
    package: np.lib.npyio.NpzFile,
    years: np.ndarray,
) -> tuple[np.ndarray, dict[str, int], Path]:
    source = replay.prior_tools.ARCHIVE / f"python_data/proxy_data_{network}.mat"
    selected = np.asarray(package[f"proxy{network}_record_indices_0based"], dtype=int)
    centers = np.asarray(package[f"proxy{network}_archived_hxb_mean_1500_1848"], dtype=float)
    observations = np.full((len(years), len(selected)), np.nan, dtype=np.float64)
    year_lookup = {int(year): index for index, year in enumerate(years)}
    audit = {
        "record_count": int(len(selected)),
        "age_metadata_cells_read": 0,
        "allowed_proxy_value_cells_read": 0,
        "sealed_proxy_value_cells_read": 0,
        "duplicate_allowed_year_assignments": 0,
    }
    with h5py.File(source, "r") as archive:
        references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        for output_index, record_index in enumerate(selected):
            proxy = archive[references[int(record_index)]]
            ages = np.asarray(proxy["age_yng"], dtype=float).reshape(-1, order="F")
            audit["age_metadata_cells_read"] += int(ages.size)
            source_indices = []
            target_indices = []
            for source_index, age in enumerate(ages):
                if not np.isfinite(age):
                    continue
                target = year_lookup.get(int(age))
                if target is not None:
                    source_indices.append(source_index)
                    target_indices.append(target)
            if not source_indices:
                continue
            data = proxy["data"]
            source_array = np.asarray(source_indices, dtype=int)
            if data.shape == (len(ages), 1):
                values = np.asarray(data[source_array, 0], dtype=float)
            elif data.shape == (1, len(ages)):
                values = np.asarray(data[0, source_array], dtype=float)
            else:
                raise RuntimeError(f"unexpected-proxy-data-shape:{network}:{record_index}:{data.shape}")
            audit["allowed_proxy_value_cells_read"] += int(len(values))
            seen: set[int] = set()
            for target, value in zip(target_indices, values):
                if target in seen:
                    audit["duplicate_allowed_year_assignments"] += 1
                seen.add(target)
                if np.isfinite(value):
                    observations[target, output_index] = float(value) - centers[output_index]
    if audit["sealed_proxy_value_cells_read"] != 0:
        raise RuntimeError("sealed-proxy-value-read")
    return observations, audit, source


def main(receipt_path: Path, output: Path) -> None:
    receipt = validate_authorization(receipt_path)
    if core.sha256(STATIC_PACKAGE) != core.EXPECTED_STATIC_PACKAGE_SHA256:
        raise RuntimeError("static-package-hash-mismatch")
    if output.exists():
        raise FileExistsError(output)

    with np.load(STATIC_PACKAGE, allow_pickle=False) as package:
        years = np.asarray(package["years"], dtype=np.int32)
        if not np.array_equal(years, ALLOWED_YEARS):
            raise RuntimeError("allowed-year-axis")
        if np.any((years >= 1921) & (years <= 1939)):
            raise RuntimeError("sealed-year-present")
        state_mean = np.asarray(package["balanced_state_forced_mean"], dtype=np.float64)
        state_residual = np.asarray(package["pooled_state_residual_library"], dtype=np.float64)
        arrays: dict[str, np.ndarray] = {
            "years": years,
            "no_proxy_balanced_background": state_mean.astype(np.float32),
            "retained_state_indices_0based": np.asarray(
                package["retained_state_indices_0based"], dtype=np.int32
            ),
            "retained_state_latitude": np.asarray(package["retained_state_latitude"], dtype=float),
            "retained_state_longitude": np.asarray(package["retained_state_longitude"], dtype=float),
            "gtng_region_codes": np.asarray(package["gtng_region_codes"]).astype("U5"),
            "gtng_region_weights": np.asarray(package["gtng_region_weights"], dtype=float),
            "hma_weights": np.asarray(package["hma_weights"], dtype=float),
        }
        source_paths: dict[str, Path] = {}
        read_audit: dict[str, dict[str, int]] = {}
        network_posteriors = []
        for network in NETWORKS:
            observations, audit, source = read_allowed_proxy_observations(network, package, years)
            operator_mean = np.asarray(
                package[f"proxy{network}_balanced_operator_forced_mean"], dtype=np.float64
            )
            operator_residual = np.asarray(
                package[f"proxy{network}_pooled_operator_residual_library"], dtype=np.float64
            )
            variance = np.asarray(package[f"proxy{network}_residual_variance"], dtype=np.float64)
            posterior, active_count, unique_masks = core.posterior_by_active_mask(
                state_mean,
                state_residual,
                operator_mean,
                operator_residual,
                variance,
                observations,
            )
            arrays[f"proxy{network}_posterior_temperature_anomaly"] = posterior.astype(np.float32)
            arrays[f"proxy{network}_centered_observations"] = observations.astype(np.float32)
            arrays[f"proxy{network}_active_proxy_count"] = active_count
            arrays[f"proxy{network}_record_indices_0based"] = np.asarray(
                package[f"proxy{network}_record_indices_0based"], dtype=np.int32
            )
            network_posteriors.append(posterior)
            audit["unique_active_masks"] = int(unique_masks)
            read_audit[f"proxy{network}"] = audit
            source_paths[f"proxy{network}_archive"] = source
        joint = 0.5 * network_posteriors[0] + 0.5 * network_posteriors[1]
        if not np.all(np.isfinite(joint)):
            raise RuntimeError("nonfinite-joint-posterior")
        arrays["candidate_b2_joint_temperature_anomaly"] = joint.astype(np.float32)

    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "execution_contract": ROOT / "LIA_EXECUTION_CONTRACT.md",
        "static_package": STATIC_PACKAGE,
        "base_contract": BASE_CONTRACT,
        "centering_contract": CENTERING_CONTRACT,
        "authorization_receipt": receipt_path,
        "driver": DRIVER,
        "core": Path(core.__file__).resolve(),
        **source_paths,
    }
    before = {name: core.sha256(path) for name, path in sources.items()}
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=output.parent))
    try:
        result_path = work / "candidate_b2_stage_c_real_proxy_output_v1.npz"
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
            "allowed_periods": [[1500, 1920], [1940, 2000]],
            "year_count": int(len(years)),
            "state_cell_count": int(joint.shape[1]),
            "pooled_member_count": int(state_residual.shape[0]),
            "network_weights": {"proxy6": 0.5, "proxy9": 0.5},
            "observation_centering": (
                "raw proxy value minus archived CESM HXb mean over 1500-1848"
            ),
            "proxy_read_audit": read_audit,
            "all_posterior_values_finite": True,
            "sealed_1921_1939_values_used": False,
            "prohibited_source": "20CRv3",
            "prohibited_source_accessed": False,
            "production_modified": False,
            "oggm_run": False,
            "output": str(output / result_path.name),
            "output_sha256": result_hash,
            "next_action": "execute frozen Stage C observational evaluation",
        }
        (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        manifest = {
            name: {"path": str(path), "sha256": before[name]}
            for name, path in sources.items()
        }
        manifest["output"] = {"path": str(output / result_path.name), "sha256": result_hash}
        manifest["sealed_proxy_value_cells_read"] = 0
        manifest["prohibited_source_accessed"] = False
        (work / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(work, output)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.authorization_receipt, args.output)
