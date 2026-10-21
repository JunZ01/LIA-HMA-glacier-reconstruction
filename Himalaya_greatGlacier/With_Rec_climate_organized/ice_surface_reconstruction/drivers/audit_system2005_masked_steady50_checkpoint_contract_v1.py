"""Verify exact checkpoint roundtrip and tamper rejection without dynamics."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from . import run_system2005_masked_steady50_checkpointed_v2 as driver


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUT = BASE / "lia_191_system2005_masked_steady50_checkpoint_contract_v1"
GEOLOGY = BASE / "lia_191_igm_planform2d_system2005_static_v1/geology.nc"
STATIC = BASE / "lia_191_igm_planform2d_system2005_static_v1/static_summary.json"
CLIMATE = BASE / "lia_191_torch_planform2d_system2005_climate_bridge_export_v1/monthly_ti_package.npz"
OBSERVATION = BASE / "lia_191_igm_planform2d_system2005_static_v1/observation_path_static.csv"
ROUTE = BASE / "lia_191_current_route_v10_140/summary.json"
STATE = ROOT / "LIA_CURRENT_STATE.json"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    state_control = json.loads(STATE.read_text(encoding="utf-8"))
    if state_control["state_id"] != "2026-08-29-v10.140-system2005-masked-climate-1y-preflight-pass":
        raise RuntimeError("unexpected-controller-state")
    config = {
        "schema": "lia-191-system2005-masked-steady50-checkpointed-config-v1",
        "project_root": str(ROOT),
        "system_id": 2005,
        "target_years": 50,
        "geology": str(GEOLOGY.relative_to(ROOT)),
        "geology_sha256": driver.engine.sha256(GEOLOGY),
        "static_summary": str(STATIC.relative_to(ROOT)),
        "static_summary_sha256": driver.engine.sha256(STATIC),
        "climate_package": str(CLIMATE.relative_to(ROOT)),
        "climate_package_sha256": driver.engine.sha256(CLIMATE),
        "observation_path": str(OBSERVATION.relative_to(ROOT)),
        "observation_path_sha256": driver.engine.sha256(OBSERVATION),
        "domain_variable": "lia_source_mask",
        "glen_a": 4.355770498109968e-23,
        "driver_sha256": driver.engine.sha256(Path(driver.__file__)),
        "expected_route": str(ROUTE.relative_to(ROOT)),
        "expected_route_sha256": driver.engine.sha256(ROUTE),
        "output_root": "not-launched-checkpoint-contract-audit",
    }
    contract_hash = driver.engine.contract_sha256(config)
    with Dataset(GEOLOGY) as dataset:
        thickness = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
        mask = np.asarray(dataset.variables["lia_source_mask"][:], dtype=float) > 0.5
    if np.any((thickness > 0.0) & ~mask):
        raise RuntimeError("initial-state-outside-mask")

    OUT.mkdir(parents=False)
    checkpoint_root = OUT / "checkpoints"
    checkpoint_root.mkdir()
    counters = {
        "accepted_steps": 851,
        "rejected_steps": 293,
        "cumulative_source_m3": -3306142.688558495,
        "maximum_error_relative_l2": 0.04824580858628062,
    }
    metrics = {
        "1": {
            "elapsed_years": 1.0,
            "volume_relative_change": -0.0004092222,
        }
    }
    manifest = driver.write_checkpoint(
        checkpoint_root,
        year=1,
        state=thickness,
        elapsed_seconds=31_536_000.0,
        proposed_dt_seconds=86_400.0,
        counters=counters,
        checkpoint_metrics=metrics,
        contract_hash=contract_hash,
    )
    restored = driver.engine.load_checkpoint(manifest, contract_hash)

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    tampered_contract = dict(manifest_data)
    tampered_contract["contract_sha256"] = "0" * 64
    tampered_contract_path = OUT / "tampered_contract_manifest.json"
    tampered_contract_path.write_bytes(driver.engine.json_bytes(tampered_contract))
    contract_rejection = None
    try:
        driver.engine.load_checkpoint(tampered_contract_path, contract_hash)
    except RuntimeError as error:
        contract_rejection = str(error)

    corrupted_state = OUT / "corrupted_state.npz"
    shutil.copyfile(manifest.parent / "state.npz", corrupted_state)
    with corrupted_state.open("ab") as stream:
        stream.write(b"intentional-checkpoint-integrity-audit")
    corrupt_manifest = dict(manifest_data)
    corrupt_manifest["state_file"] = str(corrupted_state)
    corrupt_manifest_path = OUT / "corrupt_state_manifest.json"
    corrupt_manifest_path.write_bytes(driver.engine.json_bytes(corrupt_manifest))
    state_rejection = None
    try:
        driver.engine.load_checkpoint(corrupt_manifest_path, contract_hash)
    except RuntimeError as error:
        state_rejection = str(error)

    latest = json.loads((checkpoint_root / "latest.json").read_text(encoding="utf-8"))
    temporary_entries = [str(path) for path in checkpoint_root.rglob("*.tmp*")]
    checks = {
        "controller_state_exact": True,
        "initial_state_inside_source_mask": True,
        "state_roundtrip_bitwise_equal": bool(np.array_equal(thickness, restored["state"])),
        "elapsed_roundtrip_exact": restored["elapsed_seconds"] == 31_536_000.0,
        "proposed_dt_roundtrip_exact": restored["proposed_dt_seconds"] == 86_400.0,
        "counters_roundtrip_exact": restored["counters"] == counters,
        "metrics_roundtrip_exact": restored["checkpoint_metrics"] == metrics,
        "manifest_complete": manifest_data["status"] == "checkpoint-complete",
        "manifest_state_hash_exact": driver.engine.sha256(manifest.parent / "state.npz") == manifest_data["state_sha256"],
        "latest_pointer_exact": Path(latest["manifest"]).resolve() == manifest.resolve(),
        "contract_tamper_rejected": contract_rejection == "checkpoint-contract-hash-mismatch",
        "state_tamper_rejected": state_rejection == "checkpoint-state-hash-mismatch",
        "temporary_checkpoint_entries_absent": not temporary_entries,
        "v1_never_launched": not (BASE / "lia_191_system2005_masked_steady50_checkpointed_v1").exists(),
        "production_files_modified": False,
        "dynamics_run": False,
    }
    summary = {
        "schema": "lia-191-system2005-masked-steady50-checkpoint-contract-v1",
        "status": "checkpoint-contract-pass" if all(checks.values()) else "checkpoint-contract-fail",
        "system_id": 2005,
        "tested_driver": str(Path(driver.__file__).relative_to(ROOT)),
        "tested_driver_sha256": driver.engine.sha256(Path(driver.__file__)),
        "v1_status": "superseded-before-launch-by-atomic-directory-v2",
        "contract_sha256": contract_hash,
        "state_shape": list(thickness.shape),
        "state_bytes": int(thickness.nbytes),
        "checkpoint_manifest": str(manifest.relative_to(ROOT)),
        "checkpoint_state_sha256": manifest_data["state_sha256"],
        "contract_tamper_rejection": contract_rejection,
        "state_tamper_rejection": state_rejection,
        "checks": checks,
        "next_action": "preregister one checkpointed masked 50-year steady run",
    }
    (OUT / "summary.json").write_bytes(driver.engine.json_bytes(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
