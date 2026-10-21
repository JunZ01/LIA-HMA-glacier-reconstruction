"""Prepare two linear historical configs for steady system 1880; no OGGM run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEM_ID = 1880
STEADY_ROOT = "w704steady_parent_refine5_1880_v1"
STEADY_MANIFEST = "w704steady_parent_refine5_manifest_v1.json"
STEADY_MANIFEST_SHA256 = "d5b7ae2e76622098fd7cd62a71d0eebad0d14c1580214eb12ca35cf33edc3e2f"
ENDPOINTS = ((0.75, "p075"), (1.25, "p125"))
MANIFEST_NAME = "w704hist_linear_1880_from_refine5_manifest_v1.json"
CONFIG_DIR = (
    "Himalaya_greatGlacier/With_Rec_climate_organized/"
    "ice_surface_reconstruction/config/"
    "synthetic_lia_system_standard_inversion_v1"
)
REFS = (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_controller_v1/core191_length_references.csv"
)
RUNNER = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers."
    "synthetic_lia_conservative_junction_historical_ramp_v1"
)
PYTHON = Path(r"D:\Anaconda\envs\oggm_env2\python.exe")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / CONFIG_DIR
    manifest_path = root / MANIFEST_NAME
    steady_manifest_path = root / STEADY_MANIFEST
    steady_root = root / STEADY_ROOT
    steady_config_path = steady_root / "config_snapshot.json"
    steady_summary_path = steady_root / "steady_50y_summary.json"
    steady_state_path = steady_root / "lia_equilibrium_state_50y.pkl.gz"
    refs_path = root / REFS

    _require(_sha(steady_manifest_path) == STEADY_MANIFEST_SHA256, "steady-manifest-conflict")
    _require(PYTHON.is_file(), "python-missing")
    for path in (steady_config_path, steady_summary_path, steady_state_path, refs_path):
        _require(path.is_file(), f"required-evidence-missing:{path}")
    steady_manifest = _read(steady_manifest_path)
    steady_case = next(c for c in steady_manifest["cases"] if int(c["system_id"]) == SYSTEM_ID)
    _require(str(steady_case["output_root"]) == STEADY_ROOT, "steady-root-manifest-conflict")
    source_config_path = root / str(steady_case["config"])
    _require(_read(source_config_path) == _read(steady_config_path), "steady-config-snapshot-conflict")
    steady_config = _read(steady_config_path)
    steady_summary = _read(steady_summary_path)
    _require(steady_summary.get("status") == "steady-50y-pass", "steady-not-pass")
    _require(not steady_summary.get("review_flags", []), "steady-review-flags")
    _require(int(steady_summary["final"]["year"]) == 50, "steady-incomplete")
    _require(abs(float(steady_summary["volume_change_fraction_50y"])) <= 0.02, "steady-volume-gate-conflict")
    _require(float(steady_summary["final"]["length_relative_error_lia"]) <= 0.1, "steady-length-gate-conflict")
    _require(float(steady_summary["final"]["remaining_free_downstream_m"]) >= 5000.0, "steady-domain-gate-conflict")
    _require(float(steady_config["stage2"]["delta_t_c"]) == 0.25, "steady-start-delta-conflict")
    _require(steady_config["stage2"]["prcp_anomaly_mode"] == "off", "precipitation-contract-conflict")

    inversion_name = str(steady_config["inversion_root"]["sub"])
    inversion_root = root / inversion_name
    inversion_config_path = inversion_root / "config_snapshot.json"
    inversion_summary_path = inversion_root / "inversion_summary_formal.json"
    inversion_config, inversion_summary = _read(inversion_config_path), _read(inversion_summary_path)
    _require(inversion_summary.get("status") == "inversion-pass", "inversion-not-pass")
    _require(not inversion_summary.get("review_flags", []), "inversion-review-flags")
    _require(inversion_config["stage2"] == steady_config["stage2"] | {"delta_t_c": inversion_config["stage2"]["delta_t_c"]}, "unreachable")
    _require(str(inversion_config["synthetic_rgi_id"]) == str(steady_config["synthetic_rgi_id"]), "synthetic-rgi-conflict")

    refs = pd.read_csv(refs_path)
    rows = refs[pd.to_numeric(refs["lia_unique_id"]).astype(int).eq(SYSTEM_ID)]
    _require(len(rows) == 1, f"length-reference-row-count:{len(rows)}")
    row = rows.iloc[0]
    common = {
        "experiment": "synthetic-lia-system-conservative-junction-v1",
        "stage": "remaining-route-v3-system-1880-historical-linear-screen-v1",
        "system_id": SYSTEM_ID,
        "main_rgi_id": inversion_config["main_rgi_id"],
        "synthetic_rgi_id": inversion_config["synthetic_rgi_id"],
        "inversion_root": {"anchor": "project", "sub": inversion_name},
        "steady_root": {"anchor": "project", "sub": STEADY_ROOT},
        "climate_pixels": inversion_config["climate_pixels"],
        "climate_id": inversion_config["climate_id"],
        "reconstruction_climate": inversion_config["reconstruction_climate"],
        "start_year": 1470,
        "end_year": 2000,
        "minimum_timestep_seconds": 60.0,
        "lia_registered_length_m": float(row["lia_registered_length_m"]),
        "rgi_registered_length_m": float(row["rgi_registered_length_m"]),
        "stage2": steady_config["stage2"],
    }
    targets: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    destinations = [manifest_path]
    output_roots: list[Path] = []
    for endpoint, label in ENDPOINTS:
        config_path = config_root / f"historical_linear_1880_end{label}_from_refine5_v1.json"
        output_name = f"w704hist_linear_1880_end{label}_from_refine5_v1"
        payload = {**common, "output_root": {"anchor": "project", "sub": output_name}, "forward_delta_t_end_c": endpoint}
        command = f"& '{PYTHON}' -m {RUNNER} --config '{config_path}'"
        record = {
            "system_id": SYSTEM_ID,
            "start_delta_t_c": 0.25,
            "end_delta_t_c": endpoint,
            "config": str(config_path.relative_to(root)),
            "output_root": output_name,
            "run_command_powershell": command,
        }
        targets.append((config_path, payload, record))
        destinations.append(config_path)
        output_roots.append(root / output_name)
    collisions = [str(p) for p in destinations + output_roots if p.exists()]
    _require(not collisions, "refusing-to-overwrite:" + repr(collisions))
    config_root.mkdir(parents=True, exist_ok=True)
    for path, payload, _ in targets:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        _require(_read(path) == payload, f"config-roundtrip-conflict:{path}")
    _require(all(not p.exists() for p in output_roots), "output-created-during-preparation")

    evidence_paths = {
        "steady_manifest": steady_manifest_path,
        "steady_source_config": source_config_path,
        "steady_run_config_snapshot": steady_config_path,
        "steady_summary": steady_summary_path,
        "steady_state": steady_state_path,
        "inversion_config_snapshot": inversion_config_path,
        "inversion_summary_formal": inversion_summary_path,
        "length_references": refs_path,
    }
    manifest = {
        "status": "configs-built-not-run",
        "count": 2,
        "runner_module": RUNNER,
        "contract": {"start_year": 1470, "end_year": 2000, "start_delta_t_c": 0.25, "precipitation_anomaly_mode": "off", "minimum_timestep_seconds": 60.0},
        "frozen_evidence": {name: {"path": str(path.relative_to(root)), "sha256": _sha(path)} for name, path in evidence_paths.items()},
        "oggm_started_by_builder": False,
        "static_validation": "pass",
        "cases": [record for _, _, record in targets],
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _require(_read(manifest_path) == manifest, "manifest-roundtrip-conflict")
    print(json.dumps({"status": manifest["status"], "count": 2, "manifest": str(manifest_path), "static_validation": "pass"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
