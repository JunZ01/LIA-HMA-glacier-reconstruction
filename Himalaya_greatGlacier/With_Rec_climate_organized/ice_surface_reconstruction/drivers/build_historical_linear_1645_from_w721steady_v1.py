"""Build two frozen 1645 linear histories from the formal w721 steady pass.

This builder performs provenance/static checks and exclusive JSON writes only.
It does not import or execute OGGM.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEM_ID = 1645
STEADY_ROOT = "w721steady_parent_refine1645_dt0p0625_v1"
STEADY_CONFIG_NAME = "steady_parent_refine1645_dt0p0625_v1.json"
ENDPOINTS = ((0.5625, "0p5625"), (1.0625, "1p0625"))
MANIFEST_NAME = "w722hist_linear_1645_from_w721steady_manifest_v1.json"
CONFIG_DIR = (
    "Himalaya_greatGlacier/With_Rec_climate_organized/"
    "ice_surface_reconstruction/config/"
    "synthetic_lia_system_standard_inversion_v1"
)
REFS = (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_controller_v1/core191_length_references.csv"
)
FORMAL_COLLECTOR_CONFIG = (
    "Himalaya_greatGlacier/With_Rec_climate_organized/"
    "ice_surface_reconstruction/config/"
    "collect_lia_191_active_historical_results_v1.json"
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


def _hashed(root: Path, path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": _sha(path)}


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / CONFIG_DIR
    steady_root = root / STEADY_ROOT
    source_config_path = config_root / STEADY_CONFIG_NAME
    snapshot_path = steady_root / "config_snapshot.json"
    summary_path = steady_root / "steady_50y_summary.json"
    state_path = steady_root / "lia_equilibrium_state_50y.pkl.gz"
    refs_path = root / REFS
    collector_config_path = root / FORMAL_COLLECTOR_CONFIG
    manifest_path = root / MANIFEST_NAME
    for path in (
        source_config_path,
        snapshot_path,
        summary_path,
        state_path,
        refs_path,
        collector_config_path,
    ):
        _require(path.is_file(), f"required-evidence-missing:{path}")
    _require(PYTHON.is_file(), f"python-missing:{PYTHON}")

    source_config = _read(source_config_path)
    steady_config = _read(snapshot_path)
    summary = _read(summary_path)
    _require(source_config == steady_config, "steady-config-snapshot-conflict")
    _require(int(steady_config["system_id"]) == SYSTEM_ID, "steady-system-id-conflict")
    _require(float(steady_config["stage2"]["delta_t_c"]) == 0.0625, "steady-delta-conflict")
    _require(steady_config["stage2"]["prcp_anomaly_mode"] == "off", "precipitation-contract-conflict")
    _require(summary.get("status") == "steady-50y-pass", "steady-not-pass")
    _require(not summary.get("review_flags", []), "steady-review-flags")
    _require(int(summary["final"]["year"]) == 50, "steady-incomplete")
    _require(abs(float(summary["volume_change_fraction_50y"]) + 0.015600553706338574) < 1e-15, "steady-dv-conflict")
    _require(abs(float(summary["final"]["length_relative_error_lia"]) - 0.013337183431887206) < 1e-15, "steady-length-error-conflict")
    _require(float(summary["final"]["remaining_free_downstream_m"]) == 5300.0, "steady-free-domain-evidence-conflict")
    _require(abs(float(summary["volume_change_fraction_50y"])) <= 0.02, "steady-volume-gate-conflict")
    _require(float(summary["final"]["length_relative_error_lia"]) <= 0.10, "steady-length-gate-conflict")
    _require(float(summary["final"]["remaining_free_downstream_m"]) >= 5000.0, "steady-domain-gate-conflict")

    inversion_name = str(steady_config["inversion_root"]["sub"])
    inversion_config_path = root / inversion_name / "config_snapshot.json"
    inversion_summary_path = root / inversion_name / "inversion_summary_formal.json"
    _require(inversion_config_path.is_file() and inversion_summary_path.is_file(), "inversion-evidence-missing")
    inversion_config, inversion_summary = _read(inversion_config_path), _read(inversion_summary_path)
    _require(inversion_summary.get("status") == "inversion-pass", "inversion-not-pass")
    _require(not inversion_summary.get("review_flags", []), "inversion-review-flags")
    _require(str(inversion_config["synthetic_rgi_id"]) == str(steady_config["synthetic_rgi_id"]), "synthetic-rgi-conflict")
    inv_stage2 = copy.deepcopy(inversion_config["stage2"])
    steady_stage2 = copy.deepcopy(steady_config["stage2"])
    inv_stage2.pop("delta_t_c")
    steady_stage2.pop("delta_t_c")
    _require(inv_stage2 == steady_stage2, "stage2-nondelta-physics-conflict")

    with refs_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["lia_unique_id"]) == SYSTEM_ID]
    _require(len(rows) == 1, f"length-reference-row-count:{len(rows)}")
    row = rows[0]
    collector_config = _read(collector_config_path)
    thresholds = collector_config["thresholds"]
    _require(float(thresholds["total_error_max"]) == 0.10, "formal-total-length-gate-conflict")
    _require(float(thresholds["separation_error_max"]) == 0.20, "formal-separation-length-gate-conflict")

    common = {
        "experiment": "synthetic-lia-system-conservative-junction-v1",
        "stage": "remaining-route-v3-system-1645-historical-linear-screen-v1",
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
    prepared = []
    output_roots: list[Path] = []
    destinations = [manifest_path]
    for endpoint, label in ENDPOINTS:
        config_path = config_root / f"historical_linear_1645_endp{label}_from_w721steady_v1.json"
        output_name = f"w722hist_linear_1645_endp{label}_from_w721steady_v1"
        payload = {
            **common,
            "output_root": {"anchor": "project", "sub": output_name},
            "forward_delta_t_end_c": endpoint,
        }
        prepared.append((config_path, payload, {
            "system_id": SYSTEM_ID,
            "start_delta_t_c": 0.0625,
            "end_delta_t_c": endpoint,
            "config": str(config_path.relative_to(root)),
            "output_root": output_name,
            "run_authorized": False,
        }))
        destinations.append(config_path)
        output_roots.append(root / output_name)
    frozen = []
    for _, payload, _ in prepared:
        item = copy.deepcopy(payload)
        del item["output_root"]
        del item["forward_delta_t_end_c"]
        frozen.append(item)
    _require(frozen[0] == frozen[1], "historical-pair-frozen-contract-conflict")
    collisions = [str(path) for path in destinations + output_roots if path.exists()]
    _require(not collisions, "refusing-to-overwrite:" + repr(collisions))

    config_root.mkdir(parents=True, exist_ok=True)
    for path, payload, record in prepared:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        _require(_read(path) == payload, f"config-roundtrip-conflict:{path}")
        record["new_config_sha256"] = _sha(path)
    _require(all(not path.exists() for path in output_roots), "output-created-during-preparation")

    evidence_paths = {
        "steady_source_config": source_config_path,
        "steady_run_config_snapshot": snapshot_path,
        "steady_summary": summary_path,
        "steady_state": state_path,
        "inversion_config_snapshot": inversion_config_path,
        "inversion_summary_formal": inversion_summary_path,
        "length_references": refs_path,
        "formal_collector_config": collector_config_path,
    }
    manifest = {
        "status": "configs-built-static-verified-not-run",
        "builder": str(Path(__file__).relative_to(root)),
        "count": 2,
        "runner_module_informational_only": RUNNER,
        "contract": {
            "start_year": 1470,
            "end_year": 2000,
            "start_delta_t_c": 0.0625,
            "precipitation_anomaly_mode": "off",
            "minimum_timestep_seconds": 60.0,
            "formal_total_length_error_max": 0.10,
            "formal_separation_length_error_max": 0.20,
            "lia_registered_length_m": common["lia_registered_length_m"],
            "rgi_registered_length_m": common["rgi_registered_length_m"],
        },
        "steady_pass_evidence": {
            "volume_change_fraction_50y": float(summary["volume_change_fraction_50y"]),
            "length_relative_error_lia": float(summary["final"]["length_relative_error_lia"]),
            "remaining_free_downstream_m": float(summary["final"]["remaining_free_downstream_m"]),
        },
        "frozen_evidence": {name: _hashed(root, path) for name, path in evidence_paths.items()},
        "historical_pair_allowed_changes": ["output_root", "forward_delta_t_end_c"],
        "run_authorized": False,
        "oggm_imported": False,
        "oggm_started_by_builder": False,
        "output_roots_created": False,
        "static_validation": "pass",
        "cases": [record for _, _, record in prepared],
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _require(_read(manifest_path) == manifest, "manifest-roundtrip-conflict")
    print(json.dumps({"status": manifest["status"], "count": 2, "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
