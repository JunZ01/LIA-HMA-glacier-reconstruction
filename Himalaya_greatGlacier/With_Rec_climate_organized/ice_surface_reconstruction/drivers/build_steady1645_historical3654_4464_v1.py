"""Build one steady refinement and four linear historical configs; never run OGGM."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


CONFIG_DIR = (
    "Himalaya_greatGlacier/With_Rec_climate_organized/"
    "ice_surface_reconstruction/config/"
    "synthetic_lia_system_standard_inversion_v1"
)
REFS = (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_controller_v1/core191_length_references.csv"
)
MANIFEST_NAME = "w721steady1645_hist3654_4464_manifest_v1.json"
HISTORICAL_RUNNER = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers."
    "synthetic_lia_conservative_junction_historical_ramp_v1"
)
STEADY_RUNNER = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers.synthetic_lia_system_steady_50y_v1"
)
PYTHON = Path(r"D:\Anaconda\envs\oggm_env2\python.exe")

STEADY_1645 = {
    "target_delta_t_c": 0.0625,
    "zero_root": "w719steady_parent_bisection4_1645_dt0p0_v1",
    "warm_root": "w704coldsteady_parent_m0p5_1645_v1",
    "zero_delta_t_c": 0.0,
    "warm_delta_t_c": 0.25,
    "zero_dv": 0.02917613277296381,
    "warm_dv": -0.16447427148225308,
}

HISTORICAL = {
    4464: {
        "steady_root": "w719steady_parent_bisection4_4464_dtm0p5_v1",
        "steady_config": "steady_parent_bisection4_4464_dtm0p5_v1.json",
        "start_delta_t_c": -0.5,
        "expected_dv": -0.0005563865258002515,
        "expected_length_error": 0.03074089585969102,
        "expected_free_m": 5950.0,
        "endpoints": [(0.0, "0p0"), (0.5, "0p5")],
    },
    3654: {
        "steady_root": "w719steady_parent_bisection4_3654_dt0p0_v1",
        "steady_config": "steady_parent_bisection4_3654_dt0p0_v1.json",
        "start_delta_t_c": 0.0,
        "expected_dv": -0.003919760317515904,
        "expected_length_error": 0.02035138017182619,
        "expected_free_m": 5500.0,
        "endpoints": [(0.5, "0p5"), (1.0, "1p0")],
    },
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _strip_steady_allowed(config: dict[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(config)
    del frozen["output_root"]
    del frozen["stage2"]["delta_t_c"]
    return frozen


def _hashed(root: Path, path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": _sha(path)}


def _steady_evidence(root: Path, output_name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    snapshot_path = root / output_name / "config_snapshot.json"
    summary_path = root / output_name / "steady_50y_summary.json"
    _require(snapshot_path.is_file() and summary_path.is_file(), f"missing-steady-evidence:{output_name}")
    config, summary = _read(snapshot_path), _read(summary_path)
    evidence = {
        "output_root": output_name,
        "config_snapshot": _hashed(root, snapshot_path),
        "steady_50y_summary": _hashed(root, summary_path),
        "status": summary["status"],
        "delta_t_c": float(config["stage2"]["delta_t_c"]),
        "volume_change_fraction_50y": float(summary["volume_change_fraction_50y"]),
        "length_relative_error_lia": float(summary["final"]["length_relative_error_lia"]),
        "remaining_free_downstream_m": float(summary["final"]["remaining_free_downstream_m"]),
        "final_year": int(summary["final"]["year"]),
        "review_flags": list(summary.get("review_flags", [])),
    }
    return config, summary, evidence


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / CONFIG_DIR
    refs_path = root / REFS
    manifest_path = root / MANIFEST_NAME
    _require(PYTHON.is_file(), f"python-missing:{PYTHON}")
    _require(refs_path.is_file(), "length-references-missing")
    with refs_path.open("r", encoding="utf-8-sig", newline="") as handle:
        ref_rows = {int(row["lia_unique_id"]): row for row in csv.DictReader(handle)}

    new_items: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    output_roots: list[Path] = []

    # A: one explicit interior refinement for system 1645.
    zero_cfg, zero_sum, zero_ev = _steady_evidence(root, STEADY_1645["zero_root"])
    warm_cfg, warm_sum, warm_ev = _steady_evidence(root, STEADY_1645["warm_root"])
    _require(_strip_steady_allowed(zero_cfg) == _strip_steady_allowed(warm_cfg), "1645-frozen-parent-conflict")
    _require(zero_ev["delta_t_c"] == STEADY_1645["zero_delta_t_c"], "1645-zero-delta-conflict")
    _require(warm_ev["delta_t_c"] == STEADY_1645["warm_delta_t_c"], "1645-warm-delta-conflict")
    _require(abs(zero_ev["volume_change_fraction_50y"] - STEADY_1645["zero_dv"]) < 1e-15, "1645-zero-dv-conflict")
    _require(abs(warm_ev["volume_change_fraction_50y"] - STEADY_1645["warm_dv"]) < 1e-15, "1645-warm-dv-conflict")
    _require(zero_ev["volume_change_fraction_50y"] > 0 > warm_ev["volume_change_fraction_50y"], "1645-not-bracketed")
    target = float(STEADY_1645["target_delta_t_c"])
    _require(0.0 < target < 0.25, "1645-target-outside-bracket")
    steady_name = "steady_parent_refine1645_dt0p0625_v1.json"
    steady_output = "w721steady_parent_refine1645_dt0p0625_v1"
    steady_path = config_root / steady_name
    steady_payload = copy.deepcopy(zero_cfg)
    steady_payload["stage2"]["delta_t_c"] = target
    steady_payload["output_root"] = {"anchor": "project", "sub": steady_output}
    _require(_strip_steady_allowed(steady_payload) == _strip_steady_allowed(zero_cfg), "1645-new-frozen-field-conflict")
    new_items.append((steady_path, steady_payload, {
        "kind": "steady-50y",
        "system_id": 1645,
        "decision_rule": "explicit-interior-bracket-refinement",
        "target_delta_t_c": target,
        "config": str(steady_path.relative_to(root)),
        "output_root": steady_output,
        "zero_parent_evidence": zero_ev,
        "warm_parent_evidence": warm_ev,
        "runner_module_informational_only": STEADY_RUNNER,
        "run_authorized": False,
    }))
    output_roots.append(root / steady_output)

    # B: two frozen linear histories from each formal steady-pass state.
    for system_id, spec in HISTORICAL.items():
        steady_root_name = str(spec["steady_root"])
        steady_cfg, steady_summary, steady_ev = _steady_evidence(root, steady_root_name)
        source_config_path = config_root / str(spec["steady_config"])
        steady_state_path = root / steady_root_name / "lia_equilibrium_state_50y.pkl.gz"
        _require(source_config_path.is_file() and steady_state_path.is_file(), f"historical-source-missing:{system_id}")
        _require(_read(source_config_path) == steady_cfg, f"steady-source-snapshot-conflict:{system_id}")
        _require(steady_summary.get("status") == "steady-50y-pass", f"steady-not-pass:{system_id}")
        _require(not steady_summary.get("review_flags", []), f"steady-review-flags:{system_id}")
        _require(steady_ev["final_year"] == 50, f"steady-incomplete:{system_id}")
        _require(abs(steady_ev["volume_change_fraction_50y"] - float(spec["expected_dv"])) < 1e-15, f"steady-dv-conflict:{system_id}")
        _require(abs(steady_ev["length_relative_error_lia"] - float(spec["expected_length_error"])) < 1e-15, f"steady-length-conflict:{system_id}")
        _require(steady_ev["remaining_free_downstream_m"] == float(spec["expected_free_m"]), f"steady-domain-conflict:{system_id}")
        _require(abs(steady_ev["volume_change_fraction_50y"]) <= 0.02, f"steady-volume-gate:{system_id}")
        _require(steady_ev["length_relative_error_lia"] <= 0.1, f"steady-length-gate:{system_id}")
        _require(steady_ev["remaining_free_downstream_m"] >= 5000.0, f"steady-domain-gate:{system_id}")
        _require(float(steady_cfg["stage2"]["delta_t_c"]) == float(spec["start_delta_t_c"]), f"steady-start-delta-conflict:{system_id}")
        _require(steady_cfg["stage2"]["prcp_anomaly_mode"] == "off", f"precipitation-contract-conflict:{system_id}")

        inversion_name = str(steady_cfg["inversion_root"]["sub"])
        inversion_config_path = root / inversion_name / "config_snapshot.json"
        inversion_summary_path = root / inversion_name / "inversion_summary_formal.json"
        _require(inversion_config_path.is_file() and inversion_summary_path.is_file(), f"inversion-evidence-missing:{system_id}")
        inversion_cfg, inversion_summary = _read(inversion_config_path), _read(inversion_summary_path)
        _require(inversion_summary.get("status") == "inversion-pass", f"inversion-not-pass:{system_id}")
        _require(not inversion_summary.get("review_flags", []), f"inversion-review-flags:{system_id}")
        inv_s2, steady_s2 = copy.deepcopy(inversion_cfg["stage2"]), copy.deepcopy(steady_cfg["stage2"])
        inv_s2.pop("delta_t_c")
        steady_s2.pop("delta_t_c")
        _require(inv_s2 == steady_s2, f"stage2-nondelta-physics-conflict:{system_id}")
        _require(system_id in ref_rows, f"length-reference-missing:{system_id}")
        row = ref_rows[system_id]
        common = {
            "experiment": "synthetic-lia-system-conservative-junction-v1",
            "stage": "remaining-route-v3-historical-linear-screen-v1",
            "system_id": system_id,
            "main_rgi_id": inversion_cfg["main_rgi_id"],
            "synthetic_rgi_id": inversion_cfg["synthetic_rgi_id"],
            "inversion_root": {"anchor": "project", "sub": inversion_name},
            "steady_root": {"anchor": "project", "sub": steady_root_name},
            "climate_pixels": inversion_cfg["climate_pixels"],
            "climate_id": inversion_cfg["climate_id"],
            "reconstruction_climate": inversion_cfg["reconstruction_climate"],
            "start_year": 1470,
            "end_year": 2000,
            "minimum_timestep_seconds": 60.0,
            "lia_registered_length_m": float(row["lia_registered_length_m"]),
            "rgi_registered_length_m": float(row["rgi_registered_length_m"]),
            "stage2": steady_cfg["stage2"],
        }
        pair_payloads = []
        for endpoint, label in spec["endpoints"]:
            config_name = f"historical_linear_{system_id}_endp{label}_from_w719steady_v1.json"
            output_name = f"w721hist_linear_{system_id}_endp{label}_from_w719steady_v1"
            config_path = config_root / config_name
            payload = {**common, "output_root": {"anchor": "project", "sub": output_name}, "forward_delta_t_end_c": float(endpoint)}
            pair_payloads.append(payload)
            evidence_paths = {
                "steady_source_config": source_config_path,
                "steady_run_config_snapshot": root / steady_root_name / "config_snapshot.json",
                "steady_summary": root / steady_root_name / "steady_50y_summary.json",
                "steady_state": steady_state_path,
                "inversion_config_snapshot": inversion_config_path,
                "inversion_summary_formal": inversion_summary_path,
                "length_references": refs_path,
            }
            new_items.append((config_path, payload, {
                "kind": "historical-linear-1470-2000",
                "system_id": system_id,
                "start_delta_t_c": float(spec["start_delta_t_c"]),
                "end_delta_t_c": float(endpoint),
                "config": str(config_path.relative_to(root)),
                "output_root": output_name,
                "steady_pass": steady_ev,
                "frozen_evidence": {name: _hashed(root, path) for name, path in evidence_paths.items()},
                "runner_module_informational_only": HISTORICAL_RUNNER,
                "run_authorized": False,
            }))
            output_roots.append(root / output_name)
        frozen_a, frozen_b = copy.deepcopy(pair_payloads[0]), copy.deepcopy(pair_payloads[1])
        for item in (frozen_a, frozen_b):
            del item["output_root"]
            del item["forward_delta_t_end_c"]
        _require(frozen_a == frozen_b, f"historical-pair-frozen-conflict:{system_id}")

    destinations = [manifest_path] + [path for path, _, _ in new_items]
    collisions = [str(path) for path in destinations + output_roots if path.exists()]
    _require(not collisions, "refusing-to-overwrite:" + repr(collisions))
    config_root.mkdir(parents=True, exist_ok=True)
    for path, payload, record in new_items:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        _require(_read(path) == payload, f"config-roundtrip-conflict:{path}")
        record["new_config_sha256"] = _sha(path)
    _require(all(not path.exists() for path in output_roots), "builder-created-run-output")

    manifest = {
        "status": "configs-built-static-verified-not-run",
        "builder": str(Path(__file__).relative_to(root)),
        "count": len(new_items),
        "counts": {"steady_50y": 1, "historical_linear_1470_2000": 4},
        "historical_contract": {
            "start_year": 1470,
            "end_year": 2000,
            "precipitation_anomaly_mode": "off",
            "minimum_timestep_seconds": 60.0,
        },
        "steady_allowed_config_changes": ["output_root", "stage2.delta_t_c"],
        "historical_pair_allowed_changes": ["output_root", "forward_delta_t_end_c"],
        "run_authorized": False,
        "oggm_imported": False,
        "oggm_started_by_builder": False,
        "output_roots_created": False,
        "static_validation": "pass",
        "cases": [record for _, _, record in new_items],
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _require(_read(manifest_path) == manifest, "manifest-roundtrip-conflict")
    print(json.dumps({"status": manifest["status"], "count": len(new_items), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
