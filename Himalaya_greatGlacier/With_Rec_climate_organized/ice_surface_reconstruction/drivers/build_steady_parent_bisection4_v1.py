"""Build four frozen steady-temperature bisection configs without running OGGM.

Only ``output_root`` and ``stage2.delta_t_c`` may differ from the completed
parent evidence.  The builder verifies and hashes both endpoint snapshots and
their 50-year summaries before exclusively creating new JSON files.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


CONFIG_DIR = (
    "Himalaya_greatGlacier/With_Rec_climate_organized/"
    "ice_surface_reconstruction/config/"
    "synthetic_lia_system_standard_inversion_v1"
)
MANIFEST_NAME = "w719steady_parent_bisection4_manifest_v1.json"
RUNNER_MODULE = (
    "Himalaya_greatGlacier.With_Rec_climate_organized."
    "ice_surface_reconstruction.drivers.synthetic_lia_system_steady_50y_v1"
)
PYTHON = Path(r"D:\Anaconda\envs\oggm_env2\python.exe")

# Endpoint order is warm, cold.  The first three targets are exact midpoints.
# System 4607 is the one authorized finite coldward refinement from delta=0.
CASES: dict[int, dict[str, Any]] = {
    1645: {
        "target": 0.0,
        "rule": "bisection-midpoint",
        "warm": "w704coldsteady_parent_m0p5_1645_v1",
        "cold": "w704steady_parent_refine5_1645_v1",
        "warm_delta": 0.25,
        "cold_delta": -0.25,
        "warm_dv": -0.16447427148225308,
        "cold_dv": 0.18431658996937725,
    },
    3654: {
        "target": 0.0,
        "rule": "bisection-midpoint",
        "warm": "w704coldsteady_parent_m0p5_3654_v1",
        "cold": "w704steady_parent_refine5_3654_v1",
        "warm_delta": 0.125,
        "cold_delta": -0.375,
        "warm_dv": -0.03711233934041692,
        "cold_dv": 0.09480353614695407,
    },
    4464: {
        "target": -0.5,
        "rule": "bisection-midpoint",
        "warm": "w704coldsteady_parent_m0p5_4464_v1",
        "cold": "w704steady_parent_refine4464_extra_cold_v1",
        "warm_delta": -0.25,
        "cold_delta": -0.75,
        "warm_dv": -0.05878203413715839,
        "cold_dv": 0.05610664595140614,
    },
    4607: {
        "target": -0.125,
        "rule": "finite-coldward-refinement",
        "warm": "w704steady_parent_zero_pass9_4607_v1",
        "cold": "w704steady_parent_refine5_4607_v1",
        "warm_delta": 0.5,
        "cold_delta": 0.0,
        "warm_dv": -0.20573917052859997,
        "cold_dv": -0.028160151979013295,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _strip_allowed(config: dict[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(config)
    del frozen["output_root"]
    del frozen["stage2"]["delta_t_c"]
    return frozen


def _evidence(root: Path, system_id: int, output_name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    snapshot_path = root / output_name / "config_snapshot.json"
    summary_path = root / output_name / "steady_50y_summary.json"
    _require(snapshot_path.is_file(), f"missing-snapshot:{output_name}")
    _require(summary_path.is_file(), f"missing-summary:{output_name}")
    snapshot = _read_json(snapshot_path)
    summary = _read_json(summary_path)
    _require(int(snapshot["system_id"]) == system_id, f"system-id-conflict:{output_name}")
    _require(summary.get("status") == "steady-50y-fail", f"status-conflict:{output_name}")
    _require(int(summary["final"]["year"]) == 50, f"incomplete-evidence:{output_name}")
    evidence = {
        "output_root": output_name,
        "config_snapshot": {
            "path": str(snapshot_path.relative_to(root)),
            "sha256": _sha256(snapshot_path),
        },
        "steady_50y_summary": {
            "path": str(summary_path.relative_to(root)),
            "sha256": _sha256(summary_path),
        },
        "status": summary["status"],
        "final_year": int(summary["final"]["year"]),
        "delta_t_c": float(snapshot["stage2"]["delta_t_c"]),
        "volume_change_fraction_50y": float(summary["volume_change_fraction_50y"]),
        "final_length_m": float(summary["final"]["length_m"]),
        "free_downstream_m": float(summary["final"]["free_downstream_m"]),
        "review_flags": list(summary.get("review_flags", [])),
        "runtime_error": summary.get("runtime_error"),
    }
    return snapshot, summary, evidence


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / CONFIG_DIR
    manifest_path = root / MANIFEST_NAME
    _require(PYTHON.is_file(), f"python-missing:{PYTHON}")

    destinations = [manifest_path]
    output_roots: list[Path] = []
    for system_id, spec in CASES.items():
        tag = str(spec["target"]).replace("-", "m").replace(".", "p")
        destinations.append(config_root / f"steady_parent_bisection4_{system_id}_dt{tag}_v1.json")
        output_roots.append(root / f"w719steady_parent_bisection4_{system_id}_dt{tag}_v1")
    collisions = [str(path) for path in destinations + output_roots if path.exists()]
    _require(not collisions, "refusing-to-overwrite:" + repr(collisions))

    prepared: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for system_id, spec in CASES.items():
        warm_config, _, warm_evidence = _evidence(root, system_id, str(spec["warm"]))
        cold_config, _, cold_evidence = _evidence(root, system_id, str(spec["cold"]))

        _require(_strip_allowed(warm_config) == _strip_allowed(cold_config), f"frozen-endpoint-conflict:{system_id}")
        _require(warm_evidence["delta_t_c"] == float(spec["warm_delta"]), f"warm-delta-conflict:{system_id}")
        _require(cold_evidence["delta_t_c"] == float(spec["cold_delta"]), f"cold-delta-conflict:{system_id}")
        _require(abs(warm_evidence["volume_change_fraction_50y"] - float(spec["warm_dv"])) < 1e-15, f"warm-dv-conflict:{system_id}")
        _require(abs(cold_evidence["volume_change_fraction_50y"] - float(spec["cold_dv"])) < 1e-15, f"cold-dv-conflict:{system_id}")

        if spec["rule"] == "bisection-midpoint":
            midpoint = 0.5 * (float(spec["warm_delta"]) + float(spec["cold_delta"]))
            _require(abs(midpoint - float(spec["target"])) < 1e-15, f"midpoint-conflict:{system_id}")
            _require(float(spec["warm_dv"]) < 0.0 < float(spec["cold_dv"]), f"unbracketed-dv:{system_id}")
        else:
            _require(system_id == 4607, "unexpected-non-bisection-case")
            _require(float(spec["warm_delta"]) == 0.5 and float(spec["cold_delta"]) == 0.0, "4607-parent-delta-conflict")
            _require(float(spec["warm_dv"]) < 0.0 and float(spec["cold_dv"]) < 0.0, "4607-dv-sign-conflict")
            _require(abs(float(spec["cold_dv"]) + 0.028160151979013295) < 1e-15, "4607-zero-dv-conflict")
            _require(float(spec["target"]) == -0.125, "4607-target-conflict")

        target = float(spec["target"])
        tag = str(target).replace("-", "m").replace(".", "p")
        output_name = f"w719steady_parent_bisection4_{system_id}_dt{tag}_v1"
        config_path = config_root / f"steady_parent_bisection4_{system_id}_dt{tag}_v1.json"
        payload = copy.deepcopy(warm_config)
        payload["stage2"]["delta_t_c"] = target
        payload["output_root"] = {"anchor": "project", "sub": output_name}
        _require(_strip_allowed(payload) == _strip_allowed(warm_config), f"frozen-new-config-conflict:{system_id}")

        record = {
            "system_id": system_id,
            "decision_rule": spec["rule"],
            "target_delta_t_c": target,
            "config": str(config_path.relative_to(root)),
            "output_root": output_name,
            "warm_parent_evidence": warm_evidence,
            "cold_parent_evidence": cold_evidence,
            "run_authorized": False,
        }
        prepared.append((config_path, payload, record))

    config_root.mkdir(parents=True, exist_ok=True)
    for config_path, payload, _ in prepared:
        with config_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    for config_path, payload, record in prepared:
        _require(_read_json(config_path) == payload, f"config-roundtrip-conflict:{config_path}")
        record["new_config_sha256"] = _sha256(config_path)
    _require(all(not path.exists() for path in output_roots), "builder-created-run-output")

    manifest = {
        "status": "configs-built-static-verified-not-run",
        "stage": "steady-parent-bisection-four-v1",
        "count": 4,
        "allowed_config_changes": ["output_root", "stage2.delta_t_c"],
        "runner_module_informational_only": RUNNER_MODULE,
        "python": str(PYTHON),
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
    _require(_read_json(manifest_path) == manifest, "manifest-roundtrip-conflict")
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path), "count": 4}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
