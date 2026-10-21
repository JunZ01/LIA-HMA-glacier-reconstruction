"""Prepare immutable config for system-2005 masked temperature reproduction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
STATE = ROOT / "LIA_CURRENT_STATE.json"
ROUTE = BASE / "lia_191_current_route_v10_150/summary.json"
BASE_CONFIG = BASE / "lia_191_system2005_masked_steady50_checkpointed_v3/config_snapshot.json"
PREFLIGHT = BASE / "lia_191_system2005_masked_temperature_5y_preflight_v1/summary.json"
FORMAL = BASE / "lia_191_system2005_masked_temperature_formal50_v1/summary.json"
FORMAL_OUT = BASE / "lia_191_system2005_masked_temperature_formal50_v1"
DRIVER = HERE.parent / "run_system2005_masked_temperature_reproduction_v1.py"
OUT = BASE / "lia_191_system2005_masked_temperature_reproduction_v1_prelaunch"
RUN_OUT = BASE / "lia_191_system2005_masked_temperature_reproduction_v1"
DELTA_T_K = 0.3342811518826305


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    if OUT.exists() or RUN_OUT.exists():
        raise FileExistsError(OUT if OUT.exists() else RUN_OUT)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    route = json.loads(ROUTE.read_text(encoding="utf-8"))
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    if state["state_id"] != "2026-08-29-v10.150-system2005-masked-temperature-reproduction-array-hash-queued":
        raise RuntimeError("unexpected-controller-state")
    if not state["next_action"]["launch_authorized"] or state["counts"]["queued"] != 1:
        raise RuntimeError("reproduction-not-uniquely-queued")
    if sha256(ROUTE) != state["authoritative_route"]["sha256"].upper():
        raise RuntimeError("route-hash-mismatch")
    if formal["status"] != "masked-temperature-formal50-pass" or not all(formal["checks"].values()):
        raise RuntimeError("formal-not-pass")
    gates = route["queued_run"]["hard_gates"]

    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config.update(
        {
            "schema": "lia-191-system2005-masked-temperature-formal50-config-v1",
            "run_role": "independent-reproduction",
            "target_years": 50,
            "temperature_offset_k": DELTA_T_K,
            "output_root": str(RUN_OUT.relative_to(ROOT)),
            "formal_output_root": str(FORMAL_OUT.relative_to(ROOT)),
            "driver": str(DRIVER.relative_to(ROOT)),
            "driver_sha256": sha256(DRIVER),
            "expected_route": str(ROUTE.relative_to(ROOT)),
            "expected_route_sha256": sha256(ROUTE),
            "preflight_evidence": str(PREFLIGHT.relative_to(ROOT)),
            "preflight_evidence_sha256": sha256(PREFLIGHT),
            "formal_evidence": str(FORMAL.relative_to(ROOT)),
            "formal_evidence_sha256": sha256(FORMAL),
            "expected_annual_thickness_array_sha256": gates[
                "all_50_checkpoint_thickness_array_sha256_exact"
            ],
            "expected_final_thickness_array_sha256": gates[
                "final_thickness_array_sha256_exact"
            ],
            "expected_final_domain_mask_array_sha256": gates[
                "final_domain_mask_array_sha256_exact"
            ],
        }
    )
    fingerprint_fields = {
        key: config[key]
        for key in (
            "schema",
            "run_role",
            "system_id",
            "target_years",
            "temperature_offset_k",
            "geology_sha256",
            "static_summary_sha256",
            "climate_package_sha256",
            "observation_path_sha256",
            "domain_variable",
            "glen_a",
            "driver_sha256",
            "expected_route_sha256",
            "preflight_evidence_sha256",
            "formal_evidence_sha256",
        )
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    config["run_fingerprint_sha256"] = fingerprint
    OUT.mkdir(parents=False)
    config_path = OUT / "config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    summary = {
        "schema": "lia-191-system2005-masked-temperature-reproduction-prelaunch-v1",
        "status": "prelaunch-config-pass",
        "system_id": 2005,
        "run_role": "independent-reproduction",
        "temperature_offset_k": DELTA_T_K,
        "target_years": 50,
        "run_fingerprint_sha256": fingerprint,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "output_root": str(RUN_OUT.relative_to(ROOT)),
        "source_files_modified": False,
        "production_temperature_modified": False,
        "initial_thickness_modified": False,
        "simulation_launched": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
