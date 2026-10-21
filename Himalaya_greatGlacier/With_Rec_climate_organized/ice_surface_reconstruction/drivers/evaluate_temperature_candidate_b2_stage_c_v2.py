from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_temperature_candidate_b2_stage_c_v1 as v1
import run_temperature_candidate_b2_stage_c_real_proxy_v1 as stage_c_v1
import run_temperature_candidate_b2_stage_c_real_proxy_v3 as stage_c_v3
import temperature_candidate_b2_stage_c_core_v1 as core


DRIVER = Path(__file__).resolve()
V1_EVALUATOR = Path(v1.__file__).resolve()
DEFAULT_INPUT = stage_c_v3.DEFAULT_OUT / "candidate_b2_stage_c_real_proxy_output_v1.npz"
DEFAULT_OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_evaluation_v2"


def validate_real_chain(input_path: Path) -> tuple[dict[str, object], Path]:
    summary_path = input_path.parent / "summary.json"
    if not summary_path.exists():
        raise PermissionError("real Stage C summary is absent")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    authorization = summary.get("authorization")
    if not isinstance(authorization, dict):
        raise PermissionError("real Stage C authorization chain is absent")
    expected = {
        "schema": "candidate-b2-stage-c-authorization-v1",
        "authorized": True,
        "authorization_text": stage_c_v1.AUTHORIZATION_TEXT,
        "static_package_sha256": core.EXPECTED_STATIC_PACKAGE_SHA256,
        "driver_sha256": core.sha256(Path(stage_c_v3.__file__).resolve()),
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise PermissionError(f"real Stage C authorization mismatch:{key}")
    if summary.get("active_driver_version") != "v3":
        raise PermissionError("real Stage C v3 driver evidence is absent")
    if summary.get("protected_sources_unchanged") is not True:
        raise PermissionError("real Stage C source-integrity evidence is absent")
    if summary.get("output_sha256") != core.sha256(input_path):
        raise RuntimeError("real Stage C output hash mismatch")
    audit = summary.get("proxy_read_audit", {})
    for network in ("proxy6", "proxy9"):
        row = audit.get(network, {})
        if int(row.get("allowed_proxy_value_cells_read", 0)) <= 0:
            raise PermissionError(f"real proxy read evidence absent:{network}")
        if int(row.get("sealed_proxy_value_cells_read", -1)) != 0:
            raise PermissionError(f"sealed proxy read evidence invalid:{network}")
    return summary, summary_path


def evaluation_sources(input_path: Path, summary_path: Path) -> dict[str, Path]:
    return {
        "facts": v1.ROOT / "LIA_PROJECT_FACTS.md",
        "stage_c_output": input_path,
        "stage_c_summary": summary_path,
        "early_stations": v1.EARLY_STATIONS,
        "modern_stations": v1.MODERN_STATIONS,
        "monthly_panel": v1.MONTHLY_PANEL,
        "monthly_climatology": v1.MONTHLY_CLIM,
        "evidence_matrix": v1.EVIDENCE,
        "allowed_long_curves": v1.LONG_CURVES,
        "implementation_v1": V1_EVALUATOR,
        "active_evaluator_v2": DRIVER,
    }


def main(input_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    stage_c_summary, summary_path = validate_real_chain(input_path)
    sources = evaluation_sources(input_path, summary_path)
    before = {name: core.sha256(path) for name, path in sources.items()}
    v1.main(input_path, output)
    after = {name: core.sha256(path) for name, path in sources.items()}
    if before != after:
        raise RuntimeError("evaluation-source-changed")

    result_path = output / "summary.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "experiment": output.name,
            "active_evaluator_version": "v2",
            "real_stage_c_chain_validated": True,
            "real_stage_c_driver_version": stage_c_summary["active_driver_version"],
            "evaluation_sources_unchanged": True,
        }
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest_path = output / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["full_evaluation_source_set"] = {
        name: {"path": str(path), "sha256": before[name]}
        for name, path in sources.items()
    }
    manifest["evaluation_sources_unchanged"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.input, args.output)
