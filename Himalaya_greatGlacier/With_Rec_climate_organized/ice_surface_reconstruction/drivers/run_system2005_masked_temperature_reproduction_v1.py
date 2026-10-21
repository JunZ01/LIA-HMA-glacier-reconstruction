"""Independent reproduction of the passed system-2005 masked temperature run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import run_system2005_masked_temperature_formal50_v1 as formal_driver
from . import run_system2005_masked_steady50_checkpointed_v1 as engine


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest().upper()


def run(config_path: Path, resume_manifest: Path | None = None) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("run_role") != "independent-reproduction":
        raise RuntimeError("reproduction-role-missing")
    result = formal_driver.run(config_path, resume_manifest)
    output = Path(config["project_root"]) / config["output_root"]
    expected_annual = config["expected_annual_thickness_array_sha256"]
    actual_annual = {}
    for year in range(1, 51):
        with np.load(output / f"checkpoints/year_{year:03d}/state.npz", allow_pickle=False) as archive:
            actual_annual[str(year)] = array_sha256(np.asarray(archive["thickness_m"], dtype=np.float64))
    with np.load(output / "final_state_50y.npz", allow_pickle=False) as archive:
        final_thickness_hash = array_sha256(np.asarray(archive["thickness_m"], dtype=np.float64))
        final_domain_hash = array_sha256(np.asarray(archive["domain_mask"], dtype=bool))

    checks = dict(result["checks"])
    checks.update(
        {
            "formal_result_hash_exact": engine.sha256(
                Path(config["project_root"]) / config["formal_evidence"]
            ) == config["formal_evidence_sha256"].upper(),
            "all_50_checkpoint_thickness_arrays_exact": actual_annual == expected_annual,
            "final_thickness_array_exact": final_thickness_hash
            == config["expected_final_thickness_array_sha256"].upper(),
            "final_domain_mask_array_exact": final_domain_hash
            == config["expected_final_domain_mask_array_sha256"].upper(),
            "independent_output_root": config["output_root"]
            != config["formal_output_root"],
            "run_role_exact": config["run_role"] == "independent-reproduction",
        }
    )
    result.update(
        {
            "schema": "lia-191-system2005-masked-temperature-reproduction-v1",
            "status": "masked-temperature-reproduction-pass"
            if all(checks.values())
            else "masked-temperature-reproduction-fail-stop",
            "checks": checks,
            "run_role": "independent-reproduction",
            "formal_evidence": config["formal_evidence"],
            "formal_evidence_sha256": config["formal_evidence_sha256"],
            "actual_annual_thickness_array_sha256": actual_annual,
            "final_thickness_array_sha256": final_thickness_hash,
            "final_domain_mask_array_sha256": final_domain_hash,
            "accepted_count_effect": 0,
        }
    )
    engine.atomic_bytes(output / "summary.json", engine.json_bytes(result))
    engine.atomic_bytes(
        output / "progress.json",
        engine.json_bytes({"status": result["status"], "completed_years": result["completed_years"]}),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.resume_manifest), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
