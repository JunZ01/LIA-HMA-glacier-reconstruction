from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_temperature_candidate_b2_stage_c_real_proxy_v1 as v1
import run_temperature_candidate_b2_stage_c_real_proxy_v2 as v2
import temperature_candidate_b2_stage_c_core_v1 as core


DRIVER = Path(__file__).resolve()
V1_DRIVER = Path(v1.__file__).resolve()
V2_DRIVER = Path(v2.__file__).resolve()
DEFAULT_OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_real_proxy_v3"


def protected_sources(receipt_path: Path) -> dict[str, Path]:
    sources = {
        "facts": v1.ROOT / "LIA_PROJECT_FACTS.md",
        "execution_contract": v1.ROOT / "LIA_EXECUTION_CONTRACT.md",
        "static_package": v1.STATIC_PACKAGE,
        "base_contract": v1.BASE_CONTRACT,
        "centering_contract": v1.CENTERING_CONTRACT,
        "authorization_receipt": receipt_path,
        "implementation_v1": V1_DRIVER,
        "hash_guard_wrapper_v2": V2_DRIVER,
        "active_driver_v3": DRIVER,
        "core": Path(core.__file__).resolve(),
    }
    for network in v1.NETWORKS:
        sources[f"proxy{network}_archive"] = (
            v1.replay.prior_tools.ARCHIVE / f"python_data/proxy_data_{network}.mat"
        )
    return sources


def main(receipt_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    v2.DRIVER = DRIVER
    v1.DRIVER = DRIVER
    v1.validate_authorization(receipt_path)
    sources = protected_sources(receipt_path)
    before = {name: core.sha256(path) for name, path in sources.items()}
    v2.main(receipt_path, output)
    after = {name: core.sha256(path) for name, path in sources.items()}
    if before != after:
        raise RuntimeError("protected-source-changed-during-stage-c")

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "experiment": output.name,
            "active_driver_version": "v3",
            "protected_sources_unchanged": True,
            "implementation_dependencies_hashed": [
                str(V1_DRIVER), str(V2_DRIVER), str(DRIVER), str(Path(core.__file__).resolve())
            ],
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest_path = output / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["full_protected_source_set"] = {
        name: {"path": str(path), "sha256": before[name]}
        for name, path in sources.items()
    }
    manifest["protected_sources_unchanged"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.authorization_receipt, args.output)
