"""Atomic-directory repair for the system-2005 masked steady checkpoint driver."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import run_system2005_masked_steady50_checkpointed_v1 as engine


def write_checkpoint(
    checkpoint_root: Path,
    *,
    year: int,
    state: np.ndarray,
    elapsed_seconds: float,
    proposed_dt_seconds: float,
    counters: dict[str, Any],
    checkpoint_metrics: dict[str, Any],
    contract_hash: str,
) -> Path:
    directory = checkpoint_root / f"year_{year:03d}"
    temporary = checkpoint_root / f"year_{year:03d}.tmp"
    if directory.exists() or temporary.exists():
        raise FileExistsError(directory)
    temporary.mkdir(parents=False)
    state_path = temporary / "state.npz"
    np.savez_compressed(
        state_path,
        thickness_m=np.asarray(state, dtype=np.float64),
        elapsed_seconds=np.asarray(elapsed_seconds, dtype=np.float64),
        proposed_dt_seconds=np.asarray(proposed_dt_seconds, dtype=np.float64),
        counters_json=np.asarray(json.dumps(counters, sort_keys=True, allow_nan=False)),
        checkpoint_metrics_json=np.asarray(
            json.dumps(checkpoint_metrics, sort_keys=True, allow_nan=False)
        ),
        contract_sha256=np.asarray(contract_hash),
        schema=np.asarray(engine.CHECKPOINT_SCHEMA),
    )
    manifest = {
        "schema": engine.CHECKPOINT_SCHEMA,
        "status": "checkpoint-complete",
        "year": year,
        "elapsed_seconds": elapsed_seconds,
        "contract_sha256": contract_hash,
        "state_file": "state.npz",
        "state_sha256": engine.sha256(state_path),
    }
    (temporary / "manifest.json").write_bytes(engine.json_bytes(manifest))
    os.replace(temporary, directory)
    manifest_path = directory / "manifest.json"
    engine.atomic_bytes(
        checkpoint_root / "latest.json",
        engine.json_bytes(
            {"status": "checkpoint-complete", "manifest": str(manifest_path.resolve())}
        ),
    )
    return manifest_path


def run(config_path: Path, resume_manifest: Path | None = None) -> dict[str, Any]:
    original = engine.write_checkpoint
    engine.write_checkpoint = write_checkpoint
    try:
        result = engine.run(config_path, resume_manifest)
    finally:
        engine.write_checkpoint = original
    result["schema"] = "lia-191-system2005-masked-steady50-checkpointed-v2"
    result["v1_superseded_before_launch"] = True
    result["v1_issue"] = "checkpoint directory was not published atomically"
    result["v2_change"] = "complete temporary checkpoint directory then atomic rename"
    output = Path(json.loads(config_path.read_text(encoding="utf-8"))["project_root"]) / json.loads(
        config_path.read_text(encoding="utf-8")
    )["output_root"]
    engine.atomic_bytes(output / "summary.json", engine.json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.resume_manifest), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
