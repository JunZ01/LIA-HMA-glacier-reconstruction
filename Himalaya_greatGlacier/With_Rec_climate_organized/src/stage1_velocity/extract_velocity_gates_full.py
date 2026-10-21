"""Extract ITS_LIVE Zmed gates for all modern members in the LIA manifest."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT
    / "outputs"
    / "019f730c-ee32-74d0-8b77-fdb7832f6a06"
)
VELOCITY_WORK = RUN_ROOT / "velocity_work"
AUDIT_MODULE_PATH = VELOCITY_WORK / "audit_velocity_gate_245.py"
GDIR_ROOT = (
    RUN_ROOT
    / "stage0_bed_full_farinotti_v0"
    / "oggm_workdir"
    / "per_glacier"
)
TARGET_JSON = RUN_ROOT / "thickness_work" / "target_rgi60_records.json"


def load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "velocity_gate_audit", AUDIT_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(AUDIT_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    targets = json.loads(TARGET_JSON.read_text(encoding="utf-8"))
    expected_ids = {str(record["rgi_id"]) for record in targets}
    if len(expected_ids) != 1205:
        raise ValueError(f"Expected 1205 unique members, found {len(expected_ids)}")

    audit = load_audit_module()
    audit.run_gate_audit(
        gdir_root=GDIR_ROOT,
        work=VELOCITY_WORK,
        output_stem="velocity_gate_1205",
        expected_ids=expected_ids,
    )


if __name__ == "__main__":
    main()
