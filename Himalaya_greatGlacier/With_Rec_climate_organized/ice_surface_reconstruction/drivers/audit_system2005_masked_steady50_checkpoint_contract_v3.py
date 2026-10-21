"""Correct the v2 auditor's negative-condition aggregation before authorization."""

from __future__ import annotations

import contextlib
import io
import json

from . import audit_system2005_masked_steady50_checkpoint_contract_v1 as audit
from . import run_system2005_masked_steady50_checkpointed_v3 as driver


audit.driver = driver
audit.OUT = audit.BASE / "lia_191_system2005_masked_steady50_checkpoint_contract_v3"


def main() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        audit.main()
    summary_path = audit.OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = summary["checks"]
    production_modified = checks.pop("production_files_modified")
    dynamics_run = checks.pop("dynamics_run")
    checks["production_files_unchanged"] = production_modified is False
    checks["dynamics_not_run"] = dynamics_run is False
    summary.update(
        {
            "schema": "lia-191-system2005-masked-steady50-checkpoint-contract-v3",
            "status": "checkpoint-contract-pass" if all(checks.values()) else "checkpoint-contract-fail",
            "supersedes": "v2 aggregation-label error only",
            "v2_issue": "negative facts were incorrectly aggregated as pass booleans",
            "v3_change": "express production immutability and no-dynamics facts as positive checks",
        }
    )
    summary_path.write_bytes(driver.engine.json_bytes(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
