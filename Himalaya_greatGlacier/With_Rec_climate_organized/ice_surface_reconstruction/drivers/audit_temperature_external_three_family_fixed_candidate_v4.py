"""Run the fixed audit with recovered-path and source-state compatibility."""

from pathlib import Path

import audit_temperature_external_three_family_fixed_candidate_v3 as compatibility


audit = compatibility.audit


def file_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": audit.record_base.sha256(path),
    }


audit.record_base.file_state = file_state
audit.DRIVER = Path(__file__).resolve()
audit.OUTPUT = audit.RESULTS / "climate_reconstruction_temperature_external_three_family_fixed_candidate_audit_v4"


if __name__ == "__main__":
    raise SystemExit(audit.main())
