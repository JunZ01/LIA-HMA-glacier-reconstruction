"""Schema-fix shim for the v1 pure-JSON bisection builder.

The completed steady summaries call the tail field
``remaining_free_downstream_m``.  This version preserves the v1 scientific
contract and output names while correcting only that pre-run schema lookup.
No OGGM module is imported or executed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    build_steady_parent_bisection4_v1 as v1,
)


def _evidence(
    root: Path, system_id: int, output_name: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    snapshot_path = root / output_name / "config_snapshot.json"
    summary_path = root / output_name / "steady_50y_summary.json"
    v1._require(snapshot_path.is_file(), f"missing-snapshot:{output_name}")
    v1._require(summary_path.is_file(), f"missing-summary:{output_name}")
    snapshot = v1._read_json(snapshot_path)
    summary = v1._read_json(summary_path)
    v1._require(int(snapshot["system_id"]) == system_id, f"system-id-conflict:{output_name}")
    v1._require(summary.get("status") == "steady-50y-fail", f"status-conflict:{output_name}")
    v1._require(int(summary["final"]["year"]) == 50, f"incomplete-evidence:{output_name}")
    evidence = {
        "output_root": output_name,
        "config_snapshot": {
            "path": str(snapshot_path.relative_to(root)),
            "sha256": v1._sha256(snapshot_path),
        },
        "steady_50y_summary": {
            "path": str(summary_path.relative_to(root)),
            "sha256": v1._sha256(summary_path),
        },
        "status": summary["status"],
        "final_year": int(summary["final"]["year"]),
        "delta_t_c": float(snapshot["stage2"]["delta_t_c"]),
        "volume_change_fraction_50y": float(summary["volume_change_fraction_50y"]),
        "final_length_m": float(summary["final"]["length_m"]),
        "free_downstream_m": float(summary["final"]["remaining_free_downstream_m"]),
        "review_flags": list(summary.get("review_flags", [])),
        "runtime_error": summary.get("runtime_error"),
    }
    return snapshot, summary, evidence


def main() -> int:
    v1._evidence = _evidence
    return v1.main()


if __name__ == "__main__":
    raise SystemExit(main())
