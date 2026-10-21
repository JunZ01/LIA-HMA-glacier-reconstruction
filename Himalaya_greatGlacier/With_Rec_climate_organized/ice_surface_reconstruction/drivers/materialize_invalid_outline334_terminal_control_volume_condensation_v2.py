"""Connector-gate correction wrapper for system-334 materialization v1.

V1 used an absolute 300 m connector gate even though the frozen mechanism
allows a 250 m along-line terminal cut and source OGGM connectors are already
up to 194.3 m.  V2 changes no candidate geometry.  It applies the implied
450 m absolute cap and, after materialization, additionally enforces for every
branch the exact triangle bound

    new_connector <= source_connector + removed_along_line_distance.

The v1 entry point and output schema are retained, then corrected and
version-labelled atomically before any inversion is allowed.
"""

from __future__ import annotations

import csv
import json

from . import materialize_invalid_outline334_terminal_control_volume_condensation_v1 as v1


def main() -> None:
    v1.MAXIMUM_CONNECTOR_DISTANCE_M = 450.0
    v1.main()

    summary_path = v1.OUTPUT / "geometry_summary.json"
    rows_path = v1.OUTPUT / "branch_condensation_materialization.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with rows_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    triangle_pass = all(
        float(row["connector_distance_m"])
        <= float(row["old_tail_to_receiver_distance_m"])
        + float(row["cut_distance_m"])
        + 1.0e-6
        for row in rows
    )
    checks = dict(summary["checks"])
    absolute_pass = bool(checks.pop("maximum_connector_within_300m"))
    checks["maximum_connector_within_450m"] = absolute_pass
    checks["all_connectors_within_branch_triangle_bound"] = triangle_pass
    passed = all(checks.values())
    summary.update(
        {
            "status": "geometry-pass" if passed else "geometry-fail",
            "experiment": (
                "invalid-outline334-terminal-control-volume-condensation-v2"
            ),
            "connector_gate_correction": {
                "v1_absolute_limit_m": 300.0,
                "v2_absolute_limit_m": 450.0,
                "reason": (
                    "source connectors are <=194.3m and the frozen cut gate is "
                    "<=250m; v2 also enforces the exact per-branch triangle bound"
                ),
                "scientific_geometry_changed_from_v1": False,
            },
            "checks": checks,
            "review_flags": []
            if passed
            else [key for key, value in checks.items() if not value],
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
