"""Serialization-compatible v2 wrapper for the unchanged system-5598 audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import audit_invalid_outline5598_global_topological_cell_assignment_v1 as v1


def _json_default(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"not-json-serializable:{type(value).__name__}")


def main() -> None:
    original_dumps = json.dumps

    def compatible_dumps(obj, *args, **kwargs):
        kwargs.setdefault("default", _json_default)
        return original_dumps(obj, *args, **kwargs)

    v1.OUTPUT = v1.PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline5598_global_topological_cell_assignment_v2"
    v1.json.dumps = compatible_dumps
    v1.main()
    failure = v1.PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline5598_global_topological_cell_assignment_v1"
    marker = failure / "TECHNICAL_FAILURE.json"
    if failure.exists() and not marker.exists():
        marker.write_text(
            original_dumps(
                {
                    "status": "superseded-technical-failure",
                    "reason": "numpy-bool-json-serialization-after-computation",
                    "superseded_by": str(v1.OUTPUT.relative_to(v1.PROJECT)),
                    "scientific_contract_changed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
