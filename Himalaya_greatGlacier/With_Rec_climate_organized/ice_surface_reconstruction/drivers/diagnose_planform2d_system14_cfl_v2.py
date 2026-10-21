"""Schema-safe wrapper superseding the v1 CFL diagnostic write failure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import diagnose_planform2d_system14_cfl_v1 as implementation


OUTPUT = implementation.PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_invalid_outline_planform2d_source_native_cfl_diagnostic_v2/system_14"


def main() -> None:
    original_dumps = implementation.json.dumps

    def safe_dumps(obj, *args, **kwargs):
        kwargs.setdefault(
            "default",
            lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
        return original_dumps(obj, *args, **kwargs)

    implementation.OUTPUT = OUTPUT
    implementation.json.dumps = safe_dumps
    implementation.main()
    summary_path = OUTPUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema"] = "lia-191-system14-source-native-planform2d-cfl-diagnostic-v2"
    summary["supersedes"] = "v1-json-serialization-failure-before-summary"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
