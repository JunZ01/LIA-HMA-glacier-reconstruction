"""Five-year zero-MB wrapper with duration-correct versioned adjudication."""

from __future__ import annotations

import json
from pathlib import Path

from . import run_paired_cell_hybrid_graph_zero_mb_2321_v1 as implementation


OUTPUT = (
    Path(implementation.PROJECT)
    / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_zero_mb_2321_5y_v2"
)
implementation.TARGET_YEARS = 5.0
implementation.OUTPUT = OUTPUT


if __name__ == "__main__":
    implementation.main()
    path = OUTPUT / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    passed = all(summary["gates"].values())
    summary["schema"] = "lia-191-paired-cell-hybrid-graph-zero-mb-2321-5y-v2"
    summary["status"] = "zero-mb-5y-pass-steady-not-run" if passed else "zero-mb-5y-fail-mechanism-closed"
    summary["zero_mb_5y_run"] = True
    summary["next_action"] = "versioned 50-year B-window steady pilot" if passed else "close paired-cell hybrid graph mechanism"
    summary["duration_label_corrected_by_wrapper"] = True
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"formal_v2_status": summary["status"], "elapsed_years": summary["elapsed_years"], "step_count": summary["step_count"]}))
