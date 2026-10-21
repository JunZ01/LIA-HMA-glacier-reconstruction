"""Five-year wrapper for the passed system-2321 hybrid-graph zero-MB pilot."""

from __future__ import annotations

from pathlib import Path

from . import run_paired_cell_hybrid_graph_zero_mb_2321_v1 as implementation


implementation.TARGET_YEARS = 5.0
implementation.OUTPUT = (
    Path(implementation.PROJECT)
    / "validation_data/results/paleo_surface_validation_v0/lia_191_paired_cell_hybrid_graph_zero_mb_2321_5y_v1"
)


if __name__ == "__main__":
    implementation.main()
