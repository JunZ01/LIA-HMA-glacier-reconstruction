"""Apply the active total-length screen to the spatial-gate v2 batch."""

from __future__ import annotations

from pathlib import Path

from . import summarize_lia_191_delta_boundary_extension_v1 as _base


def main() -> None:
    _base.ROOT = Path(
        "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_delta_boundary_extension_spatial_gate_v2"
    )
    _base.main()


if __name__ == "__main__":
    main()
