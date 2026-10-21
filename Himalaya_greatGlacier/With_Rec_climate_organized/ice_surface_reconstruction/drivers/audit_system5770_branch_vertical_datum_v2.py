"""Serialization-only v2 launcher for the system-5770 LP/QP audit."""

from __future__ import annotations

import json as std_json
from pathlib import Path

import numpy as np

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    audit_system5770_branch_vertical_datum_v1 as implementation,
)


PROJECT = Path(__file__).resolve().parents[4]
implementation.OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_branch_rigid_vertical_datum_5770_v2"
)
_ORIGINAL_DUMPS = std_json.dumps


def _numpy_compatible_dumps(value, *args, **kwargs):
    kwargs.setdefault(
        "default",
        lambda item: item.item() if isinstance(item, np.generic) else str(item),
    )
    return _ORIGINAL_DUMPS(value, *args, **kwargs)


def main() -> None:
    implementation.json.dumps = _numpy_compatible_dumps
    try:
        implementation.main()
    finally:
        implementation.json.dumps = _ORIGINAL_DUMPS


if __name__ == "__main__":
    main()
