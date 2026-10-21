"""JSON scalar-schema shim for the v1 branch-ensemble static builder."""

from __future__ import annotations

import numpy as np

from . import build_negative_flux2342_decoupled_branch_elevation_bands_v1 as impl


_ORIGINAL_DUMPS = impl.json.dumps


def dumps_numpy(payload, *args, **kwargs):
    kwargs["default"] = lambda value: value.item() if isinstance(value, np.generic) else str(value)
    return _ORIGINAL_DUMPS(payload, *args, **kwargs)


def main() -> None:
    impl.OUTPUT = impl.PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux2342_decoupled_branch_elevation_bands_v2"
    impl.json.dumps = dumps_numpy
    impl.main()


if __name__ == "__main__":
    main()
