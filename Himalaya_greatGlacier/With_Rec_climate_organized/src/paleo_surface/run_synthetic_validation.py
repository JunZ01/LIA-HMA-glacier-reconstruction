"""Compatibility shim for the synthetic validation driver.

The implementation now lives in
``ice_surface_reconstruction.drivers.run_synthetic_validation``. This wrapper
delegates to it so ``python src/paleo_surface/run_synthetic_validation.py``
keeps working. Do not add logic here.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ice_surface_reconstruction.drivers.run_synthetic_validation import (  # noqa: E402,F401
    main,
    profile_error,
    run,
)

if __name__ == "__main__":
    main()
