"""Compatibility shim for the validation-summary driver.

The implementation now lives in
``ice_surface_reconstruction.drivers.build_validation_summary``. This wrapper
delegates to it so ``python src/paleo_surface/build_validation_summary.py``
keeps working. Do not add logic here.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ice_surface_reconstruction.drivers.build_validation_summary import (  # noqa: E402,F401
    clean,
    main,
    run,
)

if __name__ == "__main__":
    main()
