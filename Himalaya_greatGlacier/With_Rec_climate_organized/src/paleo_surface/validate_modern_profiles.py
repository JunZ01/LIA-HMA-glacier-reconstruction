"""Compatibility shim for the modern pseudo-paleo driver.

The implementation now lives in
``ice_surface_reconstruction.drivers.validate_modern_profiles``. This wrapper
delegates to it so ``python src/paleo_surface/validate_modern_profiles.py`` and
any ``from paleo_surface.validate_modern_profiles import ...`` keep working.
Do not add logic here.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ice_surface_reconstruction.drivers.validate_modern_profiles import (  # noqa: E402,F401
    legacy_profile,
    main,
    metrics,
    run,
)

if __name__ == "__main__":
    main()
