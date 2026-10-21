"""Compatibility shim for the public-data audit driver.

The implementation now lives in
``ice_surface_reconstruction.drivers.audit_public_data``. This wrapper delegates
to it so ``python src/paleo_surface/audit_public_data.py`` keeps working.
Do not add logic here.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ice_surface_reconstruction.drivers.audit_public_data import (  # noqa: E402,F401
    audit_alps_lia,
    audit_glathida,
    audit_itmix,
    main,
)

if __name__ == "__main__":
    main()
