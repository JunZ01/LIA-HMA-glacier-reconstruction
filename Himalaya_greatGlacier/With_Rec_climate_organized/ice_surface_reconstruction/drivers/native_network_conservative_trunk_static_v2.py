"""Technical import-compatibility wrapper for trunk static preflight v1.

OGGM 1.6.3 exposes ScalarMassBalance from ``oggm.core.massbalance`` rather
than ``oggm.core.flowline``.  This wrapper provides that unchanged class at
the legacy import location before entering the v1 scientific implementation.
No geometry, threshold, gate, source, or parameter is changed.
"""

from oggm.core import flowline as _flowline
from oggm.core.massbalance import ScalarMassBalance as _ScalarMassBalance

_flowline.ScalarMassBalance = _ScalarMassBalance

from .native_network_conservative_trunk_static_v1 import main


if __name__ == "__main__":
    main()
