"""Schema-compatible v2 launcher for the system-2005 real single-step gate.

V1 stopped before matrix assembly because OGGM's stock SemiImplicitModel
constructor rejects a MixedBedFlowline.  The stock object was used only as a
container for four public coefficients.  V2 supplies those exact OGGM-source
formulae without constructing the stock solver; all graph physics and gates
remain in the v1 implementation.
"""

from __future__ import annotations

from pathlib import Path

from oggm import cfg
from oggm.cfg import G
import oggm.core.flowline as oggm_flowline

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    network_coupled_semiimplicit_system2005_step_v1 as implementation,
)


PROJECT = Path(__file__).resolve().parents[4]
implementation.OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_network_coupled_semiimplicit_system2005_step_v2"
)


class _ExactCoefficientProxy:
    """Only the four coefficients read by ``face_conductance``."""

    def __init__(self, _flowlines, *, glen_a=None, fs=None, **_kwargs):
        self.glen_n = float(cfg.PARAMS["glen_n"])
        self.glen_a = float(cfg.PARAMS["glen_a"] if glen_a is None else glen_a)
        self.fs = float(cfg.PARAMS["fs"] if fs is None else fs)
        self.rho = float(cfg.PARAMS["ice_density"])
        self._fd = 2.0 / (self.glen_n + 2.0) * self.glen_a
        self.rhog = (self.rho * float(G)) ** self.glen_n


def main() -> None:
    original = oggm_flowline.SemiImplicitModel
    oggm_flowline.SemiImplicitModel = _ExactCoefficientProxy
    try:
        implementation.main()
    finally:
        oggm_flowline.SemiImplicitModel = original


if __name__ == "__main__":
    main()
