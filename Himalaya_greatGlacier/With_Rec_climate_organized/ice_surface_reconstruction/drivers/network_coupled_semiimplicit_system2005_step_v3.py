"""MixedBed-compatible v3 launcher for system 2005's real one-step gate.

Trapezoid--trapezoid faces retain the exact stock SemiImplicit coefficient.
Faces touching a parabolic control volume use the algebraically equivalent
frozen FluxBased ``velocity * section`` coefficient, avoiding the NaN ``w0``
sentinel that marks a non-trapezoidal node.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from oggm import cfg
from oggm.cfg import G
import oggm.core.flowline as oggm_flowline

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    network_coupled_semiimplicit_system2005_step_v1 as implementation,
)


PROJECT = Path(__file__).resolve().parents[4]
implementation.OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_network_coupled_semiimplicit_system2005_step_v3"
)


class _ExactCoefficientProxy:
    def __init__(self, _flowlines, *, glen_a=None, fs=None, **_kwargs):
        self.glen_n = float(cfg.PARAMS["glen_n"])
        self.glen_a = float(cfg.PARAMS["glen_a"] if glen_a is None else glen_a)
        self.fs = float(cfg.PARAMS["fs"] if fs is None else fs)
        self.rho = float(cfg.PARAMS["ice_density"])
        self._fd = 2.0 / (self.glen_n + 2.0) * self.glen_a
        self.rhog = (self.rho * float(G)) ** self.glen_n


def _mixedbed_face_conductance(model, upstream_fl, upstream_index, downstream_fl, downstream_index):
    hu = float(upstream_fl.thick[upstream_index])
    hv = float(downstream_fl.thick[downstream_index])
    wu = float(upstream_fl.widths_m[upstream_index])
    wv = float(downstream_fl.widths_m[downstream_index])
    w0u = float(upstream_fl._w0_m[upstream_index])
    w0v = float(downstream_fl._w0_m[downstream_index])
    su = float(upstream_fl.surface_h[upstream_index])
    sv = float(downstream_fl.surface_h[downstream_index])
    distance = 0.5 * (float(upstream_fl.dx_meter) + float(downstream_fl.dx_meter))
    hstag = 0.5 * (hu + hv)
    slope = (sv - su) / distance
    n = float(model.glen_n)
    if np.isfinite(w0u) and np.isfinite(w0v):
        width_stag = 0.5 * (wu + wv)
        w0_stag = 0.5 * (w0u + w0v)
        diffusivity = (
            (float(model._fd) * hstag ** (n + 2.0) + float(model.fs) * hstag**n)
            * float(model.rhog)
            * (w0_stag + width_stag)
            / 2.0
            * abs(slope) ** (n - 1.0)
        )
    else:
        section_stag = 0.5 * (
            float(upstream_fl.section[upstream_index])
            + float(downstream_fl.section[downstream_index])
        )
        diffusivity = (
            (float(model._fd) * hstag ** (n + 1.0) + float(model.fs) * hstag ** (n - 1.0))
            * float(model.rhog)
            * section_stag
            * abs(slope) ** (n - 1.0)
        )
    return float(diffusivity / distance)


def main() -> None:
    original_solver = oggm_flowline.SemiImplicitModel
    original_face = implementation.face_conductance
    oggm_flowline.SemiImplicitModel = _ExactCoefficientProxy
    implementation.face_conductance = _mixedbed_face_conductance
    try:
        implementation.main()
    finally:
        implementation.face_conductance = original_face
        oggm_flowline.SemiImplicitModel = original_solver


if __name__ == "__main__":
    main()

