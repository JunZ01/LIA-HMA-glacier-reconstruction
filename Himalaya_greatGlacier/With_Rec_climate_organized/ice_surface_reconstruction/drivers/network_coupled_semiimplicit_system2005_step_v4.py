"""Conservative-section v4 of the system-2005 real single-step gate.

The implicit matrix solves for a linearized surface.  In MixedBed geometry the
finite-volume prognostic variable is section, so this launcher applies each
linearized storage increment to section exactly and then lets OGGM's native
MixedBed section relation recover thickness.  This is the update defined by
the pre-registered network finite-volume mechanism; no residual is clipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from oggm import cfg
from oggm.cfg import G
import oggm.core.flowline as oggm_flowline

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    network_coupled_semiimplicit_system2005_step_v1 as implementation,
)
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.network_coupled_semiimplicit_system2005_step_v3 import (
    _mixedbed_face_conductance,
)


PROJECT = Path(__file__).resolve().parents[4]
implementation.OUTPUT = PROJECT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_network_coupled_semiimplicit_system2005_step_v4"
)
_UPDATE_COUNT = 0
_MAX_REQUESTED_REALIZED_SURFACE_DIFFERENCE_M = 0.0


class _ExactCoefficientProxy:
    def __init__(self, _flowlines, *, glen_a=None, fs=None, **_kwargs):
        self.glen_n = float(cfg.PARAMS["glen_n"])
        self.glen_a = float(cfg.PARAMS["glen_a"] if glen_a is None else glen_a)
        self.fs = float(cfg.PARAMS["fs"] if fs is None else fs)
        self.rho = float(cfg.PARAMS["ice_density"])
        self._fd = 2.0 / (self.glen_n + 2.0) * self.glen_a
        self.rhog = (self.rho * float(G)) ** self.glen_n


class _ConservativeThicknessArray(np.ndarray):
    """Translate a requested linearized H increment into exact section."""

    def __new__(cls, values, owner):
        obj = np.asarray(values, dtype=float).view(cls)
        obj.owner = owner
        return obj

    def __array_finalize__(self, source):
        self.owner = getattr(source, "owner", None)

    def __setitem__(self, key, requested_value):
        global _UPDATE_COUNT, _MAX_REQUESTED_REALIZED_SURFACE_DIFFERENCE_M
        if not isinstance(key, (int, np.integer)):
            return np.ndarray.__setitem__(self, key, requested_value)
        owner = self.owner
        node = int(key)
        old_h = float(np.ndarray.__getitem__(self, node))
        requested_h = float(requested_value)
        old_section = float(owner.section[node])
        old_width = float(owner.widths_m[node])
        target_section = old_section + old_width * (requested_h - old_h)
        if target_section < -1e-10:
            raise RuntimeError("linearized update requires negative section")
        target_section = max(target_section, 0.0)
        if bool(owner.is_trapezoid[node]):
            w0 = float(owner._w0_m[node])
            lam = float(owner._lambdas[node])
            if bool(owner.is_rectangular[node]) or abs(lam) < 1e-14:
                realized_h = target_section / w0
            else:
                a = 2.0 * lam
                b = 2.0 * w0
                realized_h = (np.sqrt(b * b + 4.0 * a * target_section) - b) / a
        else:
            realized_h = (0.75 * target_section * float(owner._sqrt_bed[node])) ** (2.0 / 3.0)
        np.ndarray.__setitem__(self, node, realized_h)
        _UPDATE_COUNT += 1
        _MAX_REQUESTED_REALIZED_SURFACE_DIFFERENCE_M = max(
            _MAX_REQUESTED_REALIZED_SURFACE_DIFFERENCE_M,
            abs(realized_h - requested_h),
        )


_ORIGINAL_LOAD = implementation.load_flowlines


def _load_with_conservative_arrays():
    flowlines = _ORIGINAL_LOAD()
    for flowline in flowlines:
        flowline._thick = _ConservativeThicknessArray(flowline._thick, flowline)
    return flowlines


def main() -> None:
    original_solver = oggm_flowline.SemiImplicitModel
    original_face = implementation.face_conductance
    original_load = implementation.load_flowlines
    original_deepcopy = implementation.copy.deepcopy
    oggm_flowline.SemiImplicitModel = _ExactCoefficientProxy
    implementation.face_conductance = _mixedbed_face_conductance
    implementation.load_flowlines = _load_with_conservative_arrays
    # The loaded objects are fresh in-memory objects and are never persisted.
    # Identity here preserves the ndarray owner link through v1's diagnostic
    # copy call; it does not mutate the on-disk source pickle.
    implementation.copy.deepcopy = lambda value: value
    exit_code = 0
    try:
        try:
            implementation.main()
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
    finally:
        implementation.copy.deepcopy = original_deepcopy
        implementation.load_flowlines = original_load
        implementation.face_conductance = original_face
        oggm_flowline.SemiImplicitModel = original_solver

    summary_path = implementation.OUTPUT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["section_update_mode"] = "implicit-flux-linearized-storage-to-exact-mixedbed-section"
        summary["conservative_section_update_count"] = _UPDATE_COUNT
        summary["maximum_requested_vs_realized_thickness_difference_m"] = (
            _MAX_REQUESTED_REALIZED_SURFACE_DIFFERENCE_M
        )
        summary["runtime_clip_used"] = False
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
