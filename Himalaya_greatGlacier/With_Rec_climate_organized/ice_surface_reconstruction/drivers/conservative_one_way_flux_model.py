"""FluxBasedModel variant that conservatively enforces one-way tributary flow."""

from __future__ import annotations

import numpy as np
from oggm.core.flowline import FluxBasedModel as OGGMFluxBasedModel


class ConservativeOneWayFluxBasedModel(OGGMFluxBasedModel):
    """Remove artificial volume caused by clipped reverse tributary flux."""

    last_instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_prevented_negative_tributary_m3 = 0.0
        self.cumulative_prevented_negative_tributary_m3 = 0.0
        type(self).last_instance = self

    def step(self, dt):
        used_dt = float(super().step(dt))
        prevented = 0.0
        for index, flowline in enumerate(self.fls):
            if self._tributary_indices[index][0] is None:
                continue
            terminal_flux = float(np.asarray(self.flux_stag[index], dtype=float)[:-1][-1])
            creation = max(-terminal_flux, 0.0) * used_dt
            if creation <= 0.0:
                continue
            section = np.asarray(flowline.section, dtype=float).copy()
            remaining = creation / float(flowline.dx_meter)
            for node in range(len(section) - 1, -1, -1):
                take = min(float(section[node]), remaining)
                section[node] -= take
                remaining -= take
                if remaining <= 1e-10:
                    break
            if remaining > 1e-6:
                raise RuntimeError(
                    f"junction-correction-insufficient-section:{index}:{remaining}"
                )
            flowline.section = section
            prevented += creation
        self.last_prevented_negative_tributary_m3 = prevented
        self.cumulative_prevented_negative_tributary_m3 += prevented
        return used_dt


__all__ = ["ConservativeOneWayFluxBasedModel"]
