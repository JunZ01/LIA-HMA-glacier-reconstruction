"""Load OGGM inversion flowlines into adapter ``FlowlineInput`` objects.

Driver-layer helper. Carries the OGGM dependency (unpickles inversion flowline
lists); ``core`` never imports this.

Two OGGM gotchas are handled here so callers never trip on them (both verified
on ``RGI60-15.01141``):

* ``flowline.flows_to_indice`` and ``flowline.line.xy`` are LAZY properties that
  recompute from ``flows_to_point`` and raise ``AttributeError`` on a bare
  unpickled flowline. We read the stored ``flowline.__dict__["flows_to_point"]``
  and recover the parent native index ourselves via nearest-coordinate search,
  and we use ``line.coords`` (never ``.xy``).
* OGGM ``line.coords`` are in GRID-CELL units (col, row) on the
  ``gridded_data.nc`` pixel frame; one cell is ``map_dx`` metres and
  ``dx_meter = dx * map_dx`` (dx=2 in the pilot).

The bed carried on ``FlowlineInput.bed_elevation_m`` is the modern OGGM
``surface_h`` here (a placeholder used by the adapter only for junction audit
geometry and split provenance). Real per-scenario beds are supplied separately
via :func:`gridded_bed.load_gridded_bed`; the census/validation drivers pass the
2-D bed to the adapter for connector construction.
"""

from __future__ import annotations

import gzip
import pickle
import warnings
from pathlib import Path

import numpy as np

from .oggm_network_adapter import FlowlineInput


def _read_flowlines(inversion_flowlines_pkl: Path) -> list:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with gzip.open(inversion_flowlines_pkl, "rb") as source:
            return pickle.load(source)


def load_flowline_inputs(gdir_dir: Path) -> tuple[list[FlowlineInput], list]:
    """Return (adapter inputs, raw OGGM flowlines) for one glacier directory.

    Raises ``FileNotFoundError`` if ``inversion_flowlines.pkl`` is absent.
    """

    path = gdir_dir / "inversion_flowlines.pkl"
    if not path.exists():
        raise FileNotFoundError(f"no inversion_flowlines.pkl in {gdir_dir}")
    flowlines = _read_flowlines(path)
    coords_by_index = {
        i: np.asarray(fl.line.coords, dtype=float) for i, fl in enumerate(flowlines)
    }
    inputs: list[FlowlineInput] = []
    for i, fl in enumerate(flowlines):
        flows_to = None if fl.flows_to is None else flowlines.index(fl.flows_to)
        flows_to_indice = None
        if flows_to is not None:
            junction_point = np.asarray(
                fl.__dict__["flows_to_point"].coords[0], dtype=float
            )
            parent_coords = coords_by_index[flows_to]
            flows_to_indice = int(
                np.argmin(
                    np.hypot(
                        parent_coords[:, 0] - junction_point[0],
                        parent_coords[:, 1] - junction_point[1],
                    )
                )
            )
        surface = np.asarray(fl.surface_h, dtype=float)
        inputs.append(
            FlowlineInput(
                flowline_index=i,
                flows_to=flows_to,
                flows_to_indice=flows_to_indice,
                bed_elevation_m=surface,  # placeholder bed (audit geometry only)
                reference_surface_m=surface,
                dx_meter=float(fl.dx_meter),
                map_dx_m=float(fl.dx_meter) / 2.0,
                coords_xy=coords_by_index[i],
            )
        )
    return inputs, flowlines
