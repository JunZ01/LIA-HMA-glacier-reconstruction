"""Conditioned v5 of the conservative 2x-root candidate.

The v4 scientific construction is unchanged.  Values of trapezoid lambda
below 1e-10 are the rectangular limit and are encoded as exactly zero to avoid
catastrophic cancellation in MixedBedFlowline.section.setter.  A section-
setter roundtrip is now a mandatory static gate.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
from oggm.core.centerlines import Centerline
from oggm.core.flowline import MixedBedFlowline
from shapely.geometry import LineString, Point

from . import build_zero_mb_3660_root_coarsen2x_v2 as implementation


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_zero_mb_3660_root_coarsen2x_preflight_v5"
)
LAMBDA_RECTANGULAR_LIMIT = 1e-10


def _conditioned_build_root(root):
    surface = np.asarray(root.surface_h, dtype=float)
    bed = np.asarray(root.bed_h, dtype=float)
    section = np.asarray(root.section, dtype=float)
    width = np.asarray(root.widths_m, dtype=float)
    paired = len(surface) - 1
    a = np.arange(0, paired, 2)
    b = a + 1
    pair_width = width[a] + width[b]
    coarse_width = 0.5 * pair_width
    coarse_section = 0.5 * (section[a] + section[b])
    coarse_surface = np.divide(
        width[a] * surface[a] + width[b] * surface[b],
        pair_width,
        out=0.5 * (surface[a] + surface[b]),
        where=pair_width > 0,
    )
    coarse_bed = np.divide(
        width[a] * bed[a] + width[b] * bed[b],
        pair_width,
        out=0.5 * (bed[a] + bed[b]),
        where=pair_width > 0,
    )
    coarse_surface = np.append(coarse_surface, surface[-1])
    coarse_bed = np.append(coarse_bed, bed[-1])
    coarse_section = np.append(coarse_section, section[-1] * 0.5)
    coarse_width = np.append(coarse_width, width[-1])
    thick = coarse_surface - coarse_bed
    active = coarse_section > 0.0
    raw_lambda = np.full(len(coarse_section), np.nan)
    raw_lambda[active] = (
        2.0 * (coarse_width[active] * thick[active] - coarse_section[active])
        / thick[active] ** 2
    )
    if np.any(raw_lambda[active] < -LAMBDA_RECTANGULAR_LIMIT):
        raise RuntimeError("material-negative-trapezoid-lambda")
    lambdas = np.zeros(len(coarse_section), dtype=float)
    w0 = coarse_width.copy()
    positive = active & (raw_lambda >= LAMBDA_RECTANGULAR_LIMIT)
    rectangular = active & ~positive
    lambdas[positive] = raw_lambda[positive]
    w0[positive] = 2.0 * coarse_section[positive] / thick[positive] - coarse_width[positive]
    w0[rectangular] = coarse_section[rectangular] / thick[rectangular]
    if np.any(w0 <= 0.0):
        raise RuntimeError("nonpositive-coarse-origin-width")
    new_root = MixedBedFlowline(
        line=LineString(np.asarray(root.line.coords)[::2]),
        dx=float(root.dx) * 2.0,
        map_dx=float(root.map_dx),
        surface_h=coarse_surface,
        bed_h=coarse_bed,
        section=coarse_section,
        bed_shape=np.ones(len(coarse_section), dtype=float),
        is_trapezoid=np.ones(len(coarse_section), dtype=bool),
        lambdas=lambdas,
        widths_m=w0,
        rgi_id=root.rgi_id,
        water_level=root.water_level,
    )
    return new_root, {
        "raw_minimum_lambda": float(np.nanmin(raw_lambda[active])),
        "rectangular_limit": LAMBDA_RECTANGULAR_LIMIT,
        "rectangular_limit_node_count": int(np.count_nonzero(rectangular)),
        "positive_trapezoid_node_count": int(np.count_nonzero(positive)),
        "minimum_origin_width_m": float(np.min(new_root._w0_m)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    original_set_flows_to = Centerline.set_flows_to

    def nested_grid_set_flows_to(self, other, *args, **kwargs):
        old_index = self.__dict__.get("_lazy_flows_to_indice")
        if old_index is None or int(other.nx) != 342:
            self.__dict__.pop("_lazy_flows_to_indice", None)
            return original_set_flows_to(self, other, *args, **kwargs)
        new_index = int(math.ceil(int(old_index) / 2.0))
        self.flows_to = other
        self.flows_to_point = Point(other.line.coords[new_index])
        other.inflow_points.append(self.flows_to_point)
        other.inflows.append(self)
        self.__dict__.pop("_lazy_flows_to_indice", None)

    Centerline.set_flows_to = nested_grid_set_flows_to
    implementation._build_root = _conditioned_build_root
    implementation.OUTPUT = OUTPUT
    implementation.main()

    summary_path = OUTPUT / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    candidate_path = Path(payload["candidate_model_flowlines"])
    with gzip.open(candidate_path, "rb") as stream:
        candidate = pickle.load(stream)
    root = candidate[32]
    section0 = np.asarray(root.section, dtype=float).copy()
    volume0 = float(root.volume_m3)
    test = copy.deepcopy(root)
    test.section = section0.copy()
    section1 = np.asarray(test.section, dtype=float)
    volume_difference = float(test.volume_m3 - volume0)
    maximum_section_difference = float(np.max(np.abs(section1 - section0)))
    setter_pass = abs(volume_difference) <= 1e-5 and maximum_section_difference <= 1e-6
    payload["experiment"] = "zero-mb-3660-conservative-root-coarsen2x-static-preflight-v5"
    payload["receiver_mapping_rule"] = "old root index i maps to ceil(i/2); maximum downstream shift 50 m"
    payload["mixedbed_setter_roundtrip"] = {
        "volume_difference_m3": volume_difference,
        "maximum_absolute_section_difference_m2": maximum_section_difference,
        "pass": setter_pass,
    }
    payload["gates"]["mixedbed_setter_roundtrip"] = setter_pass
    payload["static_pass"] = bool(payload["static_pass"] and setter_pass)
    payload["next_action"] = "one-corrected-30s-5y-zeroMB" if payload["static_pass"] else "close-before-dynamics"
    payload["candidate_sha256"] = _sha256(candidate_path)
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
