"""Build auditable OGGM flowlines from reconstructed LIA cross sections.

The native ``init_present_time_glacier`` replaces every trapezoid lambda with
one global configuration value.  This module intentionally bypasses that
initialiser and constructs ``MixedBedFlowline`` objects directly, while
retaining OGGM's downstream-line and network-ordering contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrapezoidGeometry:
    lambdas: np.ndarray
    bottom_width_m: np.ndarray
    area_shape_factor: np.ndarray
    positive_ice: np.ndarray


@dataclass(frozen=True)
class FixedBedTrapezoidGeometry:
    """Closest OGGM trapezoid with fixed thickness and section area."""

    lambdas: np.ndarray
    bottom_width_m: np.ndarray
    effective_surface_width_m: np.ndarray
    area_shape_factor: np.ndarray
    width_error_fraction: np.ndarray
    regime_code: np.ndarray
    positive_ice: np.ndarray


@dataclass(frozen=True)
class FlowlineNodeFields:
    """LIA geometry assigned to one OGGM inversion flowline."""

    surface_h_m: np.ndarray
    bed_h_m: np.ndarray
    section_m2: np.ndarray
    surface_width_m: np.ndarray


@dataclass(frozen=True)
class ThicknessProjection:
    effective_thickness_m: np.ndarray
    adjusted: np.ndarray
    minimum_feasible_thickness_m: np.ndarray
    maximum_feasible_thickness_m: np.ndarray


def project_thickness_to_trapezoid_domain(
    observed_thickness_m,
    section_m2,
    surface_width_m,
    *,
    minimum_bottom_width_m: float = 10.0,
    zero_tolerance: float = 1e-8,
) -> ThicknessProjection:
    """Project representative thickness into OGGM's trapezoid domain.

    Section and surface width remain fixed.  The closest representative
    thickness is selected subject to a non-negative lambda and
    ``w0 >= minimum_bottom_width_m``.  This is an explicit model-representation
    adjustment, not a repair of the source reconstruction.
    """

    observed = np.asarray(observed_thickness_m, dtype=float)
    section = np.asarray(section_m2, dtype=float)
    width = np.asarray(surface_width_m, dtype=float)
    if not (observed.shape == section.shape == width.shape):
        raise ValueError("thickness-projection-arrays-must-have-identical-shape")
    if minimum_bottom_width_m <= 0:
        raise ValueError("minimum-bottom-width-must-be-positive")
    positive = section > zero_tolerance
    if np.any(width[positive] <= minimum_bottom_width_m):
        bad = np.flatnonzero(positive & (width <= minimum_bottom_width_m))
        raise ValueError(
            "surface-width-not-above-minimum-bottom-width-at-nodes:"
            + ",".join(map(str, bad.tolist()))
        )
    minimum_h = np.zeros_like(section)
    maximum_h = np.zeros_like(section)
    minimum_h[positive] = section[positive] / width[positive]
    maximum_h[positive] = (
        2.0
        * section[positive]
        / (width[positive] + float(minimum_bottom_width_m))
    )
    effective = np.zeros_like(section)
    effective[positive] = np.clip(
        observed[positive], minimum_h[positive], maximum_h[positive]
    )
    adjusted = np.abs(effective - observed) > zero_tolerance
    adjusted[~positive] = observed[~positive] > zero_tolerance
    return ThicknessProjection(
        effective_thickness_m=effective,
        adjusted=adjusted,
        minimum_feasible_thickness_m=minimum_h,
        maximum_feasible_thickness_m=maximum_h,
    )


def derive_trapezoid_geometry(
    thickness_m,
    section_m2,
    surface_width_m,
    *,
    default_lambda: float = 2.0,
    zero_tolerance: float = 1e-8,
) -> TrapezoidGeometry:
    """Solve the exact ``(thickness, section, surface width)`` contract.

    For positive ice, ``lambda`` and the bed width ``w0`` are uniquely given
    by the trapezoid equations.  Zero-ice nodes retain a finite bed width so
    OGGM can initialise an empty downstream or margin node.
    """

    thickness = np.asarray(thickness_m, dtype=float)
    section = np.asarray(section_m2, dtype=float)
    width = np.asarray(surface_width_m, dtype=float)
    if not (thickness.shape == section.shape == width.shape):
        raise ValueError("cross-section-arrays-must-have-identical-shape")
    if thickness.ndim != 1 or thickness.size == 0:
        raise ValueError("cross-section-arrays-must-be-nonempty-1d")
    if not np.isfinite(thickness).all():
        raise ValueError("nonfinite-cross-section-thickness")
    if not np.isfinite(section).all():
        raise ValueError("nonfinite-cross-section-area")
    if not np.isfinite(width).all():
        raise ValueError("nonfinite-cross-section-width")
    if np.any(thickness < -zero_tolerance):
        raise ValueError("negative-cross-section-thickness")
    if np.any(section < -zero_tolerance):
        raise ValueError("negative-cross-section-area")
    if np.any(width <= 0):
        raise ValueError("nonpositive-cross-section-surface-width")

    thickness = np.maximum(thickness, 0.0)
    section = np.maximum(section, 0.0)
    positive_h = thickness > zero_tolerance
    positive_a = section > zero_tolerance
    if np.any(positive_h != positive_a):
        raise ValueError("thickness-section-zero-state-mismatch")

    lambdas = np.full(thickness.shape, float(default_lambda), dtype=float)
    bottom_width = width.copy()
    shape_factor = np.full(thickness.shape, np.nan, dtype=float)
    if np.any(positive_h):
        h = thickness[positive_h]
        area = section[positive_h]
        w = width[positive_h]
        ratio = area / (w * h)
        lam = 2.0 * (w - area / h) / h
        w0 = 2.0 * area / h - w
        # A rectangle is the closed upper endpoint (ratio=1, lambda=0).
        feasible = (
            (ratio > 0.5 + zero_tolerance)
            & (ratio <= 1.0 + zero_tolerance)
            & (lam >= -zero_tolerance)
            & (w0 > zero_tolerance)
        )
        if not np.all(feasible):
            bad = np.flatnonzero(positive_h)[~feasible]
            raise ValueError(
                "trapezoid-cross-section-infeasible-at-nodes:"
                + ",".join(map(str, bad.tolist()))
            )
        lambdas[positive_h] = np.maximum(lam, 0.0)
        bottom_width[positive_h] = w0
        shape_factor[positive_h] = ratio

    return TrapezoidGeometry(
        lambdas=lambdas,
        bottom_width_m=bottom_width,
        area_shape_factor=shape_factor,
        positive_ice=positive_h,
    )


def derive_fixed_bed_trapezoid_geometry(
    thickness_m,
    section_m2,
    observed_surface_width_m,
    *,
    default_lambda: float = 2.0,
    bottom_width_fraction: float = 1e-6,
    zero_tolerance: float = 1e-8,
) -> FixedBedTrapezoidGeometry:
    """Preserve bed, thickness and section while minimising width error.

    Regime codes are 0 for no ice, 1 for ``F <= 0.5`` (near-triangular),
    2 for ``0.5 < F <= 1`` (exact observed width), and 3 for ``F > 1``
    (rectangular).  OGGM requires strictly positive trapezoid bottom widths,
    so the triangular limit uses a dimensionless numerical floor.
    """

    thickness = np.asarray(thickness_m, dtype=float)
    section = np.asarray(section_m2, dtype=float)
    observed_width = np.asarray(observed_surface_width_m, dtype=float)
    if not (thickness.shape == section.shape == observed_width.shape):
        raise ValueError("fixed-bed-cross-section-arrays-must-have-identical-shape")
    if thickness.ndim != 1 or thickness.size == 0:
        raise ValueError("fixed-bed-cross-section-arrays-must-be-nonempty-1d")
    if not np.isfinite(thickness).all():
        raise ValueError("nonfinite-fixed-bed-thickness")
    if not np.isfinite(section).all():
        raise ValueError("nonfinite-fixed-bed-section")
    if not np.isfinite(observed_width).all():
        raise ValueError("nonfinite-fixed-bed-observed-width")
    if np.any(thickness < -zero_tolerance):
        raise ValueError("negative-fixed-bed-thickness")
    if np.any(section < -zero_tolerance):
        raise ValueError("negative-fixed-bed-section")
    if np.any(observed_width <= 0):
        raise ValueError("nonpositive-fixed-bed-observed-width")
    if not 0 < bottom_width_fraction < 1:
        raise ValueError("bottom-width-fraction-must-be-between-zero-and-one")

    thickness = np.maximum(thickness, 0.0)
    section = np.maximum(section, 0.0)
    positive_h = thickness > zero_tolerance
    positive_a = section > zero_tolerance
    if np.any(positive_h != positive_a):
        raise ValueError("fixed-bed-thickness-section-zero-state-mismatch")

    lambdas = np.full(thickness.shape, float(default_lambda), dtype=float)
    bottom_width = observed_width.copy()
    effective_width = observed_width.copy()
    shape_factor = np.full(thickness.shape, np.nan, dtype=float)
    width_error = np.zeros(thickness.shape, dtype=float)
    regime = np.zeros(thickness.shape, dtype=np.int8)
    if np.any(positive_h):
        indices = np.flatnonzero(positive_h)
        h = thickness[positive_h]
        area = section[positive_h]
        width = observed_width[positive_h]
        mean_width = area / h
        factor = area / (width * h)
        low = factor <= 0.5
        exact = (factor > 0.5) & (factor <= 1.0)
        high = factor > 1.0

        local_lambda = np.zeros_like(h)
        local_bottom = mean_width.copy()
        if np.any(low):
            numerical_floor = np.maximum(
                bottom_width_fraction * mean_width[low],
                np.finfo(float).eps * np.maximum(mean_width[low], 1.0),
            )
            local_bottom[low] = numerical_floor
            local_lambda[low] = (
                2.0 * (mean_width[low] - numerical_floor) / h[low]
            )
        if np.any(exact):
            local_lambda[exact] = (
                2.0 * (width[exact] - mean_width[exact]) / h[exact]
            )
            local_bottom[exact] = 2.0 * mean_width[exact] - width[exact]
        # The closest admissible section for F > 1 is a rectangle.
        local_lambda[high] = 0.0
        local_bottom[high] = mean_width[high]

        local_effective_width = local_bottom + local_lambda * h
        reconstructed_section = (
            0.5 * (local_bottom + local_effective_width) * h
        )
        if not np.allclose(
            reconstructed_section, area, rtol=1e-12, atol=zero_tolerance
        ):
            raise RuntimeError("fixed-bed-section-reconstruction-mismatch")
        if np.any(local_bottom <= 0) or np.any(local_lambda < 0):
            raise RuntimeError("fixed-bed-trapezoid-domain-violation")

        lambdas[positive_h] = local_lambda
        bottom_width[positive_h] = local_bottom
        effective_width[positive_h] = local_effective_width
        shape_factor[positive_h] = factor
        width_error[positive_h] = local_effective_width / width - 1.0
        local_regime = np.empty_like(indices, dtype=np.int8)
        local_regime[low] = 1
        local_regime[exact] = 2
        local_regime[high] = 3
        regime[positive_h] = local_regime

    return FixedBedTrapezoidGeometry(
        lambdas=lambdas,
        bottom_width_m=bottom_width,
        effective_surface_width_m=effective_width,
        area_shape_factor=shape_factor,
        width_error_fraction=width_error,
        regime_code=regime,
        positive_ice=positive_h,
    )


def _junction_half_width(receiver_nx: int, smooth_trib_influx: bool) -> int:
    if not smooth_trib_influx:
        return 0
    if receiver_nx >= 9:
        return 4
    if receiver_nx >= 7:
        return 3
    if receiver_nx >= 5:
        return 2
    raise ValueError("tributary-receiver-has-fewer-than-five-nodes")


def audit_flowline_network(
    flowlines,
    *,
    smooth_trib_influx: bool = True,
    enforce_coordinate_spacing: bool = True,
) -> dict:
    """Validate the ordering and junction assumptions used by OGGM dynamics."""

    if not flowlines:
        raise ValueError("flowline-network-is-empty")
    roots = [flowline for flowline in flowlines if flowline.flows_to is None]
    if len(roots) != 1:
        raise ValueError(f"flowline-network-root-count:{len(roots)}")
    if flowlines[-1] is not roots[0]:
        raise ValueError("main-flowline-is-not-last")

    junction_rows = []
    maximum_coordinate_spacing_deviation_m = 0.0
    for index, flowline in enumerate(flowlines):
        coordinates = np.asarray(flowline.line.coords, dtype=float)
        if coordinates.shape != (flowline.nx, 2):
            raise ValueError(f"flowline-coordinate-count-mismatch:{index}")
        if flowline.nx > 1:
            spacing = np.hypot(
                np.diff(coordinates[:, 0]), np.diff(coordinates[:, 1])
            )
            maximum_coordinate_spacing_deviation_m = max(
                maximum_coordinate_spacing_deviation_m,
                float(np.max(np.abs(spacing - float(flowline.dx)))),
            )
            if enforce_coordinate_spacing and not np.allclose(
                spacing, float(flowline.dx), atol=1e-2, rtol=0.0
            ):
                raise ValueError(f"flowline-is-not-evenly-spaced:{index}")
        if not np.isclose(
            float(flowline.dx_meter),
            float(flowline.dx) * float(flowline.map_dx),
        ):
            raise ValueError(f"flowline-dx-meter-inconsistent:{index}")
        if not np.isfinite(flowline.section).all() or np.any(flowline.section < 0):
            raise ValueError(f"invalid-flowline-section:{index}")
        is_trapezoid = np.asarray(flowline.is_trapezoid, dtype=bool)
        if not np.all(is_trapezoid):
            raise ValueError(f"nontrapezoid-node-in-dynamic-contract:{index}")
        if not np.isfinite(flowline._lambdas[is_trapezoid]).all():
            raise ValueError(f"nonfinite-flowline-lambda:{index}")
        if np.any(flowline._w0_m[is_trapezoid] <= 0):
            raise ValueError(f"nonpositive-flowline-bottom-width:{index}")

        receiver = flowline.flows_to
        if receiver is None:
            continue
        receiver_index = flowlines.index(receiver)
        if receiver_index <= index:
            raise ValueError(f"receiver-does-not-follow-tributary:{index}")
        if int(receiver.order) <= int(flowline.order):
            raise ValueError(f"receiver-order-not-above-tributary:{index}")
        half_width = _junction_half_width(receiver.nx, smooth_trib_influx)
        junction_index = int(flowline.flows_to_indice)
        if not (half_width <= junction_index <= receiver.nx - half_width - 1):
            raise ValueError(
                f"tributary-kernel-outside-receiver:{index}:"
                f"junction={junction_index}:receiver_nx={receiver.nx}:"
                f"half_width={half_width}"
            )
        junction_rows.append(
            {
                "tributary_index": index,
                "receiver_index": receiver_index,
                "junction_index": junction_index,
                "kernel_size": 1 if half_width == 0 else 2 * half_width + 1,
                "tributary_tail_minus_receiver_surface_m": float(
                    flowline.surface_h[-1] - receiver.surface_h[junction_index]
                ),
            }
        )

    return {
        "flowline_count": len(flowlines),
        "root_count": 1,
        "root_index": len(flowlines) - 1,
        "junction_count": len(junction_rows),
        "negative_junction_surface_count": int(
            sum(
                row["tributary_tail_minus_receiver_surface_m"] < 0
                for row in junction_rows
            )
        ),
        "smooth_trib_influx": bool(smooth_trib_influx),
        "coordinate_spacing_enforced": bool(enforce_coordinate_spacing),
        "maximum_coordinate_spacing_deviation_m": float(
            maximum_coordinate_spacing_deviation_m
        ),
        "junctions": junction_rows,
    }


def build_variable_lambda_flowlines(
    centerlines,
    node_fields,
    *,
    map_dx_m: float,
    downstream_line: dict | None,
    default_lambda: float = 2.0,
    rgi_id: str | None = None,
    gdir=None,
    smooth_trib_influx: bool = True,
    cross_section_contract: str = "legacy-exact-width",
    enforce_coordinate_spacing: bool = True,
):
    """Construct and link OGGM ``MixedBedFlowline`` instances."""

    from oggm.core.centerlines import line_order
    from oggm.core.flowline import MixedBedFlowline

    if len(centerlines) != len(node_fields):
        raise ValueError("centerline-node-field-count-mismatch")
    created = []
    receiver_ids: list[int | None] = []
    cross_section_audits = []
    for index, (centerline, fields) in enumerate(zip(centerlines, node_fields)):
        surface_h = np.asarray(fields.surface_h_m, dtype=float)
        bed_h = np.asarray(fields.bed_h_m, dtype=float)
        section = np.asarray(fields.section_m2, dtype=float)
        surface_width = np.asarray(fields.surface_width_m, dtype=float)
        if not (
            len(surface_h)
            == len(bed_h)
            == len(section)
            == len(surface_width)
            == centerline.nx
        ):
            raise ValueError(f"flowline-node-field-length-mismatch:{index}")
        thickness = surface_h - bed_h
        if cross_section_contract == "legacy-exact-width":
            geometry = derive_trapezoid_geometry(
                thickness,
                section,
                surface_width,
                default_lambda=default_lambda,
            )
        elif cross_section_contract == "strip-mean-rectangular":
            positive = section > 1e-8
            reconstructed = surface_width[positive] * thickness[positive]
            if not np.allclose(
                reconstructed,
                section[positive],
                rtol=1e-10,
                atol=1e-8,
            ):
                raise ValueError(
                    f"strip-mean-contract-is-not-rectangular:{index}"
                )
            geometry = TrapezoidGeometry(
                lambdas=np.zeros_like(section),
                bottom_width_m=surface_width.copy(),
                area_shape_factor=np.where(positive, 1.0, np.nan),
                positive_ice=positive,
            )
        elif cross_section_contract == "fixed-bed-section":
            geometry = derive_fixed_bed_trapezoid_geometry(
                thickness,
                section,
                surface_width,
                default_lambda=default_lambda,
            )
        else:
            raise ValueError(
                f"unsupported-cross-section-contract:{cross_section_contract}"
            )
        lambdas = geometry.lambdas
        bottom_width = geometry.bottom_width_m
        line = centerline.line

        is_root = centerline.flows_to is None
        if is_root and downstream_line is not None:
            downstream_surface = np.asarray(
                downstream_line["surface_h"], dtype=float
            )
            downstream_width = np.asarray(downstream_line["w0s"], dtype=float)
            count = len(downstream_surface)
            surface_h = np.append(surface_h, downstream_surface)
            bed_h = np.append(bed_h, downstream_surface)
            section = np.append(section, np.zeros(count, dtype=float))
            lambdas = np.append(
                lambdas, np.full(count, float(default_lambda), dtype=float)
            )
            bottom_width = np.append(bottom_width, downstream_width)
            line = downstream_line["full_line"]

        flowline = MixedBedFlowline(
            line=line,
            dx=float(centerline.dx),
            map_dx=float(map_dx_m),
            surface_h=surface_h,
            bed_h=bed_h,
            section=section,
            bed_shape=np.full(len(section), np.nan, dtype=float),
            is_trapezoid=np.ones(len(section), dtype=bool),
            lambdas=lambdas,
            widths_m=bottom_width,
            rgi_id=rgi_id or centerline.rgi_id,
            gdir=gdir,
        )
        created.append(flowline)
        receiver_ids.append(
            centerlines.index(centerline.flows_to)
            if centerline.flows_to is not None
            else None
        )
        finite_shape = geometry.area_shape_factor[
            np.isfinite(geometry.area_shape_factor)
        ]
        cross_section_audits.append(
            {
                "source_centerline_index": index,
                "positive_ice_node_count": int(geometry.positive_ice.sum()),
                "lambda_min": float(np.min(geometry.lambdas)),
                "lambda_max": float(np.max(geometry.lambdas)),
                "shape_factor_min": (
                    float(np.min(finite_shape)) if finite_shape.size else None
                ),
                "shape_factor_max": (
                    float(np.max(finite_shape)) if finite_shape.size else None
                ),
                "cross_section_contract": cross_section_contract,
                "effective_width_error_mean_fraction": (
                    float(np.mean(geometry.width_error_fraction[geometry.positive_ice]))
                    if hasattr(geometry, "width_error_fraction")
                    and np.any(geometry.positive_ice)
                    else 0.0
                ),
            }
        )

    for flowline, receiver_id in zip(created, receiver_ids):
        if receiver_id is not None:
            flowline.set_flows_to(created[receiver_id])
    for flowline in created:
        flowline.order = line_order(flowline)
    created.sort(key=lambda flowline: int(flowline.order))
    network_audit = audit_flowline_network(
        created,
        smooth_trib_influx=smooth_trib_influx,
        enforce_coordinate_spacing=enforce_coordinate_spacing,
    )
    return created, {
        "network": network_audit,
        "cross_sections": cross_section_audits,
    }


__all__ = [
    "FlowlineNodeFields",
    "FixedBedTrapezoidGeometry",
    "TrapezoidGeometry",
    "audit_flowline_network",
    "build_variable_lambda_flowlines",
    "derive_trapezoid_geometry",
    "derive_fixed_bed_trapezoid_geometry",
    "project_thickness_to_trapezoid_domain",
]
