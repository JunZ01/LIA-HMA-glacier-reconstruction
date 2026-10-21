"""V2 connected-support step; only reserve roundoff handling differs."""

from __future__ import annotations

from . import network_capacity_connected_support_step_v1 as v1
from .network_connected_support_projection_v2 import project_flux_dag_with_reserve


MINIMUM_SUPPORT_THICKNESS_M = v1.MINIMUM_SUPPORT_THICKNESS_M
minimum_section_for_support = v1.minimum_section_for_support


def connected_support_step(flowlines, coefficients, dt_seconds, closure_limit_m3):
    original = v1.project_flux_dag_with_reserve
    v1.project_flux_dag_with_reserve = project_flux_dag_with_reserve
    try:
        return v1.connected_support_step(flowlines, coefficients, dt_seconds, closure_limit_m3)
    finally:
        v1.project_flux_dag_with_reserve = original
