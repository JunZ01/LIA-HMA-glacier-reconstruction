"""Map the audited parity schema into the v1 decomposition reader."""

from __future__ import annotations

from . import audit_system2005_masked_internal_source_elevation_feedback_v1 as audit


audit.OUT = audit.BASE / "lia_191_system2005_masked_internal_source_elevation_feedback_v2"


def main() -> None:
    original_loads = audit.json.loads

    def compatible_loads(value, *args, **kwargs):
        result = original_loads(value, *args, **kwargs)
        if isinstance(result, dict) and result.get("schema") == "lia-191-system2005-1d-2d-climate-area-budget-parity-v1":
            result["two_dimensional_budget"] = {
                "fixed_initial_geometry_50y_source_m3": result["two_d_mass_budget"][
                    "fixed_initial_geometry_cumulative_source_m3"
                ]
            }
        return result

    audit.json.loads = compatible_loads
    try:
        audit.main()
    finally:
        audit.json.loads = original_loads


if __name__ == "__main__":
    main()
