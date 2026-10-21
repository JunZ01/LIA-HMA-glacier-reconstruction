"""Run the fixed three-family audit with current recovered input paths."""

from pathlib import Path

import audit_temperature_external_three_family_fixed_candidate_v2 as compatibility


audit = compatibility.audit
audit.DRIVER = Path(__file__).resolve()
audit.OUTPUT = audit.RESULTS / "climate_reconstruction_temperature_external_three_family_fixed_candidate_audit_v3"
audit.ASSIGNMENT = (
    audit.RESULTS
    / "climate_reconstruction_hma_gtng_subregion_inventory_v1/hma_grid_gtng15_assignment.csv"
)


if __name__ == "__main__":
    raise SystemExit(audit.main())
