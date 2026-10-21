"""Run the final station-closure audit with the compatibility sampler bound."""

from pathlib import Path

import audit_temperature_external_three_family_fixed_candidate_v5 as audit_v5


audit_v5.compatibility.bilinear_points = audit_v5.audit.station_base.bilinear_points
audit_v5.DRIVER = Path(__file__).resolve()
audit_v5.OUTPUT = (
    audit_v5.RESULTS
    / "climate_reconstruction_temperature_external_three_family_fixed_candidate_audit_v6"
)


if __name__ == "__main__":
    raise SystemExit(audit_v5.main())
