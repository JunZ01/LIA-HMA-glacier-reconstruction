"""Reference-identity fix for the population-93 static equilibrium audit.

System 7 and system 79 validate the formula but are not members of the
population-93 candidate table.  Their main RGI identities are therefore read
from their immutable climate-package summaries rather than from that table.
"""

from __future__ import annotations

import json

import pandas as pd

from . import audit_population93_static_climate_state_equilibrium_v2 as audit


ORIGINAL_VALIDATE = audit.core.validate_reference_packages


def validate_with_external_reference_identities(
    rows: pd.DataFrame,
    stage2_table: pd.DataFrame,
    pixels: pd.DataFrame,
    reconstruction,
):
    supplements = []
    for system_id in (7, 79):
        summary_path = (
            audit.core.BASE
            / f"lia_191_generic_planform2d_system{system_id}_parent_1441_1470_monthly_ti_v1"
            / "summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        supplements.append(
            {
                "system_id": system_id,
                "rgi_id": str(summary["climate_pixel"]["rgi_id"]),
            }
        )
    supplemental = pd.DataFrame(supplements).set_index("system_id")
    combined = pd.concat([rows, supplemental[~supplemental.index.isin(rows.index)]])
    return ORIGINAL_VALIDATE(combined, stage2_table, pixels, reconstruction)


def main() -> int:
    audit.OUT = audit.core.BASE / "lia_191_population93_static_climate_state_equilibrium_audit_v3"
    audit.core.validate_reference_packages = validate_with_external_reference_identities
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
