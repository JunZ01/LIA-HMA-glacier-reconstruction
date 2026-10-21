"""CSV-schema correction for the system-334 zero-ice connector builder.

V2 reached serialization but added one branch-only diagnostic key that was not
present in the CSV header.  V3 uses a fresh output and removes only that
nonessential key; geometry and scientific gates are unchanged.
"""

from __future__ import annotations

import json

from . import build_system334_postinversion_zero_ice_connector_v1 as v1
from .build_system334_postinversion_zero_ice_connector_v2 import (
    _rebuild_with_zero_mass_suffix,
)


OUTPUT = (
    v1.VALIDATION
    / "lia_191_invalid_outline334_terminal_control_volume_condensation_v1/"
    "postinversion_zero_ice_connector_v3"
)


def _schema_compatible_rebuild(source, receiver_surface, receiver_point):
    candidate, diagnostics = _rebuild_with_zero_mass_suffix(
        source, receiver_surface, receiver_point
    )
    diagnostics.pop("zero_mass_suffix_only", None)
    return candidate, diagnostics


def main() -> None:
    v1.OUTPUT = OUTPUT
    v1.CANDIDATE = OUTPUT / "candidate_model_flowlines.pkl.gz"
    v1.rebuild_flowline = _schema_compatible_rebuild
    v1.main()
    summary_path = OUTPUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "schema": "lia-191-system334-postinversion-zero-ice-connector-v3",
            "mechanism": (
                "postinversion-active-terminal-condensation-plus-zero-ice-"
                "connectors-v3"
            ),
            "v1_stop": "zero-mass suffix was treated as an invalid active budget",
            "v2_stop": "CSV branch-only diagnostic field mismatch after candidate write",
            "scientific_contract_changed_from_v2": False,
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
