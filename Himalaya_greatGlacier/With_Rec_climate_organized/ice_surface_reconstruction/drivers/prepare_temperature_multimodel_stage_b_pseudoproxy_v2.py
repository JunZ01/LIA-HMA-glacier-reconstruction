from __future__ import annotations

import json
from pathlib import Path

import prepare_temperature_multimodel_stage_b_pseudoproxy_v1 as v1


DRIVER = Path(__file__).resolve()
V1_DRIVER = Path(v1.__file__).resolve()
v1.OUT = v1.LOOP / "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_preregistration_v2"


def regional_weights_unique(assignment, state_points, valid_mask):
    filtered = assignment[
        (assignment["assignment_status"] == "unique")
        & assignment["o2region"].notna()
    ].copy()
    filtered["o2region"] = filtered["o2region"].astype(str)
    return ORIGINAL_REGIONAL_WEIGHTS(filtered, state_points, valid_mask)


ORIGINAL_REGIONAL_WEIGHTS = v1.regional_weights
v1.regional_weights = regional_weights_unique
# The implementation remains v1 plus the explicit unmatched-cell filter above.
# Make the generated manifest identify this wrapper as the active driver.
v1.__file__ = str(DRIVER)


def main() -> None:
    v1.main()
    summary_path = v1.OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["supersedes_technical_failure"] = (
        "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_preregistration_v1"
    )
    summary["v2_change"] = (
        "filter 41 assignment_status=unmatched rows before sorting GTN-G region codes"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest_path = v1.OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["implementation_v1"] = {
        "path": str(V1_DRIVER),
        "sha256": v1.sha256(V1_DRIVER),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
