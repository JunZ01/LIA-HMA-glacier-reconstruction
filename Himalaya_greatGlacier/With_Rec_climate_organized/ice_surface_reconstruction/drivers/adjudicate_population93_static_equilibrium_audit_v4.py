"""Adjudicate the v3 audit's inverted unchanged-state check semantics.

The v3 scientific tables and numerical checks are immutable inputs.  This
driver copies them to a new versioned artifact and replaces four state flags
with positively named checks before deriving the overall status.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = BASE / "lia_191_population93_static_climate_state_equilibrium_audit_v3"
OUT = BASE / "lia_191_population93_static_climate_state_equilibrium_audit_v4"
TABLES = [
    "input_inventory_93.csv",
    "static_equilibrium_attribution_93.csv",
    "system7_79_reference_validation.csv",
]


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    source_summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    inventory = pd.read_csv(SOURCE / "input_inventory_93.csv")
    attribution = pd.read_csv(SOURCE / "static_equilibrium_attribution_93.csv")
    validation = pd.read_csv(SOURCE / "system7_79_reference_validation.csv")
    if len(inventory) != 93 or len(attribution) != 93 or len(validation) != 2:
        raise RuntimeError(
            f"source-row-count-mismatch:{len(inventory)}:{len(attribution)}:{len(validation)}"
        )

    OUT.mkdir(parents=True)
    for name in TABLES:
        shutil.copy2(SOURCE / name, OUT / name)

    old_checks = dict(source_summary["checks"])
    corrected_checks = {
        key: bool(value)
        for key, value in old_checks.items()
        if key not in {
            "production_temperature_modified",
            "initial_thickness_modified",
            "old_output_modified",
            "launch_authorized",
        }
    }
    corrected_checks.update(
        {
            "production_temperature_unchanged": not bool(
                old_checks["production_temperature_modified"]
            ),
            "initial_thickness_unchanged": not bool(old_checks["initial_thickness_modified"]),
            "old_outputs_unchanged": not bool(old_checks["old_output_modified"]),
            "launch_remains_unauthorized": not bool(old_checks["launch_authorized"]),
        }
    )
    passed = all(corrected_checks.values())
    summary = dict(source_summary)
    summary["schema"] = "lia-191-population93-static-climate-state-equilibrium-audit-v4"
    summary["status"] = (
        "read-only-static-equilibrium-audit-pass"
        if passed
        else "read-only-static-equilibrium-audit-fail"
    )
    summary["technical_revision_of"] = {
        "artifact": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "issue": "v3 included false unchanged-state flags directly in all(checks.values())",
        "scientific_rows_recomputed": False,
        "scientific_operator_changed": False,
        "source_status_before_semantic_adjudication": source_summary["status"],
    }
    summary["checks"] = corrected_checks
    summary["next_recommended_control_action"] = (
        "register the deterministic equilibrium-centering operator and a fixed discovery16 execution contract; keep heldout16 sealed"
        if passed
        else "do not authorize discovery; reconcile the failed static-audit checks"
    )
    summary["source_artifacts_copied_without_content_change"] = TABLES
    write_json(OUT / "summary.json", summary)

    report = [
        "# Population-93 Static Climate-State Equilibrium Audit v4",
        "",
        f"Status: {summary['status']}",
        "",
        "- All 93 systems have complete frozen geometry, Stage-2, and climate inputs.",
        "- The same endpoint-independent equilibrium operator produced a bracketed root for all 93.",
        "- System 7 and 79 reference packages and initial tendency signs were reproduced.",
        "- No dynamics, production-temperature edit, thickness edit, old-output edit, or launch occurred.",
        "- v4 changes only the semantics of four unchanged-state checks; v3 scientific rows were not recomputed.",
        "",
        "The next control transaction may preregister discovery16 with this fixed operator. Heldout16 remains sealed.",
    ]
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
