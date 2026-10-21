from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
V2 = BASE / "temperature_region_season_low_frequency_preflight_v2"
EVIDENCE = BASE / "temperature_1850_1950_low_frequency_evidence_matrix_v13"
AVAILABILITY = BASE / "temperature_original_numeric_source_availability_v2"
OUTPUT = BASE / "temperature_region_season_low_frequency_preflight_v3"
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(V2 / "constraint_matrix.csv", OUTPUT / "constraint_matrix.csv")
    shutil.copy2(V2 / "PRE_REGISTRATION.md", OUTPUT / "PRE_REGISTRATION.md")
    readiness = pd.read_csv(V2 / "input_readiness.csv")
    readiness = pd.concat(
        [
            readiness,
            pd.DataFrame(
                [
                    {
                        "input_id": "original_numeric_source_availability_v2",
                        "period": "record_specific",
                        "shape": "12_target_records",
                        "status": "public_source_search_closed",
                        "eligible_use": "primary_numeric_qualitative_and_unavailable_roles_frozen",
                        "remaining_gap": "author_delivery_needed_for_Gahai_Xiada_Dagze;Bamu_safe_parser_pending",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    readiness.to_csv(OUTPUT / "input_readiness.csv", index=False)
    summary = json.loads((V2 / "summary.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "experiment": "temperature_region_season_low_frequency_preflight_v3",
            "supersedes": "temperature_region_season_low_frequency_preflight_v2",
            "latest_evidence_matrix": "temperature_1850_1950_low_frequency_evidence_matrix_v13",
            "public_numeric_source_search_closed": True,
            "source_availability_package": "temperature_original_numeric_source_availability_v2",
            "next_read_only_action": "group_primary_evidence_by_region_native_target_and_window_then_decide_whether_any_correction_mechanism_is_identifiable",
            "quarantined_agent_result_used": False,
            "prohibited_source": "20CRv3",
            "prohibited_source_accessed": False,
        }
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Preflight v3 result\n\n"
        "The public original-numeric source search is closed and evidence matrix v13 is now the frozen source identity. "
        "The next scientific action is a read-only grouping of primary evidence by region, native target, and time window. "
        "No uniform correction, Candidate v2, EnKF replay, or OGGM run is authorized.\n\n"
        "The quarantined branch was not used. No sealed-period value was used, production was not changed, and this package did not access 20CRv3.\n",
        encoding="utf-8",
    )
    manifest = {
        "inputs": [
            {"path": str(V2 / "summary.json"), "sha256": sha256(V2 / "summary.json")},
            {"path": str(V2 / "constraint_matrix.csv"), "sha256": sha256(V2 / "constraint_matrix.csv")},
            {"path": str(V2 / "input_readiness.csv"), "sha256": sha256(V2 / "input_readiness.csv")},
            {"path": str(EVIDENCE / "summary.json"), "sha256": sha256(EVIDENCE / "summary.json")},
            {"path": str(AVAILABILITY / "summary.json"), "sha256": sha256(AVAILABILITY / "summary.json")},
        ],
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "quarantined_agent_result_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
