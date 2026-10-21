from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = (
    ROOT
    / "Himalaya_greatGlacier"
    / "With_Rec_climate_organized"
    / "ice_surface_reconstruction"
    / "drivers"
)
BASE_PATH = DRIVERS / "build_temperature_low_frequency_evidence_matrix_v2.py"
OUT = (
    ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "temperature_1850_1950_low_frequency_evidence_matrix_v3"
)


def load_base():
    spec = importlib.util.spec_from_file_location("lf_matrix_v2", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = load_base()
    base.OUT = OUT
    original_candidates = base.candidate_rows

    def revised_candidates():
        rows = original_candidates()
        for row in rows:
            if row["evidence_id"] == "RARA_CHOLANGPATI_MXD_2026":
                row["source_independence"] = "mixed_two_new_sites_plus_two_old_ITRDB_sites"
                row["admitted_role"] = "method_mechanism_evidence_and_numeric_candidate_not_independent_vote"
                row["finding"] = (
                    "classic NegEx chronology loses low-frequency trend; frequency-combined "
                    "RCS-NegEx restores significant warming and improves modern instrumental skill"
                )
                row["next_action"] = (
                    "request annual NegEx RCS combined and per-site series; test new Rara/Chola "
                    "component separately and mask 1921-1939"
                )
                row["source_url"] = "https://doi.org/10.1007/s11676-026-02104-0"
        return rows

    base.candidate_rows = revised_candidates
    base.main()

    queue_path = OUT / "author_data_request_queue.csv"
    with queue_path.open(newline="", encoding="utf-8") as stream:
        queue = list(csv.DictReader(stream))
    queue.append(
        {
            "priority": "1",
            "record": "Central Himalayan Rara-Cholangpati MXD 1775-2022",
            "contact": "nahtz@geo.uni-mainz.de",
            "request": (
                "annual NegEx, RCS, frequency-combined and per-site MXD series; prediction "
                "uncertainty, chronology/sample crosswalk, and new-site-only reconstruction; "
                "mask 1921-1939 before delivery"
            ),
        }
    )
    for row in queue:
        row["priority"] = str(int(row["priority"]) + 1) if row["record"] != "Central Himalayan Rara-Cholangpati MXD 1775-2022" else "1"
    queue.sort(key=lambda row: int(row["priority"]))
    with queue_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(queue[0]))
        writer.writeheader()
        writer.writerows(queue)

    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = OUT.name
    summary["author_requests"] = len(queue)
    summary["new_actionable_mechanism"] = "proxy_chronology_low_frequency_loss_and_frequency_preserving_recombination"
    summary["mechanism_supported_by"] = "Nahtz_et_al_2026_central_Himalaya_MXD"
    summary["mechanism_test_authorized"] = "read_only_design_only_no_production_change"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = OUT / "FINAL_RESULT.md"
    report = report_path.read_text(encoding="utf-8").replace(
        "evidence matrix v2", "evidence matrix v3"
    )
    report += """

## New actionable mechanism

The strongest new result is methodological rather than another background product. Nahtz et al. (2026) show in a central-Himalayan MXD network that a conventional negative-exponential chronology has no significant low-frequency trend, while a frequency-combined chronology that takes the low-frequency component from RCS and the high-frequency component from NegEx restores significant warming and improves agreement with modern instrumental temperature.

This provides a concrete explanation for apparent long-term drift: standardization can remove the century-scale signal before the PSM or assimilation sees it. It is scientifically preferable to test this mechanism before blending four model backgrounds or fitting a post-hoc temperature offset.

The record is not a fully independent climate vote because its final four-site composite mixes two new sites (Rara and Cholangpati) with two older ITRDB sites. Its role is therefore a mechanism test. The requested delivery separates NegEx, RCS, combined, and individual-site series, with the sealed interval masked before delivery.

## Next experiment

Run a read-only frequency-retention audit on the project's temperature proxy inputs:

1. quantify centennial variance retained by archived proxy series relative to any available raw or RCS chronology;
2. separate records for which low-frequency loss is demonstrable from records whose raw data are unavailable;
3. test whether a frequency-preserving proxy representation improves only allowed direct-station periods and the admitted eastern historical gates;
4. do not change the background, EnKF, production reconstruction, or use 1921-1939.

Only if this audit transfers across regions and seasons should a Candidate v2 reconstruction experiment be designed.
"""
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
