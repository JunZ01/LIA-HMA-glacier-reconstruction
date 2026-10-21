"""Route CPC and GHCN-D precipitation evidence without reopening tuning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
ROUND81 = LOOP / "round_81_cpc_fixed_candidate_adjudication_v1"
ROUND83 = LOOP / "round_83_ghcnd_allowed_station_precipitation_v1"
OUTPUT = LOOP / "round_84_precipitation_external_evidence_router_v1"
DRIVER = Path(__file__).resolve()


def record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"path": str(path), "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "\n".join(
            [
                "# Round 84 Contract",
                "",
                "## Objective",
                "Resolve the scientific routing conflict between Round81 candidate scores and CPC observation-product incompatibility, then combine that result with the Round83 direct-station completeness outcome.",
                "",
                "## Frozen Logic",
                "- A new gridded target cannot accept or reject candidates if zero of four western observation-compatibility paths pass.",
                "- Candidate scores against such a target remain diagnostic conflict evidence only.",
                "- Direct station adjudication cannot proceed when the predeclared minimum of ten comparison-eligible western stations is not met.",
                "- Do not tune candidates, alter gates, add localization, run EnKF, or access 1921-1939.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary81_path = ROUND81 / "summary.json"
    compatibility_path = ROUND81 / "cpc_observation_compatibility.csv"
    summary83_path = ROUND83 / "summary.json"
    summary81 = json.loads(summary81_path.read_text(encoding="utf-8"))
    summary83 = json.loads(summary83_path.read_text(encoding="utf-8"))
    compatibility = pd.read_csv(compatibility_path)
    west = compatibility[compatibility.region == "west"]
    if len(west) != 4 or int(west.passes.sum()) != 0:
        raise RuntimeError("unexpected-cpc-western-compatibility-result")
    if summary81["advancing_western_candidates"]:
        raise RuntimeError("unexpected-cpc-advancing-candidate")
    if summary83["comparison_eligible_western_station_count"] >= summary83["minimum_west_stations_to_advance"]:
        raise RuntimeError("unexpected-ghcnd-station-route")

    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "cpc_western_compatibility_paths_passing": int(west.passes.sum()),
        "cpc_candidate_scores_adjudicative": False,
        "round81_machine_close_decision_adopted": False,
        "cpc_advancing_candidates": summary81["advancing_western_candidates"],
        "ghcnd_comparison_eligible_western_station_count": summary83[
            "comparison_eligible_western_station_count"
        ],
        "ghcnd_minimum_western_station_count": summary83["minimum_west_stations_to_advance"],
        "ghcnd_station_adjudication_executed": False,
        "precipitation_candidate_status": "O_and_local_enkf_remain_provisional_regional_curves_only",
        "decision": "hold_at_external_evidence_boundary_aphrodite_or_equivalent_required",
        "next_admissible_inputs": [
            "APHRO_MA_V1101R1_0.5deg",
            "equivalent_mountain_asia_gauge_product_with_documented_coverage",
            "materially_new_independent_proxy_sites_or_chronology",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Round 84 Precipitation External-Evidence Router",
                "",
                "- CPC passes `0/4` western compatibility paths against CRU, GPCC, UDel, and PREC/L.",
                "- Therefore Round81's candidate scores are diagnostic only; its machine-generated close decision is not adopted.",
                f"- GHCN-D has `{summary83['comparison_eligible_western_station_count']}` eligible western stations versus the frozen minimum `{summary83['minimum_west_stations_to_advance']}`; station adjudication is not executed.",
                "- O and local-EnKF remain provisional western regional curves only, with no accepted spatial field or production status.",
                "- Decision: `hold_at_external_evidence_boundary_aphrodite_or_equivalent_required`.",
                "- No tuning, scaling, localization, EnKF, OGGM, or locked-period access occurred.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sources = [summary81_path, compatibility_path, summary83_path, ROUND83 / "ghcnd_station_completeness_summary.csv"]
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"sources": [record(path) for path in sources], "driver": record(DRIVER), "contract": record(contract)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
