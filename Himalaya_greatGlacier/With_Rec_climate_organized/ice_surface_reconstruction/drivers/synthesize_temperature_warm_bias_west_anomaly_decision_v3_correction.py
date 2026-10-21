"""Correct the over-strong causal interpretation in the V2 decision ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_warm_bias_west_anomaly_decision_v3_correction"
V2 = RESULTS / "climate_reconstruction_temperature_warm_bias_west_anomaly_decision_v2"
HIRPORA = RESULTS / "climate_reconstruction_temperature_hirpora_independent_pre1901_validation_v1"
WEST = RESULTS / "climate_reconstruction_temperature_hindu_kush_transition_validation_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inputs = [V2 / "summary.json", HIRPORA / "summary.json", WEST / "summary.json"]
    before = {str(path): sha256(path) for path in inputs}
    v2 = json.loads(inputs[0].read_text(encoding="utf-8"))
    hirpora = json.loads(inputs[1].read_text(encoding="utf-8"))
    west = json.loads(inputs[2].read_text(encoding="utf-8"))

    summary = {
        "status": "complete_correction",
        "supersedes_interpretation": "climate_reconstruction_temperature_warm_bias_west_anomaly_decision_v2",
        "unchanged_findings": {
            "systematic_warm_bias_verified": True,
            "long_term_warm_bias_source": v2["long_term_warm_bias"]["verified_primary_cause"],
            "western_algorithmic_pathway_verified": "temperature-operator covariance pathway creates the archived western warming increment",
            "documented_season_psm_route_rejected": True,
            "production_candidate_exists": False,
        },
        "retracted_claim": "independent Hirpora verifies that the regional western temperature covariance pathway is over-amplified and nontransferable",
        "reason_for_retraction": [
            "Hirpora is a local June-July Kashmir reconstruction whereas the tested candidate is a west-HMA lon<80E regional mean",
            "standardization and smoothing do not remove spatial or seasonal representativeness error",
            "existing western records are not unanimous: Upper Indus and Hirpora share a negative 1861-1900 trend, while Kumrat is positive",
            "alpha=0 still leaves a large Hirpora 1881-1900 mean mismatch and previously failed the frozen uniform-alpha transfer route",
        ],
        "correct_interpretation": {
            "hirpora_result": "consistent with attenuation/removal of the regional temperature covariance pathway, but not causal proof",
            "western_physical_truth": "unresolved_due_spatial_and_seasonal_heterogeneity",
            "upper_indus_late_mean_sd": west["late_1881_1900_minus_1812_1860_sd"]["upper_indus"],
            "kumrat_late_mean_sd": west["late_1881_1900_minus_1812_1860_sd"]["kumrat_hindu_kush"],
            "hirpora_late_mean_sd": -0.11294177676568895,
            "hirpora_alpha0_rmse_improvement_fraction": min(
                hirpora["classification_by_attenuated_branch"]["alpha_0p00"]["rmse_improvement_by_mask"].values()
            ),
            "hirpora_alpha0_late_mean_absolute_error_sd": 1.7308134214013482,
        },
        "goal_state": "not_closed_for_western_physical_cause_or_validated_correction",
        "next_required_test": (
            "compare local/co-located reconstruction targets and candidate grid-cell or subregional branches under a frozen spatial contract; "
            "do not infer regional operator error from a local-versus-regional comparison"
        ),
        "production_authorized": False,
        "enkf_authorized": False,
        "oggm_authorized": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    audit = pd.DataFrame(
        [
            ["Long-term HMA warm bias and background source", "retained", "verified at the existing evidence level"],
            ["Western algorithmic temperature-covariance pathway", "retained", "verified decomposition of the archived regional increment"],
            ["Hirpora proves regional covariance over-amplification", "retracted", "local seasonal record was compared with a west-HMA regional mean"],
            ["Western physical cause", "unresolved", "independent records show spatial/seasonal heterogeneity"],
            ["Uniform alpha correction", "rejected", "prior frozen transfer gates remain failed"],
        ],
        columns=["claim", "corrected_status", "basis"],
    )
    audit.to_csv(OUTPUT / "corrected_claim_audit.csv", index=False)
    report = """# Corrected temperature decision V3

The V2 synthesis overstated the Hirpora result. Hirpora is a local June-July Kashmir reconstruction, while the tested branch is a west-HMA (`lon < 80 E`) regional mean produced by transferring an ordinary-EnKF operator increment to a six-member curve. Their disagreement cannot uniquely diagnose regional covariance error.

What remains valid:

- the 1500-1899 warm-bias source attribution;
- the algorithmic decomposition showing that the temperature covariance pathway creates the archived western warming increment;
- the negative documented-season PSM result;
- rejection of all current production correction candidates.

What is unresolved:

- whether the regional western warming is physically wrong everywhere;
- whether Hirpora, Upper Indus, and Kumrat differ because of spatial climate structure, season, chronology, or reconstruction method;
- the physical cause and a validated correction for the western anomaly.

The next valid comparison must be spatially co-located. No uniform alpha, EnKF replay, production replacement, or OGGM run is authorized.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")
    after = {str(path): sha256(path) for path in inputs}
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": before == after}, indent=2),
        encoding="utf-8",
    )
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
