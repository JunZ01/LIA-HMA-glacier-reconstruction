"""Create a V2 temperature cause ledger including independent Hirpora evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_warm_bias_west_anomaly_decision_v2"
OLD = RESULTS / "climate_reconstruction_temperature_warm_bias_west_anomaly_decision_v1"
HIRPORA = RESULTS / "climate_reconstruction_temperature_hirpora_independent_pre1901_validation_v1"
ALPHA = RESULTS / "climate_reconstruction_temperature_west_b1_operator_amplitude_sensitivity_1861_1900_v3"
KALAM = RESULTS / "climate_reconstruction_temperature_west_b1_alpha0p25_kalam_blind_test_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inputs = [
        OLD / "summary.json",
        OLD / "temperature_cause_and_candidate_ledger.csv",
        OLD / "goal_completion_audit.csv",
        HIRPORA / "summary.json",
        HIRPORA / "hirpora_branch_metrics_by_mask.csv",
        ALPHA / "summary.json",
        KALAM / "summary.json",
    ]
    before = {str(path): sha256(path) for path in inputs}
    old_summary = json.loads(inputs[0].read_text(encoding="utf-8"))
    hirpora = json.loads(inputs[3].read_text(encoding="utf-8"))
    alpha = json.loads(inputs[5].read_text(encoding="utf-8"))
    kalam = json.loads(inputs[6].read_text(encoding="utf-8"))
    if not hirpora["digitization_passes"] or not hirpora["identity_independent"]:
        raise RuntimeError("hirpora-evidence-gates-failed")
    if hirpora["classification"] != "independent_hirpora_supports_overamplified_temperature_covariance":
        raise RuntimeError("unexpected-hirpora-classification")

    ledger = pd.read_csv(inputs[1])
    west_row = ledger["question"].eq("Why is the western temperature pathway physically nontransferable?")
    ledger.loc[west_row, "status"] = "verified_failure_mode"
    ledger.loc[west_row, "result"] = (
        "independent Hirpora has opposite low-frequency behavior; alpha=0 improves RMSE 26.5% "
        "and restores trend sign in all digitization masks"
    )
    ledger.loc[west_row, "causal_level"] = "independent evidence for over-amplified operator covariance"
    ledger.loc[west_row, "evidence"] = str(HIRPORA / "summary.json")
    ledger = pd.concat(
        [
            ledger,
            pd.DataFrame(
                [
                    {
                        "question": "Does the independent result authorize a uniform alpha correction?",
                        "status": "no",
                        "result": (
                            "alpha=0 is mechanism-only; the uniform alpha family failed prior transfer gates, "
                            "and alpha=0.25 improved Kalam RMSE by only 0.9%"
                        ),
                        "causal_level": "frozen candidate acceptance",
                        "evidence": str(ALPHA / "summary.json"),
                    },
                    {
                        "question": "Does Hirpora resolve the 1500-1899 HMA background warm bias?",
                        "status": "no_separate_mechanism",
                        "result": "Hirpora arbitrates the western 1861-1900 anomaly only; the background offset remains a separate prior-forcing problem",
                        "causal_level": "scope boundary",
                        "evidence": str(HIRPORA / "FINAL_RESULT.md"),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    ledger.to_csv(OUTPUT / "temperature_cause_and_candidate_ledger_v2.csv", index=False)

    audit = pd.read_csv(inputs[2])
    west_audit = audit["objective_requirement"].eq("Determine a verifiable cause of the western 1881-1900 anomaly")
    audit.loc[west_audit, "evidence_status"] = "proved_failure_mode"
    audit.loc[west_audit, "remaining_requirement"] = (
        "independent Hirpora verifies over-amplified temperature covariance; the exact proxy-physiology or covariance-estimation origin is not identified"
    )
    audit.to_csv(OUTPUT / "goal_completion_audit_v2.csv", index=False)

    a0 = hirpora["classification_by_attenuated_branch"]["alpha_0p00"]
    summary = {
        "status": "complete",
        "supersedes": "climate_reconstruction_temperature_warm_bias_west_anomaly_decision_v1",
        "long_term_warm_bias": old_summary["long_term_warm_bias"],
        "western_transition": {
            "algorithmic_mechanism_verified": True,
            "independent_failure_mode_verified": True,
            "mechanism": "archived western temperature-operator covariance pathway is over-amplified and nontransferable",
            "independent_evidence": "Hirpora June-July temperature reconstruction through 1900",
            "alpha0_rmse_improvement_fraction_range": [
                min(a0["rmse_improvement_by_mask"].values()),
                max(a0["rmse_improvement_by_mask"].values()),
            ],
            "alpha0_supported_as_production_correction": False,
            "remaining_causal_depth": "exact proxy-physiology or covariance-estimation origin not identified",
        },
        "candidate_status": {
            "production_candidate_exists": False,
            "uniform_alpha_family_closed": True,
            "alpha_sensitivity_decision": alpha["decision"],
            "kalam_alpha0p25_decision": kalam["decision"],
            "reason": "new independent evidence verifies the failure mode but does not reverse previously failed frozen transfer gates",
        },
        "goal_completion_state": "scientific_diagnosis_closed_no_validated_correction_candidate",
        "next_mainline_requirement": (
            "design a new correction family with an independently justified spatial/temporal structure and rerun modern frozen preflight; "
            "do not continue tuning uniform alpha"
        ),
        "production_authorized": False,
        "enkf_authorized": False,
        "oggm_authorized": False,
        "maximum_external_year_analyzed": 1900,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# Temperature warm-bias and western-anomaly decision V2

## What is now verified

- The 1500-1899 HMA warm offset remains verified at {old_summary['long_term_warm_bias']['gap_degC']:.3f} degC; its dominant source is the time-invariant analogue background.
- Independent Hirpora evidence verifies that the western 1861-1900 archived temperature-covariance warming is nontransferable. Alpha=0 improves frozen-mask RMSE by {100 * min(a0['rmse_improvement_by_mask'].values()):.1f}-{100 * max(a0['rmse_improvement_by_mask'].values()):.1f}% and restores the trend sign.
- The uniform alpha correction family remains rejected. Independent mechanism evidence does not erase its failed modern/external transfer gates.

## Decision

The diagnosis is closed at the actionable failure-mode level, but no production correction is authorized. The next correction family must alter the spatial/temporal covariance structure with an independent physical justification and pass modern frozen preflight before any paleo replay.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")
    after = {str(path): sha256(path) for path in inputs}
    manifest = {"inputs_before": before, "inputs_after": after, "source_hashes_unchanged": before == after}
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
