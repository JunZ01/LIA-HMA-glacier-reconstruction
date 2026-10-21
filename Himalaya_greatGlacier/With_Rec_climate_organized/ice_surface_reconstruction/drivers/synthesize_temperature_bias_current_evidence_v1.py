"""Synthesize the current temperature-bias evidence after western records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_bias_current_evidence_synthesis_v1"
INPUTS = {
    "transition_onset": RESULTS / "climate_reconstruction_temperature_transition_onset_1800_1900_v1/summary.json",
    "prior_update": RESULTS / "climate_reconstruction_temperature_transition_prior_update_1800_1900_v1/summary.json",
    "western_independent": RESULTS / "climate_reconstruction_temperature_hindu_kush_transition_validation_v2/summary.json",
    "western_holdout": RESULTS / "climate_reconstruction_temperature_west_transition_amplitude_oos_v2/summary.json",
    "old_mainline": RESULTS / "climate_reconstruction_temperature_lia_warm_bias_mainline_synthesis_v1/summary.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# Round contract\n\n"
        "- Synthesize only completed, versioned diagnostics; do not read climate values.\n"
        "- New western independent evidence may revise interpretation but cannot authorize a full-field correction.\n"
        "- Preserve older results as provenance; distinguish reference disagreement from independently demonstrated bias.\n",
        encoding="utf-8",
    )
    before = {name: sha256(path) for name, path in INPUTS.items()}
    data = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in INPUTS.items()}
    western = data["western_independent"]["late_1881_1900_minus_1812_1860_sd"]
    holdout = data["western_holdout"]
    decomposition = data["prior_update"]

    summary = {
        "status": "complete",
        "current_primary_conclusion": "The temperature discrepancy is not a uniform HMA warm offset.",
        "retained_findings": [
            "The archived project remains warmer than the four-reference family over much of the LIA.",
            "The 1860-1900 project-reference divergence is generated mainly by the EnKF increment, with a smaller prior-background contribution.",
        ],
        "revised_findings": [
            "Western late-19th-century warming is independently supported in sign by Upper Indus and Kumrat Hindu Kush records.",
            "Reference-family cooling or near-zero change in the west cannot be treated as ground truth.",
            "Project western warming amplitude is higher than both independent standardized estimates, but a one-parameter attenuation fails holdout shape and smoothness gates.",
        ],
        "western_late_1881_1900_baseline_sd": western,
        "western_fit_holdout_result": {
            "fitted_amplitude_scale": holdout["final_low_frequency_amplitude_scale"],
            "heldout_rmse_improvement_fraction": holdout["metrics"]["heldout_kumrat"]["candidate_rmse_improvement_vs_project_fraction"],
            "heldout_correlation_change": holdout["metrics"]["heldout_kumrat"]["candidate_correlation_change_vs_project"],
            "passes_all_gates": holdout["all_regional_diagnostic_gates_pass"],
        },
        "mechanism_interpretation": {
            "statement": "EnKF dominance quantifies the source of project-reference divergence, not the fraction proven erroneous.",
            "hma_increment_share_percent": 71.2,
            "west_increment_share_percent": 73.3,
            "source": str(INPUTS["prior_update"]),
        },
        "production_decision": "No uniform shift, regional scalar attenuation, production temperature field, or OGGM forcing is authorized.",
        "next_high_value_experiment": "Attribute the 1860-1900 western EnKF increment to proxy/PSM groups and test whether seasonal target mismatch explains the unsupported curve shape.",
        "older_results_policy": "Do not discard old results; retain source/closure diagnostics, but supersede any interpretation that equated reference disagreement with demonstrated physical bias.",
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
    }
    after = {name: sha256(path) for name, path in INPUTS.items()}
    summary["input_hashes_unchanged"] = before == after
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({name: {"path": str(INPUTS[name]), "sha256": before[name]} for name in INPUTS}, indent=2), encoding="utf-8")

    md = f"""# Current temperature-bias evidence synthesis

## Decision

The current discrepancy is **not** a uniform HMA warm offset. No uniform shift or production correction is authorized.

## What changed

- Upper Indus: late 1881-1900 change `{western['upper_indus']:.3f}` baseline SD.
- Kumrat Hindu Kush: `{western['kumrat_hindu_kush']:.3f}` baseline SD.
- Project west: `{western['project_west']:.3f}` baseline SD.
- Reference west: `{western['reference_west']:.3f}` baseline SD.

Both independent records support western warming in sign. The project amplitude is stronger, but a scalar correction fitted on Upper Indus improves Kumrat RMSE by only `{100 * holdout['metrics']['heldout_kumrat']['candidate_rmse_improvement_vs_project_fraction']:.1f}%` while reducing correlation by `{abs(holdout['metrics']['heldout_kumrat']['candidate_correlation_change_vs_project']):.3f}`; it fails the frozen gates.

## Mechanism

The EnKF increment creates about 71% of the HMA project-reference divergence and 73% in the west. This identifies where the difference enters; it does not prove that the same fraction is wrong.

## Next experiment

Attribute the 1860-1900 western EnKF increment to proxy and PSM groups, then test seasonal-target mismatch. Keep 1901-1920 and 1921-1939 out of fitting and scoring.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
