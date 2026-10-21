"""Adjudicate the frozen b1 amplitude family without post-hoc promotion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
SOURCE = RESULTS / "climate_reconstruction_temperature_west_b1_operator_amplitude_sensitivity_1861_1900_v3"
OUTPUT = RESULTS / "climate_reconstruction_temperature_west_b1_operator_amplitude_decision_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = {
        "summary": SOURCE / "summary.json",
        "metrics": SOURCE / "b1_amplitude_validation_metrics.csv",
        "curves": SOURCE / "b1_amplitude_candidate_curves_1812_1900.csv",
    }
    before = {name: sha256(path) for name, path in paths.items()}
    upstream = json.loads(paths["summary"].read_text(encoding="utf-8"))
    metrics = pd.read_csv(paths["metrics"])
    baseline = metrics.loc[metrics.alpha_b1_temperature.eq(1.0)].iloc[0]
    rows = []
    for row in metrics.itertuples(index=False):
        gates = {
            "upper_indus_rmse_improves": row.upper_indus_rmse_1861_1900 < baseline.upper_indus_rmse_1861_1900,
            "kumrat_rmse_improves": row.kumrat_rmse_1861_1900 < baseline.kumrat_rmse_1861_1900,
            "kumrat_correlation_regression_below_0p05": row.kumrat_correlation_1861_1900 - baseline.kumrat_correlation_1861_1900 > -0.05,
            "late_warming_sign_preserved": row.candidate_late_1881_1900_mean_sd > 0.15,
            "maximum_change_below_1p5_degC": row.maximum_absolute_change_degC < 1.5,
        }
        rows.append({
            "alpha_b1_temperature": row.alpha_b1_temperature,
            "upper_indus_rmse_improvement_fraction": 1.0 - row.upper_indus_rmse_1861_1900 / baseline.upper_indus_rmse_1861_1900,
            "kumrat_rmse_improvement_fraction": 1.0 - row.kumrat_rmse_1861_1900 / baseline.kumrat_rmse_1861_1900,
            "kumrat_correlation_change": row.kumrat_correlation_1861_1900 - baseline.kumrat_correlation_1861_1900,
            **gates,
            "all_per_alpha_diagnostic_gates_pass": all(gates.values()),
        })
    audit = pd.DataFrame(rows)
    passing = audit.loc[audit.all_per_alpha_diagnostic_gates_pass & audit.alpha_b1_temperature.ne(1.0)]
    if passing.alpha_b1_temperature.tolist() != [0.25]:
        raise RuntimeError(f"Unexpected per-alpha gate pattern: {passing.alpha_b1_temperature.tolist()}")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "ROUND_CONTRACT.md").write_text(
        "# Round contract\n\n"
        "- Audit every frozen alpha against the same gates; do not replace the preregistered Upper-Indus-only selector.\n"
        "- A post-hoc passing alpha may become a future hypothesis only, never the current validated correction.\n"
        "- New promotion requires genuinely unconsumed western evidence or a newly frozen validation target.\n",
        encoding="utf-8",
    )
    audit.to_csv(OUTPUT / "per_alpha_gate_audit.csv", index=False)
    alpha025 = audit.loc[audit.alpha_b1_temperature.eq(0.25)].iloc[0]
    summary = {
        "status": "complete",
        "preregistered_upper_indus_selector_result": {
            "selected_alpha": upstream["selected_alpha_from_upper_indus"],
            "passes_all_gates": upstream["all_diagnostic_validation_gates_pass"],
            "decision": "reject_upper_indus_only_selection_rule",
        },
        "posthoc_family_observation": {
            "only_passing_nonbaseline_alpha": 0.25,
            "upper_indus_rmse_improvement_fraction": float(alpha025.upper_indus_rmse_improvement_fraction),
            "kumrat_rmse_improvement_fraction": float(alpha025.kumrat_rmse_improvement_fraction),
            "kumrat_correlation_change": float(alpha025.kumrat_correlation_change),
            "status": "frozen_hypothesis_only_not_validated_candidate",
        },
        "scientific_interpretation": "The b1 response is nonlinear; complete removal fits Upper Indus but reverses Kumrat shape, while alpha=0.25 is a weak cross-record compromise discovered after inspection.",
        "decision": "freeze_alpha_0p25_for_future_new_blind_validation_but_do_not_modify_current_reconstruction",
        "production_release_authorized": False,
        "oggm_authorized": False,
        "input_hashes_unchanged": before == {name: sha256(path) for name, path in paths.items()},
        "phase_1901_1920_accessed_or_used": False,
        "locked_1921_1939_accessed": False,
        "original_sources_modified": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(json.dumps({name: {"path": str(paths[name]), "sha256": before[name]} for name in paths}, indent=2), encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# b1 temperature-operator amplitude decision\n\n"
        "The preregistered Upper-Indus-only selector chose alpha=0 and failed Kumrat shape replication, so that selection route is rejected.\n\n"
        f"Alpha=0.25 is the only nonbaseline grid point passing the per-alpha diagnostic gates: Upper Indus RMSE improves {100 * alpha025.upper_indus_rmse_improvement_fraction:.1f}%, Kumrat RMSE improves {100 * alpha025.kumrat_rmse_improvement_fraction:.1f}%, and Kumrat correlation changes {alpha025.kumrat_correlation_change:+.3f}. Because this point was identified after family inspection and both records were already consumed, it is frozen only as a future blind-validation hypothesis. It is not a production correction.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
