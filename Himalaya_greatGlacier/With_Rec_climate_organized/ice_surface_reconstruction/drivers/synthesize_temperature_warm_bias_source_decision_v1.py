"""Synthesize the frozen evidence for the HMA temperature warm-bias source."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_warm_bias_source_decision_v1"

INPUTS = {
    "confirmation": RESULTS / "climate_reconstruction_temperature_lia_warm_bias_confirmation_v1/summary.json",
    "confirmation_regions": RESULTS / "climate_reconstruction_temperature_lia_warm_bias_confirmation_v1/four_family_gtng15_50year_confirmation.csv",
    "prior_update": RESULTS / "climate_reconstruction_temperature_lia_regional_prior_update_decomposition_v1/summary.json",
    "transition": RESULTS / "climate_reconstruction_temperature_transition_prior_update_1800_1900_v1/summary.json",
    "psm_stability": RESULTS / "climate_reconstruction_temperature_psm_modern_stability_v1/summary.json",
    "increment_cv": RESULTS / "temperature_fold_native_oenkf_increment_cv_v5_regem_R/summary.json",
    "pmip_audit": RESULTS / "climate_reconstruction_temperature_pmip_multimodel_prior_audit_v2/summary.json",
    "pmip_stage": RESULTS / "climate_reconstruction_temperature_pmip_prior_stage_synthesis_v1/summary.json",
    "blind_test": RESULTS / "climate_reconstruction_temperature_low_frequency_blind_test_1921_1939_v1/summary.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path in INPUTS.values():
        if not path.exists():
            raise FileNotFoundError(path)

    before = {name: sha256(path) for name, path in INPUTS.items()}
    data = {
        name: read_json(path)
        for name, path in INPUTS.items()
        if path.suffix == ".json"
    }
    regional_rows = read_csv(INPUTS["confirmation_regions"])

    confirmation = data["confirmation"]
    prior_update = data["prior_update"]
    transition = data["transition"]["late_transition_1881_1900"]["hma"]
    psm_networks = data["psm_stability"]["network_summary"]
    increment = data["increment_cv"]["primary_metrics"]
    pmip_audit = data["pmip_audit"]
    pmip_stage = data["pmip_stage"]
    blind = data["blind_test"]

    if not confirmation["systematic_lia_warm_bias_confirmed"]:
        raise RuntimeError("The frozen warm-bias confirmation no longer passes")
    if prior_update["hma_mechanism"] != "background_dominant":
        raise RuntimeError("The frozen prior/update mechanism changed")
    if blind["decision"] != "reject_correction_bridge_v1":
        raise RuntimeError("The frozen blind-test decision changed")

    focus_regions = []
    for region in ("13-09", "15-03"):
        rows = [row for row in regional_rows if row["domain"] == region]
        if len(rows) != 8:
            raise RuntimeError(f"Expected 8 frozen windows for {region}, got {len(rows)}")
        focus_regions.append(
            {
                "region": region,
                "positive_gap_windows": sum(
                    float(row["project_minus_four_family_central_degC"]) > 0
                    for row in rows
                ),
                "warmer_than_at_least_3_families_windows": sum(
                    int(row["warmer_than_individual_family_count"]) >= 3
                    for row in rows
                ),
                "above_full_four_family_range_windows": sum(
                    row["project_above_four_family_range"].lower() == "true"
                    for row in rows
                ),
                "median_gap_degC": sorted(
                    float(row["project_minus_four_family_central_degC"])
                    for row in rows
                )[len(rows) // 2 - 1 : len(rows) // 2 + 1],
            }
        )
    for row in focus_regions:
        middle = row.pop("median_gap_degC")
        row["median_gap_degC"] = sum(middle) / len(middle)

    background_share = (
        prior_update["hma_median_background_minus_reference_degC"]
        / prior_update["hma_median_posterior_minus_reference_degC"]
    )
    median_stable_fraction = sum(
        item["stable_fraction"] for item in psm_networks
    ) / len(psm_networks)
    blind_shortfall = (
        0.30 - blind["hma_low11_rmse_improvement_fraction"]
    )

    evidence_rows = [
        {
            "question": "Is the 1500-1899 warm bias robust to reference family?",
            "result": "yes",
            "metric": "HMA above full four-family range",
            "value": f"{confirmation['hma_project_above_four_family_range_window_count']}/8 windows",
            "decision": "reference disagreement is not the primary explanation",
        },
        {
            "question": "Where does the long-term warm bias enter?",
            "result": "background",
            "metric": "background share of posterior-reference gap",
            "value": f"{100 * background_share:.1f}%",
            "decision": "target the low-frequency background state first",
        },
        {
            "question": "Where does the 1881-1900 excess transition enter?",
            "result": "EnKF increment",
            "metric": "increment share of transition divergence",
            "value": f"{100 * transition['increment_share_of_total']:.1f}%",
            "decision": "treat transition timing as a separate mechanism",
        },
        {
            "question": "Are current temperature PSMs temporally stable?",
            "result": "mostly no",
            "metric": "median stable fraction across proxy6/proxy9",
            "value": f"{100 * median_stable_fraction:.1f}%",
            "decision": "do not rely on the annual global operator for low-frequency repair",
        },
        {
            "question": "Does blocked OEnKF increment transfer absolutely?",
            "result": "no",
            "metric": "median absolute skill / annual correlation",
            "value": f"{100 * increment['median_absolute_skill_vs_zero']:.1f}% / {increment['median_annual_correlation']:.3f}",
            "decision": "global increment shrinkage route remains closed",
        },
        {
            "question": "Do forced models support a missing cold state?",
            "result": "yes, but amplitude is model-dependent",
            "metric": "models cooler than background / bracket fraction",
            "value": f"3/3 / {100 * pmip_stage['model_window_anchor_bracketing_fraction']:.1f}%",
            "decision": "retain forced trajectories as prior evidence, not direct truth",
        },
        {
            "question": "Did the frozen low-frequency bridge validate?",
            "result": "no",
            "metric": "HMA low11 RMSE improvement / gate shortfall",
            "value": f"{100 * blind['hma_low11_rmse_improvement_fraction']:.1f}% / {100 * blind_shortfall:.1f} percentage points",
            "decision": "v1 is rejected and 1921-1939 cannot be reused for tuning",
        },
    ]

    decision = {
        "status": "complete",
        "decision": "warm_bias_is_real_and_background_dominant_with_separate_increment_transition_error",
        "periods": {
            "long_term_diagnosis": [1500, 1899],
            "modern_blocked_mechanism_test": [1940, 2000],
            "consumed_blind_test": [1921, 1939],
        },
        "reference_family_uncertainty_is_primary": False,
        "systematic_warm_bias_confirmed": True,
        "hma_warm_gap_degC": confirmation["hma_four_family_median_project_minus_reference_degC"],
        "background_gap_degC": prior_update["hma_median_background_minus_reference_degC"],
        "enkf_increment_gap_degC": prior_update["hma_median_increment_degC"],
        "background_share_of_long_term_gap": background_share,
        "increment_share_of_1881_1900_divergence": transition["increment_share_of_total"],
        "focus_region_pre1900_reference_support": focus_regions,
        "median_modern_psm_stable_fraction": median_stable_fraction,
        "blocked_increment_absolute_skill": increment["median_absolute_skill_vs_zero"],
        "blocked_increment_annual_correlation": increment["median_annual_correlation"],
        "pmip_raw_mean_prior_authorized": pmip_audit["advance_multimodel_prior_candidate"],
        "blind_bridge_improvement": blind["hma_low11_rmse_improvement_fraction"],
        "blind_bridge_gate_shortfall": blind_shortfall,
        "closed_routes": [
            "uniform_offset",
            "raw_pmip_multimodel_mean_as_prior_correction",
            "global_annual_increment_shrinkage",
            "correction_bridge_v1",
        ],
        "next_experiment": "low_frequency_hierarchical_prior_assimilation_preflight",
        "next_experiment_scope": (
            "Use three forced-model trajectories as an uncertain low-frequency prior "
            "and only cross-family robust low-frequency proxy constraints; retain the "
            "archived annual posterior anomalies and do not use 1921-1939 for fitting."
        ),
        "production_authorized": False,
        "oggm_authorized": False,
        "source_hashes_unchanged": None,
    }

    OUTPUT.mkdir(parents=True)
    write_csv(OUTPUT / "evidence_decision_matrix.csv", evidence_rows)
    write_csv(OUTPUT / "focus_region_reference_support.csv", focus_regions)

    after = {name: sha256(path) for name, path in INPUTS.items()}
    unchanged = before == after
    decision["source_hashes_unchanged"] = unchanged
    (OUTPUT / "summary.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(
            {"before": before, "after": after, "source_hashes_unchanged": unchanged},
            indent=2,
        ),
        encoding="utf-8",
    )

    report = [
        "# Temperature Warm-Bias Source Decision",
        "",
        "## Result",
        "",
        f"The 1500-1899 HMA warm bias is robust (`{decision['hma_warm_gap_degC']:+.3f} degC`) and is not explained by one reference family. The background contributes `{decision['background_gap_degC']:+.3f} degC`, or `{100 * background_share:.1f}%` of the posterior-reference gap.",
        "",
        f"The 1881-1900 transition is a separate mechanism: the EnKF increment contributes `{100 * transition['increment_share_of_total']:.1f}%` of the project-reference divergence. It must not be repaired with the same low-frequency offset.",
        "",
        "## Modern mechanism test",
        "",
        f"Only `{100 * median_stable_fraction:.1f}%` of bivariate temperature PSMs are stable across the two modern halves. Five-block OEnKF validation has only `{100 * increment['median_absolute_skill_vs_zero']:.1f}%` median absolute skill and annual correlation `{increment['median_annual_correlation']:.3f}`. The existing annual global increment route is therefore not a credible low-frequency repair.",
        "",
        "## Blind evidence",
        "",
        f"The frozen bridge improved HMA low-frequency RMSE by `{100 * blind['hma_low11_rmse_improvement_fraction']:.1f}%`, short of the predeclared 30% gate by `{100 * blind_shortfall:.1f}` percentage points. Its direction is informative, but v1 remains rejected and 1921-1939 is consumed.",
        "",
        "## Next action",
        "",
        "Run a low-frequency hierarchical-prior preflight. Treat the three forced-model trajectories as an uncertain ensemble, constrain them only with robust low-frequency proxy information, and preserve the archived annual posterior anomalies. This tests an upstream mechanism rather than fitting another post-processing correction.",
        "",
        "No production climate field or OGGM run is authorized.",
    ]
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    if not unchanged:
        raise RuntimeError("A source result changed while synthesizing evidence")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
