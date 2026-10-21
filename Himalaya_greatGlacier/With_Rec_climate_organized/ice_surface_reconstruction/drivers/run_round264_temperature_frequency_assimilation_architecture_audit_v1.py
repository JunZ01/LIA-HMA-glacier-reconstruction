from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def unique_line(lines: list[str], token: str) -> tuple[int, str]:
    matches = [(index + 1, line.strip()) for index, line in enumerate(lines) if token in line]
    if len(matches) != 1:
        raise RuntimeError(f"expected-one-match:{token}:{len(matches)}")
    return matches[0]


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    result_root = root / "validation_data/results/paleo_surface_validation_v0"
    phyda = Path("D:/Users/JUNZ/MATLAB/Projects/untitled/PHYDA-v1-master")
    output = loop / "round_264_temperature_frequency_assimilation_architecture_audit_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "driver": Path(__file__).resolve(),
        "da_ann": phyda / "da_ann.m",
        "m_update": phyda / "M_update.m",
        "round262_summary": loop / "round_262_npl065_first_difference_anchor_psm_v1/summary.json",
        "round263_summary": loop / "round_263_temperature_difference_generalization_v1/summary.json",
        "existing_oenkf_summary": result_root / "temperature_fold_native_oenkf_increment_cv_v5_regem_R/summary.json",
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    round262 = json.loads(paths["round262_summary"].read_text(encoding="ascii"))
    round263 = json.loads(paths["round263_summary"].read_text(encoding="ascii"))
    oenkf = json.loads(paths["existing_oenkf_summary"].read_text(encoding="utf-8"))
    if round262["admitted_methods"] != ["signal_free_constrained"] or round263["external_pass_count"] != 2:
        raise RuntimeError("frequency-role-evidence-contract-not-met")
    if oenkf["ordinary_enkf_member_count"] != 500 or oenkf["passes_all_gates"]:
        raise RuntimeError("existing-oenkf-contract-changed")

    da_lines = paths["da_ann"].read_text(encoding="utf-8", errors="replace").splitlines()
    update_lines = paths["m_update"].read_text(encoding="utf-8", errors="replace").splitlines()
    evidence_specs = (
        ("da_ann", da_lines, "for t=1:reconYrs", "independent annual analysis loop"),
        ("da_ann", da_lines, "Xb=Xb_o;", "the same static background is reinitialized every target year"),
        ("da_ann", da_lines, "ni=find(~isnan(y0(:,t)));", "only current-year proxy observations are selected"),
        ("da_ann", da_lines, "HXb=HXb0(ni ,:);", "the observation operator has proxy-by-ensemble dimensions only"),
        ("da_ann", da_lines, "[Xa,Xam,diagnostics]=M_update(Xb,HXb,y,R,infl);", "the update receives one annual state"),
        ("m_update", update_lines, "Xa_m=Xb_m+K*(y-HXb_m);", "analysis mean is a same-time innovation update"),
    )
    evidence_rows = []
    for source, lines, token, meaning in evidence_specs:
        line_number, text = unique_line(lines, token)
        evidence_rows.append({
            "source": source,
            "line_number": line_number,
            "verified_code": text,
            "architectural_meaning": meaning,
        })
    loop_start = unique_line(da_lines, "for t=1:reconYrs")[0]
    loop_end = next(
        index + 1
        for index in range(loop_start, len(da_lines))
        if da_lines[index].strip().startswith("% Note progress")
    )
    active_loop = "\n".join(da_lines[loop_start - 1:loop_end])
    posterior_propagated = "Xb=Xa" in active_loop.replace(" ", "")
    if posterior_propagated:
        raise RuntimeError("posterior-propagation-contract-changed")
    pd.DataFrame(evidence_rows).to_csv(output / "verified_architecture_evidence.csv", index=False)

    candidate_rows = [
        {
            "candidate": "naive_delta_HXb_in_current_annual_EnKF",
            "mathematically_defined": False,
            "full_series_temporally_consistent": False,
            "low_frequency_level_externally_controlled": False,
            "decision": "reject",
            "reason": "current prior has no physically paired adjacent-year states; arbitrary member subtraction creates an undefined covariance",
        },
        {
            "candidate": "paired_two_time_augmented_EnKF",
            "mathematically_defined": True,
            "full_series_temporally_consistent": False,
            "low_frequency_level_externally_controlled": True,
            "decision": "retain_as_transition_unit_test_only",
            "reason": "valid for one transition using consecutive model years, but overlapping annual pairs need a common smoothing constraint",
        },
        {
            "candidate": "batch_increment_state_space_smoother",
            "mathematically_defined": True,
            "full_series_temporally_consistent": True,
            "low_frequency_level_externally_controlled": True,
            "decision": "recommended_architecture",
            "reason": "estimates annual increments jointly, integrates them under external level anchors, and can preserve model-derived temporal covariance",
        },
        {
            "candidate": "regional_modern_difference_OOF_preflight",
            "mathematically_defined": True,
            "full_series_temporally_consistent": True,
            "low_frequency_level_externally_controlled": True,
            "decision": "recommended_next_experiment",
            "reason": "tests the frequency role on 1940-2000 observations before any gridded or paleo reconstruction",
        },
    ]
    pd.DataFrame(candidate_rows).to_csv(output / "candidate_architecture_decision.csv", index=False)

    design = {
        "experiment_name": "modern_only_regional_increment_smoother_oof",
        "authorization_state": "designed_not_executed",
        "years": [1940, 2000],
        "forbidden_years": [1921, 1939],
        "state": "annual regional temperature increments for the existing HMA subregions",
        "proxy_operator": "blocked-OOF delta proxy to delta seasonal temperature; no absolute proxy level",
        "level_constraint": "fold-training instrumental multi-product regional mean and covariance only",
        "temporal_prior": "consecutive-year temperature increments from physically paired model years",
        "assimilation": "batch linear Gaussian smoother; no localization and no annual posterior reset",
        "comparators": [
            "existing level-PSM ordinary-EnKF increment",
            "instrumental/background level only",
            "frequency-separated increment smoother",
        ],
        "outer_blocks": [[1940, 1951], [1952, 1963], [1964, 1975], [1976, 1987], [1988, 2000]],
        "primary_gates": {
            "median_region_product_delta_correlation_minimum": 0.30,
            "median_region_product_delta_skill_minimum_exclusive": 0.0,
            "median_level_RMSE_improvement_vs_existing_level_PSM_minimum": 0.05,
            "worst_region_product_level_RMSE_regression_maximum": 0.05,
            "positive_outer_block_fraction_minimum": 0.70,
            "all_low_frequency_anchor_data_excluded_from_test_fold": True,
        },
        "scope_boundary": "a pass permits only a gridded non-production paired-prior experiment; production, paleo, locked-period, and OGGM remain unauthorized",
    }
    (output / "ROUND_265_DESIGN.json").write_text(json.dumps(design, indent=2), encoding="ascii")
    (output / "ROUND_265_DESIGN.md").write_text(
        "# Proposed Round 265: modern-only regional increment smoother OOF\n\n"
        "The test reconstructs regional annual temperature increments over 1940-2000. Proxy levels do not enter the operator. Each held-out block is excluded from both the proxy-difference fit and the instrumental low-frequency anchor.\n\n"
        "The temporal prior uses physically consecutive model years. A batch smoother estimates all increments jointly and integrates them under the fold-native external level constraint. No localization, production reconstruction, paleo values, locked-period values, or OGGM are involved.\n\n"
        "Passing this test would authorize only a gridded non-production experiment, not production use.\n",
        encoding="ascii",
    )

    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "verified": {
            "annual_background_reinitialized": True,
            "current_year_observations_only": True,
            "operator_has_no_adjacent_year_dimension": True,
            "posterior_propagated_to_next_year": posterior_propagated,
            "existing_oenkf_500_member_closure": True,
            "existing_oenkf_passes_all_gates": False,
        },
        "decision": "do_not_insert_delta_psm_into_current_independent_year_enkf_use_batch_increment_smoother",
        "next_experiment": "round_265_modern_only_regional_increment_smoother_oof_requires_explicit_execution_authorization",
        "locked_1921_1939_values_accessed": False,
        "proxy_or_climate_numeric_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 264 temperature frequency-assimilation architecture audit\n\n"
        "The current annual EnKF resets the same static background every year, selects only current-year observations, and has no adjacent-year state or operator dimension. A delta proxy operator therefore cannot be inserted by subtracting arbitrary prior members.\n\n"
        "Decision: `do_not_insert_delta_psm_into_current_independent_year_enkf_use_batch_increment_smoother`.\n\n"
        "Round 265 is fully designed but not executed. It is a 1940-2000 regional blocked-OOF batch increment smoother with physically consecutive prior years and external low-frequency anchors.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
