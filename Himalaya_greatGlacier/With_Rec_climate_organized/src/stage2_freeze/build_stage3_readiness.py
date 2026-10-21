"""Build the 245-system route ledger and unsqueezed Stage 2 ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BRACKET_PATTERN = re.compile(
    r"temp residuals are\s+([-+0-9.eE]+),\s+([-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: str) -> Path:
    return (REPO_ROOT / path).resolve()


def classify_mb_failure(error: str) -> tuple[str, float, float]:
    match = BRACKET_PATTERN.search(str(error))
    if not match:
        return "mb-bracket-residual-sign-unavailable", math.nan, math.nan
    lower, upper = (float(value) for value in match.groups())
    if lower < 0 and upper < 0:
        label = "mb-bracket-residuals-both-negative"
    elif lower > 0 and upper > 0:
        label = "mb-bracket-residuals-both-positive"
    else:
        label = "mb-bracket-residuals-mixed-or-zero"
    return label, lower, upper


def select_three_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    feasible = frame[
        frame["status"] == "feasible-modern-mb-equivalence-candidate"
    ].copy()
    grid = feasible[feasible["candidate_kind"] == "grid"].copy()
    baseline = feasible[
        feasible["candidate_kind"] == "stage0-baseline"
    ].copy()
    if len(grid) == 13:
        feasible = grid
    elif len(grid) == 12 and len(baseline) == 1:
        feasible = pd.concat([grid, baseline], ignore_index=True)
    else:
        raise ValueError(
            "Expected a 13-point grid or a 12-point grid plus one baseline; "
            f"found grid={len(grid)}, baseline={len(baseline)}"
        )
    feasible["tuple_key"] = list(
        zip(
            feasible["melt_f"].round(10),
            feasible["precip_prior_multiplier"].round(10),
            feasible["temp_bias"].round(10),
        )
    )
    feasible = feasible.sort_values(
        ["precip_prior_multiplier", "melt_f", "candidate_id"],
    ).drop_duplicates("tuple_key", keep="first")
    feasible = feasible.sort_values(
        ["precip_prior_multiplier", "melt_f", "candidate_id"]
    ).reset_index(drop=True)
    if len(feasible) != 13:
        raise ValueError(
            f"Expected 13 unique equivalence candidates, found {len(feasible)}"
        )
    selected = feasible.iloc[[0, 6, 12]].copy()
    selected["equivalence_ordinal"] = [1, 7, 13]
    selected["equivalence_coordinate"] = [0.0, 0.5, 1.0]
    return selected.drop(columns=["tuple_key"])


def validate_count(name: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise RuntimeError(f"{name}: expected {expected}, found {actual}")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    manifest_path = resolve(config["manifest_path"])
    f19_dir = resolve(config["farinotti_stage0_dir"])
    m22_dir = resolve(config["millan_stage0_dir"])
    turnover_dir = resolve(config["farinotti_turnover_dir"])
    attribution_dir = resolve(config["attribution_dir"])
    section_dir = resolve(config["section_audit_dir"])
    paths = {
        "manifest": manifest_path,
        "mb_system_candidates": f19_dir / "mb_system_candidates.csv",
        "mb_member_parameters": f19_dir / "mb_member_parameters.csv",
        "mb_failures": f19_dir / "climate_mb_failures.csv",
        "farinotti_inversion": f19_dir / "inversion_system_summary.csv",
        "millan_inversion": m22_dir / "inversion_system_summary.csv",
        "millan_failures": m22_dir / "inversion_failures.csv",
        "equivalence_candidates": turnover_dir / "candidate_summary.csv",
        "attribution_systems": attribution_dir / "system_attribution.csv",
        "section_consensus": section_dir / "system_consensus_audit.csv",
    }
    input_hashes = {key: sha256(path) for key, path in paths.items()}

    manifest = pd.read_csv(paths["manifest"])
    mb_system = pd.read_csv(paths["mb_system_candidates"])
    mb_members = pd.read_csv(paths["mb_member_parameters"])
    mb_failures = pd.read_csv(paths["mb_failures"])
    f19 = pd.read_csv(paths["farinotti_inversion"])
    m22 = pd.read_csv(paths["millan_inversion"])
    m22_failures = pd.read_csv(paths["millan_failures"])
    candidates = pd.read_csv(paths["equivalence_candidates"])
    attribution = pd.read_csv(paths["attribution_systems"])
    section = pd.read_csv(paths["section_consensus"])

    expected_manifest = int(config["expected_manifest_systems"])
    expected_core = int(config["expected_core_systems"])
    expected_excluded = int(config["expected_excluded_systems"])
    expected_millan = int(config["expected_millan_systems"])
    expected_members = int(config["expected_core_members"])
    validate_count("manifest systems", len(manifest), expected_manifest)
    validate_count("MB core systems", len(mb_system), expected_core)
    validate_count("MB excluded systems", len(mb_failures), expected_excluded)
    validate_count("Farinotti inversion systems", len(f19), expected_core)
    validate_count("Millan inversion systems", len(m22), expected_millan)
    validate_count("core modern members", len(mb_members), expected_members)
    if set(mb_system["lia_unique_id"]) | set(mb_failures["lia_unique_id"]) != set(
        manifest["lia_unique_id"]
    ):
        raise RuntimeError("Core and excluded branches do not partition 245 systems")

    failure_rows = []
    for row in mb_failures.itertuples(index=False):
        detail, lower, upper = classify_mb_failure(row.error)
        failure_rows.append(
            {
                "lia_unique_id": int(row.lia_unique_id),
                "stage": row.stage,
                "status": row.status,
                "error_type": row.error_type,
                "exclusion_detail": detail,
                "temperature_bracket_residual_lower": lower,
                "temperature_bracket_residual_upper": upper,
                "error": row.error,
                "member_count": int(row.member_count),
            }
        )
    excluded = pd.DataFrame(failure_rows)

    core_ids = set(mb_system["lia_unique_id"].astype(int))
    millan_ids = set(m22["lia_unique_id"].astype(int))
    strict_ids = set(
        section.loc[
            section["farinotti_strict_gate_count"]
            + section["millan_strict_gate_count"]
            > 0,
            "lia_unique_id",
        ].astype(int)
    )
    median_residual_ids = set(section["lia_unique_id"].astype(int)) - strict_ids
    m22_glen_fail_ids = set(
        m22_failures.loc[
            m22_failures["error"].str.contains(
                "Thickness target is outside", na=False
            ),
            "lia_unique_id",
        ].astype(int)
    )

    f_attr = attribution[attribution["scenario"] == "farinotti"].set_index(
        "lia_unique_id"
    )
    mb_by_id = mb_system.set_index("lia_unique_id")
    f19_by_id = f19.set_index("lia_unique_id")
    failure_by_id = excluded.set_index("lia_unique_id")

    route_rows = []
    for row in manifest.itertuples(index=False):
        system_id = int(row.lia_unique_id)
        core = system_id in core_ids
        if core:
            primary_route = "core-191-stage3-ready-three-point-ensemble"
            exclusion_detail = "none"
            exclusion_error = ""
        else:
            primary_route = "excluded-54-modern-mb-no-solution"
            exclusion_detail = failure_by_id.loc[
                system_id, "exclusion_detail"
            ]
            exclusion_error = failure_by_id.loc[system_id, "error"]

        if system_id in millan_ids:
            millan_route = "millan-scenario-available"
        elif system_id in m22_glen_fail_ids:
            millan_route = "millan-thickness-target-outside-glen-a-scan"
        else:
            millan_route = "millan-unavailable-after-modern-mb-failure"

        if not core:
            velocity_class = "not-evaluated-stage0-excluded"
            no_strong_gate = False
            velocity_available = False
        else:
            attr = f_attr.loc[system_id]
            no_strong_gate = attr["attribution_class"] == "no-evaluable-strong-gate"
            velocity_available = not no_strong_gate
            if system_id in strict_ids:
                velocity_class = "strict-observation-interval-residual"
            elif system_id in median_residual_ids:
                velocity_class = "median-only-residual-interval-overlap"
            else:
                velocity_class = str(attr["three_way_attribution_class"])

        route_rows.append(
            {
                "lia_unique_id": system_id,
                "main_rgi_id": row.main_rgi_id,
                "region_code": int(row.region_code),
                "region_label": row.region_label,
                "declared_member_count": int(row.declared_member_count),
                "primary_stage3_route": primary_route,
                "farinotti_core_191": core,
                "millan_scenario_route": millan_route,
                "millan_scenario_available": system_id in millan_ids,
                "excluded_54_detail": exclusion_detail,
                "excluded_54_error": exclusion_error,
                "velocity_evidence_class": velocity_class,
                "velocity_strict_anomaly_7": system_id in strict_ids,
                "velocity_no_evaluable_strong_gate": no_strong_gate,
                "velocity_stage2_likelihood_weight": 0.0,
                "velocity_soft_evidence_role": (
                    "structure-routing;stage4-residual-interpretation"
                ),
                "velocity_stage4_diagnostic_available": velocity_available,
                "debris_ratio_fraction": row.debris_ratio_fraction,
                "debris_ratio_class": row.debris_ratio_class,
                "debris_binary_status": row.debris_binary_status,
                "hugonnet_mb_kg_m2_yr": row.hugonnet_mb_kg_m2_yr,
                "hugonnet_err_fully_correlated_mwea": (
                    row.hugonnet_err_fully_correlated_mwea
                ),
                "stage1_identifiability_status": (
                    mb_by_id.loc[system_id, "identifiability_status"]
                    if core
                    else "not-available"
                ),
                "stage2_parameter_status": (
                    "frozen-unsqueezed-three-point-ensemble"
                    if core
                    else "not-available"
                ),
                "glen_a_source_stage": (
                    "stage0_bed_full_farinotti_v0" if core else "none"
                ),
                "mb_equivalence_source_stage": (
                    "s1_turnover_f19_full_v0" if core else "none"
                ),
                "velocity_evidence_source_stage": (
                    "attribution_preflight_v0;section_width_audit_v0"
                    if core
                    else "none"
                ),
                "written_by_stage": "stage2-freeze-stage3-routing-v0",
                "allowed_use": (
                    "stage3-read-only-parameter-input;stage4-diagnostic"
                    if core
                    else "failure-ledger-only"
                ),
            }
        )
    routes = pd.DataFrame(route_rows).sort_values("lia_unique_id")

    roles = list(config["ensemble_roles"])
    ensemble_weight = float(config["ensemble_weight"])
    f19_fields = [
        "lia_unique_id",
        "scenario_id",
        "thickness_target_product",
        "glen_a_multiplier",
        "glen_a_pa_3_s_1",
        "fs",
    ]
    f19_parameters = f19[f19_fields].set_index("lia_unique_id")
    system_ensemble_rows = []
    selection_failures = []
    for system_id, frame in candidates.groupby("lia_unique_id", sort=True):
        try:
            selected = select_three_candidates(frame)
            rheology = f19_parameters.loc[int(system_id)]
            for role, (_, candidate) in zip(roles, selected.iterrows()):
                system_ensemble_rows.append(
                    {
                        "lia_unique_id": int(system_id),
                        "ensemble_id": f"{int(system_id)}-{role}",
                        "ensemble_role": role,
                        "ensemble_weight": ensemble_weight,
                        "equivalence_ordinal": int(
                            candidate["equivalence_ordinal"]
                        ),
                        "equivalence_coordinate": float(
                            candidate["equivalence_coordinate"]
                        ),
                        "source_candidate_id": candidate["candidate_id"],
                        "source_candidate_kind": candidate["candidate_kind"],
                        "melt_f": float(candidate["melt_f"]),
                        "precip_prior_multiplier": float(
                            candidate["precip_prior_multiplier"]
                        ),
                        "temp_bias": float(candidate["temp_bias"]),
                        "target_mb_kg_m2_yr": float(candidate["target_mb_kg_m2_yr"]),
                        "modeled_mb_kg_m2_yr": float(
                            candidate["modeled_mb_kg_m2_yr"]
                        ),
                        "mb_residual_kg_m2_yr": float(
                            candidate["mb_residual_kg_m2_yr"]
                        ),
                        "glen_a_multiplier": float(rheology["glen_a_multiplier"]),
                        "glen_a_pa_3_s_1": float(rheology["glen_a_pa_3_s_1"]),
                        "fs": float(rheology["fs"]),
                        "thickness_scenario_id": rheology["scenario_id"],
                        "thickness_target_product": rheology[
                            "thickness_target_product"
                        ],
                        "velocity_selection_weight": 0.0,
                        "velocity_evidence_role": (
                            "soft-routing-and-stage4-diagnostic-only"
                        ),
                        "parameter_source_stage": (
                            "stage0_bed_full_farinotti_v0;"
                            "s1_turnover_f19_full_v0"
                        ),
                        "glen_a_source_artifact": str(paths["farinotti_inversion"]),
                        "glen_a_source_sha256": input_hashes[
                            "farinotti_inversion"
                        ],
                        "mb_equivalence_source_artifact": str(
                            paths["equivalence_candidates"]
                        ),
                        "mb_equivalence_source_sha256": input_hashes[
                            "equivalence_candidates"
                        ],
                        "written_by_stage": "stage2-freeze-stage3-routing-v0",
                        "allowed_use": (
                            "stage3-read-only-parameter-input;"
                            "stage4-diagnostic"
                        ),
                        "status": "frozen-stage2-ensemble-member",
                    }
                )
        except Exception as exc:
            selection_failures.append(
                {
                    "lia_unique_id": int(system_id),
                    "stage": "stage2-ensemble-selection",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    system_ensemble = pd.DataFrame(system_ensemble_rows)

    prcp_min = float(config["prcp_fac_min"])
    prcp_max = float(config["prcp_fac_max"])
    member_ensemble = mb_members.merge(
        system_ensemble,
        on="lia_unique_id",
        validate="many_to_many",
        suffixes=("_stage1", ""),
    )
    member_ensemble["raw_prcp_fac"] = (
        member_ensemble["winter_prcp_fac_center"]
        * member_ensemble["precip_prior_multiplier"]
    )
    member_ensemble["prcp_fac"] = member_ensemble["raw_prcp_fac"].clip(
        prcp_min, prcp_max
    )
    member_ensemble["prcp_fac_clipped"] = ~np.isclose(
        member_ensemble["raw_prcp_fac"], member_ensemble["prcp_fac"]
    )
    member_columns = [
        "lia_unique_id",
        "ensemble_id",
        "ensemble_role",
        "ensemble_weight",
        "equivalence_ordinal",
        "equivalence_coordinate",
        "rgi_id",
        "winter_prcp_fac_center",
        "precip_prior_multiplier",
        "raw_prcp_fac",
        "prcp_fac",
        "prcp_fac_clipped",
        "melt_f",
        "temp_bias",
        "glen_a_multiplier",
        "glen_a_pa_3_s_1",
        "fs",
        "thickness_scenario_id",
        "velocity_selection_weight",
        "velocity_evidence_role",
        "parameter_source_stage",
        "glen_a_source_artifact",
        "glen_a_source_sha256",
        "mb_equivalence_source_artifact",
        "mb_equivalence_source_sha256",
        "written_by_stage",
        "allowed_use",
        "status",
    ]
    member_ensemble = member_ensemble[member_columns].sort_values(
        ["lia_unique_id", "equivalence_ordinal", "rgi_id"]
    )

    failures = pd.DataFrame(
        selection_failures,
        columns=["lia_unique_id", "stage", "error_type", "error"],
    )
    validate_count("routing rows", len(routes), expected_manifest)
    validate_count(
        "core routing rows", int(routes["farinotti_core_191"].sum()), expected_core
    )
    validate_count("excluded details", len(excluded), expected_excluded)
    validate_count("system ensemble rows", len(system_ensemble), expected_core * 3)
    validate_count("member ensemble rows", len(member_ensemble), expected_members * 3)
    validate_count("selection failures", len(failures), 0)
    role_counts = system_ensemble.groupby("lia_unique_id")[
        "ensemble_role"
    ].nunique()
    if not role_counts.eq(3).all():
        raise RuntimeError("At least one core system lacks three ensemble roles")
    if not system_ensemble["velocity_selection_weight"].eq(0).all():
        raise RuntimeError("Velocity entered Stage 2 candidate selection")

    routes.to_csv(output_dir / "stage3_routing_245.csv", index=False)
    system_ensemble.to_csv(
        output_dir / "stage2_system_ensemble.csv", index=False
    )
    member_ensemble.to_csv(
        output_dir / "stage2_member_ensemble.csv", index=False
    )
    excluded.to_csv(output_dir / "excluded_54_details.csv", index=False)
    failures.to_csv(output_dir / "failures.csv", index=False)

    summary = {
        "run_id": config["run_id"],
        "status": "stage3-readiness-gate-passed-no-lia-run-started",
        "routing_system_count": int(len(routes)),
        "core_191_count": int(routes["farinotti_core_191"].sum()),
        "excluded_54_count": int((~routes["farinotti_core_191"]).sum()),
        "millan_scenario_available_count": int(
            routes["millan_scenario_available"].sum()
        ),
        "velocity_strict_anomaly_count": int(
            routes["velocity_strict_anomaly_7"].sum()
        ),
        "system_ensemble_row_count": int(len(system_ensemble)),
        "member_ensemble_row_count": int(len(member_ensemble)),
        "prcp_fac_clipped_member_row_count": int(
            member_ensemble["prcp_fac_clipped"].sum()
        ),
        "maximum_absolute_mb_residual_kg_m2_yr": float(
            system_ensemble["mb_residual_kg_m2_yr"].abs().max()
        ),
        "excluded_detail_counts": excluded[
            "exclusion_detail"
        ].value_counts().to_dict(),
        "velocity_evidence_class_counts": routes[
            "velocity_evidence_class"
        ].value_counts().to_dict(),
        "stage2_velocity_selection_weight": 0.0,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json_atomic(output_dir / "routing_summary.json", summary)
    write_json_atomic(
        output_dir / "run_manifest.json",
        {
            "run_id": config["run_id"],
            "method_version": config["method_version"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256(config_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "input_hashes": input_hashes,
            "selection_columns": [
                "melt_f",
                "precip_prior_multiplier",
                "temp_bias",
                "candidate_id",
                "candidate_kind",
                "status",
            ],
            "forbidden_selection_columns": [
                "weighted_interval_loss",
                "inside_observation_interval_count",
                "median_observed_to_modeled_ratio",
                "LIA geometry",
                "reconstructed climate",
            ],
            "velocity_policy": (
                "zero Stage 2 likelihood weight; retain soft routing and "
                "Stage 4 diagnostic evidence"
            ),
            "parameter_write_policy": "Stage 2 artifact immutable after publication",
            "output_dir": str(output_dir),
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
