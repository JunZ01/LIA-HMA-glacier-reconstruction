"""Technical revision of the population-93 static equilibrium audit.

Three special/legacy-boundary systems have no unresolved-master route row.
Their Stage-2 baseline is retained as a labeled comparison only; the common
equilibrium operator itself is unchanged and does not use master-route deltas.
"""

from __future__ import annotations

import json

import pandas as pd
import xarray as xr

from . import audit_population93_static_climate_state_equilibrium_v1 as core


OUT = core.BASE / "lia_191_population93_static_climate_state_equilibrium_audit_v2"


RENAME = {
    "registered_spinup_delta_t_c": "reference_delta_t_c",
    "static_parent_mb_registered_m_ice_yr": "static_parent_mb_reference_m_ice_yr",
    "static_parent_source_registered_m3_yr": "static_parent_source_reference_m3_yr",
    "static_mb_sensitivity_m_ice_yr_per_c_near_registered": "static_mb_sensitivity_m_ice_yr_per_c_near_reference",
    "equilibrium_correction_from_registered_c": "equilibrium_correction_from_reference_c",
    "registered_parent_ela_m": "reference_parent_ela_m",
    "registered_accumulation_area_fraction": "reference_accumulation_area_fraction",
}


def describe_group(group: pd.DataFrame) -> dict[str, object]:
    correction = group["equilibrium_correction_from_reference_c"]
    mb = group["static_parent_mb_reference_m_ice_yr"]
    return {
        "count": int(len(group)),
        "reference_mb_positive_count": int((mb > 0.01).sum()),
        "reference_mb_near_zero_count_abs_le_0p01": int((mb.abs() <= 0.01).sum()),
        "reference_mb_negative_count": int((mb < -0.01).sum()),
        "median_reference_mb_m_ice_yr": float(mb.median()),
        "median_equilibrium_correction_from_reference_c": float(correction.median()),
        "equilibrium_correction_q25_c": float(correction.quantile(0.25)),
        "equilibrium_correction_q75_c": float(correction.quantile(0.75)),
        "absolute_correction_gt_0p5_count": int((correction.abs() > 0.5).sum()),
        "absolute_correction_gt_1p0_count": int((correction.abs() > 1.0).sum()),
        "median_equilibrium_accumulation_area_fraction": float(
            group["equilibrium_accumulation_area_fraction"].median()
        ),
    }


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    OUT.mkdir(parents=True)

    cohorts = pd.read_csv(core.COHORTS)
    stage2_table = pd.read_csv(core.STAGE2)
    stage2_table["lia_unique_id"] = stage2_table["lia_unique_id"].astype(int)
    master_table = pd.read_csv(core.MASTER)
    master_table["lia_unique_id"] = master_table["lia_unique_id"].astype(int)
    pixels = pd.read_csv(core.PIXELS)
    if len(cohorts) != 93 or cohorts["system_id"].nunique() != 93:
        raise RuntimeError(f"cohort-count-mismatch:{len(cohorts)}")

    inventories: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    with xr.open_dataset(core.RECONSTRUCTION, use_cftime=True) as reconstruction:
        for cohort in cohorts.itertuples(index=False):
            cohort_series = pd.Series(cohort._asdict())
            system_id = int(cohort_series["system_id"])
            rgi_id = str(cohort_series["rgi_id"])
            stage2_rows = stage2_table[
                stage2_table["lia_unique_id"].eq(system_id)
                & stage2_table["ensemble_role"].eq("mid-turnover")
                & stage2_table["rgi_id"].astype(str).eq(rgi_id)
            ]
            master_rows = master_table[master_table["lia_unique_id"].eq(system_id)]
            pixel_rows = pixels[
                pixels["rgi_id"].astype(str).eq(rgi_id)
                & pixels["climate_id"].astype(str).eq("ensemble_rec_ref")
            ]
            if len(stage2_rows) != 1 or len(master_rows) > 1 or len(pixel_rows) != 1:
                raise RuntimeError(
                    f"registered-row-count:{system_id}:stage2={len(stage2_rows)}:"
                    f"master={len(master_rows)}:pixel={len(pixel_rows)}"
                )

            master_available = len(master_rows) == 1
            comparison_master = (
                master_rows.iloc[0]
                if master_available
                else pd.Series({"spinup_delta_t_c": 0.0})
            )
            inventory, result = core.evaluate_system(
                cohort_series,
                stage2_rows.iloc[0],
                comparison_master,
                pixel_rows.iloc[0],
                reconstruction,
            )
            inventory["master_route_row_count"] = int(len(master_rows))
            result["unresolved_master_delta_available"] = bool(master_available)
            result["reference_delta_source"] = (
                "unresolved-master-route"
                if master_available
                else "stage2-base-comparison-no-unresolved-master-row"
            )
            inventories.append(inventory)
            results.append(result)

        result_table = pd.DataFrame(results).rename(columns=RENAME)
        result_table = result_table.sort_values(["cohort", "cohort_order", "system_id"])
        reference_validation = core.validate_reference_packages(
            result_table.set_index("system_id"), stage2_table, pixels, reconstruction
        )

    inventory_table = pd.DataFrame(inventories).sort_values("system_id")
    validation_table = pd.DataFrame(reference_validation).sort_values("system_id")
    inventory_table.to_csv(OUT / "input_inventory_93.csv", index=False)
    result_table.to_csv(OUT / "static_equilibrium_attribution_93.csv", index=False)
    validation_table.to_csv(OUT / "system7_79_reference_validation.csv", index=False)

    package_formula_pass = bool(
        (validation_table["maximum_temperature_array_error_c"] <= 1.0e-5).all()
        and (validation_table["maximum_precipitation_array_error_kg_m2_month"] <= 1.0e-5).all()
        and (validation_table["reference_height_error_m"] <= 1.0e-9).all()
        and (validation_table["gradient_array_error_c_m"] <= 1.0e-12).all()
    )
    tendency_sign_pass = bool(validation_table["initial_tendency_sign_match"].all())
    all_roots_pass = bool(
        (result_table["equilibrium_residual_m_ice_yr"].abs() <= 1.0e-10).all()
        and (result_table["equilibrium_root_final_width_c"] <= 1.0e-12).all()
    )
    checks = {
        "population_count_93": len(result_table) == 93,
        "input_inventory_complete": bool(
            inventory_table[["grid_exists", "baseline_climate_exists"]].all(axis=None)
        ),
        "registered_input_rows_unambiguous": bool(
            inventory_table[
                ["stage2_main_midturnover_row_count", "climate_pixel_row_count"]
            ].eq(1).all(axis=None)
            and inventory_table["master_route_row_count"].isin([0, 1]).all()
        ),
        "reference_package_formula": package_formula_pass,
        "reference_initial_tendency_sign": tendency_sign_pass,
        "equilibrium_roots": all_roots_pass,
        "no_dynamics_run": True,
        "endpoint_targets_not_used_by_operator": True,
        "production_temperature_modified": False,
        "initial_thickness_modified": False,
        "old_output_modified": False,
        "launch_authorized": False,
    }
    group_summary = {
        "population93": describe_group(result_table),
        **{
            cohort: describe_group(group)
            for cohort, group in result_table.groupby("cohort", sort=False)
        },
    }
    missing_master = result_table.loc[
        ~result_table["unresolved_master_delta_available"], "system_id"
    ].astype(int).tolist()
    summary = {
        "schema": "lia-191-population93-static-climate-state-equilibrium-audit-v2",
        "status": (
            "read-only-static-equilibrium-audit-pass"
            if all(checks.values())
            else "read-only-static-equilibrium-audit-fail"
        ),
        "technical_revision_of": {
            "artifact": "lia_191_population93_static_climate_state_equilibrium_audit_v1",
            "failure": "three special-or-legacy contract-boundary systems have no unresolved-master route row",
            "scientific_operator_changed": False,
        },
        "population": 93,
        "operator": {
            "name": "fixed-parent-climate-area-integrated-zero-net-mb-bisection-v1",
            "inputs_allowed_to_vary_by_system": [
                "frozen Stage-2 mid-turnover melt_f, prcp_fac, temp_bias",
                "registered main-glacier W5E5 baseline climate",
                "registered reconstruction climate pixel",
                "frozen V3/F19 initial surface distribution",
            ],
            "fixed_for_all_systems": {
                "parent_climate_years": [core.START_YEAR, core.END_YEAR],
                "precipitation_anomaly_mode": "off",
                "temp_melt_c": core.T_MELT,
                "temp_solid_c": core.T_SOLID,
                "temp_liquid_c": core.T_LIQ,
                "temperature_lapse_rate_c_m": core.GRAD,
                "ice_density_kg_m3": core.RHO,
                "root_bracket_c": [core.ROOT_LOW_C, core.ROOT_HIGH_C],
                "root_iterations": core.ROOT_ITERATIONS,
            },
            "objective": "area-weighted 30-year mean surface mass balance on the frozen initial ice mask equals zero",
            "endpoint_length_targets_used": False,
            "glacier_specific_manual_tuning": False,
        },
        "checks": checks,
        "group_summary": group_summary,
        "reference_validation": reference_validation,
        "unresolved_master_delta_coverage": {
            "available": int(result_table["unresolved_master_delta_available"].sum()),
            "unavailable": len(missing_master),
            "unavailable_system_ids": missing_master,
            "policy": (
                "equilibrium operator remains Stage-2 based; missing master deltas are not "
                "imputed and Stage-2 delta 0 is used only as a labeled comparison reference"
            ),
        },
        "interpretation": [
            "The completed system7 and system79 runs conserve the climate source in the volume budget; their scientific failure is not numerical mass leakage.",
            "This audit measures the initial climate-state mismatch and its common-rule equilibrium correction. It does not claim that zero initial net mass balance alone guarantees a 50-year dynamic steady state.",
            "The discovery cohort must test the frozen operator as one class. No named glacier may receive a custom correction or search range.",
        ],
        "next_recommended_control_action": (
            "register the deterministic equilibrium-centering operator and a fixed discovery16 execution contract; keep heldout16 sealed"
            if all(checks.values())
            else "do not authorize discovery; reconcile the failed static-audit checks"
        ),
        "dynamics_run": False,
        "model_launch_performed": False,
        "launch_authorized": False,
        "accepted_count_effect": 0,
        "controller_state_modified": False,
    }
    core.write_json(OUT / "summary.json", summary)
    report = [
        "# Population-93 Static Climate-State Equilibrium Audit v2",
        "",
        f"Status: {summary['status']}",
        "",
        "- One deterministic, endpoint-independent operator was applied to all 93 systems.",
        "- Three systems without unresolved-master rows remain in the denominator and use Stage-2 base only as a labeled comparison.",
        "- No dynamics, thickness edit, old-output edit, or model launch occurred.",
        "- The equilibrium correction is a physics-derived diagnostic, not a per-glacier manual fit.",
        "- A zero initial net balance is necessary evidence, not proof of 50-year dynamic steadiness.",
        "",
        "The next control transaction may preregister discovery16 with this fixed operator. Heldout16 remains sealed.",
    ]
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
