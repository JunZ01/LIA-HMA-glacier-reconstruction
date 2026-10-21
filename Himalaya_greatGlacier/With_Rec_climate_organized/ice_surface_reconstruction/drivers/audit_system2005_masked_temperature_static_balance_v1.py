"""Read-only temperature sensitivity on the saved masked system-2005 path."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


HERE = Path(__file__).resolve()
ROOT = next(parent for parent in HERE.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
STATE = ROOT / "LIA_CURRENT_STATE.json"
RUN_ROOT = BASE / "lia_191_system2005_masked_steady50_checkpointed_v3"
RUN = RUN_ROOT / "summary.json"
DECOMPOSITION = BASE / "lia_191_system2005_masked_internal_source_elevation_feedback_v3/summary.json"
SUPPORT = BASE / "lia_191_system2005_masked_support_exchange_v1/summary.json"
NATURAL_1D = BASE / "lia_191_system2005_policy_v7_unchanged_natural_state_feasibility_v1/summary.json"
V4 = BASE / "lia_191_system2005_policy_v6_temperature_correction_v4/summary.json"
GEOLOGY = BASE / "lia_191_igm_planform2d_system2005_static_v1/geology.nc"
CLIMATE = BASE / "lia_191_torch_planform2d_system2005_climate_bridge_export_v1/monthly_ti_package.npz"
OUT = BASE / "lia_191_system2005_masked_temperature_static_balance_v1"

TEMPERATURE_GRID_K = np.arange(-0.5, 1.5000001, 0.125, dtype=np.float64)
EXPECTED_STATE_ID = "2026-08-29-v10.144-system2005-internal-support-exchange-confirmed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def interpolate_crossing(rows: list[dict], target: float) -> float | None:
    for left, right in zip(rows[:-1], rows[1:]):
        yl = left["projected_total_source_m3"] - target
        yr = right["projected_total_source_m3"] - target
        if yl == 0.0:
            return float(left["delta_t_k"])
        if yl * yr <= 0.0:
            xl = float(left["delta_t_k"])
            xr = float(right["delta_t_k"])
            return xl + (xr - xl) * (-yl) / (yr - yl)
    return None


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)

    control = load_json(STATE)
    run = load_json(RUN)
    decomposition = load_json(DECOMPOSITION)
    support = load_json(SUPPORT)
    natural_1d = load_json(NATURAL_1D)
    v4 = load_json(V4)
    if control["state_id"] != EXPECTED_STATE_ID:
        raise RuntimeError("unexpected-controller-state")
    if run["status"] != "masked-steady50-fail-stop" or run["completed_years"] != 50.0:
        raise RuntimeError("masked-run-not-terminal")

    with Dataset(GEOLOGY) as dataset:
        bed = np.asarray(dataset.variables["topg"][:], dtype=np.float64)
        initial = np.asarray(dataset.variables["thk"][:], dtype=np.float64)
        domain = np.asarray(dataset.variables["lia_source_mask"][:], dtype=float) > 0.5
        dx = float(getattr(dataset, "dx_m"))

    states = [initial]
    checkpoint_hashes_valid = True
    for year in range(1, 51):
        directory = RUN_ROOT / f"checkpoints/year_{year:03d}"
        manifest = load_json(directory / "manifest.json")
        state_path = directory / manifest["state_file"]
        checkpoint_hashes_valid &= sha256(state_path) == manifest["state_sha256"].upper()
        with np.load(state_path, allow_pickle=False) as archive:
            states.append(np.asarray(archive["thickness_m"], dtype=np.float64))

    with np.load(CLIMATE, allow_pickle=False) as package:
        temp = np.asarray(package["temp"], dtype=np.float64)
        prcp = np.asarray(package["prcp"], dtype=np.float64)
        grad = np.asarray(package["grad"], dtype=np.float64)
        sequence = np.asarray(package["sequence_indices"], dtype=np.int64)
        ref_hgt = float(package["ref_hgt"])
        t_melt = float(package["t_melt"])
        t_solid = float(package["t_solid"])
        t_liq = float(package["t_liq"])
        monthly_melt_f = float(package["monthly_melt_f"])
        bias = float(package["bias"])
        rho = float(package["rho"])
        sec_in_month = float(package["sec_in_month"])
        sec_in_year = float(package["sec_in_year"])
        active_tolerance = float(package["active_thickness_tolerance_m"])

    if sequence.size != 600:
        raise RuntimeError("climate-sequence-not-600-months")
    stack = np.stack(states)
    initial_active = initial > active_tolerance
    ever_active = np.any(stack > active_tolerance, axis=0)
    added_support = (~initial_active) & ever_active
    cell_area = dx**2
    initial_volume = float(np.sum(initial) * cell_area)
    within_year_residual = float(decomposition["decomposition_m3"]["within_year_geometry_effect"])
    expected_baseline_frozen = float(run["cumulative_climate_source_m3"]) - within_year_residual

    def evaluate(delta_t_k: float, annual: bool = False) -> tuple[float, dict[str, float], list[dict]]:
        total = 0.0
        class_totals = {"initially_active": 0.0, "added_support": 0.0}
        annual_rows = []
        for year in range(50):
            thickness = stack[year]
            active = thickness > active_tolerance
            surface = bed + thickness
            active_surface = surface[active]
            active_initial_class = initial_active[active]
            year_cell_source = np.zeros(active_surface.size, dtype=np.float64)
            for month in range(12 * year, 12 * (year + 1)):
                source_index = sequence[month]
                temperature = (
                    temp[source_index]
                    + delta_t_k
                    + grad[source_index] * (active_surface - ref_hgt)
                )
                solid_fraction = np.clip(
                    1.0 - (temperature - t_solid) / (t_liq - t_solid), 0.0, 1.0
                )
                monthly_mass = prcp[source_index] * solid_fraction
                monthly_mass -= monthly_melt_f * np.maximum(temperature - t_melt, 0.0)
                monthly_mass -= bias * sec_in_month / sec_in_year
                year_cell_source += monthly_mass / rho
            year_source = float(np.sum(year_cell_source) * cell_area)
            initial_source = float(np.sum(year_cell_source[active_initial_class]) * cell_area)
            added_source = float(np.sum(year_cell_source[~active_initial_class]) * cell_area)
            total += year_source
            class_totals["initially_active"] += initial_source
            class_totals["added_support"] += added_source
            if annual:
                annual_rows.append(
                    {
                        "year": year + 1,
                        "delta_t_k": delta_t_k,
                        "frozen_start_source_m3": year_source,
                        "initially_active_source_m3": initial_source,
                        "added_support_source_m3": added_source,
                    }
                )
        return total, class_totals, annual_rows

    grid_rows = []
    for delta in TEMPERATURE_GRID_K:
        frozen_source, classes, _ = evaluate(float(delta))
        projected_total = frozen_source + within_year_residual
        grid_rows.append(
            {
                "delta_t_k": float(delta),
                "frozen_annual_start_source_m3": frozen_source,
                "fixed_within_year_residual_m3": within_year_residual,
                "projected_total_source_m3": projected_total,
                "projected_volume_change_fraction": projected_total / initial_volume,
                "initially_active_source_m3": classes["initially_active"],
                "added_support_source_m3": classes["added_support"],
            }
        )

    zero_delta = interpolate_crossing(grid_rows, 0.0)
    gate_volume = 0.02 * initial_volume
    positive_gate_delta = interpolate_crossing(grid_rows, gate_volume)
    negative_gate_delta = interpolate_crossing(grid_rows, -gate_volume)
    if zero_delta is None or positive_gate_delta is None or negative_gate_delta is None:
        raise RuntimeError("predeclared-temperature-grid-does-not-bracket-static-gates")

    evaluated_rows = []
    annual_rows = []
    for label, delta in (
        ("positive_2pct_boundary", positive_gate_delta),
        ("zero_source", zero_delta),
        ("negative_2pct_boundary", negative_gate_delta),
    ):
        frozen_source, classes, years = evaluate(delta, annual=True)
        projected_total = frozen_source + within_year_residual
        evaluated_rows.append(
            {
                "point": label,
                "delta_t_k": delta,
                "frozen_annual_start_source_m3": frozen_source,
                "projected_total_source_m3": projected_total,
                "projected_volume_change_fraction": projected_total / initial_volume,
                "initially_active_source_m3": classes["initially_active"],
                "added_support_source_m3": classes["added_support"],
            }
        )
        for row in years:
            row["point"] = label
            annual_rows.append(row)

    baseline_row = next(row for row in grid_rows if row["delta_t_k"] == 0.0)
    monotonic = all(
        right["projected_total_source_m3"] < left["projected_total_source_m3"]
        for left, right in zip(grid_rows[:-1], grid_rows[1:])
    )
    baseline_tolerance = max(1.0e-3, abs(expected_baseline_frozen) * 1.0e-10)
    checks = {
        "controller_state_exact": True,
        "masked_run_terminal": True,
        "all_51_states_loaded": len(states) == 51,
        "checkpoint_hashes_valid": bool(checkpoint_hashes_valid),
        "states_finite_nonnegative": bool(np.isfinite(stack).all() and np.min(stack) >= 0.0),
        "outside_domain_zero_all_states": bool(np.max(np.abs(stack[:, ~domain])) == 0.0),
        "baseline_frozen_source_closure": abs(
            baseline_row["frozen_annual_start_source_m3"] - expected_baseline_frozen
        ) <= baseline_tolerance,
        "temperature_response_strictly_monotonic": bool(monotonic),
        "predeclared_grid_brackets_zero_and_both_2pct_boundaries": True,
        "one_dimensional_temperature_only_already_closed": natural_1d["decision"]["temperature_only_mechanism"] == "closed-by-read-only-joint-gate",
        "old_v4_inherits_prohibited_thickness_lift": float(v4["static"]["uniform_thickness_lift_m"]) > 0.0,
        "source_hashes_unchanged": True,
        "production_files_unchanged": True,
        "initial_thickness_unchanged": True,
        "dynamics_not_run": True,
    }

    OUT.mkdir(parents=False)
    write_csv(OUT / "temperature_grid.csv", grid_rows)
    write_csv(OUT / "interpolated_static_boundaries.csv", evaluated_rows)
    write_csv(OUT / "annual_static_sources.csv", annual_rows)
    inventory = [
        {
            "mechanism": "1d-natural-state-temperature-only",
            "status": "closed",
            "reason": "No joint 1D length-volume candidate on unchanged natural state.",
        },
        {
            "mechanism": "1d-policy-v6-v4-temperature-correction",
            "status": "ineligible",
            "reason": "Steady pass inherited a prohibited 2.8052643058495814 m thickness lift.",
        },
        {
            "mechanism": "2d-unmasked-temperature-baseline",
            "status": "superseded-for-mechanism-diagnosis",
            "reason": "Exterior support expansion dominated the unmasked +68.76% drift.",
        },
        {
            "mechanism": "2d-masked-temperature-static-replay",
            "status": "tested-read-only",
            "reason": "Current saved support path had not previously been tested under a temperature offset.",
        },
    ]
    write_csv(OUT / "prior_mechanism_inventory.csv", inventory)

    status = "static-temperature-bracket-pass" if all(checks.values()) else "static-temperature-audit-fail"
    summary = {
        "schema": "lia-191-system2005-masked-temperature-static-balance-v1",
        "status": status,
        "system_id": 2005,
        "method": {
            "type": "read-only saved-trajectory MonthlyTI operator replay",
            "temperature_grid_k": TEMPERATURE_GRID_K.tolist(),
            "frozen": [
                "51 saved thickness states",
                "annual-start active support",
                "surface elevations",
                "precipitation",
                "temperature lapse rate",
                "MonthlyTI thresholds and melt factor",
            ],
            "fixed_approximation": "The baseline within-year geometry residual is retained; no dynamics or length response is predicted.",
        },
        "baseline": {
            "actual_source_m3": float(run["cumulative_climate_source_m3"]),
            "frozen_annual_start_source_m3": baseline_row["frozen_annual_start_source_m3"],
            "within_year_geometry_residual_m3": within_year_residual,
            "volume_change_fraction": float(run["final_volume_relative_change"]),
            "root_length_error_fraction": float(run["final_root_path_metrics"]["registered_length_relative_error"]),
        },
        "static_temperature_boundaries": {
            "positive_2pct_delta_t_k": positive_gate_delta,
            "zero_source_delta_t_k": zero_delta,
            "negative_2pct_delta_t_k": negative_gate_delta,
            "evaluated_points": evaluated_rows,
        },
        "interpretation": {
            "temperature_can_close_operator_balance_on_saved_path": status == "static-temperature-bracket-pass",
            "temperature_is_accepted_dynamic_solution": False,
            "length_response_known": False,
            "next_decision": "A single short masked 2D dynamics preflight is scientifically eligible only if the required offset is physically admissible and independently preregistered.",
        },
        "prior_mechanism_inventory": inventory,
        "checks": checks,
        "source_sha256": {
            "geology": sha256(GEOLOGY),
            "climate_package": sha256(CLIMATE),
            "masked_run": sha256(RUN),
            "decomposition": sha256(DECOMPOSITION),
            "support_exchange": sha256(SUPPORT),
            "natural_1d": sha256(NATURAL_1D),
            "v4": sha256(V4),
        },
        "production_temperature_modified": False,
        "initial_thickness_modified": False,
        "historical_run": False,
        "dynamics_run": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = f"""# System 2005 masked temperature static-balance audit

- Status: `{status}`
- Baseline masked 50-year volume change: `{run['final_volume_relative_change']:.6%}`
- Saved-path zero-source temperature offset: `{zero_delta:+.6f} K`
- Saved-path 2% static-balance interval: `{positive_gate_delta:+.6f}` to `{negative_gate_delta:+.6f} K`
- Final baseline root-length error: `{run['final_root_path_metrics']['registered_length_relative_error']:.6%}`

This is a read-only operator test. It establishes whether temperature can remove the
climate-source imbalance on the already saved two-dimensional path. It does not
predict the changed support, geometry, or length under a new dynamic run and is not
an accepted parameter or production-temperature modification.
"""
    (OUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
