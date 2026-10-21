"""Build the non-executable population-93 one-dimensional route package.

This module performs manifest generation and validation only.  It never imports
OGGM, instantiates a flowline model, writes a model checkpoint, or launches a
simulation.  The generated matrices remain prospective until a later,
versioned controller route explicitly authorizes a dynamic stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASE = PROJECT_ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
SOURCE_AUDIT = BASE / "lia_191_population93_static_climate_state_equilibrium_audit_v4"
CONTROLLER = BASE / "lia_191_controller_v1"
FLOWLINE_ROOT = BASE / "lia_elevation_band_core191_length_corrected_v1"
OUT = BASE / "lia_191_flowline_population93_prospective_route_v1"
CURRENT_STATE = PROJECT_ROOT / "LIA_CURRENT_STATE.json"
RECONSTRUCTION = Path(
    "E:/3spyderProject/HMA_500/Down_scale_new/out_data/oggm_climate/"
    "Ensemble/climate_oggm_ensemble_rec_ref.nc"
)

BLOCK_YEARS = 300
MAX_SPINUP_YEARS = 3000
BLOCK_COUNT = MAX_SPINUP_YEARS // BLOCK_YEARS
SPINUP_WINDOW = "1441-1470"
FORWARD_START_YEAR = 1470
FORWARD_END_YEAR = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_population() -> list[dict[str, object]]:
    inventory = read_csv(SOURCE_AUDIT / "input_inventory_93.csv")
    static_rows = {
        int(row["system_id"]): row
        for row in read_csv(SOURCE_AUDIT / "static_equilibrium_attribution_93.csv")
    }
    status_rows = {
        int(row["lia_unique_id"]): row
        for row in read_csv(CONTROLLER / "lia_191_master_status.csv")
    }
    length_rows = {
        int(row["lia_unique_id"]): row
        for row in read_csv(CONTROLLER / "core191_length_references.csv")
    }
    pixel_rows = {
        int(row["lia_unique_id"]): row
        for row in read_csv(CONTROLLER / "core191_climate_pixels_ensemble_rec_ref.csv")
    }

    if len(inventory) != 93 or len({int(r["system_id"]) for r in inventory}) != 93:
        raise ValueError("population93 inventory must contain 93 unique systems")

    rows: list[dict[str, object]] = []
    for order, inv in enumerate(inventory, start=1):
        system_id = int(inv["system_id"])
        static = static_rows[system_id]
        status = status_rows[system_id]
        lengths = length_rows[system_id]
        pixel = pixel_rows[system_id]
        rgi_id = inv["rgi_id"]
        if not all(source["rgi_id"] == rgi_id for source in (static, status, lengths, pixel)):
            raise ValueError(f"identity mismatch for system {system_id}")
        if not all(
            as_bool(inv[key])
            for key in (
                "grid_exists",
                "baseline_climate_exists",
            )
        ):
            raise ValueError(f"incomplete population93 input row for system {system_id}")
        if any(int(inv[key]) != 1 for key in (
            "stage2_main_midturnover_row_count",
            "climate_pixel_row_count",
        )):
            raise ValueError(f"ambiguous population93 input row for system {system_id}")
        master_count = int(inv["master_route_row_count"])
        if master_count not in (0, 1):
            raise ValueError(f"ambiguous master-route coverage for system {system_id}")
        if master_count == 0 and not (
            system_id in {1044, 2711, 3584}
            and static["unresolved_master_delta_available"].lower() == "false"
            and static["reference_delta_source"]
            == "stage2-base-comparison-no-unresolved-master-row"
        ):
            raise ValueError(f"unregistered missing master delta for system {system_id}")

        flowline = FLOWLINE_ROOT / str(system_id) / "elevation_band_flowline_lia.pkl.gz"
        if not flowline.is_file():
            raise FileNotFoundError(flowline)

        stage2_temp_bias = float(static["stage2_temp_bias_c"])
        equilibrium_delta = float(static["equilibrium_delta_t_c_from_stage2_base"])
        curve_id = f"p93-{order:03d}-s{system_id}"
        rows.append(
            {
                "population_order": order,
                "curve_id": curve_id,
                "system_id": system_id,
                "lia_unique_id": system_id,
                "rgi_id": rgi_id,
                "rgi_region": int(static["rgi_region"]),
                "cohort": inv["cohort"],
                "size_class": static["size_class"],
                "retreat_class": static["retreat_class"],
                "flowline_path": flowline.relative_to(PROJECT_ROOT).as_posix(),
                "flowline_sha256": sha256(flowline),
                "flowline_bytes": flowline.stat().st_size,
                "flowline_representation": "corrected-lia-elevation-band-single-flowline",
                "stage2_melt_f": float(static["stage2_melt_f"]),
                "stage2_prcp_fac": float(static["stage2_prcp_fac"]),
                "stage2_glen_a": float(status["stage2_glen_a"]),
                "stage2_temp_bias_c": stage2_temp_bias,
                "equilibrium_delta_t_c_from_stage2_base": equilibrium_delta,
                "effective_centered_temp_bias_c": stage2_temp_bias + equilibrium_delta,
                "equilibrium_operator": "fixed-parent-climate-area-integrated-zero-net-mb-bisection-v1",
                "equilibrium_operator_residual_m_ice_yr": float(static["equilibrium_residual_m_ice_yr"]),
                "lia_registered_length_m": float(lengths["lia_registered_length_m"]),
                "rgi_registered_length_m": float(lengths["rgi_registered_length_m"]),
                "modern_root_index_observational_provenance": int(lengths["modern_root_index"]),
                "climate_lat_index": int(pixel["climate_lat_index"]),
                "climate_lon_index": int(pixel["climate_lon_index"]),
                "climate_pixel_distance_m": float(pixel["climate_pixel_distance_m"]),
            }
        )
    return rows


def build_inventory(population: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = [
        "population_order", "curve_id", "system_id", "lia_unique_id", "rgi_id",
        "rgi_region", "cohort", "size_class", "retreat_class", "flowline_path",
        "flowline_sha256", "flowline_bytes", "flowline_representation",
        "climate_lat_index", "climate_lon_index", "climate_pixel_distance_m",
        "lia_registered_length_m", "rgi_registered_length_m",
        "modern_root_index_observational_provenance",
    ]
    return [{field: row[field] for field in fields} for row in population]


def build_spinup_matrix(population: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in population:
        previous = ""
        for block_index in range(1, BLOCK_COUNT + 1):
            task_id = f"spinup-{item['curve_id']}-b{block_index:02d}"
            rows.append(
                {
                    "matrix_schema": "lia-191-flowline-population93-spinup-matrix-v1",
                    "task_id": task_id,
                    "curve_id": item["curve_id"],
                    "population_order": item["population_order"],
                    "system_id": item["system_id"],
                    "lia_unique_id": item["lia_unique_id"],
                    "rgi_id": item["rgi_id"],
                    "block_index": block_index,
                    "block_start_year": (block_index - 1) * BLOCK_YEARS,
                    "block_end_year": block_index * BLOCK_YEARS,
                    "block_years": BLOCK_YEARS,
                    "maximum_spinup_years": MAX_SPINUP_YEARS,
                    "operation": "initialize" if block_index == 1 else "resume",
                    "predecessor_task_id": previous,
                    "release_condition": (
                        "explicit-versioned-launch-authorization"
                        if block_index == 1
                        else "predecessor-terminal-and-adaptive-decision-equals-continue"
                    ),
                    "spinup_climate_window": SPINUP_WINDOW,
                    "spinup_forcing_rule": "repeat-fixed-parent-climate-with-common-equilibrium-centering",
                    "precipitation_anomaly_mode": "off",
                    "stage2_melt_f": item["stage2_melt_f"],
                    "stage2_prcp_fac": item["stage2_prcp_fac"],
                    "stage2_glen_a": item["stage2_glen_a"],
                    "stage2_fs": 0.0,
                    "stage2_temp_melt_c": -1.0,
                    "stage2_temp_bias_c": item["stage2_temp_bias_c"],
                    "equilibrium_delta_t_c_from_stage2_base": item["equilibrium_delta_t_c_from_stage2_base"],
                    "effective_centered_temp_bias_c": item["effective_centered_temp_bias_c"],
                    "flowline_path": item["flowline_path"],
                    "flowline_sha256": item["flowline_sha256"],
                    "checkpoint_contract": "atomic-end-of-block-full-state-same-fingerprint-resume",
                    "stationarity_rule": "two-consecutive-300y-blocks-pass-e1-derived-state-screen",
                    "stationarity_volume_area_length_fraction_max": 0.01,
                    "stationarity_terminal_minimum_thickness_fraction_max": 0.02,
                    "equilibrium_length_role": "output-state-coordinate-not-input-target",
                    "endpoint_length_target_used": False,
                    "glacier_specific_manual_tuning": False,
                    "prospective_output_root": f"prospective/flowline_population93/spinup/{item['curve_id']}",
                    "status": "prospective-not-authorized",
                    "launch_authorized": False,
                }
            )
            previous = task_id
    return rows


def build_forward_matrix(population: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in population:
        rows.append(
            {
                "matrix_schema": "lia-191-flowline-population93-forward-matrix-v1",
                "task_id": f"forward-{item['curve_id']}-1470-2000",
                "curve_id": item["curve_id"],
                "population_order": item["population_order"],
                "system_id": item["system_id"],
                "lia_unique_id": item["lia_unique_id"],
                "rgi_id": item["rgi_id"],
                "prerequisite_spinup_state_ref": f"spinup/{item['curve_id']}/equilibrium_state.json",
                "prerequisite_rule": "qualified-adaptive-spinup-state-and-new-versioned-forward-authorization",
                "equilibrium_length_m_source": "spinup-output-state-coordinate",
                "equilibrium_length_used_as_forcing_target": False,
                "start_year": FORWARD_START_YEAR,
                "end_year": FORWARD_END_YEAR,
                "curve_years": FORWARD_END_YEAR - FORWARD_START_YEAR,
                "forward_forcing_rule": "immutable-reconstruction-anomaly-plus-spinup-centered-offset",
                "reconstruction_climate": RECONSTRUCTION.as_posix(),
                "climate_lat_index": item["climate_lat_index"],
                "climate_lon_index": item["climate_lon_index"],
                "precipitation_anomaly_mode": "off",
                "effective_centered_temp_bias_c": item["effective_centered_temp_bias_c"],
                "extra_endpoint_ramp_or_fit": "none",
                "forward_variants_per_system": 1,
                "lia_registered_length_m_observation": item["lia_registered_length_m"],
                "rgi_registered_length_m_observation": item["rgi_registered_length_m"],
                "lia_total_error_max": 0.1,
                "lia_separation_error_max": 0.2,
                "rgi_total_error_max": 0.1,
                "rgi_separation_error_max": 0.2,
                "minimum_free_downstream_m": 5000.0,
                "checkpoint_interval_years": 1,
                "resume_contract": "annual-atomic-full-state-same-fingerprint",
                "glacier_specific_manual_tuning": False,
                "prospective_curve_path": f"prospective/flowline_population93/forward/{item['curve_id']}/annual_curve.csv",
                "status": "blocked-pending-spinup-and-explicit-authorization",
                "launch_authorized": False,
            }
        )
    return rows


def build_benchmark_plan(population: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked = sorted(population, key=lambda row: (int(row["flowline_bytes"]), int(row["system_id"])))
    representatives = [
        ("small", ranked[0]),
        ("median", ranked[len(ranked) // 2]),
        ("large", ranked[-1]),
    ]
    rows: list[dict[str, object]] = [
        {
            "benchmark_id": "B0-static-package-validation",
            "stage": "manifest-only",
            "system_id": "all-93",
            "simulated_years": 0,
            "purpose": "verify identities, hashes, matrix separation, and no-launch state",
            "release_gate": "none-read-only",
            "status": "completed-by-package-builder",
            "launch_authorized": False,
        },
        {
            "benchmark_id": "B1-checkpoint-contract-fixture",
            "stage": "serialization-fixture",
            "system_id": "synthetic-state-only",
            "simulated_years": 0,
            "purpose": "prove atomic publication, exact restore, fingerprint rejection, and append-only resume bookkeeping",
            "release_gate": "new-versioned-preflight-package",
            "status": "planned-not-authorized",
            "launch_authorized": False,
        },
    ]
    for size_label, item in representatives:
        rows.append(
            {
                "benchmark_id": f"B2-spinup-300y-{size_label}",
                "stage": "single-spinup-block",
                "system_id": item["system_id"],
                "simulated_years": BLOCK_YEARS,
                "purpose": "measure wall time, CPU time, steps, minimum CFL step, peak memory, and checkpoint bytes",
                "release_gate": "explicit-versioned-dynamic-benchmark-authorization",
                "status": "planned-not-authorized",
                "launch_authorized": False,
            }
        )
    rows.extend(
        [
            {
                "benchmark_id": "B3-median-resume-equivalence",
                "stage": "second-300y-resume-block",
                "system_id": representatives[1][1]["system_id"],
                "simulated_years": BLOCK_YEARS,
                "purpose": "measure resume overhead and compare state-content hashes with the registered uninterrupted control",
                "release_gate": "B2-median-pass-and-separate-versioned-resume-authorization",
                "status": "planned-not-authorized",
                "launch_authorized": False,
            },
            {
                "benchmark_id": "B4-population-wave-plan",
                "stage": "resource-model-only",
                "system_id": "all-93",
                "simulated_years": 0,
                "purpose": "derive concurrency and wall-time bounds from measured B2/B3 p50/p90, not legacy timings",
                "release_gate": "B2-and-B3-terminal-evidence",
                "status": "planned-not-authorized",
                "launch_authorized": False,
            },
        ]
    )
    return rows


def build_runtime_evidence() -> list[dict[str, object]]:
    root = BASE / "identified_experiments_v1" / "e1_window_state"
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("system_*/anchor_point_*/window_*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        timing = summary["timing_seconds"]
        rows.append(
            {
                "system_id": summary["lia_unique_id"],
                "spinup_window": "-".join(str(v) for v in summary["spinup_window"]),
                "spinup_years": summary["spinup_years"],
                "spinup_runtime_seconds": timing["spinup"],
                "forward_years": 2000 - summary["start_year"],
                "forward_runtime_seconds": timing["forward"],
                "status": summary["status"],
                "comparability": "historical-1d-evidence-only-rebenchmark-required",
                "source_summary": summary_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
    return rows


def validate_matrices(
    population: list[dict[str, object]],
    spinup: list[dict[str, object]],
    forward: list[dict[str, object]],
) -> None:
    if len(population) != 93 or len(spinup) != 93 * BLOCK_COUNT or len(forward) != 93:
        raise ValueError("unexpected population or matrix row count")
    if {row["task_id"] for row in spinup} & {row["task_id"] for row in forward}:
        raise ValueError("spin-up and forward task identifiers must remain disjoint")
    if any(row["launch_authorized"] for row in spinup + forward):
        raise ValueError("prospective matrices may not authorize launch")
    if any(row["endpoint_length_target_used"] for row in spinup):
        raise ValueError("spin-up may not target an observed length")
    if any(row["equilibrium_length_used_as_forcing_target"] for row in forward):
        raise ValueError("equilibrium length must remain an output state coordinate")
    if any(row["glacier_specific_manual_tuning"] for row in spinup + forward):
        raise ValueError("glacier-specific manual tuning is prohibited")


def write_design(path: Path) -> None:
    path.write_text(
        """# Prospective population-93 one-dimensional OGGM flowline route v1

Status: **design and manifest material only; no dynamic launch is authorized**.

## Scope and population

The route reuses the frozen 93-system extension pool from the population-level
equilibrium attribution audit. This is the comparable approximately-100-curve
population: one deterministic curve per `lia_unique_id`, no substitutions,
duplicates, or glacier-specific rescues. The corrected single elevation-band
flowline is the 1D model geometry; spatial LIA and RGI lengths remain distinct
observational references.

## Spin-up matrix

Spin-up is an independent matrix of 93 systems by ten conditional 300-year
blocks. Block 1 initializes from the immutable corrected 1D geometry. Blocks
2-10 are same-fingerprint resumes from the preceding atomic full-state
checkpoint. A block is released only if the prior adjudication says
`continue`; unused later blocks are recorded as skipped. The hard proposed
ceiling is 3000 years.

The common forcing rule repeats the fixed 1441-1470 parent climate with
precipitation anomalies off and the population-level equilibrium-centering
operator already registered by the read-only population93 audit. No observed
length enters this operator.

The proposed online stationarity screen inherits the earlier E1 state
thresholds: absolute fractional endpoint change and normalized trend no larger
than 1% for volume, area, and length, and 2% for terminal and minimum
thickness. Two consecutive 300-year blocks must pass. Length is used only for
state stationarity; `equilibrium_length_m` is written as an output coordinate
and is never a forcing or calibration target. Failure to converge by 3000
years is a population result, not permission to tune the glacier.

Every block must additionally retain finite/nonnegative geometry, zero
forbidden-boundary ice, the registered free-domain rule, immutable source
hashes, and exact checkpoint-contract restoration. A technical failure may be
considered only under a later exact-contract retry decision. A scientific
failure stays in the denominator.

## Forward matrix

The forward matrix is a separate 93-row artifact. A row can consume only a
qualified spin-up state fingerprint and its output coordinates. It runs one
1470-2000 curve using the immutable reconstruction anomaly plus the same
spin-up-centered offset, with precipitation anomalies off. There is no extra
endpoint ramp, per-system bracket, interpolation, or endpoint fitting.

The LIA peak and year-2000 RGI lengths are scored only after the curve exists,
under the frozen 10% total and 20% separation gates. Annual atomic checkpoints
and same-fingerprint resume are required. A formal pass would still need a
separately authorized exact reproduction before any acceptance action.

## Runtime benchmark plan

Legacy 1D E1 timings are retained as evidence but not used as production
estimates because their geometry, parameters, software state, and forcing
contract differ. After explicit authorization, benchmark the smallest,
median, and largest flowline files for one exact 300-year block, then validate
one 300-year resume on the median representative. Record wall/CPU time,
accepted and rejected steps, minimum CFL-required step, peak memory, checkpoint
size, and resume overhead. Only measured p50/p90 results may set population
wave size and wall-time bounds.

## Authorization boundary

This package does not modify `LIA_PROJECT_FACTS.md`, `LIA_CURRENT_STATE.json`,
the accepted registry, production climate, model geometry, or old output. It
does not resolve the current discovery15 retry decision. Any static fixture,
dynamic benchmark, spin-up block, forward curve, or reproduction requires its
own prospective evidence, versioned route transaction, validator pass, and
explicit launch authorization.
""",
        encoding="utf-8",
    )


def build() -> None:
    state = json.loads(CURRENT_STATE.read_text(encoding="utf-8"))
    if state.get("next_action", {}).get("launch_authorized") is not False:
        raise ValueError("builder requires the canonical no-launch state")

    population = load_population()
    inventory = build_inventory(population)
    spinup = build_spinup_matrix(population)
    forward = build_forward_matrix(population)
    benchmarks = build_benchmark_plan(population)
    runtime = build_runtime_evidence()
    validate_matrices(population, spinup, forward)

    OUT.mkdir(parents=True, exist_ok=False)
    write_csv(OUT / "evidence_inventory_93.csv", inventory, list(inventory[0]))
    write_csv(OUT / "spinup_matrix_93x10.csv", spinup, list(spinup[0]))
    write_csv(OUT / "forward_matrix_93.csv", forward, list(forward[0]))
    write_csv(OUT / "runtime_benchmark_plan.csv", benchmarks, list(benchmarks[0]))
    write_csv(OUT / "legacy_1d_runtime_evidence.csv", runtime, list(runtime[0]))
    write_design(OUT / "ROUTE_DESIGN.md")

    sources = [
        CURRENT_STATE,
        PROJECT_ROOT / "LIA_PROJECT_FACTS.md",
        PROJECT_ROOT / "LIA_EXECUTION_CONTRACT.md",
        SOURCE_AUDIT / "summary.json",
        SOURCE_AUDIT / "input_inventory_93.csv",
        SOURCE_AUDIT / "static_equilibrium_attribution_93.csv",
        CONTROLLER / "lia_191_master_status.csv",
        CONTROLLER / "core191_length_references.csv",
        CONTROLLER / "core191_climate_pixels_ensemble_rec_ref.csv",
    ]
    source_manifest = {
        "schema": "lia-191-flowline-population93-prospective-source-manifest-v1",
        "sources": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sources
        ],
        "flowline_source_count": 93,
        "flowline_hashes_embedded_in": "evidence_inventory_93.csv",
        "reconstruction_path_reference": RECONSTRUCTION.as_posix(),
        "reconstruction_hash_deferred_to_prelaunch_fixture": True,
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
    )

    generated = [
        OUT / "ROUTE_DESIGN.md",
        OUT / "evidence_inventory_93.csv",
        OUT / "spinup_matrix_93x10.csv",
        OUT / "forward_matrix_93.csv",
        OUT / "runtime_benchmark_plan.csv",
        OUT / "legacy_1d_runtime_evidence.csv",
        OUT / "source_manifest.json",
    ]
    summary = {
        "schema": "lia-191-flowline-population93-prospective-route-v1",
        "status": "prospective-design-and-matrices-complete-no-launch-authority",
        "population": 93,
        "curve_count": 93,
        "curve_policy": "one-common-rule-curve-per-system",
        "population_basis": "frozen population93 extension pool",
        "spinup_matrix_rows": len(spinup),
        "spinup_block_years": BLOCK_YEARS,
        "spinup_maximum_years": MAX_SPINUP_YEARS,
        "spinup_maximum_blocks": BLOCK_COUNT,
        "spinup_forward_matrices_separate": True,
        "equilibrium_length_role": "output-state-coordinate-not-input-target",
        "glacier_specific_manual_tuning": False,
        "forward_matrix_rows": len(forward),
        "forward_curves_per_system": 1,
        "benchmark_plan_rows": len(benchmarks),
        "legacy_runtime_evidence_rows": len(runtime),
        "canonical_state_id": state["state_id"],
        "canonical_state_status": state["current_focus"]["state"],
        "canonical_launch_authorized": False,
        "model_run_performed": False,
        "dynamic_launch_authorized": False,
        "current_state_modified": False,
        "project_facts_modified": False,
        "accepted_count_effect": 0,
        "generated_artifacts": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in generated
        ],
        "next_legal_action": (
            "read-only review of this prospective package; any fixture, benchmark, "
            "spin-up, or forward launch requires a new versioned route and explicit authorization"
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def check() -> None:
    population = load_population()
    spinup = read_csv(OUT / "spinup_matrix_93x10.csv")
    forward = read_csv(OUT / "forward_matrix_93.csv")
    if len(spinup) != 93 * BLOCK_COUNT or len(forward) != 93:
        raise ValueError("persisted matrix row count mismatch")
    if any(row["launch_authorized"].lower() != "false" for row in spinup + forward):
        raise ValueError("persisted prospective matrix authorizes launch")
    if any(row["glacier_specific_manual_tuning"].lower() != "false" for row in spinup + forward):
        raise ValueError("persisted matrix permits manual tuning")
    if any(row["endpoint_length_target_used"].lower() != "false" for row in spinup):
        raise ValueError("persisted spin-up matrix uses length targets")
    if any(row["equilibrium_length_used_as_forcing_target"].lower() != "false" for row in forward):
        raise ValueError("persisted forward matrix uses equilibrium length as a target")
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    if summary["population"] != len(population) or summary["dynamic_launch_authorized"] is not False:
        raise ValueError("persisted summary mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate an existing package")
    args = parser.parse_args()
    check() if args.check else build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
