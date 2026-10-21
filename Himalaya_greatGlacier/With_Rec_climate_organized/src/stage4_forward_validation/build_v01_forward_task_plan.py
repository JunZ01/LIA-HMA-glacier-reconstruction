"""Build the Stage 4 v0.1 LIA initial-state forward-validation dry-run plan.

The planner performs joins and path checks only. It does not launch OGGM, does
not write to glacier directories, and does not modify Stage 1/2 calibration
artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "019f730c-ee32-74d0-8b77-fdb7832f6a06"


def default_result_root() -> Path:
    return (REPO_ROOT / ".." / ".." / "validation_data" / "results").resolve()


def default_stage2_root() -> Path:
    return REPO_ROOT / "outputs" / RUN_ID / "stage2_freeze_stage3_routing_v0"


def default_stage3_root() -> Path:
    return REPO_ROOT / "outputs" / RUN_ID / "stage3_initial_state_pilot_v0"


def default_release_root() -> Path:
    return (
        default_result_root()
        / "paleo_surface_validation_v0"
        / "initial_state_release_v01_core191"
    )


def default_output_root() -> Path:
    return (
        default_result_root()
        / "paleo_surface_validation_v0"
        / "stage4_lia_v01_forward_validation_dryrun"
    )


def default_stage3_config_path() -> Path:
    return REPO_ROOT / "config" / "stage3_initial_state_pilot_v0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=default_release_root())
    parser.add_argument("--stage2-root", type=Path, default=default_stage2_root())
    parser.add_argument("--stage3-root", type=Path, default=default_stage3_root())
    parser.add_argument("--stage3-config", type=Path, default=default_stage3_config_path())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--ensemble-role", default="mid-turnover")
    parser.add_argument("--climate-id", default="ensemble_rec_ref")
    parser.add_argument(
        "--manifest-name",
        default="production_ready_default_one_member_per_system_v01.csv",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any selected task has dry-run failures.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:-6] / rgi_id[:-3] / rgi_id


def resolve_stage0_gdir_root(stage0_source_dir: Path) -> Path:
    candidates = (
        stage0_source_dir / "oggm_workdir" / "per_glacier",
        stage0_source_dir / "per_glacier",
        stage0_source_dir / "oggm_workdir",
        stage0_source_dir,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def required_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def as_int_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().astype(int)


def path_exists(value: object) -> bool:
    if pd.isna(value):
        return False
    return Path(str(value)).exists()


def first_or_empty(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    values = frame[column].dropna().astype(str).unique()
    return ";".join(sorted(values))


def build_plan(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    release_root = args.release_root.resolve()
    stage2_root = args.stage2_root.resolve()
    stage3_root = args.stage3_root.resolve()
    output_root = args.output_root.resolve()
    stage3_config_path = args.stage3_config.resolve()
    stage3_config = read_json(stage3_config_path)
    stage0_source_dir = resolve_repo_path(stage3_config["stage0_source_dir"])
    stage0_gdir_root = resolve_stage0_gdir_root(stage0_source_dir)

    manifest_path = release_root / args.manifest_name
    coverage_path = release_root / "initial_state_v01_system_coverage.csv"
    qc_routes_path = release_root / "initial_state_v01_qc_routes.csv"
    stage2_system_path = stage2_root / "stage2_system_ensemble.csv"
    stage2_member_path = stage2_root / "stage2_member_ensemble.csv"
    routing_path = stage2_root / "stage3_routing_245.csv"
    climate_registry_path = stage3_root / "climate_registry_6.csv"

    manifest = read_csv(manifest_path)
    coverage = read_csv(coverage_path)
    qc_routes = read_csv(qc_routes_path)
    stage2_system = read_csv(stage2_system_path)
    stage2_member = read_csv(stage2_member_path)
    routing = read_csv(routing_path)
    climate_registry = read_csv(climate_registry_path)

    required_columns(
        manifest,
        {
            "lia_unique_id",
            "main_rgi_id",
            "bed_scenario",
            "input_pickle",
            "volume_change_fraction_5yr_zero_mb",
            "area_change_fraction_5yr_zero_mb",
        },
        "initial-state manifest",
    )
    required_columns(
        coverage,
        {"lia_unique_id", "system_initial_state_status"},
        "system coverage",
    )
    required_columns(
        qc_routes,
        {"lia_unique_id", "bed_scenario", "initial_state_qc_status"},
        "QC routes",
    )
    required_columns(
        stage2_system,
        {
            "lia_unique_id",
            "ensemble_role",
            "ensemble_id",
            "melt_f",
            "temp_bias",
            "glen_a_pa_3_s_1",
            "fs",
            "status",
        },
        "Stage 2 system ensemble",
    )
    required_columns(
        stage2_member,
        {
            "lia_unique_id",
            "ensemble_role",
            "rgi_id",
            "prcp_fac",
            "melt_f",
            "temp_bias",
            "glen_a_pa_3_s_1",
            "fs",
            "status",
        },
        "Stage 2 member ensemble",
    )
    required_columns(
        routing,
        {"lia_unique_id"},
        "Stage 3 routing ledger",
    )
    required_columns(
        climate_registry,
        {
            "climate_id",
            "path",
            "sha256",
            "bytes",
            "time_start_year",
            "time_end_year",
            "monthly_count",
        },
        "climate registry",
    )

    for frame in (manifest, coverage, qc_routes, stage2_system, stage2_member, routing):
        frame["lia_unique_id"] = as_int_id(frame["lia_unique_id"])

    manifest = manifest.sort_values(["lia_unique_id", "bed_scenario"]).copy()
    coverage_subset = coverage[
        [
            "lia_unique_id",
            "system_initial_state_status",
            "production_ready_member_count",
            "dynamic_relaxation_review_member_count",
            "width_remap_review_member_count",
            "input_missing_member_count",
        ]
    ].copy()
    manifest = manifest.merge(
        coverage_subset,
        on="lia_unique_id",
        how="left",
        validate="one_to_one",
    )

    selected_system = stage2_system[
        stage2_system["ensemble_role"].astype(str).eq(args.ensemble_role)
    ].copy()
    selected_system = selected_system.add_prefix("stage2_system_")
    selected_system = selected_system.rename(
        columns={"stage2_system_lia_unique_id": "lia_unique_id"}
    )

    member_selected = stage2_member[
        stage2_member["ensemble_role"].astype(str).eq(args.ensemble_role)
    ].copy()
    member_summary = (
        member_selected.groupby("lia_unique_id")
        .agg(
            stage2_member_row_count=("rgi_id", "size"),
            stage2_member_rgi_ids=("rgi_id", lambda x: ";".join(sorted(map(str, x)))),
            stage2_member_source_gdir_paths=(
                "rgi_id",
                lambda x: ";".join(
                    str(gdir_path(stage0_gdir_root, str(rgi_id)))
                    for rgi_id in sorted(map(str, x))
                ),
            ),
            stage2_member_prcp_fac_min=("prcp_fac", "min"),
            stage2_member_prcp_fac_max=("prcp_fac", "max"),
            stage2_member_melt_f_min=("melt_f", "min"),
            stage2_member_melt_f_max=("melt_f", "max"),
            stage2_member_temp_bias_min=("temp_bias", "min"),
            stage2_member_temp_bias_max=("temp_bias", "max"),
            stage2_member_statuses=("status", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )

    climate_rows = climate_registry[
        climate_registry["climate_id"].astype(str).eq(args.climate_id)
    ].copy()
    if climate_rows.empty:
        climate_row = {}
    else:
        climate_row = climate_rows.iloc[0].to_dict()

    routing_cols = ["lia_unique_id"]
    for column in (
        "farinotti_core_191",
        "region_code",
        "velocity_evidence_class",
        "velocity_strict_anomaly",
        "debris_class",
        "debris_fraction",
    ):
        if column in routing.columns:
            routing_cols.append(column)
    routing_subset = routing[routing_cols].copy()
    routing_subset = routing_subset.add_prefix("routing_")
    routing_subset = routing_subset.rename(
        columns={"routing_lia_unique_id": "lia_unique_id"}
    )

    plan = manifest.merge(
        selected_system,
        on="lia_unique_id",
        how="left",
        validate="one_to_one",
    ).merge(
        member_summary,
        on="lia_unique_id",
        how="left",
        validate="one_to_one",
    ).merge(
        routing_subset,
        on="lia_unique_id",
        how="left",
        validate="one_to_one",
    )

    plan.insert(0, "task_id", [
        f"s{int(row.lia_unique_id)}-{args.ensemble_role}-{args.climate_id}"
        for row in plan.itertuples()
    ])
    plan["selected_ensemble_role"] = args.ensemble_role
    plan["selected_climate_id"] = args.climate_id
    plan["climate_path"] = str(climate_row.get("path", ""))
    plan["climate_sha256"] = str(climate_row.get("sha256", ""))
    plan["climate_time_start_year"] = climate_row.get("time_start_year", "")
    plan["climate_time_end_year"] = climate_row.get("time_end_year", "")
    plan["climate_monthly_count"] = climate_row.get("monthly_count", "")
    plan["initial_state_pickle_exists"] = plan["input_pickle"].map(path_exists)
    plan["climate_path_exists"] = path_exists(climate_row.get("path", ""))
    plan["has_stage2_system_parameter"] = plan[
        "stage2_system_ensemble_id"
    ].notna()
    plan["has_stage2_member_parameters"] = plan[
        "stage2_member_row_count"
    ].fillna(0).astype(int).gt(0)
    plan["stage0_source_dir"] = str(stage0_source_dir)
    plan["stage0_source_dir_exists"] = stage0_source_dir.exists()
    plan["stage0_gdir_root"] = str(stage0_gdir_root)
    plan["stage0_gdir_root_exists"] = stage0_gdir_root.exists()
    plan["stage2_member_source_gdir_all_exist"] = plan[
        "stage2_member_source_gdir_paths"
    ].fillna("").map(
        lambda value: bool(value)
        and all(Path(item).exists() for item in str(value).split(";") if item)
    )
    plan["has_routing_row"] = plan.filter(regex=r"^routing_").notna().any(axis=1)
    plan["system_initial_state_status"] = plan[
        "system_initial_state_status"
    ].fillna("missing-coverage-row")

    failure_columns = [
        "missing_initial_state_pickle",
        "missing_climate_file",
        "missing_stage2_system_parameter",
        "missing_stage2_member_parameter",
        "missing_stage2_member_source_gdir",
        "missing_routing_row",
        "non_default_review_category",
    ]
    plan["missing_initial_state_pickle"] = ~plan["initial_state_pickle_exists"]
    plan["missing_climate_file"] = ~plan["climate_path_exists"]
    plan["missing_stage2_system_parameter"] = ~plan["has_stage2_system_parameter"]
    plan["missing_stage2_member_parameter"] = ~plan["has_stage2_member_parameters"]
    plan["missing_stage2_member_source_gdir"] = ~plan[
        "stage2_member_source_gdir_all_exist"
    ]
    plan["missing_routing_row"] = ~plan["has_routing_row"]
    plan["non_default_review_category"] = ~plan[
        "system_initial_state_status"
    ].eq("has-production-ready-default")
    plan["dryrun_ready"] = ~plan[failure_columns].any(axis=1)

    failures = []
    for row in plan.itertuples(index=False):
        for column, failure_class in (
            ("missing_initial_state_pickle", "missing-initial-state"),
            ("missing_climate_file", "missing-climate"),
            ("missing_stage2_system_parameter", "missing-stage2-system-parameter"),
            ("missing_stage2_member_parameter", "missing-stage2-member-parameter"),
            ("missing_stage2_member_source_gdir", "missing-stage2-member-source-gdir"),
            ("missing_routing_row", "missing-routing-row"),
            ("non_default_review_category", "review-category-excluded"),
        ):
            if bool(getattr(row, column)):
                failures.append(
                    {
                        "task_id": row.task_id,
                        "lia_unique_id": row.lia_unique_id,
                        "main_rgi_id": row.main_rgi_id,
                        "bed_scenario": row.bed_scenario,
                        "failure_class": failure_class,
                        "failure_detail": column,
                    }
                )
    failure_frame = pd.DataFrame(
        failures,
        columns=[
            "task_id",
            "lia_unique_id",
            "main_rgi_id",
            "bed_scenario",
            "failure_class",
            "failure_detail",
        ],
    )

    source_paths = {
        "manifest": manifest_path,
        "coverage": coverage_path,
        "qc_routes": qc_routes_path,
        "stage2_system_ensemble": stage2_system_path,
        "stage2_member_ensemble": stage2_member_path,
        "stage3_routing": routing_path,
        "climate_registry": climate_registry_path,
        "stage3_config": stage3_config_path,
    }
    source_hashes = {
        name: {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in source_paths.items()
    }
    if climate_row:
        climate_path = Path(str(climate_row["path"]))
        source_hashes["selected_climate"] = {
            "path": str(climate_path),
            "sha256": str(climate_row.get("sha256", "")),
            "bytes": int(climate_row.get("bytes", 0)),
            "path_exists": climate_path.exists(),
            "hash_source": "registered-climate-registry",
        }

    summary = {
        "run_id": "stage4-lia-v01-forward-validation-dryrun",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "dryrun-passed" if bool(plan["dryrun_ready"].all()) else "dryrun-has-failures",
        "release_root": str(release_root),
        "stage2_root": str(stage2_root),
        "stage3_root": str(stage3_root),
        "stage3_config_path": str(stage3_config_path),
        "stage0_source_dir": str(stage0_source_dir),
        "stage0_source_dir_exists": stage0_source_dir.exists(),
        "stage0_gdir_root": str(stage0_gdir_root),
        "stage0_gdir_root_exists": stage0_gdir_root.exists(),
        "output_root": str(output_root),
        "selected_ensemble_role": args.ensemble_role,
        "selected_climate_id": args.climate_id,
        "manifest_row_count": int(len(manifest)),
        "task_count": int(len(plan)),
        "dryrun_ready_count": int(plan["dryrun_ready"].sum()),
        "failure_row_count": int(len(failure_frame)),
        "missing_initial_state_pickle_count": int(plan["missing_initial_state_pickle"].sum()),
        "missing_climate_file_count": int(plan["missing_climate_file"].sum()),
        "missing_stage2_system_parameter_count": int(plan["missing_stage2_system_parameter"].sum()),
        "missing_stage2_member_parameter_count": int(plan["missing_stage2_member_parameter"].sum()),
        "missing_stage2_member_source_gdir_count": int(plan["missing_stage2_member_source_gdir"].sum()),
        "missing_routing_row_count": int(plan["missing_routing_row"].sum()),
        "non_default_review_category_count": int(plan["non_default_review_category"].sum()),
        "stage2_member_row_count_min": int(plan["stage2_member_row_count"].min()),
        "stage2_member_row_count_max": int(plan["stage2_member_row_count"].max()),
        "climate_path": str(climate_row.get("path", "")),
        "climate_path_exists": bool(path_exists(climate_row.get("path", ""))),
        "climate_time_start_year": climate_row.get("time_start_year", ""),
        "climate_time_end_year": climate_row.get("time_end_year", ""),
        "climate_monthly_count": climate_row.get("monthly_count", ""),
        "source_hashes": source_hashes,
    }

    return plan, failure_frame, summary


def main() -> int:
    args = parse_args()
    plan, failures, summary = build_plan(args)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    plan_path = output_root / "stage4_v01_task_plan_dryrun.csv"
    failure_path = output_root / "stage4_v01_planner_failures.csv"
    summary_path = output_root / "stage4_v01_task_plan_summary.json"
    source_hash_path = output_root / "source_hashes_dryrun.json"

    write_csv_atomic(plan_path, plan)
    write_csv_atomic(failure_path, failures)
    write_json_atomic(summary_path, summary)
    write_json_atomic(source_hash_path, summary["source_hashes"])

    print(json.dumps({k: v for k, v in summary.items() if k != "source_hashes"}, indent=2, ensure_ascii=True))
    if args.strict and summary["status"] != "dryrun-passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
