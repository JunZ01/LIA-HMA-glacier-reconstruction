"""Round 00 v2: reproduce valid data and classify legacy nodata-to-zero encoding."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np

import run_climate_reconstruction_round00_production_reproduction_v1 as base


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = base.project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0" / base.EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")

    facts = root / "LIA_PROJECT_FACTS.md"
    lock_contract = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_precipitation_instrumental_baseline_v1/LOCKED_FINAL_PERIOD.json"
    plan = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/climate_reconstruction_improvement_loop_v1.md"
    driver = Path(__file__).resolve()
    members = [(variable, method, network) for variable in ("tas", "pr") for method in base.METHODS for network in (6, 9)]
    tifs = [base.source_tif(*member) for member in members]
    required = [base.NETCDF, base.STRUCTURED_MAT, base.BUILDER, base.ASSEMBLER, facts, lock_contract, plan, *tifs]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-round00-inputs:" + ";".join(missing))
    locked = json.loads(lock_contract.read_text(encoding="utf-8"))
    if locked.get("status") != "locked_not_evaluated":
        raise RuntimeError("locked-period-contract-failed")

    before = {str(path): base.stat(path) for path in required}
    hashes = {str(path): base.sha256(path) for path in [base.NETCDF, base.BUILDER, base.ASSEMBLER, *tifs]}
    lineage_rows: list[dict[str, object]] = []
    closure_rows: list[dict[str, object]] = []
    archived_members: dict[str, dict[str, np.ndarray]] = {"tas": {}, "pr": {}}
    valid_masks: dict[str, list[np.ndarray]] = {"tas": [], "pr": []}

    with netCDF4.Dataset(base.NETCDF, "r") as dataset:
        years = np.asarray(dataset.variables["time"][:], dtype=int)
        if not np.array_equal(years, np.arange(1000, 2001, dtype=int)):
            raise RuntimeError("unexpected-production-time-axis")
        allowed = (years < base.LOCKED_START) | (years > base.LOCKED_END)
        indices = np.flatnonzero(allowed)
        allowed_years = years[indices]
        if np.any((allowed_years >= base.LOCKED_START) & (allowed_years <= base.LOCKED_END)):
            raise RuntimeError("locked-period-requested")
        reference = (allowed_years >= base.REFERENCE_START) & (allowed_years <= base.REFERENCE_END)
        if int(reference.sum()) != 30:
            raise RuntimeError("reference-period-incomplete")
        bands = [int(year - 999) for year in allowed_years]
        latitudes = np.asarray(dataset.variables["lat"][:], dtype=float)
        longitudes = np.asarray(dataset.variables["lon"][:], dtype=float)
        if latitudes.shape != (43,) or longitudes.shape != (78,):
            raise RuntimeError("unexpected-production-grid")

        for variable, method, network in members:
            label = f"{method}_{network}"
            tif = base.source_tif(variable, method, network)
            source, metadata = base.read_tif_allowed(tif, bands)
            source_anomaly = source - np.nanmean(source[reference], axis=0)[None, :, :]
            archived = base.clean(dataset.variables[base.member_variable(variable, method, network)][indices, :, :])
            direct = base.compare(source_anomaly, archived)
            flipped = base.compare(source_anomaly[:, ::-1, :], archived)
            if direct["maximum_absolute_error"] <= flipped["maximum_absolute_error"]:
                orientation, rebuilt, score = "as_stored", source_anomaly, direct
            else:
                orientation, rebuilt, score = "latitude_flipped", source_anomaly[:, ::-1, :], flipped
            if score["maximum_absolute_error"] > base.MEMBER_TOLERANCE:
                raise RuntimeError(f"member-value-closure-failed:{variable}:{label}:{score}")

            source_finite = np.isfinite(rebuilt)
            archive_finite = np.isfinite(archived)
            only_archive = (~source_finite) & archive_finite
            only_source = source_finite & (~archive_finite)
            only_archive_values = archived[only_archive]
            zero_encoded = bool(
                int(only_source.sum()) == 0
                and only_archive_values.size > 0
                and float(np.max(np.abs(only_archive_values))) <= 1e-12
            )
            if int(score["finite_mask_mismatch_count"]) > 0 and not zero_encoded:
                raise RuntimeError(f"unclassified-mask-mismatch:{variable}:{label}:{score}")

            archived_members[variable][label] = archived
            valid_masks[variable].append(np.any(source_finite, axis=0))
            reference_residual = np.nanmean(archived[:, np.any(source_finite, axis=0)][reference], axis=0)
            lineage_rows.append({
                "variable": variable,
                "member": label,
                "method": method,
                "proxy_network": network,
                "source_tif": str(tif),
                "source_tif_sha256": hashes[str(tif)],
                "netcdf_variable": base.member_variable(variable, method, network),
                "orientation": orientation,
                "source_shape": "x".join(str(value) for value in metadata["shape"]),
                "allowed_year_count": int(len(allowed_years)),
                "locked_year_count_skipped": base.LOCKED_END - base.LOCKED_START + 1,
                "maximum_absolute_error_on_source_valid_values": score["maximum_absolute_error"],
                "rmse_on_source_valid_values": score["rmse"],
                "source_invalid_archive_zero_value_count": int(only_archive.sum()),
                "source_invalid_archive_zero_cell_count": int(np.any(only_archive, axis=0).sum()),
                "nodata_to_zero_encoding_confirmed": zero_encoded,
                "reference_period_valid_cell_mean_abs_max": float(np.nanmax(np.abs(reference_residual))),
            })

        for variable in ("tas", "pr"):
            for network in (6, 9):
                labels = [f"{method}_{network}" for method in base.METHODS]
                rebuilt = base.finite_mean([archived_members[variable][label] for label in labels])
                archived = base.clean(dataset.variables[f"{variable}_ensemble_mean_{network}"][indices, :, :])
                closure_rows.append({"variable": variable, "ensemble": f"network_{network}", **base.compare(rebuilt, archived)})
            labels = [f"{method}_{network}" for method in base.METHODS for network in (6, 9)]
            rebuilt = base.finite_mean([archived_members[variable][label] for label in labels])
            archived = base.clean(dataset.variables[f"{variable}_ensemble_mean_rec_result"][indices, :, :])
            closure_rows.append({"variable": variable, "ensemble": "six_member", **base.compare(rebuilt, archived)})

    if max(row["maximum_absolute_error"] for row in closure_rows) > base.ENSEMBLE_TOLERANCE:
        raise RuntimeError(f"ensemble-closure-failed:{closure_rows}")
    if any(row["finite_mask_mismatch_count"] != 0 for row in closure_rows):
        raise RuntimeError(f"ensemble-mask-closure-failed:{closure_rows}")
    after = {str(path): base.stat(path) for path in required}
    if before != after:
        raise RuntimeError("source-changed-during-round00")

    output.mkdir(parents=True)
    common_masks = {variable: np.logical_and.reduce(valid_masks[variable]) for variable in ("tas", "pr")}
    np.savez_compressed(
        output / "source_valid_domain_masks_05deg.npz",
        latitude=latitudes,
        longitude=longitudes,
        tas_common_six_member=common_masks["tas"],
        pr_common_six_member=common_masks["pr"],
    )
    config = {
        "experiment_id": base.EXPERIMENT_ID,
        "driver_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only_reproduction_audit",
        "allowed_periods": [[1000, 1920], [1940, 2000]],
        "locked_period": [base.LOCKED_START, base.LOCKED_END],
        "reference_period": [base.REFERENCE_START, base.REFERENCE_END],
        "member_tolerance": base.MEMBER_TOLERANCE,
        "ensemble_tolerance": base.ENSEMBLE_TOLERANCE,
        "python_executable": r"D:\Anaconda\envs\oggm_env2\python.exe",
        "oggm_launched": False,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({
        "before": before,
        "after": after,
        "content_sha256": hashes,
        "driver_sha256": base.sha256(driver),
        "structured_mat_note": "Metadata only; direct closure uses 12 source GeoTIFFs and skips locked bands.",
    }, indent=2), encoding="utf-8")
    write_csv(output / "member_lineage.csv", lineage_rows)
    write_csv(output / "closure_scores.csv", closure_rows)

    max_member = max(float(row["maximum_absolute_error_on_source_valid_values"]) for row in lineage_rows)
    max_ensemble = max(float(row["maximum_absolute_error"]) for row in closure_rows)
    zero_cells = {variable: int((~common_masks[variable]).sum()) for variable in common_masks}
    summary = {
        "experiment_id": base.EXPERIMENT_ID,
        "status": "round00_pass_with_legacy_nodata_zero_encoding_recorded",
        "member_count": len(lineage_rows),
        "allowed_year_count": int(len(allowed_years)),
        "locked_1921_1939_accessed": False,
        "maximum_member_source_to_netcdf_error_on_valid_values": max_member,
        "maximum_ensemble_rebuild_error": max_ensemble,
        "legacy_nodata_to_zero_encoding_confirmed_for_all_members": all(bool(row["nodata_to_zero_encoding_confirmed"]) for row in lineage_rows),
        "common_six_member_invalid_cell_count": zero_cells,
        "all_declared_sources_unchanged": before == after,
        "gate_00": "pass_on_source_valid_domain; locked-period closure deferred; source-domain mask mandatory",
        "next_round": "round_01_blocked_cv_baseline",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "ROUND_CONTRACT.md").write_text(
        "# Round 00 Contract\n\n"
        "Decision: determine whether the archived six-member temperature and precipitation product is reproducible enough to be the v2 baseline.\n\n"
        "Only 1000-1920 and 1940-2000 are read. The locked 1921-1939 values remain unopened. Source-invalid cells encoded as zero by the legacy anomaly builder are classified and excluded through a saved source-domain mask.\n",
        encoding="utf-8",
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 00 Production Reproduction\n\n"
        f"- Status: `{summary['status']}`\n"
        f"- Valid-domain source-to-member maximum error: `{max_member:.9g}`\n"
        f"- Ensemble maximum error: `{max_ensemble:.9g}`\n"
        f"- Common invalid cells: tas `{zero_cells['tas']}`, pr `{zero_cells['pr']}` of 3354\n"
        "- Locked 1921-1939 values accessed: `false`\n"
        "- Decision: Gate 00 passes on the source-valid domain. Round 01 must use the saved masks; legacy zero-coded nodata cannot be scored as climate.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
