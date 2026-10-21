"""Independently verify the materialized full posterior for frozen Candidate v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from netCDF4 import Dataset


DRIVER = Path(__file__).resolve()
PROJECT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
FREEZE = LOOP / "temperature_candidate_v2_formal_freeze_v1"
MATERIALIZED = LOOP / "temperature_candidate_v2_full_posterior_materialization_v1"
OUTPUT = LOOP / "temperature_candidate_v2_full_posterior_independent_verification_v1"
PUBLIC = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
FULL_FIELD = MATERIALIZED / "temperature_candidate_v2_full_posterior_allowed_years.nc"
MATERIALIZED_SUMMARY = MATERIALIZED / "summary.json"
FROZEN_FIELD = FREEZE / "temperature_candidate_v2_frozen_allowed_years.nc"
FROZEN_CURVE = FREEZE / "hma_temperature_candidate_v2_allowed.csv"
FROZEN_SUMMARY = FREEZE / "summary.json"
GRID_ASSIGNMENT = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "hma_grid_gtng15_assignment.csv"
)
CORE191 = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "core191_gtng15_assignment.csv"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
EXECUTION = PROJECT / "LIA_EXECUTION_CONTRACT.md"
STATE = PROJECT / "LIA_CURRENT_STATE.json"

CANDIDATE_ID = "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828"
POSTERIOR_VARIABLE = "tas_ensemble_mean_rec_result"
ALLOWED_YEARS = np.r_[np.arange(1500, 1921), np.arange(1940, 2001)]
CORRECTION_YEARS = np.r_[np.arange(1500, 1921), np.arange(1940, 1951)]
LOCKED = (1921, 1939)
FLOAT_TOLERANCE = 5e-7
CURVE_TOLERANCE = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def read_requested(
    dataset: h5py.Dataset, all_years: np.ndarray, requested_years: np.ndarray
) -> np.ndarray:
    if np.any((requested_years >= LOCKED[0]) & (requested_years <= LOCKED[1])):
        raise RuntimeError("locked-year-requested")
    lookup = {int(year): index for index, year in enumerate(all_years)}
    indices = np.asarray([lookup[int(year)] for year in requested_years], dtype=int)
    boundaries = [0, *(np.flatnonzero(np.diff(indices) != 1) + 1).tolist(), len(indices)]
    chunks = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        chunks.append(
            np.asarray(
                dataset[int(indices[left]) : int(indices[right - 1]) + 1],
                dtype=np.float32,
            )
        )
    return np.concatenate(chunks, axis=0)


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if not np.array_equal(np.isfinite(left), np.isfinite(right)):
        return float("inf")
    if not np.any(valid):
        return 0.0
    return float(np.max(np.abs(left[valid].astype(np.float64) - right[valid].astype(np.float64))))


def weighted(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(values) & mask[None]
    numerator = np.nansum(np.where(valid, values, np.nan) * weights[None], axis=(1, 2))
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return numerator / denominator


def nearest(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    return np.abs(values[:, None] - coordinates[None, :]).argmin(axis=1)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "INDEPENDENT_VERIFICATION_CONTRACT.md"
    contract.write_text(
        "# Independent full-posterior verification\n\n"
        "Recompute Candidate v2 from the authoritative project posterior and immutable frozen "
        "correction without importing the materialization driver. Verify hashes, years, grid, "
        "units, 1961-1990 baseline, field algebra, frozen HMA curve, all 15 GTN-G regions, and "
        "nearest plus four-point grid support for all Core-191 centroids. Numeric climate values "
        "from 1921-1939 must never be read. This verification cannot promote or modify production.\n",
        encoding="utf-8",
    )
    sources = {
        "authoritative_public_field": PUBLIC,
        "full_field": FULL_FIELD,
        "materialized_summary": MATERIALIZED_SUMMARY,
        "frozen_field": FROZEN_FIELD,
        "frozen_curve": FROZEN_CURVE,
        "frozen_summary": FROZEN_SUMMARY,
        "grid_assignment": GRID_ASSIGNMENT,
        "core191": CORE191,
        "facts": FACTS,
        "execution_contract": EXECUTION,
        "current_state": STATE,
        "driver": DRIVER,
        "verification_contract": contract,
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: state(path) for name, path in sources.items()}

    frozen_summary = json.loads(FROZEN_SUMMARY.read_text(encoding="utf-8"))
    materialized_summary = json.loads(MATERIALIZED_SUMMARY.read_text(encoding="utf-8"))
    if frozen_summary["candidate_id"] != CANDIDATE_ID:
        raise RuntimeError("frozen-candidate-id-failed")
    if materialized_summary["candidate_id"] != CANDIDATE_ID:
        raise RuntimeError("materialized-candidate-id-failed")
    full_field_hash = sha256(FULL_FIELD)
    if full_field_hash != materialized_summary["output_field_sha256"]:
        raise RuntimeError("materialized-full-field-hash-failed")
    if sha256(FROZEN_FIELD) != frozen_summary["frozen_field_sha256"]:
        raise RuntimeError("frozen-field-hash-failed")
    if sha256(FROZEN_CURVE) != frozen_summary["hma_curve_sha256"]:
        raise RuntimeError("frozen-curve-hash-failed")

    with Dataset(FULL_FIELD) as output:
        output_years = np.asarray(output.variables["year"][:], dtype=int)
        lat = np.asarray(output.variables["lat"][:], dtype=float)
        lon = np.asarray(output.variables["lon"][:], dtype=float)
        hma = np.asarray(output.variables["hma_mask"][:], dtype=bool)
        common = np.asarray(output.variables["candidate_common_support_mask"][:], dtype=bool)
        unsupported = np.asarray(output.variables["unsupported_hma_mask"][:], dtype=bool)
        output_posterior = np.asarray(output.variables["tas_project_posterior"][:], dtype=np.float32)
        output_correction = np.asarray(output.variables["tas_candidate_v2_correction"][:], dtype=np.float32)
        output_candidate = np.asarray(output.variables["tas_candidate_v2_posterior"][:], dtype=np.float32)
        units = {
            name: str(output.variables[name].units)
            for name in [
                "tas_project_posterior",
                "tas_candidate_v2_correction",
                "tas_candidate_v2_posterior",
            ]
        }
        output_attrs = {name: getattr(output, name) for name in output.ncattrs()}
    if not np.array_equal(output_years, ALLOWED_YEARS):
        raise RuntimeError("output-year-contract-failed")
    if np.any((output_years >= LOCKED[0]) & (output_years <= LOCKED[1])):
        raise RuntimeError("locked-year-in-output")

    with Dataset(FROZEN_FIELD) as frozen:
        frozen_years = np.asarray(frozen.variables["year"][:], dtype=int)
        frozen_lat = np.asarray(frozen.variables["lat"][:], dtype=float)
        frozen_lon = np.asarray(frozen.variables["lon"][:], dtype=float)
        frozen_hma = np.asarray(frozen.variables["hma_mask"][:], dtype=bool)
        frozen_common = np.asarray(frozen.variables["common_support_mask"][:], dtype=bool)
        frozen_correction = np.asarray(
            frozen.variables["tas_background_low_frequency_correction"][:], dtype=np.float32
        )
    if not np.array_equal(frozen_years, CORRECTION_YEARS):
        raise RuntimeError("frozen-year-contract-failed")
    grid_identity = bool(
        np.array_equal(lat, frozen_lat)
        and np.array_equal(lon, frozen_lon)
        and np.array_equal(hma, frozen_hma)
        and np.array_equal(common, frozen_common)
        and np.array_equal(unsupported, hma & ~common)
    )

    with h5py.File(PUBLIC, "r") as public:
        public_years = np.asarray(public["time"], dtype=int).ravel()
        public_lat = np.asarray(public["lat"], dtype=float).ravel()
        public_lon = np.asarray(public["lon"], dtype=float).ravel()
        independent_posterior = read_requested(
            public[POSTERIOR_VARIABLE], public_years, ALLOWED_YEARS
        )
        public_units = public[POSTERIOR_VARIABLE].attrs["units"].decode("ascii")
        public_reference = public[POSTERIOR_VARIABLE].attrs["reference_period"].decode("ascii")
    independent_posterior[independent_posterior <= -9998.0] = np.nan
    public_grid_identity = bool(
        np.array_equal(lat, public_lat) and np.array_equal(lon, public_lon)
    )

    independent_correction = np.full_like(independent_posterior, np.nan)
    independent_correction[:, common] = 0.0
    output_lookup = {int(year): index for index, year in enumerate(ALLOWED_YEARS)}
    for index, year in enumerate(CORRECTION_YEARS):
        independent_correction[output_lookup[int(year)], common] = frozen_correction[index, common]
    independent_candidate = independent_posterior.copy()
    independent_candidate[:, common] = (
        independent_posterior[:, common] + independent_correction[:, common]
    ).astype(np.float32)

    posterior_reproduction = max_abs(output_posterior, independent_posterior)
    correction_reproduction = max_abs(output_correction, independent_correction)
    candidate_reproduction = max_abs(output_candidate, independent_candidate)
    output_algebra = max_abs(
        output_candidate[:, common],
        (output_posterior[:, common] + output_correction[:, common]).astype(np.float32),
    )
    outside_candidate_identity = max_abs(
        output_candidate[:, ~common], output_posterior[:, ~common]
    )
    outside_correction_all_missing = bool(np.all(~np.isfinite(output_correction[:, ~common])))
    after_1950_zero = float(
        np.max(np.abs(output_correction[ALLOWED_YEARS >= 1951][:, common]))
    )

    baseline = (ALLOWED_YEARS >= 1961) & (ALLOWED_YEARS <= 1990)
    baseline_project = float(
        np.max(np.abs(np.mean(output_posterior[baseline][:, common], axis=0, dtype=np.float64)))
    )
    baseline_candidate = float(
        np.max(np.abs(np.mean(output_candidate[baseline][:, common], axis=0, dtype=np.float64)))
    )
    baseline_correction = float(
        np.max(np.abs(np.mean(output_correction[baseline][:, common], axis=0, dtype=np.float64)))
    )

    frozen_curve = pd.read_csv(FROZEN_CURVE)
    project_hma = weighted(output_posterior, hma, lat)
    correction_hma = weighted(output_correction, common, lat)
    contract_hma = project_hma + correction_hma
    curve_checks = {
        "project_max_abs_degC": float(
            np.max(np.abs(project_hma - frozen_curve["original_project_degC"].to_numpy(float)))
        ),
        "correction_max_abs_degC": float(
            np.max(
                np.abs(
                    correction_hma
                    - frozen_curve["candidate_background_correction_degC"].to_numpy(float)
                )
            )
        ),
        "candidate_max_abs_degC": float(
            np.max(np.abs(contract_hma - frozen_curve["improved_candidate_degC"].to_numpy(float)))
        ),
    }

    assignment = pd.read_csv(GRID_ASSIGNMENT)
    required_assignment = {"grid_row", "grid_col", "o2region"}
    if not required_assignment.issubset(assignment.columns):
        raise RuntimeError("grid-assignment-columns-failed")
    region_rows = []
    for region, cells in assignment.groupby("o2region", sort=True):
        mask = np.zeros_like(hma)
        mask[cells["grid_row"].to_numpy(int), cells["grid_col"].to_numpy(int)] = True
        region_common = mask & common
        project = weighted(output_posterior, mask, lat)
        direct = weighted(output_candidate, mask, lat)
        correction = weighted(output_correction, region_common, lat)
        contract_candidate = project + correction
        region_rows.append(
            {
                "o2region": region,
                "grid_cell_count": int(mask.sum()),
                "common_support_cell_count": int(region_common.sum()),
                "unsupported_cell_count": int((mask & ~common).sum()),
                "field_algebra_max_abs_degC": max_abs(
                    output_candidate[:, region_common],
                    (output_posterior[:, region_common] + output_correction[:, region_common]).astype(np.float32),
                ),
                "direct_field_minus_contract_hma_max_abs_degC": float(
                    np.max(np.abs(direct - contract_candidate))
                ),
            }
        )
    regional = pd.DataFrame(region_rows)
    regional.to_csv(OUTPUT / "gtng15_independent_materialization_closure.csv", index=False)
    regional_algebra_max = float(regional["field_algebra_max_abs_degC"].max())

    core = pd.read_csv(CORE191)
    if len(core) != 191 or core["lia_unique_id"].nunique() != 191:
        raise RuntimeError("core191-identity-failed")
    lat_index = nearest(core["centroid_lat"].to_numpy(float), lat)
    lon_index = nearest(core["centroid_lon"].to_numpy(float), lon)
    core_audit = core[["lia_unique_id", "centroid_lat", "centroid_lon"]].copy()
    core_audit["nearest_row"] = lat_index
    core_audit["nearest_col"] = lon_index
    core_audit["nearest_lat"] = lat[lat_index]
    core_audit["nearest_lon"] = lon[lon_index]
    core_audit["nearest_common_support"] = common[lat_index, lon_index]
    core_audit["nearest_candidate_all_years_finite"] = [
        bool(np.all(np.isfinite(output_candidate[:, row, col])))
        for row, col in zip(lat_index, lon_index)
    ]
    stencil_support = []
    for latitude, longitude in zip(
        core["centroid_lat"].to_numpy(float), core["centroid_lon"].to_numpy(float)
    ):
        rows = np.argsort(np.abs(lat - latitude), kind="stable")[:2]
        cols = np.argsort(np.abs(lon - longitude), kind="stable")[:2]
        stencil_support.append(bool(np.all(common[np.ix_(rows, cols)])))
    core_audit["nearest_four_point_stencil_common_support"] = stencil_support
    core_audit.to_csv(OUTPUT / "core191_independent_support_audit.csv", index=False)
    nearest_support_count = int(core_audit["nearest_common_support"].sum())
    nearest_finite_count = int(core_audit["nearest_candidate_all_years_finite"].sum())
    stencil_support_count = int(core_audit["nearest_four_point_stencil_common_support"].sum())

    after = {name: state(path) for name, path in sources.items()}
    unchanged = before == after
    expected_units = "degC anomaly relative to 1961-1990"
    gates = {
        "materialized_field_hash_matches_summary": full_field_hash == materialized_summary["output_field_sha256"],
        "output_and_frozen_grid_identity": grid_identity,
        "output_and_public_grid_identity": public_grid_identity,
        "posterior_reproduces_authoritative_source": posterior_reproduction == 0.0,
        "correction_reproduces_frozen_source": correction_reproduction == 0.0,
        "candidate_independent_reproduction_at_most_5e_7_degC": candidate_reproduction <= FLOAT_TOLERANCE,
        "output_field_algebra_at_most_5e_7_degC": output_algebra <= FLOAT_TOLERANCE,
        "regional_field_algebra_at_most_5e_7_degC": regional_algebra_max <= FLOAT_TOLERANCE,
        "candidate_unchanged_outside_common_support": outside_candidate_identity == 0.0,
        "correction_missing_outside_common_support": outside_correction_all_missing,
        "correction_zero_after_1950": after_1950_zero == 0.0,
        "hma_project_curve_closes": curve_checks["project_max_abs_degC"] <= CURVE_TOLERANCE,
        "hma_correction_curve_closes": curve_checks["correction_max_abs_degC"] <= CURVE_TOLERANCE,
        "hma_candidate_curve_closes": curve_checks["candidate_max_abs_degC"] <= CURVE_TOLERANCE,
        "baseline_project_cellwise_max_at_most_1e_6_degC": baseline_project <= CURVE_TOLERANCE,
        "baseline_candidate_cellwise_max_at_most_1e_6_degC": baseline_candidate <= CURVE_TOLERANCE,
        "baseline_correction_exactly_zero": baseline_correction == 0.0,
        "all_three_output_units_match": all(value == expected_units for value in units.values()),
        "public_source_units_and_reference_match": public_units == "degrees_Celsius" and public_reference == "1961-1990",
        "all_core191_nearest_cells_supported": nearest_support_count == 191,
        "all_core191_nearest_series_finite": nearest_finite_count == 191,
        "all_core191_four_point_stencils_supported": stencil_support_count == 191,
        "sealed_years_absent": not np.any((output_years >= LOCKED[0]) & (output_years <= LOCKED[1])),
        "declared_sources_unchanged": unchanged,
        "output_metadata_retains_nonproduction_boundary": output_attrs.get("status") == "formally frozen candidate; not production",
    }
    passes = bool(all(gates.values()))
    decision = (
        "independent_full_field_verification_pass_candidate_v2_release_addendum_frozen_not_production"
        if passes
        else "independent_full_field_verification_failed_do_not_promote_candidate_v2"
    )
    summary = {
        "schema": "hma-temperature-candidate-v2-independent-full-field-verification-v1",
        "status": "complete",
        "candidate_id": CANDIDATE_ID,
        "decision": decision,
        "passes_all_independent_gates": passes,
        "field_reproduction": {
            "posterior_max_abs_degC": posterior_reproduction,
            "correction_max_abs_degC": correction_reproduction,
            "candidate_max_abs_degC": candidate_reproduction,
            "output_algebra_max_abs_degC": output_algebra,
            "outside_common_support_candidate_difference_max_abs_degC": outside_candidate_identity,
        },
        "baseline_1961_1990_cellwise_max_abs_degC": {
            "project": baseline_project,
            "candidate": baseline_candidate,
            "correction": baseline_correction,
        },
        "hma_curve_closure": curve_checks,
        "gtng15_region_count": int(len(regional)),
        "gtng15_regional_field_algebra_max_abs_degC": regional_algebra_max,
        "core191": {
            "count": 191,
            "nearest_common_support_count": nearest_support_count,
            "nearest_all_years_finite_count": nearest_finite_count,
            "four_point_stencil_common_support_count": stencil_support_count,
        },
        "gates": gates,
        "locked_1921_1939_numeric_values_accessed": 0,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    summary_path = OUTPUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    release_manifest = {
        "schema": "hma-temperature-candidate-v2-full-field-release-addendum-v1",
        "candidate_id": CANDIDATE_ID,
        "status": "verified-frozen-candidate-not-production",
        "full_posterior_field": state(FULL_FIELD),
        "materialization_summary": state(MATERIALIZED_SUMMARY),
        "independent_verification_summary": state(summary_path),
        "frozen_background_field": state(FROZEN_FIELD),
        "frozen_hma_curve": state(FROZEN_CURVE),
        "authoritative_source": state(PUBLIC),
        "production_promotion_authorized": False,
        "sealed_1921_1939_values_present": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (OUTPUT / "RELEASE_ADDENDUM.json").write_text(
        json.dumps(release_manifest, indent=2), encoding="utf-8"
    )
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate v2 independent full-field verification\n\n"
        f"Decision: `{decision}`.\n\n"
        f"- Posterior reproduction: `{posterior_reproduction:.3e} degC`.\n"
        f"- Frozen correction reproduction: `{correction_reproduction:.3e} degC`.\n"
        f"- Candidate reproduction: `{candidate_reproduction:.3e} degC`.\n"
        f"- HMA candidate-curve closure: `{curve_checks['candidate_max_abs_degC']:.3e} degC`.\n"
        f"- GTN-G regions independently checked: `{len(regional)}`.\n"
        f"- Core-191 nearest/four-point support: `{nearest_support_count}/191`, `{stencil_support_count}/191`.\n"
        f"- All independent gates pass: `{str(passes).lower()}`.\n"
        "- The release addendum freezes the derived full field as a non-production candidate. "
        "No production file, sealed year, EnKF, or OGGM run was used.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("declared-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
