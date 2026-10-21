"""Materialize the frozen Candidate v2 on the authoritative posterior grid."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DRIVER = Path(__file__).resolve()
PROJECT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
FREEZE = LOOP / "temperature_candidate_v2_formal_freeze_v1"
OUTPUT = LOOP / "temperature_candidate_v2_full_posterior_materialization_v1"
PUBLIC = Path(
    r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
)
CORE191 = RESULTS / (
    "climate_reconstruction_hma_gtng_subregion_inventory_v1/"
    "core191_gtng15_assignment.csv"
)
FROZEN_FIELD = FREEZE / "temperature_candidate_v2_frozen_allowed_years.nc"
FROZEN_CURVE = FREEZE / "hma_temperature_candidate_v2_allowed.csv"
FROZEN_SUMMARY = FREEZE / "summary.json"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
EXECUTION = PROJECT / "LIA_EXECUTION_CONTRACT.md"
STATE = PROJECT / "LIA_CURRENT_STATE.json"

CANDIDATE_ID = "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828"
POSTERIOR_VARIABLE = "tas_ensemble_mean_rec_result"
ALLOWED_YEARS = np.r_[np.arange(1500, 1921), np.arange(1940, 2001)]
CORRECTION_YEARS = np.r_[np.arange(1500, 1921), np.arange(1940, 1951)]
LOCKED = (1921, 1939)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def file_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def read_allowed_chunks(
    dataset: h5py.Dataset, all_years: np.ndarray, requested_years: np.ndarray
) -> np.ndarray:
    if np.any((requested_years >= LOCKED[0]) & (requested_years <= LOCKED[1])):
        raise RuntimeError("locked-year-requested")
    lookup = {int(year): index for index, year in enumerate(all_years)}
    indices = np.asarray([lookup[int(year)] for year in requested_years], dtype=int)
    starts = [0, *(np.flatnonzero(np.diff(indices) != 1) + 1).tolist(), len(indices)]
    chunks = []
    for left, right in zip(starts[:-1], starts[1:]):
        first = int(indices[left])
        last = int(indices[right - 1]) + 1
        chunks.append(np.asarray(dataset[first:last], dtype=np.float64))
    return np.concatenate(chunks, axis=0)


def weighted_series(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(values) & mask[None]
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    numerator = np.nansum(np.where(valid, values, np.nan) * weights[None], axis=(1, 2))
    return numerator / denominator


def nearest_grid_indices(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    return np.abs(values[:, None] - coordinates[None, :]).argmin(axis=1)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "MATERIALIZATION_CONTRACT.md"
    contract.write_text(
        "# Candidate v2 full-posterior materialization contract\n\n"
        f"Materialize `{CANDIDATE_ID}` from the authoritative public project variable "
        f"`{POSTERIOR_VARIABLE}` plus the immutable frozen correction. Read numeric posterior "
        "values only for 1500-1920 and 1940-2000. Apply the correction only on its frozen "
        "common-support mask through 1950; retain the original posterior after 1950 and outside "
        "that support. Do not fill the single unsupported HMA boundary cell. The output is an "
        "independent research artifact, not a production file or OGGM forcing. No EnKF is run.\n",
        encoding="utf-8",
    )
    sources = {
        "authoritative_public_field": PUBLIC,
        "frozen_field": FROZEN_FIELD,
        "frozen_curve": FROZEN_CURVE,
        "frozen_summary": FROZEN_SUMMARY,
        "core191": CORE191,
        "facts": FACTS,
        "execution_contract": EXECUTION,
        "current_state": STATE,
        "driver": DRIVER,
        "materialization_contract": contract,
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: file_state(path) for name, path in sources.items()}

    frozen_summary = json.loads(FROZEN_SUMMARY.read_text(encoding="utf-8"))
    if frozen_summary.get("candidate_id") != CANDIDATE_ID:
        raise RuntimeError("candidate-id-mismatch")
    if frozen_summary.get("status") != "formally-frozen-candidate":
        raise RuntimeError("candidate-not-formally-frozen")
    if sha256(FROZEN_FIELD) != frozen_summary["frozen_field_sha256"]:
        raise RuntimeError("frozen-field-hash-mismatch")
    if sha256(FROZEN_CURVE) != frozen_summary["hma_curve_sha256"]:
        raise RuntimeError("frozen-curve-hash-mismatch")

    with Dataset(FROZEN_FIELD) as frozen:
        frozen_years = np.asarray(frozen.variables["year"][:], dtype=int)
        lat = np.asarray(frozen.variables["lat"][:], dtype=float)
        lon = np.asarray(frozen.variables["lon"][:], dtype=float)
        hma_mask = np.asarray(frozen.variables["hma_mask"][:], dtype=bool)
        common = np.asarray(frozen.variables["common_support_mask"][:], dtype=bool)
        frozen_correction = np.asarray(
            frozen.variables["tas_background_low_frequency_correction"][:], dtype=np.float64
        )
        frozen_background = np.asarray(
            frozen.variables["tas_background_archived"][:], dtype=np.float64
        )
        frozen_candidate_background = np.asarray(
            frozen.variables["tas_background_candidate_v2"][:], dtype=np.float64
        )
        frozen_units = str(frozen.variables["tas_background_low_frequency_correction"].units)
    if not np.array_equal(frozen_years, CORRECTION_YEARS):
        raise RuntimeError("frozen-correction-year-contract-failed")
    if np.any((frozen_years >= LOCKED[0]) & (frozen_years <= LOCKED[1])):
        raise RuntimeError("locked-year-in-frozen-field")
    if int(hma_mask.sum()) != 1581 or int(common.sum()) != 1580:
        raise RuntimeError("frozen-mask-count-changed")
    unsupported_hma = hma_mask & ~common
    if int(unsupported_hma.sum()) != 1:
        raise RuntimeError("unsupported-hma-cell-contract-changed")

    with h5py.File(PUBLIC, "r") as public:
        public_years = np.asarray(public["time"], dtype=int).ravel()
        public_lat = np.asarray(public["lat"], dtype=float).ravel()
        public_lon = np.asarray(public["lon"], dtype=float).ravel()
        posterior_dataset = public[POSTERIOR_VARIABLE]
        posterior_units = posterior_dataset.attrs["units"].decode("ascii")
        posterior_reference = posterior_dataset.attrs["reference_period"].decode("ascii")
        posterior = read_allowed_chunks(posterior_dataset, public_years, ALLOWED_YEARS)
    posterior[posterior <= -9998.0] = np.nan
    if not np.array_equal(public_years, np.arange(1000, 2001)):
        raise RuntimeError("public-year-contract-failed")
    if not np.array_equal(public_lat, lat) or not np.array_equal(public_lon, lon):
        raise RuntimeError("public-and-frozen-grid-mismatch")
    if posterior.shape != (len(ALLOWED_YEARS), len(lat), len(lon)):
        raise RuntimeError("posterior-shape-failed")
    if posterior_units != "degrees_Celsius" or posterior_reference != "1961-1990":
        raise RuntimeError("posterior-unit-or-reference-contract-failed")
    if frozen_units != "degC anomaly relative to 1961-1990":
        raise RuntimeError("frozen-correction-unit-contract-failed")

    correction = np.full_like(posterior, np.nan)
    correction[:, common] = 0.0
    correction_lookup = {int(year): index for index, year in enumerate(ALLOWED_YEARS)}
    for source_index, year in enumerate(frozen_years):
        correction[correction_lookup[int(year)], common] = frozen_correction[source_index, common]
    candidate = posterior.copy()
    candidate[:, common] = posterior[:, common] + correction[:, common]

    finite_posterior_common = bool(np.all(np.isfinite(posterior[:, common])))
    finite_correction_common = bool(np.all(np.isfinite(correction[:, common])))
    finite_candidate_common = bool(np.all(np.isfinite(candidate[:, common])))
    correction_after_1950_max = float(
        np.max(np.abs(correction[ALLOWED_YEARS >= 1951][:, common]))
    )
    unchanged_after_1950_max = float(
        np.max(np.abs(candidate[ALLOWED_YEARS >= 1951] - posterior[ALLOWED_YEARS >= 1951]))
    )
    unchanged_outside_common_max = float(
        np.nanmax(np.abs(candidate[:, ~common] - posterior[:, ~common]))
    )
    field_closure_max = float(
        np.max(np.abs(candidate[:, common] - posterior[:, common] - correction[:, common]))
    )
    background_closure = float(
        np.max(
            np.abs(
                frozen_candidate_background[:, common]
                - frozen_background[:, common]
                - frozen_correction[:, common]
            )
        )
    )
    increment_invariance = float(
        np.max(
            np.abs(
                (candidate[: len(CORRECTION_YEARS), common] - frozen_candidate_background[:, common])
                - (posterior[: len(CORRECTION_YEARS), common] - frozen_background[:, common])
            )
        )
    )

    frozen_curve = pd.read_csv(FROZEN_CURVE)
    if not np.array_equal(frozen_curve["year"].to_numpy(int), ALLOWED_YEARS):
        raise RuntimeError("frozen-curve-year-contract-failed")
    project_hma = weighted_series(posterior, hma_mask, lat)
    correction_common = weighted_series(correction, common, lat)
    candidate_contract_hma = project_hma + correction_common
    candidate_direct_hma = weighted_series(candidate, hma_mask, lat)
    curve = pd.DataFrame(
        {
            "year": ALLOWED_YEARS,
            "authoritative_project_hma_degC": project_hma,
            "frozen_correction_common_support_degC": correction_common,
            "candidate_contract_hma_degC": candidate_contract_hma,
            "candidate_direct_materialized_hma_degC": candidate_direct_hma,
            "frozen_csv_original_project_degC": frozen_curve["original_project_degC"],
            "frozen_csv_correction_degC": frozen_curve["candidate_background_correction_degC"],
            "frozen_csv_candidate_degC": frozen_curve["improved_candidate_degC"],
        }
    )
    curve["direct_minus_contract_degC"] = (
        curve["candidate_direct_materialized_hma_degC"]
        - curve["candidate_contract_hma_degC"]
    )
    curve.to_csv(OUTPUT / "hma_materialization_closure_allowed.csv", index=False)
    project_curve_closure = float(
        np.max(
            np.abs(
                curve["authoritative_project_hma_degC"]
                - curve["frozen_csv_original_project_degC"]
            )
        )
    )
    correction_curve_closure = float(
        np.max(
            np.abs(
                curve["frozen_correction_common_support_degC"]
                - curve["frozen_csv_correction_degC"]
            )
        )
    )
    candidate_curve_closure = float(
        np.max(
            np.abs(
                curve["candidate_contract_hma_degC"]
                - curve["frozen_csv_candidate_degC"]
            )
        )
    )
    direct_support_effect = float(np.max(np.abs(curve["direct_minus_contract_degC"])))

    core = pd.read_csv(CORE191)
    if len(core) != 191 or core["lia_unique_id"].nunique() != 191:
        raise RuntimeError("core191-count-failed")
    lat_index = nearest_grid_indices(core["centroid_lat"].to_numpy(float), lat)
    lon_index = nearest_grid_indices(core["centroid_lon"].to_numpy(float), lon)
    core_support = core[["lia_unique_id", "centroid_lat", "centroid_lon"]].copy()
    core_support["nearest_grid_lat"] = lat[lat_index]
    core_support["nearest_grid_lon"] = lon[lon_index]
    core_support["nearest_grid_in_hma_mask"] = hma_mask[lat_index, lon_index]
    core_support["nearest_grid_in_candidate_common_support"] = common[lat_index, lon_index]
    core_support["distance_deg"] = np.hypot(
        core_support["centroid_lat"] - core_support["nearest_grid_lat"],
        core_support["centroid_lon"] - core_support["nearest_grid_lon"],
    )
    core_support.to_csv(OUTPUT / "core191_candidate_support_audit.csv", index=False)
    core_common_count = int(core_support["nearest_grid_in_candidate_common_support"].sum())

    output_nc = OUTPUT / "temperature_candidate_v2_full_posterior_allowed_years.nc"
    with Dataset(output_nc, "w", format="NETCDF4") as dataset:
        dataset.createDimension("year", len(ALLOWED_YEARS))
        dataset.createDimension("lat", len(lat))
        dataset.createDimension("lon", len(lon))
        dataset.createVariable("year", "i4", ("year",))[:] = ALLOWED_YEARS
        dataset.createVariable("lat", "f8", ("lat",))[:] = lat
        dataset.createVariable("lon", "f8", ("lon",))[:] = lon
        dataset.createVariable("hma_mask", "i1", ("lat", "lon"))[:] = hma_mask.astype(np.int8)
        dataset.createVariable("candidate_common_support_mask", "i1", ("lat", "lon"))[:] = common.astype(np.int8)
        dataset.createVariable("unsupported_hma_mask", "i1", ("lat", "lon"))[:] = unsupported_hma.astype(np.int8)
        for name, values, long_name in [
            ("tas_project_posterior", posterior, "Authoritative project posterior"),
            ("tas_candidate_v2_correction", correction, "Frozen Candidate v2 background correction"),
            ("tas_candidate_v2_posterior", candidate, "Materialized Candidate v2 posterior"),
        ]:
            variable = dataset.createVariable(
                name,
                "f4",
                ("year", "lat", "lon"),
                zlib=True,
                complevel=4,
                chunksizes=(1, len(lat), len(lon)),
                fill_value=np.float32(np.nan),
            )
            variable[:] = values.astype(np.float32)
            variable.units = "degC anomaly relative to 1961-1990"
            variable.long_name = long_name
        dataset.title = "HMA Temperature Candidate v2 full posterior, allowed years only"
        dataset.candidate_id = CANDIDATE_ID
        dataset.status = "formally frozen candidate; not production"
        dataset.source_posterior_variable = POSTERIOR_VARIABLE
        dataset.candidate_definition = "authoritative project posterior plus immutable frozen background correction"
        dataset.support_rule = "apply correction on candidate_common_support_mask; retain original elsewhere"
        dataset.locked_years = "1921-1939 absent and unread"
        dataset.production_file_modified = "false"
        dataset.enkf_run = "false"
        dataset.oggm_run = "false"

    mean_correction = np.nanmean(correction[ALLOWED_YEARS <= 1920], axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    for start, end in [(1500, 1920), (1940, 2000)]:
        rows = curve[curve["year"].between(start, end)]
        axes[0].plot(rows["year"], rows["authoritative_project_hma_degC"], color="#D55E00", lw=0.9, alpha=0.75)
        axes[0].plot(rows["year"], rows["candidate_contract_hma_degC"], color="#0072B2", lw=1.1, alpha=0.85)
    axes[0].set_ylabel("HMA anomaly (degC)")
    axes[0].set_title("Authoritative project posterior and frozen Candidate v2")
    axes[0].plot([], [], color="#D55E00", label="Project posterior")
    axes[0].plot([], [], color="#0072B2", label="Candidate v2")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(curve["year"], curve["direct_minus_contract_degC"], color="#009E73", lw=1.2)
    axes[1].set_ylabel("Direct minus contract (degC)")
    axes[1].set_title("One-cell support boundary effect on HMA mean")
    image = axes[2].imshow(
        mean_correction,
        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        origin="upper" if lat[0] > lat[-1] else "lower",
        cmap="RdBu_r",
        vmin=-max(0.1, float(np.nanmax(np.abs(mean_correction)))),
        vmax=max(0.1, float(np.nanmax(np.abs(mean_correction)))),
        aspect="auto",
    )
    missing_row, missing_col = np.where(unsupported_hma)
    axes[2].scatter(lon[missing_col], lat[missing_row], marker="x", color="#111111", s=45, label="Unsupported HMA cell")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    axes[2].set_title("Mean frozen correction, 1500-1920")
    axes[2].legend(frameon=False)
    fig.colorbar(image, ax=axes[2], label="degC")
    for axis in axes[:2]:
        axis.axhline(0, color="#777777", lw=0.6)
        axis.grid(axis="y", color="#DDDDDD", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure = OUTPUT / "candidate_v2_full_posterior_materialization.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    output_hash = sha256(output_nc)
    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = before == after
    gates = {
        "authoritative_project_curve_closure_at_most_1e_6_degC": project_curve_closure <= 1e-6,
        "frozen_correction_curve_closure_at_most_1e_6_degC": correction_curve_closure <= 1e-6,
        "frozen_candidate_curve_closure_at_most_1e_6_degC": candidate_curve_closure <= 1e-6,
        "field_algebra_closure_at_most_1e_6_degC": field_closure_max <= 1e-6,
        "background_algebra_closure_at_most_1e_6_degC": background_closure <= 1e-6,
        "archived_increment_invariance_at_most_1e_6_degC": increment_invariance <= 1e-6,
        "posterior_finite_on_common_support": finite_posterior_common,
        "correction_finite_on_common_support": finite_correction_common,
        "candidate_finite_on_common_support": finite_candidate_common,
        "candidate_unchanged_after_1950": unchanged_after_1950_max == 0.0,
        "candidate_unchanged_outside_common_support": unchanged_outside_common_max == 0.0,
        "all_core191_nearest_cells_on_common_support": core_common_count == 191,
        "declared_sources_unchanged": unchanged,
        "sealed_years_absent": not np.any((ALLOWED_YEARS >= LOCKED[0]) & (ALLOWED_YEARS <= LOCKED[1])),
    }
    passes = bool(all(gates.values()))
    decision = (
        "candidate_v2_full_posterior_materialized_and_production_ready_for_explicit_promotion_decision"
        if passes
        else "candidate_v2_materialized_but_requires_boundary_or_closure_resolution_before_promotion"
    )
    missing_row, missing_col = np.where(unsupported_hma)
    summary = {
        "schema": "hma-temperature-candidate-v2-full-posterior-materialization-v1",
        "status": "complete",
        "candidate_id": CANDIDATE_ID,
        "decision": decision,
        "passes_all_materialization_gates": passes,
        "year_segments": ["1500-1920", "1940-2000"],
        "grid_shape": [int(len(lat)), int(len(lon))],
        "hma_cell_count": int(hma_mask.sum()),
        "candidate_common_support_cell_count": int(common.sum()),
        "unsupported_hma_cells": [
            {"lat": float(lat[row]), "lon": float(lon[col])}
            for row, col in zip(missing_row, missing_col)
        ],
        "core191_nearest_cell_common_support_count": core_common_count,
        "closure": {
            "project_curve_max_abs_degC": project_curve_closure,
            "correction_curve_max_abs_degC": correction_curve_closure,
            "candidate_contract_curve_max_abs_degC": candidate_curve_closure,
            "direct_materialized_vs_contract_hma_max_abs_degC": direct_support_effect,
            "field_algebra_max_abs_degC": field_closure_max,
            "background_algebra_max_abs_degC": background_closure,
            "archived_increment_invariance_max_abs_degC": increment_invariance,
        },
        "unchanged_checks": {
            "correction_after_1950_max_abs_degC": correction_after_1950_max,
            "candidate_after_1950_max_abs_difference_degC": unchanged_after_1950_max,
            "candidate_outside_common_support_max_abs_difference_degC": unchanged_outside_common_max,
        },
        "gates": gates,
        "output_field": str(output_nc),
        "output_field_sha256": output_hash,
        "locked_1921_1939_numeric_values_accessed": 0,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate v2 full-posterior materialization\n\n"
        f"Decision: `{decision}`.\n\n"
        f"- Authoritative project-curve closure: `{project_curve_closure:.3e} degC`.\n"
        f"- Frozen correction-curve closure: `{correction_curve_closure:.3e} degC`.\n"
        f"- Frozen candidate-curve closure: `{candidate_curve_closure:.3e} degC`.\n"
        f"- Direct-field versus frozen HMA contract effect from the one unsupported boundary cell: `{direct_support_effect:.3e} degC`.\n"
        f"- Archived increment invariance: `{increment_invariance:.3e} degC`.\n"
        f"- Core-191 nearest cells on frozen common support: `{core_common_count}/191`.\n"
        f"- All materialization gates pass: `{str(passes).lower()}`.\n"
        "- Production files remain unchanged; no sealed-year value, EnKF, or OGGM run was used.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("declared-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
