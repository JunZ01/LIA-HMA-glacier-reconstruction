"""Audit Candidate v2 correction discontinuities adjacent to the sealed period."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
FULL = LOOP / (
    "temperature_candidate_v2_full_posterior_materialization_v1/"
    "temperature_candidate_v2_full_posterior_allowed_years.nc"
)
FULL_SUMMARY = LOOP / "temperature_candidate_v2_full_posterior_materialization_v1/summary.json"
VERIFY = LOOP / (
    "temperature_candidate_v2_full_posterior_independent_verification_v2/"
    "RELEASE_ADDENDUM.json"
)
FROZEN_SUMMARY = LOOP / "temperature_candidate_v2_formal_freeze_v1/summary.json"
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
OUTPUT = LOOP / "temperature_candidate_v2_sealed_boundary_transition_audit_v1"

CANDIDATE_ID = "HMA-TEMP-CANDIDATE-V2-FROZEN-20260828"
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


def weighted_mean(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(values) & mask[None]
    return np.nansum(values * weights[None], axis=(1, 2)) / np.sum(
        valid * weights[None], axis=(1, 2)
    )


def weighted_rms(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(values) & mask[None]
    return np.sqrt(
        np.nansum(values * values * weights[None], axis=(1, 2))
        / np.sum(valid * weights[None], axis=(1, 2))
    )


def nearest(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    return np.abs(values[:, None] - coordinates[None, :]).argmin(axis=1)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("inf")


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "BOUNDARY_AUDIT_CONTRACT.md"
    contract.write_text(
        "# Candidate v2 sealed-boundary transition audit\n\n"
        "Use only the allowed Candidate v2 correction through 1920 and from 1940 onward. "
        "Do not read, infer, interpolate, score, or plot any 1921-1939 climate value. Quantify "
        "the correction discontinuity that would be imposed if a future continuous file blindly "
        "retained the original project posterior in the sealed interval, which is algebraically "
        "equivalent to setting correction to zero there. Compare each endpoint discontinuity "
        "with the largest and 99th-percentile annual correction change already present within "
        "the allowed 1500-1920 segment. Report HMA, 15 GTN-G regions, and Core-191 centroids. "
        "This audit does not choose a bridge, alter Candidate v2, authorize production, or run "
        "EnKF/OGGM.\n",
        encoding="utf-8",
    )
    sources = {
        "full_candidate": FULL,
        "full_summary": FULL_SUMMARY,
        "independent_release": VERIFY,
        "frozen_summary": FROZEN_SUMMARY,
        "grid_assignment": GRID_ASSIGNMENT,
        "core191": CORE191,
        "facts": FACTS,
        "execution_contract": EXECUTION,
        "current_state": STATE,
        "driver": DRIVER,
        "contract": contract,
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: file_state(path) for name, path in sources.items()}

    full_summary = json.loads(FULL_SUMMARY.read_text(encoding="utf-8"))
    release = json.loads(VERIFY.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_SUMMARY.read_text(encoding="utf-8"))
    if any(item["candidate_id"] != CANDIDATE_ID for item in [full_summary, release, frozen]):
        raise RuntimeError("candidate-id-chain-failed")
    if release["status"] != "independently-verified-frozen-candidate-not-production":
        raise RuntimeError("independent-release-not-valid")
    if sha256(FULL) != full_summary["output_field_sha256"]:
        raise RuntimeError("full-field-hash-failed")

    with Dataset(FULL) as dataset:
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        common = np.asarray(dataset.variables["candidate_common_support_mask"][:], dtype=bool)
        correction = np.asarray(
            dataset.variables["tas_candidate_v2_correction"][:], dtype=np.float64
        )
    if np.any((years >= LOCKED[0]) & (years <= LOCKED[1])):
        raise RuntimeError("locked-year-in-candidate")
    year_index = {int(year): index for index, year in enumerate(years)}
    for required in [1500, 1920, 1940, 1950, 1951, 2000]:
        if required not in year_index:
            raise RuntimeError(f"required-year-missing:{required}")

    early = (years >= 1500) & (years <= 1920)
    early_correction = correction[early]
    early_years = years[early]
    early_changes = np.diff(early_correction, axis=0)
    hma_series = weighted_mean(early_correction, common, lat)
    hma_changes = np.diff(hma_series)
    hma_abs_changes = np.abs(hma_changes)
    hma_internal_max = float(np.max(hma_abs_changes))
    hma_internal_p99 = float(np.quantile(hma_abs_changes, 0.99))
    hma_internal_rms_max = float(np.max(weighted_rms(early_changes, common, lat)))

    field_1920 = correction[year_index[1920]]
    field_1940 = correction[year_index[1940]]
    field_1950 = correction[year_index[1950]]
    field_1951 = correction[year_index[1951]]
    boundary_fields = {
        "1920_to_hypothetical_zero_1921": -field_1920,
        "hypothetical_zero_1939_to_1940": field_1940,
        "1950_to_actual_zero_1951": field_1951 - field_1950,
    }
    hma_boundary = {}
    for name, values in boundary_fields.items():
        mean_value = float(weighted_mean(values[None], common, lat)[0])
        rms_value = float(weighted_rms(values[None], common, lat)[0])
        max_value = float(np.nanmax(np.abs(values[common])))
        hma_boundary[name] = {
            "area_weighted_mean_degC": mean_value,
            "absolute_area_weighted_mean_degC": abs(mean_value),
            "area_weighted_spatial_rms_degC": rms_value,
            "cell_max_abs_degC": max_value,
            "mean_jump_over_internal_max": safe_ratio(abs(mean_value), hma_internal_max),
            "mean_jump_over_internal_p99": safe_ratio(abs(mean_value), hma_internal_p99),
            "spatial_rms_jump_over_internal_rms_max": safe_ratio(rms_value, hma_internal_rms_max),
        }

    assignment = pd.read_csv(GRID_ASSIGNMENT)
    region_rows = []
    for region, cells in assignment.groupby("o2region", sort=True):
        mask = np.zeros_like(common)
        mask[cells["grid_row"].to_numpy(int), cells["grid_col"].to_numpy(int)] = True
        mask &= common
        regional_series = weighted_mean(early_correction, mask, lat)
        regional_internal = np.abs(np.diff(regional_series))
        internal_max = float(np.max(regional_internal))
        internal_p99 = float(np.quantile(regional_internal, 0.99))
        for boundary, values in boundary_fields.items():
            mean_jump = float(weighted_mean(values[None], mask, lat)[0])
            region_rows.append(
                {
                    "o2region": region,
                    "boundary": boundary,
                    "common_cell_count": int(mask.sum()),
                    "area_weighted_mean_jump_degC": mean_jump,
                    "absolute_mean_jump_degC": abs(mean_jump),
                    "allowed_internal_annual_change_max_abs_degC": internal_max,
                    "allowed_internal_annual_change_p99_abs_degC": internal_p99,
                    "jump_over_internal_max": safe_ratio(abs(mean_jump), internal_max),
                    "jump_over_internal_p99": safe_ratio(abs(mean_jump), internal_p99),
                }
            )
    regions = pd.DataFrame(region_rows)
    regions.to_csv(OUTPUT / "gtng15_sealed_boundary_jump_audit.csv", index=False)

    core = pd.read_csv(CORE191)
    if len(core) != 191 or core["lia_unique_id"].nunique() != 191:
        raise RuntimeError("core191-contract-failed")
    rows = nearest(core["centroid_lat"].to_numpy(float), lat)
    cols = nearest(core["centroid_lon"].to_numpy(float), lon)
    if not np.all(common[rows, cols]):
        raise RuntimeError("core191-outside-common-support")
    core_early = early_correction[:, rows, cols]
    core_internal_changes = np.abs(np.diff(core_early, axis=0))
    core_audit = core[["lia_unique_id", "centroid_lat", "centroid_lon", "o2region"]].copy()
    core_audit["nearest_lat"] = lat[rows]
    core_audit["nearest_lon"] = lon[cols]
    core_audit["allowed_internal_change_max_abs_degC"] = np.max(core_internal_changes, axis=0)
    core_audit["allowed_internal_change_p99_abs_degC"] = np.quantile(core_internal_changes, 0.99, axis=0)
    for boundary, values in boundary_fields.items():
        jump = values[rows, cols]
        label = {
            "1920_to_hypothetical_zero_1921": "jump_1920_to_zero",
            "hypothetical_zero_1939_to_1940": "jump_zero_to_1940",
            "1950_to_actual_zero_1951": "jump_1950_to_1951",
        }[boundary]
        core_audit[f"{label}_degC"] = jump
        core_audit[f"{label}_abs_over_internal_max"] = np.abs(jump) / core_audit[
            "allowed_internal_change_max_abs_degC"
        ].replace(0, np.nan)
    core_audit.to_csv(OUTPUT / "core191_sealed_boundary_jump_audit.csv", index=False)

    first_boundary = core_audit["jump_1920_to_zero_abs_over_internal_max"]
    second_boundary = core_audit["jump_zero_to_1940_abs_over_internal_max"]
    actual_1951 = core_audit["jump_1950_to_1951_abs_over_internal_max"]
    core_summary = {
        "systems": 191,
        "1920_to_zero_jump_exceeds_own_internal_max_count": int((first_boundary > 1).sum()),
        "1920_to_zero_jump_exceeds_5x_own_internal_max_count": int((first_boundary > 5).sum()),
        "zero_to_1940_jump_exceeds_own_internal_max_count": int((second_boundary > 1).sum()),
        "actual_1950_to_1951_jump_exceeds_own_internal_max_count": int((actual_1951 > 1).sum()),
        "1920_to_zero_ratio_median": float(first_boundary.median()),
        "1920_to_zero_ratio_max": float(first_boundary.max()),
        "zero_to_1940_ratio_median": float(second_boundary.median()),
        "actual_1950_to_1951_ratio_median": float(actual_1951.median()),
    }

    region_1920 = regions[regions["boundary"].eq("1920_to_hypothetical_zero_1921")]
    region_1940 = regions[regions["boundary"].eq("hypothetical_zero_1939_to_1940")]
    hma_1920_exceeds_internal = (
        hma_boundary["1920_to_hypothetical_zero_1921"]["mean_jump_over_internal_max"] > 1
    )
    hma_1940_exceeds_internal = (
        hma_boundary["hypothetical_zero_1939_to_1940"]["mean_jump_over_internal_max"] > 1
    )
    classification = (
        "blind_zero_correction_gap_would_create_out_of_family_boundary_discontinuity_requires_new_explicit_transition_contract"
        if hma_1920_exceeds_internal or hma_1940_exceeds_internal
        else "blind_zero_correction_gap_not_larger_than_allowed_internal_correction_changes"
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    focus = (early_years >= 1880) & (early_years <= 1920)
    axes[0].plot(early_years[focus], hma_series[focus], color="#0072B2", lw=2.0)
    axes[0].scatter([1920], [hma_series[-1]], color="#D55E00", zorder=3, label="1920 allowed endpoint")
    axes[0].scatter(
        [1940],
        [weighted_mean(field_1940[None], common, lat)[0]],
        color="#009E73",
        zorder=3,
        label="1940 allowed endpoint",
    )
    axes[0].axvspan(1921, 1939, color="#EEEEEE", alpha=0.9)
    axes[0].text(1930, 0.5 * sum(axes[0].get_ylim()), "sealed: no values read", ha="center", color="#555555")
    axes[0].set_xlim(1880, 1950)
    axes[0].set_ylabel("HMA correction (degC)")
    axes[0].set_title("Allowed Candidate v2 correction endpoints around the sealed interval")
    axes[0].legend(frameon=False)
    order = region_1920.sort_values("jump_over_internal_max", ascending=False)
    x = np.arange(len(order))
    axes[1].bar(x - 0.18, order["jump_over_internal_max"], 0.36, color="#D55E00", label="1920 to zero")
    lookup = region_1940.set_index("o2region")["jump_over_internal_max"]
    axes[1].bar(x + 0.18, order["o2region"].map(lookup), 0.36, color="#009E73", label="zero to 1940")
    axes[1].axhline(1, color="#333333", lw=0.9, ls="--")
    axes[1].set_xticks(x, order["o2region"], rotation=45, ha="right")
    axes[1].set_ylabel("Boundary jump / allowed internal max")
    axes[1].set_title("Regional boundary discontinuity relative to Candidate v2's own annual smoothness")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    figure = OUTPUT / "candidate_v2_sealed_boundary_transition_audit.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: file_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary = {
        "schema": "hma-temperature-candidate-v2-sealed-boundary-transition-audit-v1",
        "status": "complete",
        "candidate_id": CANDIDATE_ID,
        "classification": classification,
        "assessed_policy": "hypothetical zero correction in sealed interval, equivalent to blind retention of original posterior",
        "hma_allowed_internal_correction_change": {
            "annual_max_abs_degC": hma_internal_max,
            "annual_p99_abs_degC": hma_internal_p99,
            "spatial_rms_annual_max_degC": hma_internal_rms_max,
        },
        "hma_boundaries": hma_boundary,
        "regions": {
            "count": int(regions["o2region"].nunique()),
            "1920_to_zero_exceeds_internal_max_count": int((region_1920["jump_over_internal_max"] > 1).sum()),
            "zero_to_1940_exceeds_internal_max_count": int((region_1940["jump_over_internal_max"] > 1).sum()),
        },
        "core191": core_summary,
        "minimum_scientific_decision_before_continuous_production": (
            "Choose and preregister a transition contract for 1921-1939. Blindly retaining the "
            "original posterior is not automatically equivalent to a smooth continuation of "
            "Candidate v2. Any bridge is a new candidate version, not Candidate v2."
        ),
        "locked_1921_1939_numeric_values_accessed": 0,
        "locked_1921_1939_values_inferred_or_interpolated": 0,
        "all_declared_sources_unchanged": unchanged,
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
        "# Candidate v2 sealed-boundary transition audit\n\n"
        f"Classification: `{classification}`.\n\n"
        f"- HMA 1920-to-zero mean jump: `{hma_boundary['1920_to_hypothetical_zero_1921']['area_weighted_mean_degC']:+.4f} degC`, "
        f"`{hma_boundary['1920_to_hypothetical_zero_1921']['mean_jump_over_internal_max']:.2f}x` the largest allowed internal annual correction change.\n"
        f"- HMA zero-to-1940 mean jump: `{hma_boundary['hypothetical_zero_1939_to_1940']['area_weighted_mean_degC']:+.4f} degC`, "
        f"`{hma_boundary['hypothetical_zero_1939_to_1940']['mean_jump_over_internal_max']:.2f}x` the internal maximum.\n"
        f"- Regions exceeding their own internal maximum at the two boundaries: "
        f"`{int((region_1920['jump_over_internal_max'] > 1).sum())}/15` and "
        f"`{int((region_1940['jump_over_internal_max'] > 1).sum())}/15`.\n"
        f"- Core-191 systems exceeding their own internal maximum: "
        f"`{core_summary['1920_to_zero_jump_exceeds_own_internal_max_count']}/191` and "
        f"`{core_summary['zero_to_1940_jump_exceeds_own_internal_max_count']}/191`.\n"
        "- No sealed-period value was read or generated. Candidate v2 and production remain unchanged.\n",
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("declared-source-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
