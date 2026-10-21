"""Read-only identity and PSM metadata audit for the archived proxy9 network.

This stage establishes which original combined-proxy series every archived
proxy9 record actually uses. It reports metadata mismatches only where an
exact LMR metadata record is available; records without such metadata are
explicitly left unresolved rather than inferred from name or location.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


DRIVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DRIVER_DIR))
import audit_enkf_reconstruction_provenance as provenance


COMBINED = Path(r"E:\2Data\2Rec_ref\a_new_two_proxy\process_past_20000_proxy\combined_proxydata6.mat")
LMR_METADATA = Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\proxy-data\LMR_NCDC_database_v.0.2.0\lmr_meta.csv")
PROXY9 = provenance.DEFAULT_ARCHIVE_ROOT / "python_data" / "proxy_data_9.mat"
OUTPUT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm\validation_data\results\paleo_surface_validation_v0\climate_reconstruction_proxy_network_psm_audit_v1")
ANNUAL_MONTHS = [1, 12]
MIN_CORRELATION = 0.999999999
MAX_AFFINE_RESIDUAL = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(dataset: h5py.Dataset) -> str:
    return "".join(chr(int(value)) for value in np.asarray(dataset).reshape(-1, order="F") if int(value)).strip()


def affine_closure(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float]:
    valid = np.isfinite(left) & np.isfinite(right)
    x, y = right[valid], left[valid]
    if len(x) < 20:
        return np.nan, np.nan, np.nan, np.nan
    matrix = np.column_stack([x, np.ones(len(x))])
    slope, intercept = np.linalg.lstsq(matrix, y, rcond=None)[0]
    residual = y - matrix @ np.array([slope, intercept])
    return float(np.corrcoef(x, y)[0, 1]), float(slope), float(intercept), float(np.max(np.abs(residual)))


def combined_series(years: np.ndarray, values: np.ndarray, requested_years: np.ndarray) -> np.ndarray:
    lookup = {int(year): float(value) for year, value in zip(years, values) if np.isfinite(value)}
    return np.asarray([lookup.get(int(year), np.nan) for year in requested_years], dtype=float)


def expected_sign(relation: object) -> int | None:
    text = str(relation).strip().lower()
    if text == "positive":
        return 1
    if text == "negative":
        return -1
    return None


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    inputs = [COMBINED, LMR_METADATA, PROXY9]
    before = {str(path): sha256(path) for path in inputs}
    (OUTPUT / "input_manifest_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")

    combined = loadmat(COMBINED, squeeze_me=True, struct_as_record=False)
    years = np.asarray(combined["year"], dtype=int).reshape(-1)
    matrix = np.asarray(combined["combined6_data"], dtype=float)
    names = np.asarray(combined["dataName"]).reshape(-1)
    lat = np.asarray(combined["p_lat"], dtype=float).reshape(-1)
    lon = np.asarray(combined["p_lon"], dtype=float).reshape(-1)
    flags = np.asarray(combined["source_flag"], dtype=int).reshape(-1)
    lmr = pd.read_csv(LMR_METADATA)
    lmr_index = lmr.set_index("NCDC ID", drop=False)

    rows: list[dict[str, object]] = []
    with h5py.File(PROXY9, "r") as handle:
        references = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        train = np.asarray(handle["proxy_data/temp_proxy_indices"], dtype=int)
        all_indices = set(range(1, len(references) + 1))
        holdouts = {index: 0 for index in all_indices}
        for sequence in train:
            for index in all_indices.difference({int(value) for value in sequence if int(value) > 0}):
                holdouts[index] += 1
        for position, reference in enumerate(references, start=1):
            proxy = handle[reference]
            location = np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")
            proxy_years = np.asarray(proxy["age_yng"], dtype=int).reshape(-1, order="F")
            proxy_values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            coefficients = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            candidates = np.flatnonzero(np.isclose(lat, location[0], atol=1e-6) & np.isclose(lon, location[1], atol=1e-6))
            scored = []
            for column in candidates:
                target = combined_series(years, matrix[:, column], proxy_years)
                corr, slope, intercept, residual = affine_closure(proxy_values, target)
                scored.append((abs(corr) if np.isfinite(corr) else -np.inf, corr, slope, intercept, residual, int(column)))
            scored.sort(reverse=True)
            if not scored:
                raise RuntimeError(f"No coordinate candidate for proxy9 record {position}")
            _, corr, slope, intercept, residual, column = scored[0]
            if not np.isfinite(corr) or corr < MIN_CORRELATION or residual > MAX_AFFINE_RESIDUAL:
                raise RuntimeError(f"Identity closure failed for proxy9 record {position}: corr={corr}, residual={residual}")
            public_id = str(names[column]).strip()
            meta = lmr_index.loc[public_id] if public_id in lmr_index.index else None
            if isinstance(meta, pd.DataFrame):
                raise RuntimeError(f"Ambiguous LMR metadata ID: {public_id}")
            climate_variable = meta["climateVariable"] if meta is not None else None
            relation = meta["Relation_to_climateVariable"] if meta is not None else None
            seasonality = meta["Seasonality"] if meta is not None else None
            expected = expected_sign(relation) if meta is not None else None
            coefficient = np.nan
            if climate_variable == "temperature":
                coefficient = float(coefficients[1]) if len(coefficients) > 1 else np.nan
            elif climate_variable == "precipitation":
                coefficient = float(coefficients[2]) if len(coefficients) > 2 else np.nan
            sign_match = None if expected is None or not np.isfinite(coefficient) or coefficient == 0 else bool(np.sign(coefficient) == expected)
            rows.append({
                "proxy9_record_1based": position,
                "combined_column_1based": column + 1,
                "combined_name": public_id,
                "combined_source_flag": int(flags[column]),
                "latitude": float(location[0]),
                "longitude": float(location[1]),
                "proxy_type": decode(proxy["ptype"]),
                "psm_method": decode(proxy["psm_mthd"]),
                "psm_b0": float(coefficients[0]) if len(coefficients) else np.nan,
                "psm_temp_coefficient_b1": float(coefficients[1]) if len(coefficients) > 1 else np.nan,
                "psm_prcp_coefficient_b2": float(coefficients[2]) if len(coefficients) > 2 else np.nan,
                "psm_correlation": float(np.asarray(proxy["pcr"]).reshape(-1)[0]),
                "psm_temperature_correlation": float(np.asarray(proxy["pcr_t"]).reshape(-1)[0]),
                "psm_precipitation_correlation": float(np.asarray(proxy["pcr_p"]).reshape(-1)[0]),
                "psm_p_value": float(np.asarray(proxy["pval"]).reshape(-1)[0]),
                "first_finite_year": int(np.min(proxy_years[np.isfinite(proxy_values)])),
                "last_finite_year": int(np.max(proxy_years[np.isfinite(proxy_values)])),
                "holdout_frequency_across_10_repeats": holdouts[position],
                "identity_candidate_count_same_coordinate": len(candidates),
                "identity_correlation": corr,
                "identity_affine_slope": slope,
                "identity_affine_intercept": intercept,
                "identity_maximum_abs_residual": residual,
                "lmr_metadata_available": meta is not None,
                "lmr_climate_variable": climate_variable,
                "lmr_relation": relation,
                "lmr_seasonality": seasonality,
                "lmr_site_name": meta["Site name"] if meta is not None else None,
                "lmr_archive_type": meta["Archive type"] if meta is not None else None,
                "lmr_measurement": meta["Proxy measurement"] if meta is not None else None,
                "metadata_target_coefficient": coefficient,
                "metadata_expected_sign": expected,
                "metadata_coefficient_sign_match": sign_match,
                "archived_psm_months": "Jan-Dec",
                "seasonality_matches_archived_annual": None if meta is None else str(seasonality).strip() in {"[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]", "annual", "Annual"},
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "proxy9_identity_and_psm_metadata.csv", index=False)
    known = frame[frame.lmr_metadata_available].copy()
    sign_known = known[known.metadata_coefficient_sign_match.notna()].copy()
    seasonal = known[known.lmr_seasonality.notna()].copy()
    summary = {
        "proxy9_records": int(len(frame)),
        "identity_closure_minimum_correlation": float(frame.identity_correlation.min()),
        "identity_closure_maximum_affine_residual": float(frame.identity_maximum_abs_residual.max()),
        "records_with_multiple_same_coordinate_candidates": int((frame.identity_candidate_count_same_coordinate > 1).sum()),
        "lmr_metadata_exact_id_matches": int(len(known)),
        "lmr_metadata_unmatched": int(len(frame) - len(known)),
        "metadata_sign_testable_records": int(len(sign_known)),
        "metadata_sign_mismatches": int((~sign_known.metadata_coefficient_sign_match.astype(bool)).sum()),
        "metadata_nonannual_seasonality_records": int((~seasonal.seasonality_matches_archived_annual.astype(bool)).sum()),
        "records_never_held_out": int((frame.holdout_frequency_across_10_repeats == 0).sum()),
        "archived_psm_months": ANNUAL_MONTHS,
        "interpretation": "Metadata differences are candidate operator issues, not automatic corrections. Only exact-ID LMR matches are assessed here.",
    }
    pd.DataFrame([summary]).to_csv(OUTPUT / "summary.csv", index=False)
    flagged = known[(known.metadata_coefficient_sign_match == False) | (known.seasonality_matches_archived_annual == False)].copy()
    flagged.to_csv(OUTPUT / "lmr_metadata_operator_flags.csv", index=False)
    after = {str(path): sha256(path) for path in inputs}
    (OUTPUT / "input_manifest_after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("Declared input changed during audit")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Proxy Network PSM Audit v1\n\n"
        "All 297 archived proxy9 records are sequence-mapped to the original combined proxy archive by exact affine closure. "
        "The LMR metadata comparison is intentionally limited to exact public-ID matches and does not infer metadata for other records. "
        "See `summary.csv`, `proxy9_identity_and_psm_metadata.csv`, and `lmr_metadata_operator_flags.csv`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
