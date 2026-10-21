"""Trace proxy9 records 203 and 241 back to public proxy identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_char(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(value)) for value in values if int(value)).strip()


def load_proxy_records(path: Path, record_ids: list[int]) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    with h5py.File(path, "r") as archive:
        references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        for record_id in record_ids:
            proxy = archive[references[record_id - 1]]
            years = np.asarray(proxy["age_yng"], dtype=float).reshape(-1, order="F")
            values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            finite = np.isfinite(years) & np.isfinite(values)
            records[record_id] = {
                "years": years[finite].astype(int),
                "values": values[finite],
                "location": np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")[:2],
                "ptype": decode_char(proxy["ptype"]),
                "psm_method": decode_char(proxy["psm_mthd"]),
            }
    return records


def affine_closure(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(left) & np.isfinite(right)
    x = right[valid]
    y = left[valid]
    matrix = np.column_stack([x, np.ones(len(x))])
    slope, intercept = np.linalg.lstsq(matrix, y, rcond=None)[0]
    residual = y - matrix @ np.array([slope, intercept])
    return {
        "n": int(valid.sum()),
        "correlation": float(np.corrcoef(y, x)[0, 1]),
        "slope": float(slope),
        "intercept": float(intercept),
        "maximum_abs_residual": float(np.max(np.abs(residual))),
    }


def align(years: np.ndarray, values: np.ndarray, target_years: np.ndarray) -> np.ndarray:
    by_year = {int(year): float(value) for year, value in zip(years, values) if np.isfinite(value)}
    return np.array([by_year.get(int(year), np.nan) for year in target_years], dtype=float)


def scalar_text(value: object) -> str:
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return str(value).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    args = parser.parse_args()
    root = project_root()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {name: Path(value) for name, value in cfg["sources"].items()}
    output = root / cfg["result_root"]
    output.mkdir(parents=True, exist_ok=True)

    hashes_before = {name: sha256(path) for name, path in paths.items()}
    targets = load_proxy_records(paths["proxy9"], [203, 241])
    gates = cfg["gates"]
    closure_rows: list[dict[str, object]] = []
    combined_matches: dict[int, dict[str, object]] = {}

    for source_name in ("combined_desktop", "combined_archive"):
        combined = scipy.io.loadmat(paths[source_name], simplify_cells=True)
        years = np.asarray(combined["year"], dtype=float).reshape(-1).astype(int)
        data = np.asarray(combined["combined6_data"], dtype=float)
        lat = np.asarray(combined["p_lat"], dtype=float).reshape(-1)
        lon = np.asarray(combined["p_lon"], dtype=float).reshape(-1)
        archive = np.atleast_1d(combined["archive"])
        names = np.asarray(combined["dataName"]).reshape(-1)
        flags = np.asarray(combined["source_flag"], dtype=int).reshape(-1)
        for record_id, target in targets.items():
            expected = cfg["records"][str(record_id)]
            location = np.asarray(expected["expected_location"], dtype=float)
            candidates = np.flatnonzero(np.isclose(lat, location[0]) & np.isclose(lon, location[1]))
            if len(candidates) != 1:
                raise RuntimeError(f"combined-coordinate-match-{source_name}-{record_id}-{len(candidates)}")
            column = int(candidates[0])
            source_aligned = align(years, data[:, column], np.asarray(target["years"]))
            closure = affine_closure(np.asarray(target["values"]), source_aligned)
            closure_rows.append({"record": record_id, "stage": source_name, "column_1based": column + 1, **closure})
            if closure["correlation"] < gates["minimum_correlation"] or closure["maximum_abs_residual"] > gates["maximum_affine_residual"]:
                raise RuntimeError(f"combined-affine-closure-failed-{source_name}-{record_id}")
            match = {
                "combined_column_1based": column + 1,
                "combined_name": scalar_text(names[column]),
                "combined_source_flag": int(flags[column]),
                "combined_archive": scalar_text(archive[column]),
            }
            previous = combined_matches.setdefault(record_id, match)
            if previous != match:
                raise RuntimeError(f"combined-candidate-disagreement-{record_id}")

    mann = scipy.io.loadmat(paths["mann_source"], simplify_cells=True)
    mann_ids = np.atleast_1d(mann["paleoData_pages2kID"])
    mann_index = np.flatnonzero(np.array([scalar_text(value) == cfg["records"]["203"]["mann_pages2k_id"] for value in mann_ids]))
    if len(mann_index) != 1:
        raise RuntimeError(f"mann-id-match-count-{len(mann_index)}")
    mann_index = int(mann_index[0])
    mann_years = np.asarray(np.atleast_1d(mann["year"])[mann_index], dtype=float).reshape(-1).astype(int)
    mann_values = np.asarray(np.atleast_1d(mann["paleoData_values"])[mann_index], dtype=float).reshape(-1)
    mann_aligned = align(mann_years, mann_values, np.asarray(targets[203]["years"]))
    mann_closure = affine_closure(np.asarray(targets[203]["values"]), mann_aligned)
    closure_rows.append({"record": 203, "stage": "mann_source", "column_1based": mann_index + 1, **mann_closure})
    if mann_closure["correlation"] < gates["minimum_correlation"] or mann_closure["maximum_abs_residual"] > gates["maximum_affine_residual"]:
        raise RuntimeError("mann-source-affine-closure-failed")

    lmr_meta = pd.read_csv(paths["lmr_metadata"])
    lmr_id = cfg["records"]["241"]["lmr_id"]
    lmr_row = lmr_meta.loc[lmr_meta["NCDC ID"].eq(lmr_id)]
    if len(lmr_row) != 1:
        raise RuntimeError(f"lmr-metadata-match-count-{len(lmr_row)}")
    lmr_row = lmr_row.iloc[0]
    lmr_data = pd.read_csv(paths["lmr_series"], usecols=["Year C.E.", lmr_id])
    lmr_aligned = align(lmr_data["Year C.E."].to_numpy(), lmr_data[lmr_id].to_numpy(), np.asarray(targets[241]["years"]))
    lmr_closure = affine_closure(np.asarray(targets[241]["values"]), lmr_aligned)
    closure_rows.append({"record": 241, "stage": "lmr_series", "column_1based": int(lmr_meta.index[lmr_meta["NCDC ID"].eq(lmr_id)][0]) + 1, **lmr_closure})
    if lmr_closure["correlation"] < gates["minimum_correlation"] or lmr_closure["maximum_abs_residual"] > gates["maximum_affine_residual"]:
        raise RuntimeError("lmr-source-affine-closure-failed")

    parameters = paths["pipeline_parameters"].read_text(encoding="utf-8", errors="replace")
    active_parameters = "\n".join(line.split("%", 1)[0] for line in parameters.splitlines())
    annual_match = re.search(r"mon_avg_o\s*=\s*(\d+)\s*;\s*mon_avg_f\s*=\s*(\d+)", active_parameters)
    calibration_match = re.search(r"clb_prd\s*=\s*\[\s*(\d+)\s+(\d+)\s*\]", active_parameters)
    if annual_match is None or calibration_match is None:
        raise RuntimeError("pipeline-parameter-parse-failed")
    pipeline_months = [int(annual_match.group(1)), int(annual_match.group(2))]
    calibration_period = [int(calibration_match.group(1)), int(calibration_match.group(2))]

    identity_rows = [
        {
            "proxy9_record": 203,
            **combined_matches[203],
            "public_id": scalar_text(mann_ids[mann_index]),
            "site_name": cfg["records"]["203"]["public_site_name"],
            "latitude": float(np.asarray(mann["geo_meanLat"]).reshape(-1)[mann_index]),
            "longitude": float(np.asarray(mann["geo_meanLon"]).reshape(-1)[mann_index]),
            "elevation_m": float(np.asarray(mann["geo_meanElev"]).reshape(-1)[mann_index]),
            "species": cfg["records"]["203"]["public_species"],
            "measurement": scalar_text(np.atleast_1d(mann["paleoData_proxy"])[mann_index]),
            "variable": scalar_text(np.atleast_1d(mann["paleoData_variableName"])[mann_index]),
            "first_year": int(np.min(mann_years)),
            "last_year": int(np.max(mann_years)),
            "seasonality": "not encoded in local Mann_2008 source",
            "seasonality_evidence": cfg["records"]["203"]["seasonality_boundary"],
            "citation": cfg["records"]["203"]["dataset_citation"],
            "primary_investigators": cfg["records"]["203"]["primary_chronology_investigators"],
        },
        {
            "proxy9_record": 241,
            **combined_matches[241],
            "public_id": lmr_id,
            "site_name": scalar_text(lmr_row["Site name"]),
            "latitude": float(lmr_row["Lat (N)"]),
            "longitude": float(lmr_row["Lon (E)"]),
            "elevation_m": float(lmr_row["Elev"]),
            "species": cfg["records"]["241"]["public_species"],
            "measurement": scalar_text(lmr_row["Proxy measurement"]),
            "variable": scalar_text(lmr_row["climateVariable"]),
            "first_year": int(lmr_row["Oldest (C.E.)"]),
            "last_year": int(lmr_row["Youngest (C.E.)"]),
            "seasonality": scalar_text(lmr_row["Seasonality"]),
            "seasonality_evidence": "LMR metadata; JJA months [6, 7, 8]",
            "citation": cfg["records"]["241"]["dataset_citation"],
            "primary_investigators": scalar_text(lmr_row["Investigators"]),
        },
    ]
    pd.DataFrame(identity_rows).to_csv(output / "record_public_identity.csv", index=False)
    pd.DataFrame(closure_rows).to_csv(output / "affine_closure.csv", index=False)

    hashes_after = {name: sha256(path) for name, path in paths.items()}
    pure = hashes_before == hashes_after
    maximum_residual = max(float(row["maximum_abs_residual"]) for row in closure_rows)
    minimum_correlation = min(float(row["correlation"]) for row in closure_rows)
    status = "pass" if pure and maximum_residual <= gates["maximum_affine_residual"] and minimum_correlation >= gates["minimum_correlation"] else "fail"
    summary = {
        "experiment_id": cfg["experiment_id"],
        "status": status,
        "source_hashes_unchanged": pure,
        "minimum_correlation": minimum_correlation,
        "maximum_affine_residual": maximum_residual,
        "pipeline_climate_months": pipeline_months,
        "pipeline_calibration_period": calibration_period,
        "identity": identity_rows,
        "important_boundary": {
            "record_203": "The exact proxy9 series is Mann_2008/PAGES2k ID indi002x (MXD), not the shorter LMR Breitenmoser Gulmarg ring-width series at the same coordinates.",
            "record_241": "The exact proxy9 series is the LMR Breitenmoser record asia_indi007B:trsgi_breit (Thijwas A), whose metadata seasonality is JJA.",
            "pipeline": "The archived EnKF parameter file uses January-December climate means, so record 241 has a documented seasonality mismatch that warrants a controlled sensitivity experiment."
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    (output / "input_manifest.json").write_text(json.dumps({"before": hashes_before, "after": hashes_after}, indent=2), encoding="utf-8")
    table = pd.DataFrame(identity_rows)[["proxy9_record", "public_id", "site_name", "species", "measurement", "first_year", "last_year", "seasonality", "combined_column_1based"]]
    (output / "FINAL_RESULT.md").write_text(
        "# Proxy9 Record Provenance\n\n"
        f"Status: **{status}**. All source hashes unchanged: **{pure}**. "
        f"Minimum sequence correlation: `{minimum_correlation:.12f}`; maximum affine residual: `{maximum_residual:.3e}`.\n\n"
        "## Identities\n\n"
        f"{table.to_markdown(index=False)}\n\n"
        "## Decision-relevant result\n\n"
        "Record 203 is the Mann_2008/PAGES2k `indi002x` maximum-latewood-density series from Gulmarg. "
        "Record 241 is the LMR/Breitenmoser `asia_indi007B:trsgi_breit` series from Thijwas A, with JJA metadata seasonality. "
        "The archived reconstruction parameter file instead uses January-December climate means. This is a concrete, testable PSM seasonality mismatch for record 241; it is not evidence that the chronology itself is invalid.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    if status != "pass":
        raise RuntimeError("provenance-hard-gate-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
