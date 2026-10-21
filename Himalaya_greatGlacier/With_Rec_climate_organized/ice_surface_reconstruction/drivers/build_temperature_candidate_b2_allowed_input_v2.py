from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import acquire_round67_mode_ra_allowed_modern_subset_v2 as range_access
import build_temperature_candidate_b2_allowed_input_v1 as v1
import mode_sim_cdf2_parser_v1 as cdf2


v1.OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annual_mode(variable: str, points: np.ndarray, audit: list[dict[str, Any]]) -> np.ndarray:
    source = range_access.ExactRangeSource(
        v1.MODE[variable], f"candidate_b2_mode_{variable}", audit
    )
    try:
        header = cdf2.parse_header(
            source.read(0, range_access.HEADER_BYTES - 1, "header_only", None)
        )
        lat_var = range_access.coordinate_variable(header, ("lat", "latitude", "y"))
        lon_var = range_access.coordinate_variable(header, ("lon", "longitude", "x"))
        data_name = "temp2" if variable == "tas" else "totprec"
        data_var = header.variables[data_name]
        lat = range_access.read_fixed_variable(source, header, lat_var)
        lon = range_access.normalize_longitude(
            range_access.read_fixed_variable(source, header, lon_var)
        )
        lat_ids, lon_ids = v1.bbox(lat, lon, points)
        spatial_dims = list(data_var.dimensions[1:])
        lat_axis = spatial_dims.index(lat_var.dimensions[0])
        lon_axis = spatial_dims.index(lon_var.dimensions[0])
        annual = np.empty((len(v1.YEARS), len(lat_ids), len(lon_ids)), dtype=float)
        cursor = 0
        for start, end in v1.PERIODS:
            first = start - 1420
            last = end - 1420
            byte_start = header.record_base + first * header.record_size
            byte_end = header.record_base + (last + 1) * header.record_size - 1
            payload = source.read(
                byte_start, byte_end, f"allowed_{variable}_records", (start, end)
            )
            for local in range(end - start + 1):
                raw = range_access.record_segment(payload, header, data_var, local)
                field = range_access.decode_numeric(
                    raw, data_var, header.shape(data_var)[1:]
                )
                if (lat_axis, lon_axis) != (0, 1):
                    field = np.moveaxis(field, (lat_axis, lon_axis), (0, 1))
                annual[cursor + local] = field[np.ix_(lat_ids, lon_ids)]
            cursor += end - start + 1
        values = v1.interpolate(annual, lat[lat_ids], lon[lon_ids], points)
        units = str(data_var.attributes.get("units", ""))
    finally:
        source.close()
    if variable == "tas" and values.mean() > 100:
        values -= 273.15
    if variable == "pr":
        normalized = units.lower().replace(" ", "")
        if "kg" in normalized and ("s-1" in normalized or "/s" in normalized):
            values *= 86400.0
        elif values.mean() < 0.01:
            values *= 86400.0
    return values


def main() -> None:
    v1.annual_mode = annual_mode
    v1.main()
    summary_path = v1.OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = v1.OUT.name
    summary["supersedes_technical_failure"] = (
        "temperature_balanced_pooled_candidate_b2_allowed_input_v1"
    )
    summary["v2_change"] = "use returned CDF2 coordinate variable objects directly"
    summary["technical_retry_driver_sha256"] = sha256(Path(__file__).resolve())
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
