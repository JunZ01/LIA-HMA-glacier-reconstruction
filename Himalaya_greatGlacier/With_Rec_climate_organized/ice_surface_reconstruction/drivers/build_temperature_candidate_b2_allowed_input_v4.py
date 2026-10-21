from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import acquire_round67_mode_ra_allowed_modern_subset_v2 as access
import build_temperature_candidate_b2_allowed_input_v3 as v3
import mode_sim_cdf2_parser_v1 as cdf2


v1 = v3.v1
v1.OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v4"


def annual_mode(variable: str, points: np.ndarray, audit: list[dict[str, Any]]) -> np.ndarray:
    source = access.ExactRangeSource(v1.MODE[variable], f"candidate_b2_mode_{variable}", audit)
    try:
        header = cdf2.parse_header(source.read(0, access.HEADER_BYTES - 1, "header_only", None))
        lat_var = access.coordinate_variable(header, ("lat", "latitude", "y"))
        lon_var = access.coordinate_variable(header, ("lon", "longitude", "x"))
        data_var = header.variables["temp2" if variable == "tas" else "totprec"]
        lat = access.read_fixed_variable(source, header, lat_var)
        lon = access.normalize_longitude(access.read_fixed_variable(source, header, lon_var))
        lat_ids, lon_ids = v1.bbox(lat, lon, points)
        dims = list(data_var.dimensions[1:])
        axes = (dims.index(lat_var.dimensions[0]), dims.index(lon_var.dimensions[0]))
        annual = np.empty((len(v1.YEARS), len(lat_ids), len(lon_ids)), float)
        lookup = {int(year): i for i, year in enumerate(v1.YEARS)}
        for period_start, period_end in v1.PERIODS:
            for start in range(period_start, period_end + 1, 10):
                end = min(start + 9, period_end)
                first, last = start - 1420, end - 1420
                payload = source.read(
                    header.record_base + first * header.record_size,
                    header.record_base + (last + 1) * header.record_size - 1,
                    f"allowed_{variable}_ten_year_block", (start, end),
                )
                for local, year in enumerate(range(start, end + 1)):
                    raw = access.record_segment(payload, header, data_var, local)
                    field = access.decode_numeric(raw, data_var, header.shape(data_var)[1:])
                    if axes != (0, 1):
                        field = np.moveaxis(field, axes, (0, 1))
                    annual[lookup[year]] = field[np.ix_(lat_ids, lon_ids)]
                print(f"DONE ModE-Sim {variable} {start}-{end}", flush=True)
        values = v1.interpolate(annual, lat[lat_ids], lon[lon_ids], points)
        units = str(data_var.attributes.get("units", "")).lower().replace(" ", "")
    finally:
        source.close()
    if variable == "tas" and values.mean() > 100:
        values -= 273.15
    if variable == "pr" and (("kg" in units and ("s-1" in units or "/s" in units)) or values.mean() < 0.01):
        values *= 86400.0
    return values


def main() -> None:
    v1.annual_mode = annual_mode
    v1.annual_pmip = v3.annual_pmip
    v1.main()
    path = v1.OUT / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["experiment"] = v1.OUT.name
    summary["supersedes_transport_failure"] = "temperature_balanced_pooled_candidate_b2_allowed_input_v3"
    summary["v4_change"] = "split ModE-Sim into ten-year HTTP Range blocks"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
