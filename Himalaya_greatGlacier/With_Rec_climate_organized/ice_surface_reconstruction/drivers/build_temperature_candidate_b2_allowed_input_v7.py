from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np

import build_temperature_candidate_b2_allowed_input_v3 as v3
import build_temperature_candidate_b2_allowed_input_v6 as v6
from test_esgf_retrying_range_file_v1 import RetryingRangeFile


v1 = v3.v1
v1.OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v7"


def annual_pmip(model: str, variable: str, points: np.ndarray, audit: list[dict[str, Any]]) -> np.ndarray:
    key = model.lower().replace("-", "_")
    cache_name = f"{key}_{variable}_{len(points)}points_482years"
    def build() -> np.ndarray:
        total = np.zeros((len(v1.YEARS), len(points)), float)
        count = np.zeros(len(v1.YEARS), np.int16)
        lookup = {int(year): i for i, year in enumerate(v1.YEARS)}
        for item in v1.pmip_items(model, variable):
            source_start, source_end = int(item["start_year_ce"]), int(item["end_year_ce"])
            url = v3.globus_url(str(item["url"]), model, variable, str(item["experiment_role"]))
            print(f"OPEN {model} {variable} {source_start}-{source_end}", flush=True)
            remote = RetryingRangeFile(url)
            try:
                with h5py.File(remote, "r") as handle:
                    lat = v3.decode(handle["lat"], handle["lat"][:])
                    lon = v3.decode(handle["lon"], handle["lon"][:])
                    lat_ids, lon_ids = v1.bbox(lat, lon, points)
                    data = handle[variable]
                    if data.chunks is None or int(data.chunks[0]) != 1:
                        raise RuntimeError(f"non-monthly-chunk:{model}:{variable}:{data.chunks}")
                    for allowed_start, allowed_end in v1.PERIODS:
                        start, end = max(source_start, allowed_start), min(source_end, allowed_end)
                        if start > end: continue
                        first, last = (start-source_start)*12, (end-source_start+1)*12
                        raw = v3.decode(data, data[first:last, lat_ids.min():lat_ids.max()+1, lon_ids.min():lon_ids.max()+1])
                        raw = raw.reshape(end-start+1, 12, len(lat_ids), len(lon_ids)).mean(axis=1)
                        if variable == "pr": raw *= 86400.0
                        values = v1.interpolate(raw, lat[lat_ids], lon[lon_ids], points)
                        if variable == "tas": values -= 273.15
                        for row, year in enumerate(range(start, end+1)):
                            total[lookup[year]] += values[row]; count[lookup[year]] += 1
                        audit.append({"model":model,"variable":variable,"source":url,"transport":"retrying_exact_https_range","time_chunk_months":1,"requested_year_start":start,"requested_year_end":end,"http_range_requests":remote.requests,"sealed_climate_values_requested":0})
                        print(f"DONE {model} {variable} {start}-{end}", flush=True)
            finally:
                remote.close()
        if not np.all(count == 1):
            raise RuntimeError(f"coverage:{model}:{variable}:{v1.YEARS[count != 1].tolist()[:20]}")
        return total
    return v6.cached(cache_name, build)


def main() -> None:
    v1.annual_mode = v6.mode
    v1.annual_pmip = annual_pmip
    v1.main()
    path = v1.OUT / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["experiment"] = v1.OUT.name
    summary["v7_transport"] = "retrying exact HTTPS Range file object with one-month HDF5 chunks"
    summary["cache_directory"] = str(v6.CACHE)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
