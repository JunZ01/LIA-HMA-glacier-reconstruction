from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import fsspec
import h5py
import numpy as np

import build_temperature_candidate_b2_allowed_input_v2 as v2


v1 = v2.v1
v1.OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_allowed_input_v3"
GLOBUS = "https://g-52ba3.fd635.8443.data.globus.org/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def globus_url(url: str, model: str, variable: str, role: str) -> str:
    marker = "/css03_data/"
    if marker not in url:
        raise RuntimeError(f"unexpected-esgf-url:{url}")
    path = "css03_data/" + url.split(marker, 1)[1]
    if model == "MIROC-ES2L" and role == "past":
        path = path.replace("/v20231115/", "/v20200318/")
    return GLOBUS + path


def decode(dataset: h5py.Dataset, values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    fill = dataset.attrs.get("_FillValue")
    if fill is not None:
        out[np.isclose(out, float(np.asarray(fill).reshape(-1)[0]))] = np.nan
    out *= float(np.asarray(dataset.attrs.get("scale_factor", 1.0)).reshape(-1)[0])
    out += float(np.asarray(dataset.attrs.get("add_offset", 0.0)).reshape(-1)[0])
    return out


def annual_pmip(
    model: str, variable: str, points: np.ndarray, audit: list[dict[str, Any]]
) -> np.ndarray:
    total = np.zeros((len(v1.YEARS), len(points)), dtype=float)
    count = np.zeros(len(v1.YEARS), dtype=np.int16)
    year_lookup = {int(year): index for index, year in enumerate(v1.YEARS)}
    for item in v1.pmip_items(model, variable):
        source_start = int(item["start_year_ce"])
        source_end = int(item["end_year_ce"])
        url = globus_url(str(item["url"]), model, variable, str(item["experiment_role"]))
        print(f"OPEN {model} {variable} {source_start}-{source_end}", flush=True)
        remote = fsspec.open(url, "rb", block_size=1, cache_type="none").open()
        try:
            with h5py.File(remote, "r") as handle:
                lat = decode(handle["lat"], handle["lat"][:])
                lon = decode(handle["lon"], handle["lon"][:])
                lat_ids, lon_ids = v1.bbox(lat, lon, points)
                data = handle[variable]
                if data.chunks is None or int(data.chunks[0]) != 1:
                    raise RuntimeError(f"non-monthly-time-chunk:{model}:{variable}:{data.chunks}")
                for allowed_start, allowed_end in v1.PERIODS:
                    start = max(source_start, allowed_start)
                    end = min(source_end, allowed_end)
                    if start > end:
                        continue
                    first = (start - source_start) * 12
                    last = (end - source_start + 1) * 12
                    raw = decode(
                        data,
                        data[
                            first:last,
                            lat_ids.min() : lat_ids.max() + 1,
                            lon_ids.min() : lon_ids.max() + 1,
                        ],
                    )
                    raw = raw.reshape(
                        end - start + 1, 12, len(lat_ids), len(lon_ids)
                    ).mean(axis=1)
                    if variable == "pr":
                        raw *= 86400.0
                    values = v1.interpolate(raw, lat[lat_ids], lon[lon_ids], points)
                    if variable == "tas":
                        values -= 273.15
                    for row, year in enumerate(range(start, end + 1)):
                        target = year_lookup[year]
                        total[target] += values[row]
                        count[target] += 1
                    audit.append({
                        "model": model, "variable": variable, "source": url,
                        "transport": "fsspec_https_range_h5py",
                        "time_chunk_months": int(data.chunks[0]),
                        "requested_year_start": start, "requested_year_end": end,
                        "requested_month_start_index": first,
                        "requested_month_end_index": last - 1,
                        "sealed_climate_values_requested": 0,
                    })
                    print(f"DONE {model} {variable} {start}-{end}", flush=True)
        finally:
            remote.close()
    if not np.all(count == 1):
        raise RuntimeError(f"coverage:{model}:{variable}:{v1.YEARS[count != 1].tolist()[:30]}")
    return total


def main() -> None:
    v1.annual_mode = v2.annual_mode
    v1.annual_pmip = annual_pmip
    v1.main()
    path = v1.OUT / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["experiment"] = v1.OUT.name
    summary["supersedes_transport_failure"] = (
        "temperature_balanced_pooled_candidate_b2_allowed_input_v2"
    )
    summary["v3_transport"] = "official ESGF Globus HTTPS Range with one-month HDF5 chunks"
    summary["v3_driver_sha256"] = sha256(Path(__file__).resolve())
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
