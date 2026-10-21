from __future__ import annotations

import io
import json
import time
from pathlib import Path

import h5py
import numpy as np
import requests


URL = "https://g-52ba3.fd635.8443.data.globus.org/css03_data/CMIP6/PMIP/MPI-M/MPI-ESM1-2-LR/past2k/r1i1p1f1/Amon/tas/gn/v20210714/tas_Amon_MPI-ESM1-2-LR_past2k_r1i1p1f1_gn_849101-851012.nc"
OUT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm\validation_data\results\paleo_surface_validation_v0\climate_reconstruction_improvement_loop_v2\temperature_candidate_b2_esgf_retrying_range_fixture_v1")


class RetryingRangeFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        head = self.session.head(url, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        self.pos = 0
        self.requests = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos
    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)
    def read(self, size=-1):
        if size < 0: size = self.size - self.pos
        if size == 0 or self.pos >= self.size: return b""
        end = min(self.pos + size, self.size) - 1
        last = None
        for attempt in range(1, 9):
            try:
                response = self.session.get(
                    self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=(30, 120)
                )
                if response.status_code not in (200, 206): response.raise_for_status()
                data = response.content
                expected = end - self.pos + 1
                if len(data) != expected: raise IOError(f"range-length:{len(data)}:{expected}")
                self.pos = end + 1
                self.requests += 1
                return data
            except Exception as error:
                last = error
                time.sleep(min(3 * attempt, 20))
                self.session.close()
                self.session = requests.Session()
        raise last
    def close(self):
        self.session.close()
        super().close()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    remote = RetryingRangeFile(URL)
    with h5py.File(remote, "r") as handle:
        data = handle["tas"]
        values = np.asarray(data[108:120, 60:61, 40:41], dtype=float).reshape(-1)
        result = {
            "status": "complete", "shape": list(data.shape), "chunks": list(data.chunks),
            "fixture_months": 12, "fixture_source_year": 8500,
            "finite": bool(np.all(np.isfinite(values))), "minimum_K": float(values.min()),
            "maximum_K": float(values.max()), "http_range_request_count": remote.requests,
            "sealed_values_read": False, "production_modified": False,
            "enkf_run": False, "oggm_run": False,
        }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
