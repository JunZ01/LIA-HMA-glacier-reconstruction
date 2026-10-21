from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUT = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
    / "round_208_setp_cru_tmin_annual_target_audit_v1"
)
RAW = OUT / "cru_ts401_tmn_allowed_decades"
BASE = (
    "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.01/"
    "cruts.1709081022.v4.01/tmn/"
)
DECADES = (
    "1901.1910",
    "1911.1920",
    "1941.1950",
    "1951.1960",
    "1961.1970",
    "1971.1980",
    "1981.1990",
    "1991.2000",
)
FORBIDDEN_TOKENS = ("1921.1930", "1931.1940")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    records = []
    for decade in DECADES:
        filename = f"cru_ts4.01.{decade}.tmn.dat.nc.gz"
        url = BASE + filename
        if any(token in url for token in FORBIDDEN_TOKENS):
            raise RuntimeError(f"Forbidden CRU decade requested: {url}")
        compressed = RAW / filename
        partial = compressed.with_suffix(compressed.suffix + ".part")
        netcdf = RAW / filename.removesuffix(".gz")

        if not compressed.exists():
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "LIA-HMA-read-only-validation/1.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {url}")
                with partial.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
            partial.replace(compressed)

        with compressed.open("rb") as stream:
            if stream.read(2) != b"\x1f\x8b":
                raise RuntimeError(f"Not a gzip file: {compressed}")
        if not netcdf.exists():
            with gzip.open(compressed, "rb") as source, netcdf.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        with netcdf.open("rb") as stream:
            magic = stream.read(4)
        if magic not in (b"CDF\x01", b"CDF\x02", b"\x89HDF"):
            raise RuntimeError(f"Unexpected NetCDF magic {magic!r}: {netcdf}")

        records.append(
            {
                "url": url,
                "allowed_year_range": decade.replace(".", "-"),
                "compressed_path": str(compressed),
                "compressed_bytes": compressed.stat().st_size,
                "compressed_sha256": sha256(compressed),
                "netcdf_path": str(netcdf),
                "netcdf_bytes": netcdf.stat().st_size,
                "netcdf_sha256": sha256(netcdf),
                "numeric_values_parsed": False,
            }
        )

    summary = {
        "dataset": "CRU TS4.01 monthly minimum temperature",
        "official_base_url": BASE,
        "downloaded_allowed_decades": list(DECADES),
        "forbidden_decades_not_requested": list(FORBIDDEN_TOKENS),
        "file_count": len(records),
        "numeric_values_parsed": False,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
        "files": records,
    }
    (OUT / "cru_tmn_acquisition_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
