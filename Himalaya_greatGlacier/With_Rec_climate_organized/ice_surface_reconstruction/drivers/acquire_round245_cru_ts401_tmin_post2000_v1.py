from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


BASE_URL = (
    "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.01/"
    "cruts.1709081022.v4.01/tmn/"
)
DECADES = ("2001.2010", "2011.2016")
FORBIDDEN = ("1921.1930", "1931.1940")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/"
        "round_245_cru_ts401_tmin_post2000_acquisition_v1"
    )
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw_unopened"
    extracted = output / "extracted"
    raw.mkdir()
    extracted.mkdir()
    inputs = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round245_cru_ts401_tmin_post2000_acquisition_v1_contract.md",
        "driver": Path(__file__).resolve(),
    }
    before = {name: sha256(path) for name, path in inputs.items()}
    records = []
    for decade in DECADES:
        filename = f"cru_ts4.01.{decade}.tmn.dat.nc.gz"
        url = BASE_URL + filename
        if any(token in url for token in FORBIDDEN):
            raise RuntimeError(f"forbidden-decade-request:{url}")
        compressed = raw / filename
        partial = compressed.with_suffix(compressed.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "LIA-HMA-read-only-validation/1.0"})
        with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as stream:
            if response.status != 200:
                raise RuntimeError(f"http-status:{response.status}:{url}")
            shutil.copyfileobj(response, stream, length=1024 * 1024)
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "")
            last_modified = response.headers.get("Last-Modified", "")
        partial.replace(compressed)
        with compressed.open("rb") as stream:
            if stream.read(2) != b"\x1f\x8b":
                raise RuntimeError(f"not-gzip:{compressed.name}")
        netcdf = extracted / filename.removesuffix(".gz")
        with gzip.open(compressed, "rb") as source, netcdf.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        with netcdf.open("rb") as stream:
            magic = stream.read(4)
        if magic not in (b"CDF\x01", b"CDF\x02", b"\x89HDF"):
            raise RuntimeError(f"unexpected-netcdf-magic:{netcdf.name}:{magic!r}")
        records.append({
            "url": url,
            "final_url": final_url,
            "content_type": content_type,
            "last_modified": last_modified,
            "allowed_year_range": decade.replace(".", "-"),
            "compressed_path": str(compressed),
            "compressed_bytes": compressed.stat().st_size,
            "compressed_sha256": sha256(compressed),
            "netcdf_path": str(netcdf),
            "netcdf_bytes": netcdf.stat().st_size,
            "netcdf_sha256": sha256(netcdf),
            "numeric_values_parsed": False,
        })
    after = {name: sha256(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("input-hash-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "dataset": "CRU TS4.01 monthly minimum temperature",
        "official_base_url": BASE_URL,
        "downloaded_allowed_decades": list(DECADES),
        "forbidden_decades_not_requested": list(FORBIDDEN),
        "files": records,
        "numeric_values_parsed": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "source_inputs_unchanged": True,
    }
    (output / "acquisition_manifest.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 245 CRU TS4.01 post-2000 Tmin acquisition\n\n"
        f"- Official decade archives acquired: `{len(records)}`.\n"
        "- Numeric climate values parsed: `false`.\n"
        "- Forbidden decades requested: `false`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
