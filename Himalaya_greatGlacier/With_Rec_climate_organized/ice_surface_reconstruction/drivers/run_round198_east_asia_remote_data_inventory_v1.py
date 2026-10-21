from __future__ import annotations

import csv
import hashlib
import json
import struct
import urllib.request
from collections import Counter
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUT = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "round_198_east_asia_remote_data_inventory_v1"
)
METADATA = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "round_197_east_asia_temperature_metadata_v2/zenodo_record_18477496.json"
)
EXPECTED_SIZE = 1_177_089_719
EXPECTED_MD5 = "46fa321968bc240e7aa4a4e273b20854"


def get_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "LIA-HMA-zip-inventory/1.0",
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", None)
        if status != 206:
            raise RuntimeError(f"Range request not honored; status={status}")
        data = response.read()
        expected = end - start + 1
        if len(data) != expected:
            raise RuntimeError(f"Range length mismatch: {len(data)} != {expected}")
        return data


def parse_central_directory(payload: bytes, expected_entries: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    offset = 0
    header = struct.Struct("<4s6H3I5H2I")
    while offset < len(payload):
        if payload[offset : offset + 4] != b"PK\x01\x02":
            raise RuntimeError(f"Invalid central-directory signature at {offset}")
        values = header.unpack_from(payload, offset)
        (
            _, made_by, needed, flags, method, mod_time, mod_date, crc32,
            compressed_size, uncompressed_size, name_len, extra_len, comment_len,
            disk_start, internal_attr, external_attr, local_offset,
        ) = values
        start = offset + header.size
        name_bytes = payload[start : start + name_len]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = name_bytes.decode(encoding)
        rows.append(
            {
                "member": name,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "compression_method": method,
                "crc32": f"{crc32:08x}",
                "flags": flags,
                "local_header_offset": local_offset,
                "is_directory": name.endswith("/"),
            }
        )
        offset = start + name_len + extra_len + comment_len
    if len(rows) != expected_entries:
        raise RuntimeError(f"Entry count mismatch: {len(rows)} != {expected_entries}")
    return rows


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent result: {OUT}")
    OUT.mkdir(parents=True)

    metadata_bytes = METADATA.read_bytes()
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    files = {item["key"]: item for item in metadata["files"]}
    item = files["Data.zip"]
    if item["size"] != EXPECTED_SIZE or item["checksum"] != f"md5:{EXPECTED_MD5}":
        raise RuntimeError("Data.zip metadata changed")
    url = item["links"].get("content", item["links"].get("self"))
    if not url:
        raise RuntimeError("Data.zip content URL missing")

    tail_size = min(131_072, EXPECTED_SIZE)
    tail_start = EXPECTED_SIZE - tail_size
    tail = get_range(url, tail_start, EXPECTED_SIZE - 1)
    signature = b"PK\x05\x06"
    relative = tail.rfind(signature)
    if relative < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    eocd = struct.unpack_from("<4s4H2IH", tail, relative)
    _, disk_no, cd_disk, disk_entries, total_entries, cd_size, cd_offset, comment_len = eocd
    if any(value in (0xFFFF, 0xFFFFFFFF) for value in (total_entries, cd_size, cd_offset)):
        raise RuntimeError("ZIP64 central directory requires a new parser contract")
    if disk_no != 0 or cd_disk != 0 or disk_entries != total_entries:
        raise RuntimeError("Multi-disk ZIP is not supported")

    central = get_range(url, cd_offset, cd_offset + cd_size - 1)
    rows = parse_central_directory(central, total_entries)
    with (OUT / "data_zip_central_directory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    file_rows = [row for row in rows if not row["is_directory"]]
    suffix_counts = Counter(Path(str(row["member"])).suffix.lower() or "[none]" for row in file_rows)
    model_names = [
        str(row["member"]) for row in file_rows
        if "CMIP6_PMIP4" in str(row["member"]) or "Historical" in str(row["member"])
    ]
    proxy_candidates = [
        str(row["member"]) for row in file_rows
        if "prox" in str(row["member"]).lower()
    ]
    cru_candidates = [
        str(row["member"]) for row in file_rows
        if "cru" in str(row["member"]).lower()
    ]

    summary = {
        "round": 198,
        "record": "10.5281/zenodo.18477496",
        "data_zip_declared_size": EXPECTED_SIZE,
        "data_zip_declared_md5": EXPECTED_MD5,
        "http_bytes_read": len(tail) + len(central),
        "full_archive_downloaded": False,
        "member_payload_opened": False,
        "entry_count": len(rows),
        "file_count": len(file_rows),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "model_member_names": model_names,
        "proxy_member_candidates": proxy_candidates,
        "cru_member_candidates": cru_candidates,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
        "decision": (
            "advance_targeted_proxy_identity_metadata"
            if proxy_candidates and model_names
            else "insufficient_source_inventory"
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "zenodo_metadata_path": str(METADATA),
                "zenodo_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "remote_data_zip": {"size": EXPECTED_SIZE, "md5": EXPECTED_MD5},
                "remote_ranges_sha256": {
                    f"{tail_start}-{EXPECTED_SIZE - 1}": hashlib.sha256(tail).hexdigest(),
                    f"{cd_offset}-{cd_offset + cd_size - 1}": hashlib.sha256(central).hexdigest(),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = [
        "# Round 198 remote Data.zip inventory",
        "",
        f"- Decision: `{summary['decision']}`.",
        f"- ZIP entries / files: `{len(rows)}` / `{len(file_rows)}`.",
        f"- HTTP bytes read: `{summary['http_bytes_read']}` of `{EXPECTED_SIZE}`.",
        f"- Proxy candidates: `{proxy_candidates}`.",
        f"- CRU candidates: `{cru_candidates}`.",
        f"- Model-path members: `{len(model_names)}`.",
        "- No ZIP member payload or climate/proxy value was opened.",
        "- The full archive, locked period, and Jionglacuo were not accessed.",
    ]
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
