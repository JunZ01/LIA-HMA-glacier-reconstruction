from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import struct
import sys
import urllib.parse
import zipfile
from io import BytesIO
from pathlib import Path


HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "run_temperature_west_central_independent_evidence_acquisition_v1.py"
HMA_SCREENING_BOX = (70.0, 28.0, 90.0, 42.0)


def load_v1():
    spec = importlib.util.spec_from_file_location("west_central_evidence_v1", V1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {V1_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.urllib.request.urlparse = urllib.parse.urlparse
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_dbf(data: bytes) -> list[dict[str, str]]:
    count = struct.unpack_from("<I", data, 4)[0]
    header_length = struct.unpack_from("<H", data, 8)[0]
    record_length = struct.unpack_from("<H", data, 10)[0]
    fields = []
    cursor = 32
    while data[cursor] != 0x0D:
        descriptor = data[cursor : cursor + 32]
        name = descriptor[:11].split(b"\x00", 1)[0].decode("ascii")
        length = descriptor[16]
        fields.append((name, length))
        cursor += 32
    records = []
    for index in range(count):
        start = header_length + index * record_length
        record = data[start : start + record_length]
        if not record or record[0:1] == b"*":
            continue
        offset = 1
        values = {}
        for name, length in fields:
            values[name] = record[offset : offset + length].decode("utf-8", errors="replace").strip()
            offset += length
        records.append(values)
    return records


def read_polygon_bboxes(data: bytes) -> list[tuple[float, float, float, float]]:
    stream = BytesIO(data)
    stream.seek(100)
    boxes = []
    while True:
        record_header = stream.read(8)
        if not record_header:
            break
        if len(record_header) != 8:
            raise RuntimeError("Truncated shapefile record header")
        _, words = struct.unpack(">2i", record_header)
        content = stream.read(words * 2)
        if len(content) != words * 2:
            raise RuntimeError("Truncated shapefile record")
        shape_type = struct.unpack_from("<i", content, 0)[0]
        if shape_type == 0:
            boxes.append((float("nan"),) * 4)
        elif shape_type in (5, 15, 25, 31):
            boxes.append(struct.unpack_from("<4d", content, 4))
        else:
            raise RuntimeError(f"Unexpected shape type {shape_type}")
    return boxes


def intersects(box: tuple[float, float, float, float], target: tuple[float, float, float, float]) -> bool:
    xmin, ymin, xmax, ymax = box
    txmin, tymin, txmax, tymax = target
    return xmax >= txmin and xmin <= txmax and ymax >= tymin and ymin <= tymax


def inspect_polygon_zip(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    members = []
    features = []
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad_member}")
        for info in archive.infolist():
            members.append(
                {
                    "member": info.filename,
                    "size_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                }
            )
        for stem in ("geographical_division_15region", "geographical_division_3region"):
            dbf = read_dbf(archive.read(f"shp/{stem}.dbf"))
            boxes = read_polygon_bboxes(archive.read(f"shp/{stem}.shp"))
            if len(dbf) != len(boxes):
                raise RuntimeError(f"DBF/SHP count mismatch for {stem}: {len(dbf)} vs {len(boxes)}")
            for index, (attributes, box) in enumerate(zip(dbf, boxes), start=1):
                features.append(
                    {
                        "layer": stem,
                        "record_index": index,
                        **attributes,
                        "xmin": box[0],
                        "ymin": box[1],
                        "xmax": box[2],
                        "ymax": box[3],
                        "intersects_hma_screening_box": intersects(box, HMA_SCREENING_BOX),
                    }
                )
    return members, features


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(source_raw_dir: str, output_dir: str) -> None:
    v1 = load_v1()
    source_raw = Path(source_raw_dir).resolve()
    out = Path(output_dir).resolve()
    raw = out / "official_sources_unmodified"
    if out.exists():
        raise FileExistsError(out)
    shutil.copytree(source_raw, raw)

    manifest = []
    for path in sorted(raw.iterdir()):
        if path.is_file():
            manifest.append(
                {
                    "source_id": path.stem,
                    "local_path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "copied_from_interrupted_v1": True,
                }
            )

    allowed_rows = []
    headers = {}
    for key in ("reaches_annual", "reaches_summer", "reaches_winter", "reaches_record_count"):
        rows, header = v1.parse_allowed_table(raw / f"{key}.txt", key)
        allowed_rows.extend(rows)
        headers[key] = header
    write_csv(out / "reaches_allowed_rows.csv", allowed_rows)

    members, features = inspect_polygon_zip(raw / "reaches_polygons.zip")
    write_csv(out / "reaches_polygon_members.csv", members)
    write_csv(out / "reaches_polygon_features.csv", features)

    hma_features = [row for row in features if row["intersects_hma_screening_box"]]
    summary = {
        "experiment": "temperature_west_central_independent_evidence_acquisition_v2",
        "supersedes": "temperature_west_central_independent_evidence_acquisition_v1_interrupted_on_windows_path_length",
        "role": "metadata_and_allowed_year_acquisition_only",
        "sealed_period": [v1.SEALED_START, v1.SEALED_END],
        "allowed_windows": [list(window) for window in v1.ALLOWED_WINDOWS],
        "sealed_values_converted_to_numeric": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_file_count": len(manifest),
        "allowed_reaches_row_count": len(allowed_rows),
        "reaches_headers": headers,
        "polygon_member_count": len(members),
        "polygon_feature_count": len(features),
        "hma_screening_box": list(HMA_SCREENING_BOX),
        "hma_intersecting_polygon_feature_count": len(hma_features),
        "next_gate": "reaches_region_record_density_and_candidate_source_identity_audit",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_temperature_west_central_independent_evidence_acquisition_v3.py SOURCE_RAW_DIR OUTPUT_DIR")
    main(sys.argv[1], sys.argv[2])
