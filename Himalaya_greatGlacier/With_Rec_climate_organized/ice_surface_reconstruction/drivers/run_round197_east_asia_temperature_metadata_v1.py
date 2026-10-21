from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUT = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "round_197_east_asia_temperature_metadata_v1"
)
RECORD_URL = "https://zenodo.org/api/records/18477496"
EXPECTED_CODE_MD5 = "b91d32c7b5467361ad5934b99c952f31"

KEYWORDS = {
    "existing_reconstruction_families": [
        "phyda", "lmr", "modE", "pages2k", "cru", "berkeley", "hadcrut",
        "gistemp", "noaa", "reanalysis",
    ],
    "model_families": [
        "pmip", "cmip", "past1000", "historical", "model", "simulation",
    ],
    "proxy_families": [
        "tree", "ring", "mxd", "ice", "document", "proxy", "chronology",
    ],
    "method_terms": [
        "reconstruct", "regress", "analog", "ensemble", "weight", "eof",
        "pca", "svd", "kalman", "assimilat", "random forest", "matlab",
    ],
    "temperature_semantics": [
        "temperature", "anomaly", "climatology", "baseline", "absolute",
        "degc", "celsius",
    ],
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "LIA-HMA-metadata-audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def text_members(archive: zipfile.ZipFile) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        suffix = Path(info.filename).suffix.lower()
        is_text = suffix in {".m", ".txt", ".md", ".csv", ".json", ".py", ".r"}
        raw = archive.read(info)
        decoded = ""
        if is_text:
            for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
                try:
                    decoded = raw.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
        rows.append(
            {
                "member": info.filename,
                "size": info.file_size,
                "crc32": f"{info.CRC:08x}",
                "is_text": is_text,
                "line_count": len(decoded.splitlines()) if decoded else 0,
                "text": decoded,
            }
        )
    return rows


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite independent result: {OUT}")
    OUT.mkdir(parents=True)

    metadata_path = OUT / "zenodo_record_18477496.json"
    fetch(RECORD_URL, metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    files = {item["key"]: item for item in metadata.get("files", [])}
    if "Code.zip" not in files:
        raise RuntimeError("Zenodo metadata does not expose Code.zip")
    if set(files) != {"Code.zip", "Data.zip", "Results.zip"}:
        raise RuntimeError(f"Unexpected deposited file inventory: {sorted(files)}")

    code_path = OUT / "Code.zip"
    fetch(files["Code.zip"]["links"]["content"], code_path)
    code_md5 = digest(code_path, "md5")
    if code_md5 != EXPECTED_CODE_MD5:
        raise RuntimeError(f"Code.zip MD5 mismatch: {code_md5}")

    with zipfile.ZipFile(code_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Code.zip CRC failure: {bad}")
        members = text_members(archive)

    with (OUT / "code_member_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["member", "size", "crc32", "is_text", "line_count"]
        )
        writer.writeheader()
        for row in members:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    hits: list[dict[str, object]] = []
    for row in members:
        text = str(row["text"])
        if not text:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            for family, words in KEYWORDS.items():
                matched = sorted({word for word in words if word.lower() in lowered})
                if matched:
                    hits.append(
                        {
                            "member": row["member"],
                            "line": line_number,
                            "family": family,
                            "keywords": "|".join(matched),
                            "text": re.sub(r"\s+", " ", line).strip()[:500],
                        }
                    )

    with (OUT / "code_keyword_hits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["member", "line", "family", "keywords", "text"]
        )
        writer.writeheader()
        writer.writerows(hits)

    description = metadata.get("metadata", {}).get("description", "")
    title = metadata.get("metadata", {}).get("title", "")
    declared = (title + " " + re.sub("<[^>]+>", " ", description)).lower()
    coverage_gate = all(token in declared for token in ("1400", "2000", "70", "150"))
    hma_gate = coverage_gate and "temperature" in declared
    reproducible_code_gate = any(row["is_text"] and row["line_count"] for row in members)
    direct_copy_flag = any(
        hit["family"] == "existing_reconstruction_families"
        and any(token in str(hit["keywords"]).lower() for token in ("phyda", "lmr", "mode"))
        for hit in hits
    )

    decision = (
        "advance_to_source_overlap_adjudication"
        if hma_gate and reproducible_code_gate and not direct_copy_flag
        else "do_not_advance_numeric_field"
    )
    summary = {
        "round": 197,
        "record": "10.5281/zenodo.18477496",
        "record_url": RECORD_URL,
        "metadata_sha256": digest(metadata_path),
        "code_zip_size": code_path.stat().st_size,
        "code_zip_md5": code_md5,
        "code_zip_sha256": digest(code_path),
        "deposited_files": {
            key: {"size": value.get("size"), "checksum": value.get("checksum")}
            for key, value in sorted(files.items())
        },
        "code_member_count": len(members),
        "text_code_member_count": sum(bool(row["is_text"] and row["line_count"]) for row in members),
        "keyword_hit_count": len(hits),
        "coverage_gate": coverage_gate,
        "hma_temperature_gate": hma_gate,
        "reproducible_code_gate": reproducible_code_gate,
        "direct_existing_product_copy_flag": direct_copy_flag,
        "numeric_temperature_values_read": False,
        "data_zip_downloaded": False,
        "results_zip_downloaded": False,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
        "decision": decision,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "sources": [
                    {"path": str(metadata_path), "sha256": digest(metadata_path)},
                    {"path": str(code_path), "sha256": digest(code_path), "md5": code_md5},
                ],
                "excluded_remote_files": ["Data.zip", "Results.zip"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    family_counts: dict[str, int] = {}
    for hit in hits:
        family_counts[str(hit["family"])] = family_counts.get(str(hit["family"]), 0) + 1
    report = [
        "# Round 197 East Asia temperature metadata audit",
        "",
        f"- Decision: `{decision}`.",
        f"- Code members / text members: `{len(members)}` / `{summary['text_code_member_count']}`.",
        f"- Keyword-hit families: `{family_counts}`.",
        f"- Direct existing-product copy flag: `{str(direct_copy_flag).lower()}`.",
        "- `Data.zip` and `Results.zip` were not downloaded or opened.",
        "- No annual temperature, proxy, model, locked-period, or Jionglacuo value was accessed.",
        "",
        "This is a metadata/code preflight only. Source overlap still requires manual adjudication",
        "before any numeric field may be read.",
    ]
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
