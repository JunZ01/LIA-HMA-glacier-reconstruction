from __future__ import annotations

import hashlib
import json
import re
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
LOCKED_YEAR = re.compile(r"(?<!\d)(?:192[1-9]|193\d)(?!\d)")
KEYWORDS = re.compile(
    r"Ranwu|Sygera|Duan|sample|site|maximum latewood density|MXD|chronolog|"
    r"coordinate|latitude|longitude|spruce|Picea|data|archive|repository|"
    r"supplement|figure|fig\.|table",
    re.IGNORECASE,
)


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_of(element: ET.Element) -> str:
    return " ".join(
        "".join(node.text or "" for node in element.iter(f"{W}t")).split()
    )


def image_size(payload: bytes) -> tuple[int | None, int | None]:
    if payload.startswith(b"\x89PNG\r\n\x1a\n") and len(payload) >= 24:
        return struct.unpack(">II", payload[16:24])
    if payload.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(payload):
            if payload[index] != 0xFF:
                index += 1
                continue
            marker = payload[index + 1]
            index += 2
            if marker in (0xD8, 0xD9):
                continue
            if index + 2 > len(payload):
                break
            length = int.from_bytes(payload[index : index + 2], "big")
            if marker in range(0xC0, 0xC4) and index + 7 < len(payload):
                height = int.from_bytes(payload[index + 3 : index + 5], "big")
                width = int.from_bytes(payload[index + 5 : index + 7], "big")
                return width, height
            index += max(length, 2)
    return None, None


def safe_core_properties(payload: bytes) -> dict[str, str]:
    root = ET.fromstring(payload)
    result: dict[str, str] = {}
    for child in root:
        value = " ".join((child.text or "").split())
        if value and not LOCKED_YEAR.search(value):
            result[child.tag.rsplit("}", 1)[-1]] = value
    return result


def main() -> int:
    root = project_root()
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    source = loop / (
        "round_169_huang2025_mxd_public_data_recovery_v1/raw_unopened/"
        "huang2025_supplement_mmc1.docx"
    )
    output = loop / "round_170_huang2025_source_identity_metadata_v1"
    output.mkdir(parents=True, exist_ok=False)
    before_hash = sha256_path(source)

    safe_paragraphs: list[dict[str, object]] = []
    discarded_locked = 0
    media_inventory: list[dict[str, object]] = []
    external_relationships: list[dict[str, str]] = []
    with zipfile.ZipFile(source) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        for index, paragraph in enumerate(document.iter(f"{W}p"), start=1):
            text = text_of(paragraph)
            if not text:
                continue
            if LOCKED_YEAR.search(text):
                discarded_locked += 1
                continue
            if KEYWORDS.search(text):
                safe_paragraphs.append({"paragraph_1based": index, "text": text})

        for name in sorted(
            item for item in archive.namelist() if item.startswith("word/media/")
        ):
            payload = archive.read(name)
            width, height = image_size(payload)
            media_inventory.append(
                {
                    "member": name,
                    "size_bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                    "width_px": width,
                    "height_px": height,
                }
            )

        rel_name = "word/_rels/document.xml.rels"
        if rel_name in archive.namelist():
            rel_root = ET.fromstring(archive.read(rel_name))
            for relation in rel_root.iter(f"{PKG_REL}Relationship"):
                if relation.attrib.get("TargetMode") == "External":
                    target = relation.attrib.get("Target", "")
                    if target and not LOCKED_YEAR.search(target):
                        external_relationships.append(
                            {
                                "id": relation.attrib.get("Id", ""),
                                "type": relation.attrib.get("Type", ""),
                                "target": target,
                            }
                        )

        properties = (
            safe_core_properties(archive.read("docProps/core.xml"))
            if "docProps/core.xml" in archive.namelist()
            else {}
        )

    after_hash = sha256_path(source)
    if before_hash != after_hash:
        raise RuntimeError("source-hash-changed")

    identity_terms = " ".join(row["text"] for row in safe_paragraphs)
    direct_overlap = bool(re.search(r"same (?:samples?|site|chronology)", identity_terms, re.I))
    direct_distinct = bool(re.search(r"new(?:ly)? (?:collected |developed )?(?:samples?|site|chronology)", identity_terms, re.I))
    decision = (
        "overlap_established"
        if direct_overlap
        else "distinct_source_established"
        if direct_distinct
        else "source_identity_unresolved_request_raw_data_and_crosswalk"
    )
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source_path": str(source),
        "source_sha256_before": before_hash,
        "source_sha256_after": after_hash,
        "source_hash_unchanged": before_hash == after_hash,
        "safe_keyword_paragraph_count": len(safe_paragraphs),
        "locked_paragraphs_discarded_before_output_or_numeric_conversion": discarded_locked,
        "locked_1921_1939_numeric_values_converted": False,
        "media_count": len(media_inventory),
        "figures_rendered_or_ocr_or_digitized": False,
        "external_relationship_count": len(external_relationships),
        "core_properties": properties,
        "direct_overlap_phrase_found": direct_overlap,
        "direct_distinct_phrase_found": direct_distinct,
        "decision": decision,
        "annual_values_parsed": False,
        "paleo_trajectory_inspected": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (output / "safe_keyword_paragraphs.json").write_text(
        json.dumps(safe_paragraphs, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "media_inventory.json").write_text(
        json.dumps(media_inventory, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "external_relationships.json").write_text(
        json.dumps(external_relationships, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 170 Huang 2025 Source-Identity Metadata Audit\n\n"
        f"- Safe keyword paragraphs: `{len(safe_paragraphs)}`.\n"
        f"- Locked paragraphs discarded before output: `{discarded_locked}`.\n"
        f"- Media members structurally inventoried: `{len(media_inventory)}`.\n"
        "- Figures rendered, OCR-processed, or digitized: `false`.\n"
        f"- Source hash unchanged: `{str(before_hash == after_hash).lower()}`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
