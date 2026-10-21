from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


STUDIES = {
    "NEPA001": ("4422", "5632"),
    "NEPA002": ("4453", "6975"),
    "NEPA037": ("3794", "4254"),
    "NEPA043": ("22394", "20522"),
    "NEPA055": ("22408", "20550"),
    "NPL065": ("40542", "81629"),
    "GANESH_O18": ("22551", "20722"),
    "HUMLA_O18": ("22547", "20714"),
    "SOUTHERN_HIMALAYA_O18": ("23830", "21974"),
    "WACHE_BHUTAN_O18": ("22548", "20716"),
}
KEYWORDS = (
    "temperature", "precipitation", "hydroclimate", "moisture", "drought",
    "density", "maximum latewood", "mxd", "oxygen", "isotope", "delta18",
    "d18", "ring width", "reconstruction", "datafile", "fileurl", "url",
    "earliest", "latest", "start", "end", "season", "variable", "site",
)


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


def walk(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield from walk(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, f"{path}[{index}]")
    elif value is not None:
        yield path, str(value)


def fetch(xml_id: str) -> tuple[bytes, str, str]:
    url = "https://www.ncei.noaa.gov/access/paleo-search/study/search.json?xmlId=" + xml_id
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read(), response.geturl(), response.headers.get("Content-Type", "")


def main() -> int:
    root = project_root()
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    output = loop / "round_236_central_east_hma_noaa_candidate_metadata_audit_v2"
    output.mkdir(parents=True, exist_ok=False)
    metadata_dir = output / "raw"
    metadata_dir.mkdir()
    inputs = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": Path(__file__).with_name("round236_central_east_hma_noaa_candidate_metadata_audit_v2_contract.md"),
        "round235": loop / "round_235_noaa_hma_tree_ring_time_filter_gap_inventory_v1/summary.json",
    }
    before = {name: sha256(path) for name, path in inputs.items()}

    context_rows: list[dict[str, str]] = []
    study_rows: list[dict[str, object]] = []
    for label, (noaa_id, xml_id) in STUDIES.items():
        payload, final_url, content_type = fetch(xml_id)
        path = metadata_dir / f"{label.lower()}.json"
        path.write_bytes(payload)
        document = json.loads(payload.decode("utf-8"))
        studies = document.get("study", [])
        if len(studies) != 1:
            raise RuntimeError(f"unexpected-study-count:{label}:{len(studies)}")
        study = studies[0]
        returned_id = str(study.get("NOAAStudyId", ""))
        if returned_id != noaa_id:
            raise RuntimeError(f"study-id-mismatch:{label}:{returned_id}")
        flattened = list(walk(study))
        selected = []
        for scalar_path, scalar_value in flattened:
            combined = f"{scalar_path} {scalar_value}".lower()
            matched = [keyword for keyword in KEYWORDS if keyword in combined]
            if matched:
                selected.append((scalar_path, scalar_value, ";".join(matched)))
                context_rows.append({
                    "candidate": label,
                    "noaa_study_id": noaa_id,
                    "xml_id": xml_id,
                    "metadata_path": scalar_path,
                    "metadata_value": scalar_value,
                    "matched_keywords": ";".join(matched),
                })
        joined = " ".join(value for _, value in flattened).lower()
        study_rows.append({
            "candidate": label,
            "noaa_study_id": noaa_id,
            "xml_id": xml_id,
            "study_name": study.get("studyName", ""),
            "selected_metadata_context_count": len(selected),
            "mentions_temperature": "temperature" in joined,
            "mentions_precipitation": "precipitation" in joined,
            "mentions_density_or_mxd": any(term in joined for term in ("density", "maximum latewood", "mxd")),
            "mentions_oxygen_isotope": any(term in joined for term in ("oxygen", "isotope", "delta18", "d18")),
            "mentions_reconstruction": "reconstruction" in joined,
            "metadata_sha256": sha256(path),
            "api_final_url": final_url,
            "api_content_type": content_type,
        })

    with (output / "study_metadata_flags.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(study_rows[0]))
        writer.writeheader()
        writer.writerows(study_rows)
    with (output / "selected_metadata_contexts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(context_rows[0]))
        writer.writeheader()
        writer.writerows(context_rows)
    after = {name: sha256(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("input-hash-changed")
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "study_count": len(study_rows),
        "study_metadata_flags": study_rows,
        "selected_metadata_context_count": len(context_rows),
        "linked_data_file_urls_followed_or_opened": False,
        "annual_proxy_or_climate_values_downloaded_or_parsed": False,
        "locked_1921_1939_numeric_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "input_hashes_unchanged": True,
        "decision": "adjudicate_metadata_flags_before_any_value_download",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 236 Central/East HMA NOAA Candidate Metadata Audit\n\n"
        f"- Exact NOAA studies inspected: `{len(study_rows)}`.\n"
        f"- Selected metadata contexts: `{len(context_rows)}`.\n"
        "- Linked annual data files opened: `false`.\n\n"
        "Decision: `adjudicate_metadata_flags_before_any_value_download`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

