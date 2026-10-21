from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


XML_IDS = {
    "CHIN067": "14352",
    "CHIN068": "14353",
    "CHIN070": "14355",
    "CHIN071": "14356",
}


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


def fetch_metadata(xml_id: str) -> tuple[bytes, dict[str, object]]:
    url = (
        "https://www.ncei.noaa.gov/access/paleo-search/study/search.json?xmlId="
        + xml_id
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    return payload, json.loads(payload.decode("utf-8"))


def main() -> int:
    root = project_root()
    base = root / "validation_data/results/paleo_surface_validation_v0"
    loop = base / "climate_reconstruction_improvement_loop_v2"
    r171 = loop / "round_171_noaa_hma_temperature_raw_inventory_v1"
    output = loop / "round_172_noaa_hma_temperature_inventory_closure_v1"
    output.mkdir(parents=True, exist_ok=False)
    inputs = {
        "round171_inventory": r171 / "candidate_inventory.csv",
        "round171_summary": r171 / "summary.json",
        "ledger": loop
        / "round_110_unresolved_proxy_metadata_recovery_v4"
        / "proxy_evidence_master_ledger_metadata_recovered.csv",
        "bt021_identity": base
        / "climate_reconstruction_bt021_identity_closure_20260819_v2"
        / "summary.json",
        "bt021_oof": base
        / "climate_reconstruction_bt021_low_frequency_modern_oof_20260819_v1"
        / "summary.json",
    }
    hashes_before = {name: sha256(path) for name, path in inputs.items()}
    with inputs["round171_inventory"].open(
        "r", encoding="ascii", newline=""
    ) as stream:
        inventory = list(csv.DictReader(stream))
    review = [row for row in inventory if row["disposition"] == "metadata_review_required"]
    if {row["itrdb_code"] for row in review} != {
        "BT006",
        "BT021",
        "CHIN067",
        "CHIN068",
        "CHIN070",
        "CHIN071",
    }:
        raise RuntimeError("unexpected-round171-review-set")

    with inputs["ledger"].open("r", encoding="utf-8", newline="") as stream:
        ledger = list(csv.DictReader(stream))
    camgang = [
        row
        for row in ledger
        if row["network"] == "proxy6"
        and row["formal_metadata_identity"].startswith("Asia-CAMGSP::")
        and row["first_year"] == "1400"
        and row["last_year"] == "2005"
    ]
    if len(camgang) != 1:
        raise RuntimeError("bt006-camgang-ledger-identity-not-unique")

    bt021_identity = json.loads(inputs["bt021_identity"].read_text(encoding="utf-8"))
    bt021_oof = json.loads(inputs["bt021_oof"].read_text(encoding="utf-8"))
    if "close" not in str(bt021_oof.get("decision", "")).lower():
        raise RuntimeError("bt021-oof-not-closed")

    resolutions: list[dict[str, object]] = [
        {
            "itrdb_code": "BT006",
            "resolution": "existing_proxy_network_source",
            "evidence": camgang[0]["formal_metadata_identity"],
        },
        {
            "itrdb_code": "BT021",
            "resolution": "closed_existing_candidate_modern_oof",
            "evidence": bt021_oof.get("decision", ""),
            "identity_decision": bt021_identity.get("decision", ""),
        },
    ]
    metadata_dir = output / "noaa_full_study_metadata_unmodified"
    metadata_dir.mkdir()
    for code, xml_id in XML_IDS.items():
        payload, document = fetch_metadata(xml_id)
        path = metadata_dir / f"{code.lower()}_xmlid_{xml_id}.json"
        path.write_bytes(payload)
        studies = document.get("study", [])
        if len(studies) != 1:
            raise RuntimeError(f"unexpected-study-count:{code}")
        study = studies[0]
        notes = str(study.get("studyNotes", ""))
        publications = study.get("publication", [])
        if isinstance(publications, dict):
            publications = [publications]
        titles = [str(row.get("title", "")) for row in publications]
        joined = " ".join([notes, *titles]).lower()
        if "precipitation" not in joined:
            raise RuntimeError(f"precipitation-purpose-not-established:{code}")
        resolutions.append(
            {
                "itrdb_code": code,
                "resolution": "not_temperature_candidate_documented_precipitation",
                "evidence": notes.strip(),
                "publication_titles": titles,
                "metadata_sha256": sha256(path),
            }
        )

    unresolved = [
        row for row in resolutions if str(row["resolution"]).startswith("unresolved")
    ]
    hashes_after = {name: sha256(path) for name, path in inputs.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("input-hash-changed")
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "round171_review_count": len(review),
        "resolved_count": len(resolutions),
        "unresolved_count": len(unresolved),
        "resolutions": resolutions,
        "annual_proxy_or_climate_values_downloaded_or_parsed": False,
        "linked_noaa_data_files_opened": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "decision": "close_current_noaa_catalogue_as_source_of_new_temperature_candidate",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 172 NOAA HMA Temperature-Inventory Closure\n\n"
        f"- Round 171 review rows resolved: `{len(resolutions)}` of `{len(review)}`.\n"
        "- BT006: existing PAGES2k Camgang source.\n"
        "- BT021: previously closed by modern OOF.\n"
        "- CHIN067/068/070/071: documented annual-precipitation sources.\n"
        "- Annual values or linked NOAA data files opened: `false`.\n\n"
        "Decision: `close_current_noaa_catalogue_as_source_of_new_temperature_candidate`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
