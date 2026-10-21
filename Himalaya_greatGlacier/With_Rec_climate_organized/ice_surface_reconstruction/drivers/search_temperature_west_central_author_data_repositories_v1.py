"""Search public repositories for four priority west/central HMA temperature series."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / (
    "climate_reconstruction_improvement_loop_v2/"
    "temperature_west_central_author_repository_search_v1"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
DRIVER = Path(__file__).resolve()

CANDIDATES = [
    {
        "id": "upper_indus_mxd_2024",
        "title": "Tree-ring maximum latewood density reveals unprecedented warming and long-term summer temperature in the upper Indus Basin northern Pakistan",
        "doi": "10.1016/j.scitotenv.2024.177393",
    },
    {
        "id": "pakistan_western_himalaya_2026",
        "title": "Tree-ring based reconstruction of pre-monsoon temperatures variability in Pakistan's Western Himalayan since 1855 CE",
        "doi": "10.1088/2515-7620/ae718e",
    },
    {
        "id": "rara_cholangpati_mxd_2026",
        "title": "Central Himalayan tree-ring density network reveals warm-season temperature variability back to 1775 CE",
        "doi": "10.1007/s11676-026-02104-0",
    },
    {
        "id": "sano_humla_2005",
        "title": "Temperature variations since the mid-18th century for western Nepal as reconstructed from tree-ring width and density of Abies spectabilis",
        "doi": "10.1016/j.dendro.2005.08.003",
    },
]

HEADERS = {"User-Agent": "LIA-HMA-climate-audit/1.0 (public metadata search)"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens(value: str) -> set[str]:
    stop = {"the", "a", "an", "of", "in", "and", "from", "to", "for", "based", "since"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2 and token not in stop
    }


def similarity(left: str, right: str) -> float:
    a = tokens(left)
    b = tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def get_json(url: str) -> tuple[int, object]:
    response = requests.get(url, headers=HEADERS, timeout=90)
    status = response.status_code
    response.raise_for_status()
    return status, response.json()


def post_json(url: str, payload: dict[str, object]) -> tuple[int, object]:
    response = requests.post(url, headers=HEADERS, json=payload, timeout=90)
    status = response.status_code
    response.raise_for_status()
    return status, response.json()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    raw = OUTPUT / "raw_repository_responses"
    raw.mkdir(parents=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# West/central HMA author-data repository search\n\n"
        "Search public metadata only for four already admitted author-request candidates. "
        "Query DataCite, Zenodo, Figshare, Crossref, and OpenAlex using frozen titles and "
        "DOIs. Do not open annual climate or proxy values. A plausible repository hit must "
        "have title-token Jaccard similarity at least 0.40 or an explicit DOI relation. "
        "Repository discovery alone does not admit a record; any file requires a separate "
        "identity and locked-safe parser audit. No EnKF, production, or OGGM run is allowed.\n",
        encoding="utf-8",
    )
    sources = {"facts": FACTS, "driver": DRIVER, "contract": contract}
    before = {name: sha256(path) for name, path in sources.items()}

    requests_log: list[dict[str, object]] = []
    hits: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for candidate in CANDIDATES:
        candidate_id = candidate["id"]
        title = candidate["title"]
        doi = candidate["doi"]
        queries = [
            (
                "datacite",
                f"https://api.datacite.org/dois?query={quote(title)}&page[size]=25",
                None,
            ),
            (
                "zenodo",
                f"https://zenodo.org/api/records?q={quote(title)}&size=25",
                None,
            ),
            (
                "figshare",
                "https://api.figshare.com/v2/articles/search",
                {"search_for": title, "limit": 25, "order": "published_date", "order_direction": "desc"},
            ),
            (
                "crossref",
                f"https://api.crossref.org/works/{quote(doi, safe='')}",
                None,
            ),
            (
                "openalex",
                f"https://api.openalex.org/works/https://doi.org/{doi}",
                None,
            ),
        ]
        for repository, url, payload in queries:
            try:
                status, result = post_json(url, payload) if payload is not None else get_json(url)
                response_path = raw / f"{candidate_id}_{repository}.json"
                response_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                requests_log.append(
                    {
                        "candidate_id": candidate_id,
                        "repository": repository,
                        "url": url,
                        "http_status": status,
                        "response_file": str(response_path),
                        "response_sha256": sha256(response_path),
                    }
                )

                if repository == "datacite":
                    for item in result.get("data", []):
                        attributes = item.get("attributes", {})
                        found_title = " ".join(
                            value.get("title", "") for value in attributes.get("titles", [])
                        )
                        hits.append(
                            {
                                "candidate_id": candidate_id,
                                "repository": repository,
                                "record_id": item.get("id"),
                                "title": found_title,
                                "url": attributes.get("url"),
                                "resource_type": attributes.get("types", {}).get("resourceTypeGeneral"),
                                "title_similarity": similarity(title, found_title),
                                "explicit_doi_relation": doi.lower() in json.dumps(item).lower(),
                            }
                        )
                elif repository == "zenodo":
                    for item in result.get("hits", {}).get("hits", []):
                        found_title = item.get("metadata", {}).get("title", "")
                        hits.append(
                            {
                                "candidate_id": candidate_id,
                                "repository": repository,
                                "record_id": item.get("id"),
                                "title": found_title,
                                "url": item.get("links", {}).get("html"),
                                "resource_type": item.get("metadata", {}).get("resource_type", {}).get("type"),
                                "title_similarity": similarity(title, found_title),
                                "explicit_doi_relation": doi.lower() in json.dumps(item).lower(),
                            }
                        )
                elif repository == "figshare":
                    for item in result:
                        found_title = item.get("title", "")
                        hits.append(
                            {
                                "candidate_id": candidate_id,
                                "repository": repository,
                                "record_id": item.get("id"),
                                "title": found_title,
                                "url": item.get("url_public_html") or item.get("url"),
                                "resource_type": "figshare_article",
                                "title_similarity": similarity(title, found_title),
                                "explicit_doi_relation": doi.lower() in json.dumps(item).lower(),
                            }
                        )
                elif repository == "crossref":
                    message = result.get("message", {})
                    relation = message.get("relation", {})
                    for relation_type, items in relation.items():
                        for item in items:
                            hits.append(
                                {
                                    "candidate_id": candidate_id,
                                    "repository": repository,
                                    "record_id": item.get("id"),
                                    "title": relation_type,
                                    "url": item.get("id"),
                                    "resource_type": item.get("id-type"),
                                    "title_similarity": 0.0,
                                    "explicit_doi_relation": True,
                                }
                            )
                elif repository == "openalex":
                    for location in result.get("locations", []):
                        source = location.get("source") or {}
                        landing = location.get("landing_page_url")
                        pdf = location.get("pdf_url")
                        if source.get("type") == "repository" or (landing and "dataset" in landing.lower()):
                            hits.append(
                                {
                                    "candidate_id": candidate_id,
                                    "repository": repository,
                                    "record_id": source.get("id"),
                                    "title": source.get("display_name", "repository_location"),
                                    "url": pdf or landing,
                                    "resource_type": source.get("type"),
                                    "title_similarity": 0.0,
                                    "explicit_doi_relation": True,
                                }
                            )
            except Exception as exc:
                errors.append(
                    {"candidate_id": candidate_id, "repository": repository, "error": repr(exc)}
                )

    requests_frame = pd.DataFrame(requests_log)
    hits_frame = pd.DataFrame(hits)
    errors_frame = pd.DataFrame(errors)
    requests_frame.to_csv(OUTPUT / "repository_request_log.csv", index=False)
    hits_frame.to_csv(OUTPUT / "all_repository_hits.csv", index=False)
    errors_frame.to_csv(OUTPUT / "repository_errors.csv", index=False)
    if hits_frame.empty:
        plausible = hits_frame.copy()
    else:
        plausible = hits_frame[
            (hits_frame["title_similarity"] >= 0.40) | hits_frame["explicit_doi_relation"]
        ].copy()
    plausible.to_csv(OUTPUT / "plausible_repository_hits.csv", index=False)

    counts = {
        candidate["id"]: int((plausible["candidate_id"] == candidate["id"]).sum())
        if not plausible.empty
        else 0
        for candidate in CANDIDATES
    }
    summary = {
        "status": "complete",
        "candidate_count": len(CANDIDATES),
        "successful_repository_requests": int(len(requests_frame)),
        "failed_repository_requests": int(len(errors_frame)),
        "plausible_hit_counts": counts,
        "machine_readable_public_dataset_found": bool(any(counts.values())),
        "decision": (
            "inspect_plausible_repository_hits_under_separate_contract"
            if any(counts.values())
            else "public_repository_search_closed_author_request_required"
        ),
        "annual_values_opened": False,
        "locked_1921_1939_values_opened": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# West/Central HMA Author-Data Repository Search",
        "",
        f"Decision: `{summary['decision']}`.",
        "",
    ]
    for candidate in CANDIDATES:
        report.append(f"- {candidate['id']}: plausible public repository hits `{counts[candidate['id']]}`.")
    report.extend(
        [
            "",
            f"- Successful repository requests: `{len(requests_frame)}`.",
            f"- Failed repository requests: `{len(errors_frame)}`.",
            "- No annual value or locked-period value was opened.",
            "- A metadata hit must still expose a machine-readable annual file and pass identity review.",
        ]
    )
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    after = {name: sha256(path) for name, path in sources.items()}
    unchanged = before == after
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("Source file changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
