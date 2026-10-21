from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "round_224_central_himalaya_mxd2026_public_recovery_v1"
RAW = OUT / "official_sources_unmodified"
CONTRACT = OUT / "ROUND_CONTRACT.md"
DRIVER = Path(__file__).resolve()
DOI = "10.1007/s11676-026-02104-0"
ARTICLE_URL = f"https://link.springer.com/article/{DOI}"
PDF_URL = f"https://link.springer.com/content/pdf/{DOI}.pdf"
CROSSREF_URL = f"https://api.crossref.org/works/{DOI}"
DATACITE_URL = "https://api.datacite.org/dois?query=titles.title:%22Central%20Himalayan%20tree-ring%20density%20network%22&page[size]=100"
ZENODO_URL = "https://zenodo.org/api/records?q=%22Central%20Himalayan%20tree-ring%20density%20network%22&size=100"
HEADERS = {"User-Agent": "HMA-LIA-reconstruction-audit/1.0 (research data recovery)"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path)}


def download(url: str, path: Path) -> dict[str, object]:
    response = requests.get(url, headers=HEADERS, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return {
        "url": url,
        "status": response.status_code,
        "content_type": response.headers.get("content-type"),
        **state(path),
    }


def pdf_container(path: Path) -> dict[str, object]:
    reader = PdfReader(path)
    embedded_names = []
    try:
        embedded_names = sorted(reader.attachments.keys())
    except Exception:
        embedded_names = []
    return {
        "page_count": len(reader.pages),
        "embedded_file_names": embedded_names,
        "embedded_file_count": len(embedded_names),
        "metadata": {str(key): str(value) for key, value in (reader.metadata or {}).items()},
        "page_text_extracted": False,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    tracked = {"facts": ROOT / "LIA_PROJECT_FACTS.md", "contract": CONTRACT, "driver": DRIVER}
    before = {name: state(path) for name, path in tracked.items()}

    downloads = {}
    downloads["article_html"] = download(ARTICLE_URL, RAW / "springer_article.html")
    downloads["article_pdf"] = download(PDF_URL, RAW / "springer_article.pdf")
    downloads["crossref"] = download(CROSSREF_URL, RAW / "crossref.json")
    downloads["datacite"] = download(DATACITE_URL, RAW / "datacite_search.json")
    downloads["zenodo"] = download(ZENODO_URL, RAW / "zenodo_search.json")

    html = (RAW / "springer_article.html").read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    candidate_links = []
    for link in soup.find_all("a", href=True):
        href = urljoin(ARTICLE_URL, link["href"])
        label = " ".join(link.get_text(" ", strip=True).split())
        if "supplement" in label.lower() or "supplement" in href.lower() or "moesm" in href.lower():
            candidate_links.append({"label": label, "url": href})
    unique_links = []
    seen = set()
    for item in candidate_links:
        if item["url"] not in seen:
            unique_links.append(item)
            seen.add(item["url"])
    if not unique_links:
        raise RuntimeError("Official supplementary link not found")
    supplement_url = unique_links[0]["url"]
    downloads["supplement_pdf"] = download(supplement_url, RAW / "springer_supplement.pdf")

    crossref = json.loads((RAW / "crossref.json").read_text(encoding="utf-8"))["message"]
    datacite = json.loads((RAW / "datacite_search.json").read_text(encoding="utf-8"))
    zenodo = json.loads((RAW / "zenodo_search.json").read_text(encoding="utf-8"))
    relation_dois = sorted(
        {
            str(item.get("DOI", "")).lower()
            for item in crossref.get("relation", {}).values()
            if isinstance(item, list)
            for item in item
            if isinstance(item, dict) and item.get("DOI")
        }
    )
    datacite_hits = len(datacite.get("data", []))
    zenodo_hits = int(zenodo.get("hits", {}).get("total", 0))
    containers = {
        "article_pdf": pdf_container(RAW / "springer_article.pdf"),
        "supplement_pdf": pdf_container(RAW / "springer_supplement.pdf"),
    }
    public_machine_readable_numeric = bool(
        relation_dois
        or datacite_hits
        or zenodo_hits
        or containers["article_pdf"]["embedded_file_count"]
        or containers["supplement_pdf"]["embedded_file_count"]
    )

    site_lineage = [
        {"site": "Ghorepani", "source": "ITRDB", "sampling": "late_1970s", "new_for_2026_study": False},
        {"site": "Kalinchowk", "source": "ITRDB", "sampling": "late_1970s", "new_for_2026_study": False},
        {"site": "Jumla", "source": "Braeuning_2003", "sampling": "late_1990s", "new_for_2026_study": False},
        {"site": "Rara", "source": "Nahtz_et_al_2026", "sampling": "2023", "new_for_2026_study": True},
        {"site": "Cholangpati", "source": "Nahtz_et_al_2026", "sampling": "2023", "new_for_2026_study": True},
    ]
    (OUT / "site_lineage_inventory.json").write_text(json.dumps(site_lineage, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "publisher_link_inventory.json").write_text(json.dumps(unique_links, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "pdf_container_inventory.json").write_text(json.dumps(containers, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "download_manifest.json").write_text(json.dumps(downloads, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if public_machine_readable_numeric:
        decision = "machine_readable_candidate_found_requires_locked_safe_content_audit"
    else:
        decision = "central_hma_post1775_diagnostic_admitted_numeric_series_not_public"
    summary = {
        "round": 224,
        "decision": decision,
        "doi": DOI,
        "publication_date": "2026-07-15",
        "region": {"longitude": [82.0, 86.1], "latitude": [28.1, 29.5]},
        "proxy_type": "maximum_latewood_density",
        "target": "April-September mean temperature",
        "reconstruction_years": [1775, 2022],
        "calibration_years": [1951, 2022],
        "reported_calibration_spearman": 0.64,
        "newly_sampled_2023_sites": ["Rara", "Cholangpati"],
        "legacy_sites": ["Ghorepani", "Kalinchowk", "Jumla"],
        "begins_by_1500_gate": False,
        "modern_coverage_gate_possible": True,
        "public_machine_readable_numeric_series_found": public_machine_readable_numeric,
        "crossref_relation_dois": relation_dois,
        "datacite_exact_title_hits": datacite_hits,
        "zenodo_exact_title_hits": zenodo_hits,
        "pdf_containers": containers,
        "figure_digitization_performed": False,
        "source_states_unchanged": True,
        "locked_1921_1939_values_read": False,
        "project_proxy_values_read": False,
        "jionglacuo_accessed": False,
        "enkf_run": False,
        "production_changed": False,
        "oggm_run": False,
    }
    after = {name: state(path) for name, path in tracked.items()}
    if before != after:
        raise RuntimeError("A tracked source changed")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (OUT / "FINAL_RESULT.md").write_text(
        "# Round 224 Result\n\n"
        f"`{decision}`\n\n"
        "Nahtz et al. (2026) is a newly published Central-Himalaya MXD network over 82.0-86.1 E. "
        "Rara and Cholangpati were sampled in 2023 and provide genuinely new source material, while three sites are legacy records. "
        "The published reconstruction covers only 1775-2022, so it cannot pass the begin-by-1500 production gate.\n\n"
        f"Crossref linked-data DOI count is `{len(relation_dois)}`, DataCite exact-title hits `{datacite_hits}`, Zenodo exact-title hits `{zenodo_hits}`, "
        f"and PDF embedded-file counts are `{containers['article_pdf']['embedded_file_count']}` / `{containers['supplement_pdf']['embedded_file_count']}`. "
        "No figure was digitized and no annual temperature value was opened.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
