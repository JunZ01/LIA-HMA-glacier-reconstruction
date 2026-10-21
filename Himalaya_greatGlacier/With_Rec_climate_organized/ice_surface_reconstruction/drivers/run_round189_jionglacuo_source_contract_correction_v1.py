"""Correct the Round 184 source identity using metadata and official HTML only."""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
from pathlib import Path

import requests


ARTICLE_URL = "https://www.nature.com/articles/s41612-025-01254-y"
ARTICLE_DOI = "10.1038/s41612-025-01254-y"
DATASET_DOI = "10.17632/s7t4zc34sr.2"


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def visible_text(payload: bytes) -> str:
    text = payload.decode("utf-8", errors="replace")
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def main() -> None:
    root = project_root()
    loop = (
        root
        / "validation_data/results/paleo_surface_validation_v0"
        / "climate_reconstruction_improvement_loop_v2"
    )
    drivers = (
        root
        / "Himalaya_greatGlacier/With_Rec_climate_organized"
        / "ice_surface_reconstruction/drivers"
    )
    output = loop / "round_189_jionglacuo_source_contract_correction_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = drivers / "round189_jionglacuo_source_contract_correction_v1_contract.md"
    snapshot_path = (
        loop
        / "round_178_mendeley_east_hma_temperature_candidates_v1"
        / "metadata_unmodified/s7t4zc34sr_v2_snapshot.json"
    )
    round184_summary_path = (
        loop / "round_184_east_tp_external_temperature_bias_audit_v1/summary.json"
    )
    round184_manifest_path = (
        loop
        / "round_184_east_tp_external_temperature_bias_audit_v1/source_manifest.json"
    )
    inputs = {
        "metadata_snapshot": snapshot_path,
        "round184_summary": round184_summary_path,
        "round184_source_manifest": round184_manifest_path,
        "contract": contract,
    }
    before = {name: sha256_file(path) for name, path in inputs.items()}
    shutil.copyfile(contract, output / "ROUND_CONTRACT.md")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    round184_summary = json.loads(round184_summary_path.read_text(encoding="utf-8"))
    round184_manifest = json.loads(round184_manifest_path.read_text(encoding="utf-8"))
    if snapshot.get("doi") != DATASET_DOI:
        raise RuntimeError("dataset-doi-mismatch")
    related = [item.get("href", "") for item in snapshot.get("related_links", [])]
    metadata_links_article = any(ARTICLE_DOI in value for value in related)
    if not metadata_links_article:
        raise RuntimeError("dataset-article-link-not-established")

    response = requests.get(
        ARTICLE_URL,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 climate-source-identity-audit/1.0"},
    )
    response.raise_for_status()
    article_bytes = response.content
    (output / "official_nature_article.html").write_bytes(article_bytes)
    article_text = visible_text(article_bytes)
    site_mentions = len(re.findall(r"\bJionglacuo\b", article_text, flags=re.I))
    methods_identity = bool(
        re.search(
            r"collected\s+79\s+core\s+samples.*?at\s+Jionglacuo",
            article_text,
            flags=re.I | re.S,
        )
    )
    figure_identity = bool(
        re.search(r"Balfour\s+spruce\s+sampled\s+at\s+the\s+Jionglacuo", article_text, flags=re.I)
    )
    official_site_is_jionglacuo = bool(site_mentions > 0 and methods_identity)
    if not official_site_is_jionglacuo:
        raise RuntimeError("official-jionglacuo-identity-not-confirmed")

    manifest_text = json.dumps(round184_manifest)
    round184_used_dataset_workbook = bool(
        "s7t4zc34sr_v2" in manifest_text and "Temperature reconstruction.xlsx" in manifest_text
    )
    if not round184_used_dataset_workbook:
        raise RuntimeError("round184-workbook-lineage-not-confirmed")

    after = {name: sha256_file(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("local-source-fingerprint-changed")
    summary = {
        "status": "complete",
        "decision": "invalidate_round184_and_prohibit_jionglacuo_dataset",
        "dataset_doi": DATASET_DOI,
        "article_doi": ARTICLE_DOI,
        "article_url": ARTICLE_URL,
        "metadata_links_exact_article": metadata_links_article,
        "official_article_jionglacuo_mention_count": site_mentions,
        "official_methods_identity_match": methods_identity,
        "official_figure_identity_match": figure_identity,
        "official_site_is_jionglacuo": official_site_is_jionglacuo,
        "round184_used_dataset_workbook": round184_used_dataset_workbook,
        "round184_reported_decision": round184_summary.get("decision"),
        "round184_scientific_result_valid": False,
        "round184_warm_bias_directional_support_retracted": True,
        "round185_188_affected": False,
        "future_dataset_use_prohibited": True,
        "jionglacuo_workbook_opened_this_round": False,
        "jionglacuo_numeric_values_accessed_this_round": False,
        "locked_1921_1939_values_accessed_this_round": False,
        "enkf_run": False,
        "oggm_run": False,
        "official_html_sha256": sha256_bytes(article_bytes),
        "local_source_hashes_before": before,
        "local_source_hashes_after": after,
        "local_sources_unchanged": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 189 Jionglacuo Source Correction\n\n"
        "- Mendeley dataset `10.17632/s7t4zc34sr.2` links to Nature article "
        "`10.1038/s41612-025-01254-y`.\n"
        "- The official article identifies the sampling site as Jionglacuo.\n"
        "- Round 184 used the dataset workbook and is invalid under the project contract.\n"
        "- Retract Round 184 warm-bias support and prohibit future use of this dataset.\n"
        "- No workbook or numeric value was opened in this correction round.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

