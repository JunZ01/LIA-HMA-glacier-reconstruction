from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DOI = "10.1016/j.palaeo.2011.02.017"
PII = "S003101821100085X"
EXTENSIONS = ("xls", "xlsx", "csv", "txt", "zip", "doc", "docx", "pdf")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def request_bytes(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    content_type: str | None = None,
) -> tuple[int | None, bytes, dict[str, str], str, str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return (
                response.status,
                response.read(),
                dict(response.headers.items()),
                response.geturl(),
                "",
            )
    except urllib.error.HTTPError as error:
        return error.code, b"", dict(error.headers.items()), url, str(error)
    except Exception as error:
        return None, b"", {}, url, f"{type(error).__name__}: {error}"


def main() -> int:
    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/"
        "round_173_zhu2011_ranwu_public_data_recovery_v1"
    )
    raw = output / "raw_unopened"
    metadata = output / "repository_metadata"
    raw.mkdir(parents=True, exist_ok=False)
    metadata.mkdir()

    probes: list[dict[str, object]] = []
    attachments: list[dict[str, object]] = []
    for number in (1, 2):
        for extension in EXTENSIONS:
            url = (
                "https://ars.els-cdn.com/content/image/1-s2.0-"
                f"{PII}-mmc{number}.{extension}"
            )
            status, payload, headers, final_url, error = request_bytes(url)
            probe = {
                "url": url,
                "status": status,
                "content_type": headers.get("Content-Type", ""),
                "content_length": len(payload),
                "final_url": final_url,
                "error": error,
            }
            probes.append(probe)
            if status == 200 and payload:
                path = raw / f"zhu2011_mmc{number}.{extension}"
                path.write_bytes(payload)
                attachment = {
                    **probe,
                    "saved_path": str(path),
                    "sha256": sha256_bytes(payload),
                    "opened_or_parsed": False,
                }
                attachments.append(attachment)

    query = urllib.parse.quote(DOI, safe="")
    endpoints = {
        "crossref": f"https://api.crossref.org/works/{query}",
        "datacite": f"https://api.datacite.org/dois?query={query}",
        "zenodo": f"https://zenodo.org/api/records?q={query}&size=25",
    }
    repository_results: dict[str, dict[str, object]] = {}
    for name, url in endpoints.items():
        status, payload, headers, final_url, error = request_bytes(url)
        path = metadata / f"{name}.json"
        if payload:
            path.write_bytes(payload)
        repository_results[name] = {
            "url": url,
            "status": status,
            "content_type": headers.get("Content-Type", ""),
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload) if payload else None,
            "final_url": final_url,
            "error": error,
        }

    figshare_url = "https://api.figshare.com/v2/articles/search"
    figshare_payload = json.dumps({"search_for": DOI, "limit": 100}).encode("utf-8")
    status, payload, headers, final_url, error = request_bytes(
        figshare_url,
        data=figshare_payload,
        content_type="application/json",
    )
    if payload:
        (metadata / "figshare.json").write_bytes(payload)
    repository_results["figshare"] = {
        "url": figshare_url,
        "status": status,
        "content_type": headers.get("Content-Type", ""),
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload) if payload else None,
        "final_url": final_url,
        "error": error,
    }

    dataset_hits: dict[str, int | None] = {}
    for name in ("datacite", "zenodo", "figshare"):
        path = metadata / f"{name}.json"
        if not path.exists():
            dataset_hits[name] = None
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if name == "datacite":
            dataset_hits[name] = len(document.get("data", []))
        elif name == "zenodo":
            dataset_hits[name] = int(document.get("hits", {}).get("total", 0))
        else:
            dataset_hits[name] = len(document) if isinstance(document, list) else None

    public_data_found = bool(attachments) or any(
        count is not None and count > 0 for count in dataset_hits.values()
    )
    decision = (
        "advance_public_candidate_to_boundary_safe_structure_audit"
        if public_data_found
        else "close_zhu2011_anonymous_public_data_recovery"
    )
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "doi": DOI,
        "pii": PII,
        "official_attachment_probe_count": len(probes),
        "official_attachment_hit_count": len(attachments),
        "official_attachments": attachments,
        "repository_results": repository_results,
        "repository_exact_query_hit_counts": dataset_hits,
        "public_data_candidate_found": public_data_found,
        "candidate_files_opened_or_parsed": False,
        "annual_proxy_or_climate_values_accessed": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "decision": decision,
    }
    (output / "attachment_probes.json").write_text(
        json.dumps(probes, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 173 Zhu 2011 Ranwu Public-Data Recovery\n\n"
        f"- Official attachment paths probed: `{len(probes)}`.\n"
        f"- Official attachment hits: `{len(attachments)}`.\n"
        f"- Repository exact-query hits: `{dataset_hits}`.\n"
        "- Candidate files opened or parsed: `false`.\n"
        "- Annual values accessed: `false`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
