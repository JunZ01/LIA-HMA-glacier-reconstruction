from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
V1 = BASE / "temperature_original_numeric_source_availability_v1"
OUTPUT = BASE / "temperature_original_numeric_source_availability_v2"
DRIVER = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update(frame: pd.DataFrame, record: str, values: dict[str, object]) -> None:
    index = frame.index[frame.record == record]
    if len(index) != 1:
        raise RuntimeError(f"record-identity-failed:{record}:{len(index)}")
    for key, value in values.items():
        frame.loc[index[0], key] = value


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(V1 / "original_numeric_source_availability.csv", keep_default_na=False)
    update(
        frame,
        "Kusai Lake JJA temperature",
        {
            "source_status": "official_article_and_DOC_supplement_verified_no_public_year_temperature_table",
            "original_author_data": False,
            "machine_readable": False,
            "current_safe_use": "published_qualitative_direction_context_only_not_numeric_metric",
            "direct_url": "https://www.nature.com/articles/srep06672",
        },
    )
    update(
        frame,
        "Yamzhog Yumco pollen temperature",
        {
            "source_status": "official_article_verified_no_public_original_numeric_temperature_table",
            "original_author_data": False,
            "machine_readable": False,
            "current_safe_use": "published_cold_warm_stage_context_only_not_numeric_metric",
            "direct_url": "https://www.sciencedirect.com/science/article/pii/S0031018217311057",
        },
    )
    frame.to_csv(OUTPUT / "original_numeric_source_availability.csv", index=False)
    summary = {
        "experiment": "temperature_original_numeric_source_availability_v2",
        "status": "complete",
        "supersedes": "temperature_original_numeric_source_availability_v1",
        "record_count": int(len(frame)),
        "kusai_web_only_verdict": (
            "Official article provides the JJA calibration and a plotted 350-2010 reconstruction; the official DOC "
            "supplement is identified as supplementary tables and figures, with no public machine-readable year-temperature table."
        ),
        "yamzhog_web_only_verdict": (
            "Official article provides pollen-based cold/warm stages but no public machine-readable calibrated temperature series."
        ),
        "quarantined_agent_result_used": False,
        "web_only_reverification_used": True,
        "decision": "public_numeric_source_search_closed;author_requests_or_future_repository_deposits_are_required_for_unavailable_records",
        "sealed_period_values_used": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "input": {
            "path": str(V1 / "original_numeric_source_availability.csv"),
            "sha256": sha256(V1 / "original_numeric_source_availability.csv"),
        },
        "official_web_sources": [
            "https://www.nature.com/articles/srep06672",
            "https://www.sciencedirect.com/science/article/pii/S0031018217311057",
        ],
        "driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
        "quarantined_agent_result_used": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Original numeric temperature-source availability v2",
        "",
        "The quarantined Kusai/Yamzhog search result was not used. Both records were independently rechecked using official article pages only.",
        "",
        "| Record | Status | Current safe use |",
        "|---|---|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(f"| {row.record} | {row.source_status} | {row.current_safe_use} |")
    lines.extend(
        [
            "",
            "The public numeric search is now closed. Records without original machine tables require author delivery or a future official repository deposit.",
            "",
            "No sealed-period value was used. Production was not changed, EnKF and OGGM were not run, and 20CRv3 was not accessed.",
        ]
    )
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
