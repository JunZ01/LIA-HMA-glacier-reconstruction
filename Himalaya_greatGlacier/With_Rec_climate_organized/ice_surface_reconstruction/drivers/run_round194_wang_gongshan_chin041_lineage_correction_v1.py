from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


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


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_194_wang_gongshan_chin041_lineage_correction_v1"
    output.mkdir(parents=True, exist_ok=False)
    inputs = {
        "round190_summary": loop / "round_190_wang2015_source_overlap_audit_v1/summary.json",
        "round190_sites": loop / "round_190_wang2015_source_overlap_audit_v1/wang12_project_overlap.csv",
        "round191_summary": loop / "round_191_chin041_hengduan_sample_identity_v1/summary.json",
        "round193_summary": loop / "round_193_east_hma_dual_index_wang_shape_v1/summary.json",
    }
    hashes = {name: sha256(path) for name, path in inputs.items()}
    r190 = json.loads(inputs["round190_summary"].read_text(encoding="ascii"))
    r191 = json.loads(inputs["round191_summary"].read_text(encoding="ascii"))
    r193 = json.loads(inputs["round193_summary"].read_text(encoding="ascii"))
    with inputs["round190_sites"].open("r", encoding="utf-8-sig", newline="") as stream:
        sites = list(csv.DictReader(stream))
    gongshan = next(row for row in sites if row["wang_site"] == "Gongshan")
    if r190["wang_site_count"] != 12 or r191["chin041_sample_count"] != 35 or r193["passed"]:
        raise RuntimeError("unexpected-upstream-state")

    correction = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "authoritative_sources": {
            "wang2015_primary": "https://journals.ametsoc.org/abstract/journals/clim/28/13/jcli-d-14-00738.1.xml",
            "fan2010_primary_pdf": "https://xtbg.cas.cn/sourcedb/zw/lw/201009/P020100908412545830723.pdf",
            "chin041_noaa_metadata": "https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/noaa-tree-14910/html",
            "chin041_noaa_raw": "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/asia/chin041.rwl",
        },
        "wang_gongshan": {
            "species": "Larix speciosa",
            "source": "Fan et al. 2010 DL1 plus DL2 regional chronology",
            "sample_series_count": 68,
            "reliable_period": "1585-2005",
            "detrending": "negative exponential function",
        },
        "chin041": {
            "site": "Hengduan Mountains 22DL_T_r",
            "species": "Tsuga dumosa",
            "dated_series_count": 35,
            "period": "1530-2005",
        },
        "identities_same": False,
        "round190_gongshan_nearest_pages_record": gongshan["nearest_pages2kv2_name"],
        "round191_scientific_scope_corrected_to": "CHIN041_vs_new_Hengduan_exact_sample_identity_only",
        "wang_gongshan_vs_new_hengduan_dll": "unresolved_pending_Fan2010_DL1_DL2_raw_identity",
        "round193_validity": "valid_failed_partially_source_overlapping_shape_benchmark",
        "round193_dual_index_promoted": False,
        "annual_values_decoded": False,
        "locked_1921_1939_numeric_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "oggm_run": False,
        "decision": "close_chin041_as_wang_identity_route_and_keep_gongshan_dll_independence_unresolved",
    }
    (output / "lineage_correction.json").write_text(json.dumps(correction, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"inputs": hashes}, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 194 Wang Gongshan--CHIN041 Lineage Correction\n\n"
        "- Wang Gongshan: `Larix speciosa`, Fan 2010 `DL1+DL2`, 68 series.\n"
        "- CHIN041: `Tsuga dumosa`, `22DL_T_r`, 35 dated series.\n"
        "- They are not the same chronology or collection.\n"
        "- Round 191 excludes only direct CHIN041 reuse; it does not resolve Gongshan--DLL independence.\n"
        "- Round 193 remains a valid failed, partially source-overlapping benchmark test.\n\n"
        "Decision: keep Gongshan--DLL independence unresolved pending Fan 2010 DL1/DL2 raw identity.\n",
        encoding="ascii",
    )
    print(json.dumps(correction, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

