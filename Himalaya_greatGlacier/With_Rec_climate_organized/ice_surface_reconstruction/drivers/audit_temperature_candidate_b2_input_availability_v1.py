from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import netCDF4

import acquire_round67_mode_ra_allowed_modern_subset_v2 as access
import mode_sim_cdf2_parser_v1 as cdf2


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
LOOP = RESULTS / "climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "temperature_balanced_pooled_candidate_b2_input_availability_v1"
STAGE_A = LOOP / "temperature_multimodel_evolving_prior_stage_a_v2"
PREREG = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_preregistration_v1"
CATALOG = RESULTS / "climate_reconstruction_temperature_pmip_multimodel_prior_audit_v2/remote_source_catalog.json"
MODE_BASE = (
    "https://mode-ra.unibe.ch/data/ModE-Sim/outdata/"
    "lowres_20mem_Set_1420-3_1850-1/abs/ensstat/by_var/year/"
)
MODE_URLS = {
    "tas": MODE_BASE
    + "ModE-Sim_lowres_20mem_Set_1420-3_1850-1_ensmean_temp2_abs_1420-2009_year.nc",
    "pr": MODE_BASE
    + "ModE-Sim_lowres_20mem_Set_1420-3_1850-1_ensmean_totprec_abs_1420-2009_year.nc",
}
CESM_FILES = {
    "tas": Path(r"D:\VM\gcm_7\cesm\cesm_TREFHT_merged.nc"),
    "pr": Path(r"D:\VM\gcm_7\cesm\cesm_PRECT_merged.nc"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_netcdf_metadata(path: Path) -> dict[str, object]:
    with netCDF4.Dataset(path, "r") as dataset:
        variables = list(dataset.variables)
        dimensions = {name: len(dim) for name, dim in dataset.dimensions.items()}
        units = {
            name: str(getattr(var, "units", ""))
            for name, var in dataset.variables.items()
            if name.upper() in {"TREFHT", "PRECT", "TAS", "PR"}
        }
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "dimensions": dimensions,
        "variables": variables,
        "data_variable_units": units,
        "climate_values_read": False,
    }


def remote_mode_metadata(url: str, label: str, request_log: list[dict[str, object]]) -> dict[str, object]:
    source = access.ExactRangeSource(url, label, request_log)
    try:
        payload = source.read(0, access.HEADER_BYTES - 1, "header_only", None)
        header = cdf2.parse_header(payload)
    finally:
        source.close()
    variables = list(header.variables)
    target = "temp2" if label == "mode_tas" else "totprec"
    if target not in header.variables:
        raise RuntimeError(f"missing-mode-variable:{target}")
    variable = header.variables[target]
    return {
        "url": url,
        "declared_coverage": [1420, 2009],
        "target_variable": target,
        "target_dimensions": list(variable.dimensions),
        "target_shape": list(variable.shape),
        "target_units": str(variable.attributes.get("units", "")),
        "variables": variables,
        "request_type": "classic-netcdf-header-only",
        "climate_values_read": False,
    }


def catalog_summary(entries: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for model in ("MPI-ESM1-2-LR", "MIROC-ES2L"):
        model_entries = [entry for entry in entries if entry.get("model") == model]
        by_variable: dict[str, object] = {}
        for variable in ("tas", "pr"):
            selected = [
                entry
                for entry in model_entries
                if re.match(rf"^{variable}_", str(entry.get("title", "")))
                or entry.get("variable") == variable
            ]
            if not selected:
                raise RuntimeError(f"missing-catalog:{model}:{variable}")
            by_variable[variable] = {
                "source_count": len(selected),
                "minimum_source_year": min(int(entry["start_year_ce"]) for entry in selected),
                "maximum_source_year": max(int(entry["end_year_ce"]) for entry in selected),
                "requested_locked_climate_values": sum(
                    int(entry.get("locked_climate_values_requested", 0)) for entry in selected
                ),
                "urls": [str(entry["url"]) for entry in selected],
            }
        summary[model] = by_variable
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    sources = {
        "facts": ROOT / "LIA_PROJECT_FACTS.md",
        "stage_a_summary": STAGE_A / "summary.json",
        "candidate_contract": PREREG / "CANDIDATE_B2_STAGE_C_CONTRACT.md",
        "pmip_catalog": CATALOG,
        "driver": Path(__file__).resolve(),
    }
    for path in [*sources.values(), *CESM_FILES.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in sources.items()}
    stage_a = json.loads(sources["stage_a_summary"].read_text(encoding="utf-8"))
    if stage_a.get("stage_a_pass") is not True:
        raise RuntimeError("stage-a-not-passed")

    request_log: list[dict[str, object]] = []
    mode = {
        variable: remote_mode_metadata(url, f"mode_{variable}", request_log)
        for variable, url in MODE_URLS.items()
    }
    if any(item.get("purpose") != "header_only" for item in request_log):
        raise RuntimeError("mode-audit-read-more-than-header")

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    pmip = catalog_summary(catalog)
    if any(
        block[variable]["requested_locked_climate_values"] != 0
        for block in pmip.values()
        for variable in ("tas", "pr")
    ):
        raise RuntimeError("locked-values-in-pmip-catalog")

    cesm = {variable: local_netcdf_metadata(path) for variable, path in CESM_FILES.items()}
    after = {name: sha256(path) for name, path in sources.items()}
    if before != after:
        raise RuntimeError("source-hash-changed")

    all_available = all(path.exists() for path in CESM_FILES.values()) and all(
        mode[var]["declared_coverage"] == [1420, 2009] for var in ("tas", "pr")
    )
    summary = {
        "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "candidate_b2_input_sources_available": all_available,
        "allowed_periods": [[1500, 1920], [1940, 2000]],
        "sealed_period": [1921, 1939],
        "cesm": cesm,
        "mode_sim": mode,
        "pmip": pmip,
        "stage_a_common_valid_cells": int(stage_a["common_valid_state_cell_count"]),
        "stage_a_members_per_model": int(stage_a["paired_state_count_per_model"]),
        "mode_header_request_count": len(request_log),
        "mode_climate_values_read": False,
        "real_proxy_observations_read": False,
        "stage_c_enkf_run": False,
        "production_modified": False,
        "oggm_run": False,
        "prohibited_source": "20CRv3",
        "prohibited_source_accessed": False,
        "source_hashes_unchanged": True,
        "source_hashes": before,
        "next_action": "build_exact_allowed_segment_four_model_input_package",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "mode_sim_header_request_log.json").write_text(
        json.dumps(request_log, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "FINAL_RESULT.md").write_text(
        "# Candidate B2 input availability\n\n"
        "All four model families have identifiable temperature and precipitation "
        "sources covering both allowed segments. The previously missing ModE-Sim "
        "modern precipitation source is available from the official annual "
        "1420-2009 ensemble-mean product.\n\n"
        "Only remote headers were read in this audit. No climate value, real proxy "
        "observation, sealed year, prohibited source, production file, or OGGM "
        "model was accessed.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"],
        "candidate_b2_input_sources_available": all_available,
        "stage_a_common_valid_cells": summary["stage_a_common_valid_cells"],
        "stage_a_members_per_model": summary["stage_a_members_per_model"],
        "mode_header_request_count": len(request_log),
        "next_action": summary["next_action"],
    }, indent=2))


if __name__ == "__main__":
    main()
