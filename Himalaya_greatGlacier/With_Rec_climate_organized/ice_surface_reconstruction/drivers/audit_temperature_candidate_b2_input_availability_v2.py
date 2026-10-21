from __future__ import annotations

import hashlib
import json
from pathlib import Path

import acquire_round67_mode_ra_allowed_modern_subset_v2 as access
import audit_temperature_candidate_b2_input_availability_v1 as v1
import mode_sim_cdf2_parser_v1 as cdf2


v1.OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_input_availability_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_mode_metadata(
    url: str, label: str, request_log: list[dict[str, object]]
) -> dict[str, object]:
    source = access.ExactRangeSource(url, label, request_log)
    try:
        payload = source.read(0, access.HEADER_BYTES - 1, "header_only", None)
        header = cdf2.parse_header(payload)
    finally:
        source.close()
    target = "temp2" if label == "mode_tas" else "totprec"
    if target not in header.variables:
        raise RuntimeError(f"missing-mode-variable:{target}")
    variable = header.variables[target]
    shape = [
        header.num_records if header.dimensions[name] is None else header.dimensions[name]
        for name in variable.dimensions
    ]
    return {
        "url": url,
        "declared_coverage": [1420, 2009],
        "target_variable": target,
        "target_dimensions": list(variable.dimensions),
        "target_shape": shape,
        "target_units": str(variable.attributes.get("units", "")),
        "variables": list(header.variables),
        "request_type": "classic-netcdf-header-only",
        "climate_values_read": False,
    }


def main() -> None:
    v1.remote_mode_metadata = remote_mode_metadata
    v1.main()
    summary_path = v1.OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment"] = v1.OUT.name
    summary["supersedes_technical_failure"] = (
        "temperature_balanced_pooled_candidate_b2_input_availability_v1"
    )
    summary["v2_change"] = "derive CDF2 variable shape from header dimensions"
    summary["technical_retry_driver_sha256"] = sha256(Path(__file__).resolve())
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
