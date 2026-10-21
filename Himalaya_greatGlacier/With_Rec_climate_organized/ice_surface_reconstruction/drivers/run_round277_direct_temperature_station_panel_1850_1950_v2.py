from __future__ import annotations

import json
import shutil
import tempfile as stdlib_tempfile
from collections import defaultdict
from pathlib import Path

import run_round277_direct_temperature_station_panel_1850_1950_v1 as v1


V1_DRIVER = Path(v1.__file__).resolve()
V1_RESULT = v1.LOOP / "round_277_direct_temperature_station_panel_1850_1950_v1"
V2_RESULT = v1.LOOP / "round_277_direct_temperature_station_panel_1850_1950_v2"
V2_CONTRACT = v1.DRIVERS / "round277_direct_temperature_station_panel_1850_1950_v2_contract.md"
V2_DRIVER = Path(__file__).resolve()
V2_TEST = v1.DRIVERS / "test_round277_direct_temperature_station_panel_1850_1950_v2.py"

_V1_PARSE = v1.parse_ghcnm_lines
_V1_RENDER = v1.render_final


def parse_ghcnm_lines_v2(lines, target_ids, series, audit=None):
    rows = _V1_PARSE(lines, target_ids, series, audit)
    for row in rows:
        if row["temperature_c"] != "":
            row["temperature_c"] = float(row["temperature_c"]) / 10.0
        if row["analysis_temperature_c"] != "":
            row["analysis_temperature_c"] = float(row["analysis_temperature_c"]) / 10.0
    return rows


def run_poison_test_v2() -> dict[str, object]:
    station_id = next(iter(v1.STATIONS))
    locked_audit: dict[str, int] = defaultdict(int)
    locked_rows = parse_ghcnm_lines_v2(
        [v1._synthetic_line(station_id, 1930, "XXXXX")],
        {station_id},
        "synthetic",
        locked_audit,
    )
    if locked_rows or locked_audit["monthly_cells_accessed"] != 0:
        raise AssertionError("Locked poison reached the monthly value path")
    allowed_poison_raised = False
    try:
        parse_ghcnm_lines_v2(
            [v1._synthetic_line(station_id, 1920, "XXXXX")],
            {station_id},
            "synthetic",
            defaultdict(int),
        )
    except ValueError:
        allowed_poison_raised = True
    if not allowed_poison_raised:
        raise AssertionError("Allowed poison did not reach the monthly value path")
    valid_audit: dict[str, int] = defaultdict(int)
    valid_rows = parse_ghcnm_lines_v2(
        [v1._synthetic_line(station_id, 1940, "  123")],
        {station_id},
        "synthetic",
        valid_audit,
    )
    if len(valid_rows) != 12 or valid_rows[0]["temperature_c"] != 1.23:
        raise AssertionError("Hundredths-of-degree conversion failed")
    return {
        "passed": True,
        "unit": "degree_Celsius",
        "integer_scale": 0.01,
        "locked_year_tested": 1930,
        "locked_output_row_count": len(locked_rows),
        "locked_monthly_cells_accessed": locked_audit["monthly_cells_accessed"],
        "locked_rows_skipped_before_value_access": locked_audit[
            "locked_target_rows_skipped_before_value_access"
        ],
        "allowed_year_poison_raised_value_error": allowed_poison_raised,
        "allowed_valid_monthly_cells_accessed": valid_audit["monthly_cells_accessed"],
        "allowed_valid_temperature_c": valid_rows[0]["temperature_c"],
    }


class SameVolumeTempfile:
    @staticmethod
    def mkdtemp(prefix: str) -> str:
        V2_RESULT.parent.mkdir(parents=True, exist_ok=True)
        return stdlib_tempfile.mkdtemp(prefix=prefix, dir=V2_RESULT.parent)


def render_final_v2(summary, coverage, lineage):
    text = _V1_RENDER(summary, coverage, lineage)
    text = text.replace("# Round 277 final result", "# Round 277 v2 final result", 1)
    marker = "## Safety result\n"
    correction = (
        "## Unit correction\n\n"
        "GHCN-M v4 fixed-width values are converted from hundredths of degrees "
        "Celsius using `/100`. The v1 derived panel used `/10` and is superseded; "
        "its preserved official source archives remain valid.\n\n"
    )
    return text.replace(marker, correction + marker, 1)


def main() -> None:
    if V2_RESULT.exists():
        raise FileExistsError(f"Independent v2 result already exists: {V2_RESULT}")
    v1.OUT = V2_RESULT
    v1.CONTRACT = V2_CONTRACT
    v1.DRIVER = V2_DRIVER
    v1.TEST = V2_TEST
    v1.parse_ghcnm_lines = parse_ghcnm_lines_v2
    v1.run_poison_test = run_poison_test_v2
    v1.render_final = render_final_v2
    v1.tempfile = SameVolumeTempfile
    v1.main()

    summary_path = V2_RESULT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "experiment": "round_277_direct_temperature_station_panel_1850_1950_v2",
            "temperature_unit": "degree_Celsius",
            "ghcn_integer_scale": 0.01,
            "supersedes": V1_RESULT.name,
            "supersession_reason": "v1 divided GHCN-M hundredths-of-degree integers by 10 instead of 100",
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest_path = V2_RESULT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment"] = summary["experiment"]
    manifest["unit_conversion"] = "fixed_width_integer / 100.0 degree_Celsius"
    manifest["supersedes"] = {
        "result": str(V1_RESULT),
        "reason": summary["supersession_reason"],
        "source_archives_remain_valid": True,
        "derived_temperature_values_valid": False,
    }
    manifest["implementation_dependency"] = {
        "path": str(V1_DRIVER),
        "size_bytes": V1_DRIVER.stat().st_size,
        "sha256": v1.sha256(V1_DRIVER),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    supersession = (
        "# V1 supersession\n\n"
        f"`{V1_RESULT.name}` is invalid for derived temperature magnitudes because "
        "it divided GHCN-M v4 hundredths-of-degree integers by 10. V2 uses `/100`. "
        "The v1 source archives and hashes remain valid; do not use its monthly "
        "temperature or qcu-qcf adjustment columns.\n"
    )
    (V2_RESULT / "V1_SUPERSESSION.md").write_text(supersession, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

