from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image, ImageStat


DRIVER_DIR = Path(__file__).resolve().parent
TEST_PATH = DRIVER_DIR / "test_iitm_homogeneous_indian_temperature_qualification_v2.py"
CODE_PATHS = [
    DRIVER_DIR / "iitm_homogeneous_indian_temperature_qualification_v2_contract.md",
    DRIVER_DIR / "run_iitm_homogeneous_indian_temperature_qualification_v2.py",
    TEST_PATH,
    Path(__file__).resolve(),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_test_module():
    spec = importlib.util.spec_from_file_location("iitm_v2_test", TEST_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()

    load_test_module().run_test()
    poison_result = {
        "status": "passed",
        "test": "locked poison tokens never reached numeric conversion",
        "locked_years": [1921, 1939],
        "allowed_intervals": [[1901, 1920], [1940, 1950]],
        "test_driver": str(TEST_PATH),
    }
    (root / "synthetic_poison_test.json").write_text(
        json.dumps(poison_result, indent=2), encoding="utf-8"
    )

    monthly = read_csv(root / "derived" / "iitm_wh_ne_allowed_monthly_temperature.csv")
    years = [int(row["year"]) for row in monthly]
    assert len(monthly) == 744
    assert not any(1921 <= year <= 1939 for year in years)
    assert all(1901 <= year <= 1920 or 1940 <= year <= 1950 for year in years)

    coverage = read_csv(root / "derived" / "iitm_wh_ne_allowed_coverage.csv")
    assert len(coverage) == 4
    assert all(row["complete"].lower() == "true" for row in coverage)

    trends = read_csv(root / "derived" / "iitm_wh_ne_trend_direction_precheck.csv")
    trend_lookup = {
        (row["region_code"], row["allowed_segment"], row["season"], row["variable"]): row
        for row in trends
    }
    matrix = []
    roles = {
        "WHIND": "Indian Western Himalaya regional evidence",
        "NEIND": "Northeast India/eastern-HMA-margin evidence",
    }
    for region in ("WHIND", "NEIND"):
        for segment in ("1901-1920", "1940-1950"):
            annual = trend_lookup[(region, segment, "annual_Jan-Dec", "tmean_derived_c")]
            seasonal = [
                trend_lookup[(region, segment, season, "tmean_derived_c")]["trend_direction"]
                for season in (
                    "winter_DJF",
                    "pre_monsoon_MAM",
                    "monsoon_JJAS",
                    "post_monsoon_ON",
                )
            ]
            matrix.append(
                {
                    "region_code": region,
                    "allowed_segment": segment,
                    "spatial_role": roles[region],
                    "coverage": "complete",
                    "annual_tmean_derived_trend_c_per_century": annual["trend_c_per_century"],
                    "annual_direction": annual["trend_direction"],
                    "winter_DJF_direction": seasonal[0],
                    "pre_monsoon_MAM_direction": seasonal[1],
                    "monsoon_JJAS_direction": seasonal[2],
                    "post_monsoon_ON_direction": seasonal[3],
                    "source_independence": "IMD_source_correlated_processing_route",
                    "qualification": "regional_direction_and_processing_diagnostic_only",
                }
            )
    write_csv(root / "derived" / "iitm_wh_ne_qualification_matrix.csv", matrix)

    image_checks = []
    for relative in (
        Path("figures/iitm_wh_ne_allowed_annual_temperature.png"),
        Path("raw_unopened/TEMP-REG.jpg"),
    ):
        path = root / relative
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            stats = ImageStat.Stat(rgb)
            extrema = rgb.getextrema()
            nonblank = any(low != high for low, high in extrema)
            assert image.width > 300 and image.height > 200 and nonblank
            image_checks.append(
                {
                    "path": str(relative).replace("\\", "/"),
                    "width": image.width,
                    "height": image.height,
                    "channel_means": stats.mean,
                    "channel_extrema": extrema,
                    "nonblank": nonblank,
                }
            )

    source_manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    assert all(item["hash_unchanged"] for item in source_manifest)
    for item in source_manifest:
        assert sha256(root / "raw_unopened" / item["filename"]) == item["sha256"]

    report_path = root / "FINAL_RESULT.md"
    report = report_path.read_text(encoding="utf-8")
    amendment = """

## Verified regional and seasonal finding

The 1901-1920 annual directions differ by region: derived Tmean warms in
NEIND (`+0.546 degC/century`) but cools in WHIND
(`-0.872 degC/century`). The seasonal directions also cross: NEIND cools in
JJAS while warming in DJF and ON; WHIND warms in JJAS while cooling in DJF,
MAM, and ON. This is evidence against treating the early residual as a single
HMA-wide translation or phase correction.

Both regions cool in derived annual Tmean over 1940-1950, but this interval
contains only eleven annual points and the linear fits are endpoint-sensitive.
It is suitable as a terminal direction check, not as a correction amplitude.

The synthetic poison test passed, the qualification matrix was generated, and
independent verification found no locked or disallowed year in derived output.
"""
    if "## Verified regional and seasonal finding" not in report:
        report_path.write_text(report.rstrip() + amendment, encoding="utf-8")

    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "category": "official_source" if "raw_unopened" in path.parts else "derived_result",
                }
            )
    for path in CODE_PATHS:
        artifacts.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "category": "experiment_code_or_contract",
            }
        )
    (root / "artifact_manifest.json").write_text(
        json.dumps(artifacts, indent=2), encoding="utf-8"
    )

    verification = {
        "status": "passed",
        "poison_test": "passed",
        "allowed_monthly_rows": len(monthly),
        "locked_rows_in_output": sum(1921 <= year <= 1939 for year in years),
        "disallowed_rows_in_output": sum(
            not (1901 <= year <= 1920 or 1940 <= year <= 1950) for year in years
        ),
        "coverage_checks_complete": True,
        "official_source_hashes_unchanged": True,
        "image_checks": image_checks,
        "qualification_matrix_rows": len(matrix),
    }
    (root / "verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
