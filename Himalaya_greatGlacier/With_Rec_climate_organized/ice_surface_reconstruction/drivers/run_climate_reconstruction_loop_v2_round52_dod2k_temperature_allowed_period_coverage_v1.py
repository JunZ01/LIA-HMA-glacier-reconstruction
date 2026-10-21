"""Audit allowed-period coverage of new DoD2k HMA temperature candidates."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WINDOWS = tuple((start, start + 49) for start in range(1400, 1900, 50))
WEAK_WINDOWS = ((1400, 1449), (1450, 1499), (1550, 1599), (1650, 1699))
LOCKED_START = 1921
LOCKED_END = 1939


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def read_candidate_rows(path: Path, candidate_ids: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) != 2 or header[0] != "datasetId":
            raise RuntimeError(f"unexpected-header:{path.name}:{header}")
        for row in reader:
            if row and row[0] in candidate_ids:
                result[row[0]] = row[1:]
    missing = candidate_ids - set(result)
    if missing:
        raise RuntimeError(f"candidate-rows-missing:{path.name}:{sorted(missing)[:5]}")
    return result


def finite_float(token: str) -> float:
    try:
        value = float(token)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_52_dod2k_temperature_allowed_period_coverage_v1"
    round35 = loop / "round_35_dod2k_hma_hydroclimate_coverage_audit_v1"
    round51 = loop / "round_51_dod2k_hma_temperature_metadata_expansion_v1"
    paths = {
        "year_matrix": round35 / "dod2k_v2.0_compact_year.csv",
        "value_matrix": round35 / "dod2k_v2.0_compact_paleoData_values.csv",
        "candidate_metadata": round51 / "dod2k_hma_temperature_metadata_candidates.csv",
        "round51_summary": round51 / "summary.json",
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}

    metadata_all = pd.read_csv(paths["candidate_metadata"])
    metadata = metadata_all[~metadata_all.already_routed_by_round34.astype(bool)].copy()
    round51_summary = json.loads(paths["round51_summary"].read_text(encoding="utf-8"))
    if len(metadata) != int(round51_summary["new_temperature_candidate_count"]):
        raise RuntimeError("round51-candidate-count-failed")
    if metadata.datasetId.duplicated().any():
        raise RuntimeError("candidate-id-duplicate")
    metadata["datasetId"] = metadata.datasetId.astype(str)
    candidate_ids = set(metadata.datasetId)
    year_rows = read_candidate_rows(paths["year_matrix"], candidate_ids)
    value_rows = read_candidate_rows(paths["value_matrix"], candidate_ids)

    coverage_rows: list[dict[str, object]] = []
    allowed_rows: list[dict[str, object]] = []
    locked_positions_excluded = 0
    locked_values_parsed = 0
    for item in metadata.itertuples(index=False):
        year_tokens = year_rows[item.datasetId]
        value_tokens = value_rows[item.datasetId]
        if len(year_tokens) != len(value_tokens):
            raise RuntimeError(f"year-value-length-mismatch:{item.datasetId}")
        allowed_years: list[float] = []
        allowed_values: list[float] = []
        for year_token, value_token in zip(year_tokens, value_tokens):
            year = finite_float(year_token)
            if not np.isfinite(year):
                continue
            if LOCKED_START <= year <= LOCKED_END:
                locked_positions_excluded += 1
                continue
            value = finite_float(value_token)
            if not np.isfinite(value):
                continue
            allowed_years.append(year)
            allowed_values.append(value)
            allowed_rows.append(
                {"datasetId": item.datasetId, "year_ce": year, "paleo_value": value}
            )
        years = np.asarray(allowed_years, dtype=float)
        values = np.asarray(allowed_values, dtype=float)
        order = np.argsort(years, kind="stable")
        years = years[order]
        values = values[order]
        lia = (years >= 1400) & (years <= 1899)
        modern = (years >= 1940) & (years <= 2000)
        window_counts = [
            int(np.sum((years >= start) & (years <= end))) for start, end in WINDOWS
        ]
        occupied_windows = int(np.sum(np.asarray(window_counts) > 0))
        weak_window_count = int(
            np.sum(
                [
                    np.any((years >= start) & (years <= end))
                    for start, end in WEAK_WINDOWS
                ]
            )
        )
        lia_unique = np.unique(years[lia])
        modern_unique = np.unique(years[modern])
        lia_spacing = (
            float(np.median(np.diff(lia_unique))) if len(lia_unique) >= 2 else np.nan
        )
        modern_spacing = (
            float(np.median(np.diff(modern_unique)))
            if len(modern_unique) >= 2
            else np.nan
        )
        lia_eligible = bool(
            occupied_windows >= 3 and np.isfinite(lia_spacing) and lia_spacing <= 25
        )
        low_frequency = bool(lia_eligible and int(lia.sum()) >= 10)
        modern_capable = bool(int(modern.sum()) >= 20)
        coverage_rows.append(
            {
                "datasetId": item.datasetId,
                "archiveType": item.archiveType,
                "dataSetName": item.dataSetName,
                "geo_meanLat": item.geo_meanLat,
                "geo_meanLon": item.geo_meanLon,
                "geo_siteName": item.geo_siteName,
                "candidate_site_status": item.candidate_site_status,
                "new_temperature_evidence_priority": item.new_temperature_evidence_priority,
                "interpretation_direction": item.interpretation_direction,
                "interpretation_seasonality": item.interpretation_seasonality,
                "interpretation_variable": item.interpretation_variable,
                "interpretation_variableDetail": item.interpretation_variableDetail,
                "paleoData_proxy": item.paleoData_proxy,
                "paleoData_variableName": item.paleoData_variableName,
                "originalDatabase": item.originalDatabase,
                "originalDataURL": item.originalDataURL,
                "allowed_total_finite_count": int(len(years)),
                "allowed_first_year_ce": float(np.min(years)) if len(years) else np.nan,
                "allowed_last_year_ce": float(np.max(years)) if len(years) else np.nan,
                "lia_1400_1899_finite_count": int(lia.sum()),
                "lia_occupied_50year_window_count": occupied_windows,
                "weak_round30_window_support_count": weak_window_count,
                "lia_median_spacing_years": lia_spacing,
                "modern_1940_2000_finite_count": int(modern.sum()),
                "modern_median_spacing_years": modern_spacing,
                **{
                    f"window_{start}_{end}_count": count
                    for (start, end), count in zip(WINDOWS, window_counts)
                },
                "lia_evidence_eligible": lia_eligible,
                "low_frequency_validation_eligible": low_frequency,
                "modern_calibration_capable": modern_capable,
                "both_lia_and_modern_capable": bool(lia_eligible and modern_capable),
            }
        )

    coverage = pd.DataFrame(coverage_rows)
    allowed = pd.DataFrame(allowed_rows)
    coverage.to_csv(output / "dod2k_hma_temperature_coverage.csv", index=False)
    allowed.to_csv(output / "dod2k_hma_temperature_allowed_values.csv", index=False)
    eligible = coverage[
        coverage.lia_evidence_eligible | coverage.modern_calibration_capable
    ].copy()
    eligible.to_csv(output / "dod2k_hma_temperature_coverage_eligible.csv", index=False)
    both = coverage[coverage.both_lia_and_modern_capable].copy()
    grouped = (
        coverage.groupby(["originalDatabase", "archiveType"])
        .agg(
            record_count=("datasetId", "size"),
            lia_evidence_count=("lia_evidence_eligible", "sum"),
            modern_calibration_count=("modern_calibration_capable", "sum"),
            both_lia_modern_count=("both_lia_and_modern_capable", "sum"),
        )
        .reset_index()
    )
    grouped.to_csv(output / "coverage_counts_by_source_archive.csv", index=False)
    summary = {
        "status": "complete",
        "candidate_count": int(len(coverage)),
        "locked_1921_1939_year_positions_excluded_before_value_parse": int(
            locked_positions_excluded
        ),
        "locked_1921_1939_numeric_proxy_values_parsed": int(locked_values_parsed),
        "lia_evidence_eligible_count": int(coverage.lia_evidence_eligible.sum()),
        "low_frequency_validation_eligible_count": int(
            coverage.low_frequency_validation_eligible.sum()
        ),
        "modern_calibration_capable_count": int(
            coverage.modern_calibration_capable.sum()
        ),
        "both_lia_and_modern_capable_count": int(len(both)),
        "new_site_both_lia_and_modern_count": int(
            np.sum(
                both.candidate_site_status.eq("new_site_candidate")
            )
        ),
        "non_wood_both_lia_and_modern_count": int(
            np.sum(~both.archiveType.fillna("").str.lower().eq("wood"))
        ),
        "both_capable_supporting_all_four_weak_windows_count": int(
            np.sum(both.weak_round30_window_support_count == 4)
        ),
        "both_lia_modern_source_counts": {
            str(key): int(value) for key, value in both.originalDatabase.value_counts().items()
        },
        "both_lia_modern_archive_counts": {
            str(key): int(value) for key, value in both.archiveType.value_counts().items()
        },
        "decision": (
            "advance_both_capable_records_to_identity_chronology_direction_review"
            if len(both)
            else (
                "use_new_temperature_records_as_low_frequency_evidence_only"
                if coverage.low_frequency_validation_eligible.any()
                else "close_dod2k_temperature_expansion"
            )
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    matrix_columns = [f"window_{start}_{end}_count" for start, end in WINDOWS]
    matrix = coverage.sort_values(
        ["both_lia_and_modern_capable", "low_frequency_validation_eligible", "datasetId"],
        ascending=[False, False, True],
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 7.5), constrained_layout=True)
    image = axes[0].imshow(
        np.log1p(matrix[matrix_columns].to_numpy(float)),
        aspect="auto",
        cmap="viridis",
    )
    axes[0].set(
        xticks=np.arange(len(WINDOWS)),
        xticklabels=[str(start) for start, _ in WINDOWS],
        xlabel="50-year window start",
        ylabel="Candidate row",
        title="Temperature-candidate LIA sampling density",
    )
    figure.colorbar(image, ax=axes[0], shrink=0.8)
    labels = ["LIA", "Low-frequency", "Modern", "Both"]
    values = [
        int(coverage.lia_evidence_eligible.sum()),
        int(coverage.low_frequency_validation_eligible.sum()),
        int(coverage.modern_calibration_capable.sum()),
        int(len(both)),
    ]
    axes[1].bar(labels, values, color=["#0072b2", "#009e73", "#e69f00", "#d55e00"])
    axes[1].set(ylabel="Record count", title="Allowed-period eligibility")
    for index, value in enumerate(values):
        axes[1].text(index, value + 0.5, str(value), ha="center")
    figure.savefig(output / "dod2k_hma_temperature_coverage.png", dpi=180)
    plt.close(figure)

    lines = [
        "# DoD2k HMA Temperature Allowed-Period Coverage",
        "",
        f"- Candidates: `{len(coverage)}`; locked positions skipped before value parsing: `{locked_positions_excluded}`.",
        f"- LIA eligible: `{summary['lia_evidence_eligible_count']}`; modern capable: `{summary['modern_calibration_capable_count']}`.",
        f"- Both LIA and modern capable: `{len(both)}`; new-site both-capable: `{summary['new_site_both_lia_and_modern_count']}`; non-wood both-capable: `{summary['non_wood_both_lia_and_modern_count']}`.",
        f"- Both-capable records covering all four weak Round 30 windows: `{summary['both_capable_supporting_all_four_weak_windows_count']}`.",
        "",
        f"Decision: `{summary['decision']}`.",
    ]
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "sources_before": before,
                "sources_after": after,
                "source_states_unchanged": unchanged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
