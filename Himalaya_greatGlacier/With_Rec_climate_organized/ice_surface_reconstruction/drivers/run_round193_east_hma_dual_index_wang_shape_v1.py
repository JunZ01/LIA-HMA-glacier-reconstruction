from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
LOCKED_START = 1921
LOCKED_END = 1939
HIST_YEARS = np.arange(1500, 1900)
MODERN_YEARS = np.arange(1940, 2001)
WINDOWS = [(start, start + 49) for start in range(1500, 1900, 50)]


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


def workbook_sheet_path(archive: zipfile.ZipFile, title: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relationships.findall(f"{{{NS_PKG_REL}}}Relationship")
    }
    for sheet in workbook.findall(f".//{{{NS_MAIN}}}sheet"):
        if sheet.attrib["name"] != title:
            continue
        target = targets[sheet.attrib[f"{{{NS_REL}}}id"]].replace("\\", "/")
        if target.startswith("/"):
            return target.lstrip("/")
        return target if target.startswith("xl/") else f"xl/{target}"
    raise KeyError(title)


def cell_numeric(cell: ET.Element | None) -> float:
    if cell is None:
        raise ValueError("missing-cell")
    node = cell.find(f"{{{NS_MAIN}}}v")
    if node is None or node.text is None:
        raise ValueError("missing-numeric-value")
    return float(node.text)


def parse_allowed_chronology(path: Path, sheet_title: str) -> tuple[pd.Series, dict[str, Any]]:
    values: dict[int, float] = {}
    locked_rows_skipped = 0
    outside_allowed_values_skipped = 0
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read(workbook_sheet_path(archive, sheet_title)))
        for row in root.findall(f".//{{{NS_MAIN}}}row"):
            cells = {cell.attrib.get("r", "")[0]: cell for cell in row.findall(f"{{{NS_MAIN}}}c")}
            if "A" not in cells:
                continue
            try:
                year = int(round(cell_numeric(cells["A"])))
            except ValueError:
                continue
            if LOCKED_START <= year <= LOCKED_END:
                locked_rows_skipped += 1
                continue
            if not (1500 <= year <= 1899 or 1940 <= year <= 2000):
                outside_allowed_values_skipped += int("B" in cells)
                continue
            values[year] = cell_numeric(cells.get("B"))
    series = pd.Series(values, dtype=float).sort_index()
    if any(LOCKED_START <= int(year) <= LOCKED_END for year in series.index):
        raise RuntimeError("locked-year-entered-series")
    return series, {
        "sheet_title": sheet_title,
        "historical_value_count": int(series.reindex(HIST_YEARS).notna().sum()),
        "modern_value_count": int(series.reindex(MODERN_YEARS).notna().sum()),
        "locked_rows_skipped_before_column_b_conversion": locked_rows_skipped,
        "outside_allowed_column_b_values_skipped": outside_allowed_values_skipped,
        "locked_column_b_values_decoded": False,
    }


def correlation(left: pd.Series, right: pd.Series) -> float:
    joined = pd.concat([left, right], axis=1).dropna()
    if len(joined) < 3 or joined.iloc[:, 0].std() <= 1e-12 or joined.iloc[:, 1].std() <= 1e-12:
        return float("nan")
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def modern_zscore(series: pd.Series) -> pd.Series:
    modern = series.reindex(MODERN_YEARS)
    if modern.isna().any():
        raise RuntimeError("modern-standardization-gap")
    return (series - float(modern.mean())) / float(modern.std(ddof=1))


def centered_rolling(series: pd.Series, width: int) -> pd.Series:
    return series.rolling(width, center=True, min_periods=width).mean()


def window_means(series: pd.Series) -> pd.Series:
    return pd.Series(
        {f"{start}-{end}": float(series.loc[start:end].mean()) for start, end in WINDOWS},
        dtype=float,
    )


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_193_east_hma_dual_index_wang_shape_v1"
    output.mkdir(parents=True, exist_ok=False)
    hengduan_xlsx = loop / (
        "round_178_mendeley_east_hma_temperature_candidates_v1/raw_unopened/889n6ccx4z_v1/Upload information.xlsx"
    )
    haizi_xlsx = loop / (
        "round_181_haizi_903y_temperature_supplement_v2/raw_unopened/41598_2024_79096_MOESM1_ESM.xlsx"
    )
    wang_csv = loop / "round_149_noaa_temperature_candidate_identity_v1/tibet2015_allowed.csv"
    r180_modern = loop / "round_180_hengduan_chronology_modern_oof_v2/modern_candidate_series.csv"
    r183_modern = loop / "round_183_haizi_903y_chronology_modern_oof_v2/modern_candidate_series.csv"
    round192 = loop / "round_192_east_hma_dual_chronology_thermal_index_oof_v1/summary.json"
    sources = [hengduan_xlsx, haizi_xlsx, wang_csv, r180_modern, r183_modern, round192]
    hashes_before = {str(path): sha256(path) for path in sources}

    hengduan, hengduan_parse = parse_allowed_chronology(hengduan_xlsx, "Tree-ring width chronology")
    haizi, haizi_parse = parse_allowed_chronology(haizi_xlsx, "Sheet1")
    h_mod = pd.read_csv(r180_modern).set_index("year").chronology.reindex(MODERN_YEARS)
    z_mod = pd.read_csv(r183_modern).set_index("year").chronology.reindex(MODERN_YEARS)
    h_error = float(np.max(np.abs(hengduan.reindex(MODERN_YEARS).to_numpy() - h_mod.to_numpy())))
    z_error = float(np.max(np.abs(haizi.reindex(MODERN_YEARS).to_numpy() - z_mod.to_numpy())))
    if h_error > 1e-12 or z_error > 1e-12:
        raise RuntimeError(f"modern-parser-closure:{h_error}:{z_error}")

    wang_frame = pd.read_csv(wang_csv)
    if wang_frame.year.between(LOCKED_START, LOCKED_END).any():
        raise RuntimeError("locked-year-in-wang-allowed-csv")
    wang = pd.Series(
        wang_frame.value_1.replace(-99.99, np.nan).to_numpy(float),
        index=wang_frame.year.to_numpy(int),
        dtype=float,
    ).sort_index()
    hengduan_z = modern_zscore(hengduan)
    haizi_z = modern_zscore(haizi)
    dual = 0.5 * hengduan_z + 0.5 * haizi_z
    wang_z = modern_zscore(wang)
    historical = pd.DataFrame(
        {
            "hengduan_modern_z": hengduan_z.reindex(HIST_YEARS),
            "haizi_modern_z": haizi_z.reindex(HIST_YEARS),
            "dual_index": dual.reindex(HIST_YEARS),
            "wang2015_modern_z": wang_z.reindex(HIST_YEARS),
        },
        index=HIST_YEARS,
    )
    if historical.isna().any().any():
        raise RuntimeError(f"historical-gap:{historical.isna().sum().to_dict()}")
    historical.index.name = "year"
    historical.to_csv(output / "historical_allowed_curves.csv")

    metrics: list[dict[str, Any]] = []
    windows = pd.DataFrame(index=[f"{start}-{end}" for start, end in WINDOWS])
    windows.index.name = "window"
    for name in ["hengduan_modern_z", "haizi_modern_z", "dual_index"]:
        candidate = historical[name]
        reference = historical["wang2015_modern_z"]
        candidate_windows = window_means(candidate)
        reference_windows = window_means(reference)
        windows[name] = candidate_windows
        windows["wang2015_modern_z"] = reference_windows
        candidate_changes = np.diff(candidate_windows.to_numpy(float))
        reference_changes = np.diff(reference_windows.to_numpy(float))
        metrics.append(
            {
                "candidate": name,
                "annual_correlation": correlation(candidate, reference),
                "centered_31yr_correlation": correlation(centered_rolling(candidate, 31), centered_rolling(reference, 31)),
                "fifty_year_correlation": correlation(candidate_windows, reference_windows),
                "adjacent_50yr_sign_agreement_count": int(np.sum(np.sign(candidate_changes) == np.sign(reference_changes))),
                "adjacent_50yr_change_count": len(candidate_changes),
            }
        )
    windows.to_csv(output / "fifty_year_means.csv")
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output / "shape_metrics.csv", index=False)
    dual_metrics = next(row for row in metrics if row["candidate"] == "dual_index")
    gate = {
        "fifty_year_correlation_ge_0_30": bool(dual_metrics["fifty_year_correlation"] >= 0.30),
        "centered_31yr_correlation_ge_0_20": bool(dual_metrics["centered_31yr_correlation"] >= 0.20),
        "adjacent_50yr_sign_agreement_ge_5_of_7": bool(dual_metrics["adjacent_50yr_sign_agreement_count"] >= 5),
    }
    passed = all(gate.values())

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(HIST_YEARS, centered_rolling(historical.dual_index, 31), color="#1261A0", lw=2.1, label="Dual chronology index")
    axes[0].plot(HIST_YEARS, centered_rolling(historical.wang2015_modern_z, 31), color="#C33C54", lw=1.8, label="Wang 2015 ETP JJA")
    axes[0].plot(HIST_YEARS, centered_rolling(historical.hengduan_modern_z, 31), color="#5B8C5A", lw=1.0, alpha=0.55, label="Hengduan")
    axes[0].plot(HIST_YEARS, centered_rolling(historical.haizi_modern_z, 31), color="#7A5195", lw=1.0, alpha=0.55, label="Haizi")
    axes[0].axhline(0, color="#888888", lw=0.7)
    axes[0].set_ylabel("Modern-period z score")
    axes[0].set_title("1500-1899 centered 31-year means")
    axes[0].legend(ncol=4, frameon=False, fontsize=9)
    x = np.arange(len(windows))
    axes[1].plot(x, windows.dual_index, marker="o", color="#1261A0", lw=2.1, label="Dual chronology index")
    axes[1].plot(x, windows.wang2015_modern_z, marker="o", color="#C33C54", lw=1.8, label="Wang 2015 ETP JJA")
    axes[1].set_xticks(x, windows.index, rotation=30, ha="right")
    axes[1].set_ylabel("50-year mean z score")
    axes[1].set_title("Non-overlapping 50-year means")
    axes[1].axhline(0, color="#888888", lw=0.7)
    axes[1].legend(frameon=False)
    figure.savefig(output / "historical_shape.png", dpi=180)
    plt.close(figure)

    hashes_after = {str(path): sha256(path) for path in sources}
    if hashes_before != hashes_after:
        raise RuntimeError("source-hash-changed")
    decision = (
        "retain_dual_index_as_shape_consistent_mechanism_candidate_but_require_independent_absolute_scale"
        if passed
        else "close_dual_index_on_historical_shape_transport"
    )
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "historical_period": "1500-1899",
        "benchmark_role": "partially_source_overlapping_regional_trajectory_benchmark_only",
        "hengduan_parse": hengduan_parse,
        "haizi_parse": haizi_parse,
        "modern_parser_closure_max_abs_error": {"hengduan": h_error, "haizi": z_error},
        "shape_metrics": metrics,
        "gate": gate,
        "passed": passed,
        "absolute_temperature_amplitude_validated": False,
        "locked_1921_1939_numeric_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": True,
        "decision": decision,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"sources": hashes_after}, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 193 Dual Thermal Index Historical Shape\n\n"
        f"- Dual index annual correlation with Wang 2015: `{dual_metrics['annual_correlation']:.3f}`.\n"
        f"- Centered 31-year correlation: `{dual_metrics['centered_31yr_correlation']:.3f}`.\n"
        f"- Eight-window 50-year correlation: `{dual_metrics['fifty_year_correlation']:.3f}`.\n"
        f"- Adjacent 50-year sign agreement: `{dual_metrics['adjacent_50yr_sign_agreement_count']}/7`.\n"
        f"- Passed frozen historical-shape gate: `{str(passed).lower()}`.\n"
        "- Absolute temperature amplitude validated: `false`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

