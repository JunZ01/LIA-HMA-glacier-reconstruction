"""Compare an external east-TP summer reconstruction with frozen HMA curves."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOCKED_START = 1921
LOCKED_END = 1939
PRIMARY_WINDOWS = tuple((start, start + 49) for start in range(1500, 1900, 50))


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def fingerprint(path: Path) -> dict[str, object]:
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


def parse_external(path: Path) -> tuple[pd.Series, dict[str, object]]:
    values: dict[int, float] = {}
    locked_rows = 0
    outside_rows = 0
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        with archive.open("xl/worksheets/sheet1.xml") as handle:
            for _, row in ET.iterparse(handle, events=("end",)):
                if row.tag != f"{ns}row":
                    continue
                cells: dict[str, str] = {}
                for cell in row.findall(f"{ns}c"):
                    ref = cell.attrib.get("r", "")
                    column = "".join(char for char in ref if char.isalpha())
                    node = cell.find(f"{ns}v")
                    if node is not None and node.text is not None:
                        cells[column] = node.text
                if "A" not in cells:
                    row.clear()
                    continue
                year = int(float(cells["A"]))
                if LOCKED_START <= year <= LOCKED_END:
                    locked_rows += 1
                    row.clear()
                    continue
                if not (1210 <= year <= 1920 or 1940 <= year <= 2000):
                    outside_rows += int("B" in cells)
                    row.clear()
                    continue
                if "B" not in cells:
                    raise RuntimeError(f"missing-external-value:{year}")
                values[year] = float(cells["B"])
                row.clear()
    series = pd.Series(values, dtype=float).sort_index()
    if any(LOCKED_START <= int(year) <= LOCKED_END for year in series.index):
        raise RuntimeError("locked-year-entered-external-series")
    required = list(range(1500, 1900)) + list(range(1961, 1991))
    missing = [year for year in required if year not in series.index]
    if missing:
        raise RuntimeError(f"external-required-years-missing:{missing[:10]}")
    return series, {
        "decoded_value_count": int(series.size),
        "locked_rows_skipped_before_value_conversion": locked_rows,
        "outside_contract_value_cells_skipped": outside_rows,
        "locked_values_decoded": False,
    }


def mean_for(series: pd.Series, start: int, end: int) -> float:
    selected = series.reindex(range(start, end + 1))
    if selected.isna().any():
        raise RuntimeError(f"incomplete-window:{start}-{end}")
    return float(selected.mean())


def corr(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    drivers = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers"
    output = loop / "round_184_east_tp_external_temperature_bias_audit_v1"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round184_east_tp_external_temperature_bias_audit_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "external": loop / "round_178_mendeley_east_hma_temperature_candidates_v1/raw_unopened/s7t4zc34sr_v2/Temperature reconstruction.xlsx",
        "round23": loop / "round_23_corrected_reconstruction_reference_envelope_v1/corrected_reconstruction_reference_curves.csv",
        "round23_summary": loop / "round_23_corrected_reconstruction_reference_envelope_v1/summary.json",
        "round178_summary": loop / "round_178_mendeley_east_hma_temperature_candidates_v1/summary.json",
        "round179_summary": loop / "round_179_mendeley_temperature_structure_identity_v1/summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: fingerprint(path) for name, path in paths.items()}

    external, parse = parse_external(paths["external"])
    external = external - mean_for(external, 1961, 1990)
    curves = pd.read_csv(paths["round23"])
    locked = curves.year.between(LOCKED_START, LOCKED_END)
    value_columns = [column for column in curves.columns if column != "year"]
    if not curves.loc[locked, value_columns].isna().all().all():
        raise RuntimeError("round23-locked-values-present")
    curves = curves.set_index("year")
    project = curves.corrected_six_member_hma_degC
    reference = curves.reference_family_balanced_hma_degC
    family_columns = [
        "reference_local_enkf_hma_degC",
        "reference_lmr_hma_degC",
        "reference_neukom_hma_degC",
        "reference_phyda_hma_degC",
    ]

    windows = [(*window, "primary_50year") for window in PRIMARY_WINDOWS]
    windows += [(1900, 1920, "diagnostic_allowed"), (1940, 1989, "diagnostic_allowed")]
    rows: list[dict[str, object]] = []
    for start, end, scope in windows:
        family_means = np.asarray([mean_for(curves[column], start, end) for column in family_columns])
        ext = mean_for(external, start, end)
        proj = mean_for(project, start, end)
        ref = mean_for(reference, start, end)
        rows.append({
            "start_year": start,
            "end_year": end,
            "scope": scope,
            "external_east_tp_jun_sep_degC": ext,
            "project_six_member_hma_annual_degC": proj,
            "reference_four_family_hma_annual_degC": ref,
            "reference_family_min_degC": float(family_means.min()),
            "reference_family_max_degC": float(family_means.max()),
            "project_minus_external_degC": proj - ext,
            "reference_minus_external_degC": ref - ext,
            "external_colder_than_project": bool(ext < proj),
            "external_inside_reference_family_range": bool(family_means.min() <= ext <= family_means.max()),
        })
    table = pd.DataFrame(rows)
    table.to_csv(output / "external_temperature_window_comparison.csv", index=False)
    primary = table[table.scope == "primary_50year"].copy()
    ext_values = primary.external_east_tp_jun_sep_degC.to_numpy(float)
    project_values = primary.project_six_member_hma_annual_degC.to_numpy(float)
    reference_values = primary.reference_four_family_hma_annual_degC.to_numpy(float)
    colder_count = int(primary.external_colder_than_project.sum())
    external_to_reference = float(np.median(np.abs(ext_values - reference_values)))
    external_to_project = float(np.median(np.abs(ext_values - project_values)))
    gates = {
        "external_colder_than_project_in_at_least_6_of_8_windows": colder_count >= 6,
        "external_median_distance_closer_to_reference_than_project": external_to_reference < external_to_project,
    }
    supports = bool(all(gates.values()))
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "external_target": "eastern Tibetan Plateau June-September mean temperature",
        "project_target": "HMA annual mean temperature",
        "spatial_seasonal_target_mismatch_explicit": True,
        "external_parse": parse,
        "primary_window_count": int(len(primary)),
        "external_colder_than_project_window_count": colder_count,
        "external_colder_than_project_window_fraction": float(colder_count / len(primary)),
        "median_project_minus_external_degC": float(np.median(project_values - ext_values)),
        "median_external_absolute_distance_to_reference_degC": external_to_reference,
        "median_external_absolute_distance_to_project_degC": external_to_project,
        "external_project_window_correlation": corr(ext_values, project_values),
        "external_reference_window_correlation": corr(ext_values, reference_values),
        "external_inside_reference_family_range_count": int(primary.external_inside_reference_family_range.sum()),
        "gates": gates,
        "external_supports_existing_project_warm_bias_diagnosis": supports,
        "decision": "retain_project_warm_bias_as_independently_supported" if supports else "external_record_does_not_confirm_project_warm_bias",
        "external_used_for_fitting_or_correction": False,
        "locked_1921_1939_numeric_values_decoded": False,
        "jionglacuo_opened": False,
        "enkf_run": False,
        "oggm_run": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    centers = (primary.start_year + primary.end_year) / 2
    fig, axis = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    axis.fill_between(centers, primary.reference_family_min_degC, primary.reference_family_max_degC, color="#B8D8D8", alpha=0.55, label="Four-family HMA range")
    axis.plot(centers, reference_values, color="#176B6B", marker="o", linewidth=2, label="Four-family HMA central")
    axis.plot(centers, project_values, color="#B23A48", marker="s", linewidth=2, label="Project six-member HMA")
    axis.plot(centers, ext_values, color="#2541B2", marker="^", linewidth=2, label="East TP Jun-Sep external")
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set(xlabel="50-year window center", ylabel="Temperature anomaly (degC; 1961-1990)", title="1500-1899 external temperature bias audit")
    axis.legend(frameon=False, ncol=2)
    fig.savefig(output / "external_temperature_bias_1500_1899.png", dpi=180)
    plt.close(fig)

    (output / "FINAL_RESULT.md").write_text(
        "# Eastern Tibetan Plateau External Temperature Bias Audit\n\n"
        f"- External record colder than project: `{colder_count}/8` primary windows.\n"
        f"- Median project minus external: `{summary['median_project_minus_external_degC']:+.3f} degC`.\n"
        f"- Median external distance to four-family central/project: `{external_to_reference:.3f}/{external_to_project:.3f} degC`.\n"
        f"- Window correlations with four-family central/project: `{summary['external_reference_window_correlation']:.3f}/{summary['external_project_window_correlation']:.3f}`.\n\n"
        "The external series is eastern Tibetan Plateau June--September temperature, whereas the project and reference curves are HMA annual means. It is external low-frequency evidence, not field validation or a correction target.\n\n"
        f"Decision: `{summary['decision']}`.\n",
        encoding="utf-8",
    )
    after = {name: fingerprint(path) for name, path in paths.items()}
    manifest = {"sources_before": before, "sources_after": after, "source_states_unchanged": before == after}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("source-fingerprint-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
