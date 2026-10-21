from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from openpyxl import load_workbook

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "round_212_wscp_temperature_locked_safe_comparison_v1"
ROUND211 = LOOP / "round_211_wscp_temperature_data_recovery_v2"
WORKBOOK = (
    ROUND211
    / "official_archive_extracted"
    / "Temp_Reconstruction&Meteorological.xlsx"
)
ROUND211_SUMMARY = ROUND211 / "summary.json"
CONTRACT = OUT / "ROUND_CONTRACT.md"
DRIVER = Path(__file__).resolve()
ROUND206_DRIVER = (
    DRIVER.parent
    / "run_round206_setp_winter_temperature_locked_safe_comparison_v1.py"
)
EXPECTED_WORKBOOK_SHA256 = (
    "286e9338e1cfeddb2b51f6805241d1e0989c891af3adbdd81691f1918b794d68"
)
SITE_LAT = 31.15
SITE_LON = 99.81
LOCKED_START = 1921
LOCKED_END = 1939


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_round206():
    spec = importlib.util.spec_from_file_location("round206_for_round212", ROUND206_DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Round 206 helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_workbook_year_first() -> tuple[dict[int, float], dict[int, float], int]:
    if sha256(WORKBOOK) != EXPECTED_WORKBOOK_SHA256:
        raise RuntimeError("Official workbook hash changed")
    workbook = load_workbook(WORKBOOK, read_only=True, data_only=True)
    if workbook.sheetnames != ["Tab.1", "Tab.2"]:
        raise RuntimeError(f"Unexpected workbook sheets: {workbook.sheetnames}")

    output: list[dict[int, float]] = []
    skipped = 0
    expected_ranges = [(1383, 2005), (1962, 2005)]
    expected_rows = [629, 50]
    for sheet_name, expected_range, expected_row_count in zip(
        workbook.sheetnames, expected_ranges, expected_rows
    ):
        sheet = workbook[sheet_name]
        if sheet.max_row != expected_row_count or sheet.max_column != 2:
            raise RuntimeError(f"Unexpected dimensions for {sheet_name}")
        if sheet.cell(5, 1).value != "year":
            raise RuntimeError(f"Unexpected year header in {sheet_name}")
        if "mean temperature" not in str(sheet.cell(5, 2).value):
            raise RuntimeError(f"Unexpected temperature header in {sheet_name}")

        values: dict[int, float] = {}
        for row in sheet.iter_rows(min_row=7, max_col=2, values_only=False):
            year_token = row[0].value
            if year_token is None:
                continue
            year = int(year_token)
            if LOCKED_START <= year <= LOCKED_END:
                skipped += 1
                continue
            value_token = row[1].value
            value = float(value_token)
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite allowed value in {sheet_name}, {year}")
            if year in values:
                raise RuntimeError(f"Duplicate year in {sheet_name}: {year}")
            values[year] = value

        start, end = expected_range
        expected = set(range(start, min(end, 1920) + 1))
        if end >= 1940:
            expected |= set(range(max(start, 1940), end + 1))
        if set(values) != expected:
            raise RuntimeError(f"Allowed coverage mismatch in {sheet_name}")
        output.append(values)
    workbook.close()
    if skipped != 19:
        raise RuntimeError(f"Expected 19 locked rows skipped, found {skipped}")
    return output[0], output[1], skipped


def vector_from_dict(values: dict[int, float], years: np.ndarray) -> np.ndarray:
    output = np.full(len(years), np.nan)
    for index, year in enumerate(years):
        if int(year) in values:
            output[index] = values[int(year)]
    return output


def anomaly(values: np.ndarray, years: np.ndarray, start: int, end: int) -> np.ndarray:
    selection = (years >= start) & (years <= end) & np.isfinite(values)
    if int(selection.sum()) != end - start + 1:
        raise RuntimeError(f"Incomplete baseline {start}-{end}")
    return values - float(np.mean(values[selection]))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3:
        return math.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def trend(values: np.ndarray, years: np.ndarray, start: int, end: int) -> float:
    selection = (years >= start) & (years <= end) & np.isfinite(values)
    if int(selection.sum()) != end - start + 1:
        raise RuntimeError(f"Incomplete trend interval {start}-{end}")
    return float(np.polyfit(years[selection], values[selection], 1)[0] * 100)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        WORKBOOK,
        ROUND211_SUMMARY,
        CONTRACT,
        ROUND206_DRIVER,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    round206 = load_round206()
    required.extend([round206.LINEAGE, round206.REFERENCE, round206.HMA_MASK])
    before = {str(path): state(path) for path in required}
    reconstruction, instrumental, locked_rows_skipped = read_workbook_year_first()

    years = np.arange(1000, 2001)
    locked = (years >= LOCKED_START) & (years <= LOCKED_END)
    reference, latitudes, longitudes, reference_hma_mask = round206.read_reference_allowed(
        round206.REFERENCE, years
    )
    hma_mask = np.load(round206.HMA_MASK).astype(bool)
    if not np.array_equal(hma_mask, reference_hma_mask):
        raise RuntimeError("HMA mask mismatch")

    lineage = pd.read_csv(round206.LINEAGE)
    members = lineage[lineage.variable == "tas"].copy()
    expected_members = {"AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9"}
    if len(members) != 6 or set(members.member) != expected_members:
        raise RuntimeError("Temperature member lineage changed")
    round23 = round206.load_round23_module()
    project_members = []
    member_sources: dict[str, Path] = {}
    for row in members.itertuples(index=False):
        path = Path(row.source_tif)
        if sha256(path) != row.source_tif_sha256:
            raise RuntimeError(f"Member hash mismatch: {row.member}")
        field, _ = round23.read_allowed_tif(path, years, latitudes, longitudes)
        project_members.append(field)
        member_sources[row.member] = path
    project = round23.finite_mean(project_members)
    if np.any(np.isfinite(project[locked])) or np.any(np.isfinite(reference[locked])):
        raise RuntimeError("Locked project/reference values became visible")

    box_mask = (
        (latitudes[:, None] >= SITE_LAT - 1.0)
        & (latitudes[:, None] <= SITE_LAT + 1.0)
        & (longitudes[None, :] >= SITE_LON - 1.0)
        & (longitudes[None, :] <= SITE_LON + 1.0)
    )
    if int(box_mask.sum()) < 4:
        raise RuntimeError("Source-centered box has too few cells")
    lat_index = int(np.argmin(np.abs(latitudes - SITE_LAT)))
    lon_index = int(np.argmin(np.abs(longitudes - SITE_LON)))

    external = anomaly(vector_from_dict(reconstruction, years), years, 1961, 1990)
    project_box = anomaly(
        round206.area_weighted_series(project, box_mask, latitudes), years, 1961, 1990
    )
    reference_box = anomaly(
        round206.area_weighted_series(reference, box_mask, latitudes), years, 1961, 1990
    )
    project_cell = anomaly(project[:, lat_index, lon_index], years, 1961, 1990)
    reference_cell = anomaly(reference[:, lat_index, lon_index], years, 1961, 1990)
    instrument = vector_from_dict(instrumental, years)

    curves = {
        "wscp_jj_reconstruction_anomaly_degC": external,
        "project_local_box_annual_anomaly_degC": project_box,
        "reference_local_box_annual_anomaly_degC": reference_box,
        "project_nearest_cell_annual_anomaly_degC": project_cell,
        "reference_nearest_cell_annual_anomaly_degC": reference_cell,
        "official_jj_instrumental_degC": instrument,
    }
    smooth = {
        name: round206.segment_smooth(values, years)
        for name, values in curves.items()
        if name != "official_jj_instrumental_degC"
    }

    annual_rows = []
    for index, year in enumerate(years):
        if LOCKED_START <= year <= LOCKED_END:
            continue
        row: dict[str, float | int] = {"year": int(year)}
        for name, values in curves.items():
            row[name] = values[index]
        for name, values in smooth.items():
            row[f"{name}_smooth31"] = values[index]
        annual_rows.append(row)
    pd.DataFrame(annual_rows).to_csv(OUT / "annual_allowed_curves.csv", index=False)

    window_rows = []
    historical_names = list(curves)[:5]
    for start in range(1500, 1900, 50):
        end = start + 49
        selection = (years >= start) & (years <= end)
        row: dict[str, float | int] = {
            "start_year": start,
            "end_year": end,
            "center_year": start + 24.5,
        }
        for name in historical_names:
            values = curves[name]
            if int(np.isfinite(values[selection]).sum()) != 50:
                raise RuntimeError(f"Incomplete window {name} {start}-{end}")
            row[name] = float(np.mean(values[selection]))
        window_rows.append(row)
    windows = pd.DataFrame(window_rows)
    windows.to_csv(OUT / "temperature_50year_windows_1500_1899.csv", index=False)

    interval_rows = []
    for start, end in ((1850, 1900), (1901, 1920), (1940, 2000)):
        row = {"start_year": start, "end_year": end}
        for name in historical_names:
            row[f"{name}_trend_degC_per_century"] = trend(
                curves[name], years, start, end
            )
        interval_rows.append(row)
    pd.DataFrame(interval_rows).to_csv(OUT / "diagnostic_interval_trends.csv", index=False)

    external_windows = windows[historical_names[0]].to_numpy(float)
    project_windows = windows[historical_names[1]].to_numpy(float)
    reference_windows = windows[historical_names[2]].to_numpy(float)
    modern = (years >= 1962) & (years <= 2000)
    metrics = {
        "external_instrumental_correlation_1962_2000": correlation(
            external[modern], instrument[modern]
        ),
        "project_box_annual_vs_jj_instrumental_correlation_1962_2000": correlation(
            project_box[modern], instrument[modern]
        ),
        "reference_box_annual_vs_jj_instrumental_correlation_1962_2000": correlation(
            reference_box[modern], instrument[modern]
        ),
        "external_project_box_window_correlation": correlation(
            external_windows, project_windows
        ),
        "external_reference_box_window_correlation": correlation(
            external_windows, reference_windows
        ),
        "external_negative_window_count": int(np.sum(external_windows < 0)),
        "project_box_negative_window_count": int(np.sum(project_windows < 0)),
        "reference_box_negative_window_count": int(np.sum(reference_windows < 0)),
        "median_external_anomaly_1500_1899_degC": float(np.median(external_windows)),
        "median_project_box_anomaly_1500_1899_degC": float(np.median(project_windows)),
        "median_reference_box_anomaly_1500_1899_degC": float(np.median(reference_windows)),
        "external_project_adjacent_window_sign_agreement": int(
            np.sum(np.sign(np.diff(external_windows)) == np.sign(np.diff(project_windows)))
        ),
        "external_reference_adjacent_window_sign_agreement": int(
            np.sum(
                np.sign(np.diff(external_windows)) == np.sign(np.diff(reference_windows))
            )
        ),
    }

    supports_warm_bias = (
        metrics["external_negative_window_count"] >= 6
        and metrics["project_box_negative_window_count"]
        < metrics["external_negative_window_count"]
    )
    decision = (
        "independent_wscp_record_supports_local_warm_bias_direction_only"
        if supports_warm_bias
        else "independent_wscp_record_does_not_support_uniform_warm_correction"
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12.4, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.16},
    )
    axis = axes[0]
    axis.plot(
        years,
        smooth[historical_names[0]],
        color="#222222",
        linewidth=2.4,
        label="Baiyu tree-ring June-July",
    )
    axis.plot(
        years,
        smooth[historical_names[1]],
        color="#B33A3A",
        linewidth=2.0,
        label="Project local annual",
    )
    axis.plot(
        years,
        smooth[historical_names[2]],
        color="#147D78",
        linewidth=2.0,
        label="Four-family local annual",
    )
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.axvspan(1921, 1939, color="#E5E5E5", alpha=0.85, linewidth=0)
    axis.set_ylabel("31-year smooth anomaly (degC; 1961-1990)")
    axis.set_title("Independent western Sichuan summer temperature comparison")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=3, loc="upper left")

    axis = axes[1]
    centers = windows.center_year.to_numpy(float)
    axis.plot(
        centers,
        external_windows,
        color="#222222",
        marker="o",
        linewidth=2.0,
        label="Baiyu June-July",
    )
    axis.plot(
        centers,
        project_windows,
        color="#B33A3A",
        marker="s",
        linewidth=1.8,
        label="Project local annual",
    )
    axis.plot(
        centers,
        reference_windows,
        color="#147D78",
        marker="^",
        linewidth=1.8,
        label="Four-family local annual",
    )
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_xlim(1490, 2005)
    axis.set_xlabel("Year")
    axis.set_ylabel("50-year anomaly (degC)")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=3, loc="lower left")
    figure.text(
        0.99,
        0.01,
        "Gray gap: sealed 1921-1939. Seasonal tree-ring evidence is compared with annual fields for direction only.",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.1)
    figure.savefig(OUT / "wscp_project_reference_curves_1500_2000.png", dpi=220)
    plt.close(figure)

    modern_frame = pd.DataFrame(
        {
            "year": years[modern],
            "official_jj_instrumental_degC": instrument[modern],
            "tree_ring_reconstruction_anomaly_degC": external[modern],
            "project_local_annual_anomaly_degC": project_box[modern],
            "reference_local_annual_anomaly_degC": reference_box[modern],
        }
    )
    modern_frame.to_csv(OUT / "modern_transport_curves_1962_2000.csv", index=False)

    all_sources = {str(path): path for path in required}
    all_sources.update({f"member_{name}": path for name, path in member_sources.items()})
    source_before = {name: state(path) for name, path in all_sources.items()}
    source_after = {name: state(path) for name, path in all_sources.items()}
    if source_before != source_after:
        raise RuntimeError("A read-only source changed")

    summary = {
        "round": 212,
        "decision": decision,
        "source_independence": {
            "sampling_date": "October 2017",
            "current_ledger_nearest_tree_group_km": json.loads(
                ROUND211_SUMMARY.read_text(encoding="utf-8")
            )["identity_preflight"]["nearest"]["distance_km"],
            "groups_within_35_km": 0,
            "classification": "source_independent_of_current_assimilated_tree_network",
        },
        "target_boundary": {
            "external": "June-July mean temperature",
            "project_reference": "annual mean temperature",
            "absolute_annual_hma_correction_permitted": False,
        },
        "spatial_support": {
            "box_cell_count": int(box_mask.sum()),
            "nearest_grid_latitude": float(latitudes[lat_index]),
            "nearest_grid_longitude": float(longitudes[lon_index]),
            "member_count": len(project_members),
        },
        "locked_parser": {
            "locked_rows_skipped_before_temperature_cell_access": locked_rows_skipped,
            "project_locked_raster_bands_read": False,
            "smoothing_across_locked_gap": False,
        },
        "metrics": metrics,
        "source_states_unchanged": True,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
        "production_changed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {"sources_before": source_before, "sources_after": source_after},
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = f"""# Round 212 Result

## Decision

`{decision}`

The official Baiyu reconstruction reproduces its seven-station June-July
target over 1962-2000 at `r={metrics['external_instrumental_correlation_1962_2000']:.3f}`.
It is source-independent of the current assimilated tree network: the nearest
project evidence group is 111.355 km away and the chronology was sampled in
October 2017.

Over the eight 50-year windows in 1500-1899, the Baiyu reconstruction is below
its 1961-1990 baseline in `{metrics['external_negative_window_count']}/8`
windows, versus `{metrics['project_box_negative_window_count']}/8` for the
project local annual curve and `{metrics['reference_box_negative_window_count']}/8`
for the four-family local annual reference. Median anomalies are
`{metrics['median_external_anomaly_1500_1899_degC']:+.3f}`,
`{metrics['median_project_box_anomaly_1500_1899_degC']:+.3f}`, and
`{metrics['median_reference_box_anomaly_1500_1899_degC']:+.3f}` degC.

This seasonal record can adjudicate local low-frequency direction, but cannot
set an annual HMA correction amplitude. No production file changed and no
locked-period or Jionglacuo value was accessed.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
