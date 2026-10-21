from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
OUT = LOOP / "round_206_setp_winter_temperature_locked_safe_comparison_v1"
ROUND205 = LOOP / "round_205_setp_winter_temperature_data_recovery_v1"
SERIES = ROUND205 / "Changdu_winter_Tmin.csv"
IDENTITY_SUMMARY = ROUND205 / "summary.json"
LINEAGE = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v1"
    / "round_00_production_reproduction/member_lineage.csv"
)
REFERENCE = (
    LOOP
    / "round_21_temperature_reference_field_v1"
    / "temperature_reference_family_balanced_allowed_v1.nc"
)
HMA_MASK = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
)
CONTRACT = OUT / "ROUND_CONTRACT.md"
DRIVER = Path(__file__).resolve()
ROUND23_DRIVER = (
    DRIVER.parent
    / "run_climate_reconstruction_loop_v2_round23_corrected_reconstruction_reference_envelope_v1.py"
)

EXPECTED_SERIES_SHA256 = (
    "9957b1717ac11f0ecafac739963ae77c7e96a47c65c9068b372ef17cfe4924d6"
)
SET_BOX = {"lon_min": 94.5, "lon_max": 97.5, "lat_min": 29.5, "lat_max": 31.5}
LOCKED_START = 1921
LOCKED_END = 1939


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_round23_module():
    spec = importlib.util.spec_from_file_location("round23_locked_reader", ROUND23_DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Round 23 reader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_external_year_first(path: Path) -> tuple[dict[int, float], int]:
    if sha256(path) != EXPECTED_SERIES_SHA256:
        raise RuntimeError("Official CASEarth CSV hash changed")

    values: dict[int, float] = {}
    locked_rows_skipped = 0
    with path.open("rb") as stream:
        header = stream.readline().decode("utf-8-sig").rstrip("\r\n")
        if not header.startswith("year,Tmin.recon,"):
            raise RuntimeError(f"Unexpected CSV header: {header!r}")
        for line_number, raw_line in enumerate(stream, start=2):
            stripped = raw_line.rstrip(b"\r\n")
            if not stripped:
                continue
            separator = stripped.find(b",")
            if separator <= 0:
                raise RuntimeError(f"Missing year separator at line {line_number}")
            year = int(stripped[:separator].decode("ascii"))
            if LOCKED_START <= year <= LOCKED_END:
                locked_rows_skipped += 1
                continue

            # The remainder is decoded only after the locked-year branch exits.
            row = next(csv.reader([stripped.decode("utf-8")]))
            if int(row[0]) != year:
                raise RuntimeError(f"Year mismatch at line {line_number}")
            value = float(row[1])
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite external value in allowed year {year}")
            if year in values:
                raise RuntimeError(f"Duplicate external year {year}")
            values[year] = value

    if locked_rows_skipped != 19:
        raise RuntimeError(f"Expected 19 locked rows, skipped {locked_rows_skipped}")
    expected = set(range(1340, 1921)) | set(range(1940, 2008))
    if set(values) != expected:
        missing = sorted(expected - set(values))[:10]
        extra = sorted(set(values) - expected)[:10]
        raise RuntimeError(f"External year coverage mismatch: missing={missing}, extra={extra}")
    return values, locked_rows_skipped


def read_reference_allowed(
    path: Path, years: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output = None
    with netcdf_file(path, "r", mmap=False) as dataset:
        file_years = np.asarray(dataset.variables["year"].data, dtype=int).copy()
        latitudes = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
        longitudes = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
        file_mask = np.asarray(dataset.variables["hma_mask"].data, dtype=bool).copy()
        if not np.array_equal(file_years, years):
            raise RuntimeError("Reference year coordinate changed")
        variable = dataset.variables["tas_reference_family_balanced"]
        output = np.full((len(years), len(latitudes), len(longitudes)), np.nan)
        output[years <= 1920] = np.asarray(variable.data[:921], dtype=float).copy()
        output[years >= 1940] = np.asarray(variable.data[940:], dtype=float).copy()
    return output, latitudes, longitudes, file_mask


def area_weighted_series(
    field: np.ndarray, mask: np.ndarray, latitudes: np.ndarray
) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))[:, None] * mask
    valid = np.isfinite(field) & mask[None]
    numerator = np.nansum(field * weights[None], axis=(1, 2))
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return numerator / np.where(denominator == 0, np.nan, denominator)


def segment_smooth(values: np.ndarray, years: np.ndarray, width: int = 31) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=float)
    for start, end in ((1000, 1920), (1940, 2000)):
        selection = (years >= start) & (years <= end)
        segment = pd.Series(values[selection])
        output[selection] = segment.rolling(
            width, center=True, min_periods=width
        ).mean().to_numpy()
    return output


def baseline_anomaly(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    baseline = (years >= 1961) & (years <= 1990) & np.isfinite(values)
    if int(baseline.sum()) != 30:
        raise RuntimeError("1961-1990 baseline incomplete")
    return values - float(np.mean(values[baseline]))


def modern_z(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    baseline = (years >= 1940) & (years <= 2000) & np.isfinite(values)
    scale = float(np.std(values[baseline], ddof=1))
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid modern standard deviation")
    return values / scale


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3:
        return math.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def trend_per_century(values: np.ndarray, years: np.ndarray, start: int, end: int) -> float:
    selection = (years >= start) & (years <= end) & np.isfinite(values)
    if int(selection.sum()) != end - start + 1:
        raise RuntimeError(f"Incomplete trend interval {start}-{end}")
    return float(np.polyfit(years[selection], values[selection], 1)[0] * 100)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        SERIES,
        IDENTITY_SUMMARY,
        LINEAGE,
        REFERENCE,
        HMA_MASK,
        CONTRACT,
        ROUND23_DRIVER,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    external, locked_rows_skipped = read_external_year_first(SERIES)
    years = np.arange(1000, 2001)
    locked = (years >= LOCKED_START) & (years <= LOCKED_END)

    round23 = load_round23_module()
    lineage = pd.read_csv(LINEAGE)
    members = lineage[lineage.variable == "tas"].copy()
    expected_members = {"AO_6", "AO_9", "HGAO_6", "HGAO_9", "O_6", "O_9"}
    if len(members) != 6 or set(members.member) != expected_members:
        raise RuntimeError("Six-member temperature lineage changed")

    member_paths = {row.member: Path(row.source_tif) for row in members.itertuples(index=False)}
    all_sources = {
        "series": SERIES,
        "identity_summary": IDENTITY_SUMMARY,
        "lineage": LINEAGE,
        "reference": REFERENCE,
        "hma_mask": HMA_MASK,
        "contract": CONTRACT,
        "round23_driver": ROUND23_DRIVER,
        **{f"member_{name}": path for name, path in member_paths.items()},
    }
    before = {name: state(path) for name, path in all_sources.items()}
    for row in members.itertuples(index=False):
        if before[f"member_{row.member}"]["sha256"] != row.source_tif_sha256:
            raise RuntimeError(f"Member source hash mismatch: {row.member}")

    reference, latitudes, longitudes, reference_hma_mask = read_reference_allowed(
        REFERENCE, years
    )
    hma_mask = np.load(HMA_MASK).astype(bool)
    if not np.array_equal(hma_mask, reference_hma_mask):
        raise RuntimeError("HMA mask mismatch between Round 21 and spatial audit")

    project_members = []
    orientations = set()
    for name, path in sorted(member_paths.items()):
        field, orientation = round23.read_allowed_tif(
            path, years, latitudes, longitudes
        )
        project_members.append(field)
        orientations.add(orientation)
    if len(orientations) != 1:
        raise RuntimeError(f"Member orientation mismatch: {orientations}")
    project = round23.finite_mean(project_members)
    if np.any(np.isfinite(project[locked])) or np.any(np.isfinite(reference[locked])):
        raise RuntimeError("Locked project/reference values are present")

    box_mask = (
        (latitudes[:, None] >= SET_BOX["lat_min"])
        & (latitudes[:, None] <= SET_BOX["lat_max"])
        & (longitudes[None, :] >= SET_BOX["lon_min"])
        & (longitudes[None, :] <= SET_BOX["lon_max"])
    )
    if int(box_mask.sum()) < 4:
        raise RuntimeError("SETP comparison box has too few cells")

    project_local = area_weighted_series(project, box_mask, latitudes)
    reference_local = area_weighted_series(reference, box_mask, latitudes)
    project_hma = area_weighted_series(project, hma_mask, latitudes)
    reference_hma = area_weighted_series(reference, hma_mask, latitudes)

    external_raw = np.full(len(years), np.nan)
    for index, year in enumerate(years):
        if year in external:
            external_raw[index] = external[year]
    external_anomaly = baseline_anomaly(external_raw, years)

    curves = {
        "external_winter_tmin_anomaly_degC": external_anomaly,
        "project_setp_annual_anomaly_degC": project_local,
        "reference_setp_annual_anomaly_degC": reference_local,
        "project_hma_annual_anomaly_degC": project_hma,
        "reference_hma_annual_anomaly_degC": reference_hma,
    }
    smooth = {name: segment_smooth(values, years) for name, values in curves.items()}
    z_smooth = {name: modern_z(values, years) for name, values in smooth.items()}

    allowed_rows = []
    for index, year in enumerate(years):
        if LOCKED_START <= year <= LOCKED_END:
            continue
        row = {"year": int(year)}
        for name, values in curves.items():
            row[name] = values[index]
            row[f"{name}_smooth31"] = smooth[name][index]
            row[f"{name}_smooth31_z1940_2000"] = z_smooth[name][index]
        allowed_rows.append(row)
    pd.DataFrame(allowed_rows).to_csv(OUT / "annual_allowed_curves.csv", index=False)

    window_rows = []
    for start in range(1500, 1900, 50):
        end = start + 49
        selection = (years >= start) & (years <= end)
        row = {"start_year": start, "end_year": end, "center_year": start + 24.5}
        for name, values in curves.items():
            if int(np.isfinite(values[selection]).sum()) != 50:
                raise RuntimeError(f"Incomplete 50-year window: {name} {start}-{end}")
            row[name] = float(np.mean(values[selection]))
        window_rows.append(row)
    windows = pd.DataFrame(window_rows)
    windows.to_csv(OUT / "temperature_50year_windows_1500_1899.csv", index=False)

    ext_windows = windows.external_winter_tmin_anomaly_degC.to_numpy(float)
    project_windows = windows.project_setp_annual_anomaly_degC.to_numpy(float)
    reference_windows = windows.reference_setp_annual_anomaly_degC.to_numpy(float)
    ext_changes = np.diff(ext_windows)
    project_changes = np.diff(project_windows)
    reference_changes = np.diff(reference_windows)

    interval_rows = []
    for start, end in ((1850, 1900), (1901, 1920), (1940, 2000)):
        interval_rows.append(
            {
                "start_year": start,
                "end_year": end,
                "external_winter_tmin_trend_degC_per_century": trend_per_century(
                    external_anomaly, years, start, end
                ),
                "project_setp_annual_trend_degC_per_century": trend_per_century(
                    project_local, years, start, end
                ),
                "reference_setp_annual_trend_degC_per_century": trend_per_century(
                    reference_local, years, start, end
                ),
            }
        )
    pd.DataFrame(interval_rows).to_csv(OUT / "diagnostic_interval_trends.csv", index=False)

    paleo = (years >= 1500) & (years <= 1899)
    metrics = {
        "window_correlation_external_project_setp": correlation(
            ext_windows, project_windows
        ),
        "window_correlation_external_reference_setp": correlation(
            ext_windows, reference_windows
        ),
        "window_adjacent_change_sign_agreement_external_project_setp": int(
            np.sum(np.sign(ext_changes) == np.sign(project_changes))
        ),
        "window_adjacent_change_sign_agreement_external_reference_setp": int(
            np.sum(np.sign(ext_changes) == np.sign(reference_changes))
        ),
        "smooth31_correlation_external_project_setp_1500_1899": correlation(
            smooth["external_winter_tmin_anomaly_degC"][paleo],
            smooth["project_setp_annual_anomaly_degC"][paleo],
        ),
        "smooth31_correlation_external_reference_setp_1500_1899": correlation(
            smooth["external_winter_tmin_anomaly_degC"][paleo],
            smooth["reference_setp_annual_anomaly_degC"][paleo],
        ),
        "external_negative_window_count": int(np.sum(ext_windows < 0)),
        "project_setp_negative_window_count": int(np.sum(project_windows < 0)),
        "reference_setp_negative_window_count": int(np.sum(reference_windows < 0)),
        "median_external_winter_anomaly_1500_1899_degC": float(np.median(ext_windows)),
        "median_project_setp_annual_anomaly_1500_1899_degC": float(
            np.median(project_windows)
        ),
        "median_reference_setp_annual_anomaly_1500_1899_degC": float(
            np.median(reference_windows)
        ),
    }

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12.5, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.16},
    )
    axis = axes[0]
    axis.plot(
        years,
        z_smooth["external_winter_tmin_anomaly_degC"],
        color="#222222",
        linewidth=2.4,
        label="Huang 2019 SETP winter Tmin",
    )
    axis.plot(
        years,
        z_smooth["project_setp_annual_anomaly_degC"],
        color="#B33A3A",
        linewidth=2.0,
        label="Project six-member SETP annual",
    )
    axis.plot(
        years,
        z_smooth["reference_setp_annual_anomaly_degC"],
        color="#147D78",
        linewidth=2.0,
        label="Four-family SETP annual",
    )
    axis.plot(
        years,
        z_smooth["project_hma_annual_anomaly_degC"],
        color="#C89021",
        linewidth=1.2,
        alpha=0.8,
        label="Project HMA annual context",
    )
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.axvspan(1921, 1939, color="#E5E5E5", alpha=0.8, linewidth=0)
    axis.set_ylabel("31-year smooth (SD of 1940-2000)")
    axis.set_title("Southeastern Tibetan winter temperature versus frozen project curves")
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    axis = axes[1]
    centers = windows.center_year.to_numpy(float)
    axis.plot(centers, ext_windows, color="#222222", marker="o", linewidth=2.0, label="Winter Tmin")
    axis.plot(centers, project_windows, color="#B33A3A", marker="s", linewidth=1.8, label="Project SETP annual")
    axis.plot(centers, reference_windows, color="#147D78", marker="^", linewidth=1.8, label="Four-family SETP annual")
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_xlim(1490, 2005)
    axis.set_xlabel("Year")
    axis.set_ylabel("50-year anomaly (degC)\nrelative to 1961-1990")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=3, loc="lower left")
    figure.text(
        0.99,
        0.01,
        "Gray gap: sealed 1921-1939; no smoothing or interpolation crosses it. Seasonal targets differ.",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.1)
    figure.savefig(OUT / "setp_winter_project_curves_1500_2000.png", dpi=220)
    plt.close(figure)

    after = {name: state(path) for name, path in all_sources.items()}
    if before != after:
        raise RuntimeError("A read-only source changed during Round 206")

    supports_direction = (
        metrics["external_negative_window_count"] >= 6
        and metrics["project_setp_negative_window_count"]
        < metrics["external_negative_window_count"]
    )
    summary = {
        "round": 206,
        "decision": (
            "external_winter_record_supports_setp_warm_bias_direction_only"
            if supports_direction
            else "external_winter_record_does_not_confirm_setp_warm_bias_direction"
        ),
        "target_boundary": {
            "external": "previous-November to current-February Tmin",
            "project_and_reference": "annual mean temperature",
            "region": SET_BOX,
            "absolute_offset_validation_permitted": False,
        },
        "locked_parser": {
            "locked_rows_skipped_before_temperature_conversion": locked_rows_skipped,
            "expected_locked_rows": 19,
            "project_locked_raster_bands_read": False,
            "interpolation_or_smoothing_across_locked_gap": False,
        },
        "spatial_support": {
            "box_cell_count": int(box_mask.sum()),
            "hma_cell_count": int(hma_mask.sum()),
            "member_count": len(project_members),
            "member_orientation": next(iter(orientations)),
        },
        "metrics": metrics,
        "source_identity": json.loads(IDENTITY_SUMMARY.read_text(encoding="utf-8"))[
            "identity_audit"
        ],
        "interpretation": (
            "The comparison diagnoses low-frequency direction and phase only. "
            "A seasonal CRU-calibrated reconstruction cannot set the absolute "
            "annual HMA correction amplitude."
        ),
        "source_states_unchanged": True,
        "locked_period_values_accessed": False,
        "jionglacuo_accessed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {"sources_before": before, "sources_after": after},
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = f"""# Round 206 final result

## Decision

`{summary['decision']}`

## Main 1500-1899 result

- External winter Tmin is below its 1961-1990 mean in
  `{metrics['external_negative_window_count']}/8` 50-year windows.
- The frozen project SETP annual curve is below its baseline in
  `{metrics['project_setp_negative_window_count']}/8` windows; the four-family
  SETP reference is below baseline in
  `{metrics['reference_setp_negative_window_count']}/8` windows.
- External/project and external/reference 50-year correlations are
  `{metrics['window_correlation_external_project_setp']:.3f}` and
  `{metrics['window_correlation_external_reference_setp']:.3f}`.
- Adjacent-window sign agreements are
  `{metrics['window_adjacent_change_sign_agreement_external_project_setp']}/7`
  and `{metrics['window_adjacent_change_sign_agreement_external_reference_setp']}/7`.

## Boundary

Huang et al. reconstruct winter Tmin, whereas the project and four-family
fields are annual mean temperature. The source is calibrated to CRU TS 4.01.
Therefore this result can adjudicate regional low-frequency direction and
phase, but it cannot validate a uniform absolute HMA temperature offset or set
a production correction amplitude.

## Integrity

All 19 rows for 1921-1939 were discarded after decoding the year token and
before converting the temperature token. Project raster bands for those years
were never read. No interpolation or 31-year smoothing crosses the sealed gap.
All source hashes are unchanged. No Jionglacuo, fitting, correction, EnKF,
production reconstruction, or OGGM access occurred.
"""
    (OUT / "FINAL_RESULT.md").write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
