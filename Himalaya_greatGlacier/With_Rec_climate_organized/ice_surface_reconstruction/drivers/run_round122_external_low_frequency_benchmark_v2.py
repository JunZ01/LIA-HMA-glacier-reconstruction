"""Compare the frozen reconstruction with new independent low-frequency benchmarks."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
LOCKED = (1921, 1939)
PERIODS = {
    "prelock_1500_1920": (1500, 1920),
    "early_1901_1920": (1901, 1920),
    "modern_1940_2000": (1940, 2000),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-spec-failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "sha256": sha256(path)}


def allowed_year(year: float) -> bool:
    return 1500 <= year <= 2000 and not (LOCKED[0] <= year <= LOCKED[1])


def read_noaa_columns(
    path: Path,
    age_column: str,
    value_columns: list[str],
    age_scale: str = "CE",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    header = None
    indices: dict[str, int] = {}
    years = []
    values = {column: [] for column in value_columns}
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("\t")
            if header is None:
                if age_column not in parts or not all(column in parts for column in value_columns):
                    continue
                header = parts
                indices = {column: parts.index(column) for column in [age_column, *value_columns]}
                continue
            if len(parts) <= max(indices.values()):
                continue
            try:
                raw_age = float(parts[indices[age_column]])
            except ValueError:
                continue
            year = 1950.0 - raw_age if age_scale == "BP1950" else raw_age
            if not allowed_year(year):
                continue
            row = {}
            valid = True
            for column in value_columns:
                token = parts[indices[column]].strip()
                if token.upper() in {"", "NA", "NAN", "-999", "-999.0"}:
                    valid = False
                    break
                try:
                    row[column] = float(token)
                except ValueError:
                    valid = False
                    break
            if valid:
                years.append(year)
                for column in value_columns:
                    values[column].append(row[column])
    if header is None:
        raise RuntimeError(f"header-not-found:{path}")
    result = {}
    for column in value_columns:
        result[column] = (np.asarray(years, dtype=float), np.asarray(values[column], dtype=float))
    return result


def shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.iter(f"{{{NS_MAIN}}}t"))
        for item in root.findall(f"{{{NS_MAIN}}}si")
    ]


def first_sheet_target(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{{{NS_PKG_REL}}}Relationship")
    }
    sheet = workbook.find(f".//{{{NS_MAIN}}}sheet")
    if sheet is None:
        raise RuntimeError("xlsx-no-sheet")
    target = targets[sheet.attrib[f"{{{NS_REL}}}id"]].replace("\\", "/")
    if target.startswith("/"):
        return target.lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def cell_text(cell: ET.Element, strings: list[str]) -> str:
    cell_type = cell.attrib.get("t", "n")
    node = cell.find(f"{{{NS_MAIN}}}v")
    if cell_type == "s" and node is not None:
        return strings[int(node.text or "0")]
    inline = cell.find(f"{{{NS_MAIN}}}is")
    if cell_type == "inlineStr" and inline is not None:
        return "".join(item.text or "" for item in inline.iter(f"{{{NS_MAIN}}}t"))
    return "" if node is None else node.text or ""


def read_xlsx_ab(path: Path) -> tuple[np.ndarray, np.ndarray]:
    years = []
    values = []
    with zipfile.ZipFile(path) as archive:
        strings = shared_strings(archive)
        with archive.open(first_sheet_target(archive)) as stream:
            for _, row in ET.iterparse(stream, events=("end",)):
                if row.tag != f"{{{NS_MAIN}}}row":
                    continue
                cells = {
                    cell.attrib.get("r", "")[:1].upper(): cell
                    for cell in row.findall(f"{{{NS_MAIN}}}c")
                }
                if "A" not in cells:
                    row.clear()
                    continue
                try:
                    year = float(cell_text(cells["A"], strings))
                except ValueError:
                    row.clear()
                    continue
                if not allowed_year(year):
                    row.clear()
                    continue
                if "B" not in cells:
                    raise RuntimeError(f"xlsx-missing-column-b:{path.name}:{year}")
                try:
                    value = float(cell_text(cells["B"], strings))
                except ValueError:
                    row.clear()
                    continue
                years.append(year)
                values.append(value)
                row.clear()
    if not values:
        raise RuntimeError(f"xlsx-no-allowed-values:{path}")
    return np.asarray(years, dtype=float), np.asarray(values, dtype=float)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3 or np.nanstd(left[finite]) <= 1e-12 or np.nanstd(right[finite]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def slope_per_century(years: np.ndarray, values: np.ndarray) -> float:
    finite = np.isfinite(years) & np.isfinite(values)
    if int(finite.sum()) < 3:
        return np.nan
    return float(np.polyfit(years[finite], values[finite], 1)[0] * 100.0)


def metric_rows(
    name: str,
    variable: str,
    region: str,
    years: np.ndarray,
    values: np.ndarray,
    current_years: np.ndarray,
    current_values: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    order = np.argsort(years)
    years = years[order]
    values = values[order]
    spacing = np.diff(np.unique(years))
    annual = bool(len(spacing) and np.nanmedian(spacing) <= 1.1)
    for period, (start, end) in PERIODS.items():
        selected = (years >= start) & (years <= end)
        y = years[selected]
        ext = values[selected]
        if len(y) < 3:
            rows.append(
                {
                    "benchmark": name,
                    "climate_variable": variable,
                    "region": region,
                    "period": period,
                    "sample_count": int(len(y)),
                    "annual_resolution": annual,
                    "raw_phase_correlation": np.nan,
                    "lowpass_31y_correlation": np.nan,
                    "benchmark_trend_per_century_native_units": np.nan,
                    "current_trend_per_century_native_units": np.nan,
                    "trend_sign_agreement": False,
                }
            )
            continue
        current = np.interp(y, current_years, current_values)
        lowpass = np.nan
        if annual and len(y) >= 31:
            frame = pd.DataFrame({"year": y.astype(int), "external": ext, "current": current})
            frame = frame.groupby("year", as_index=False).mean().set_index("year")
            smooth = frame.rolling(31, center=True, min_periods=21).mean().dropna()
            lowpass = correlation(smooth.external.to_numpy(), smooth.current.to_numpy())
        ext_slope = slope_per_century(y, ext)
        current_slope = slope_per_century(y, current)
        rows.append(
            {
                "benchmark": name,
                "climate_variable": variable,
                "region": region,
                "period": period,
                "sample_count": int(len(y)),
                "annual_resolution": annual,
                "raw_phase_correlation": correlation(ext, current),
                "lowpass_31y_correlation": lowpass,
                "benchmark_trend_per_century_native_units": ext_slope,
                "current_trend_per_century_native_units": current_slope,
                "trend_sign_agreement": bool(np.sign(ext_slope) == np.sign(current_slope)),
            }
        )
    return rows


def standardize(values: np.ndarray) -> np.ndarray:
    return (values - np.nanmean(values)) / np.nanstd(values, ddof=1)


def main() -> int:
    driver = Path(__file__).resolve().parent
    base = load_module(
        "round122_base",
        driver / "run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1.py",
    )
    common = base.load_common(driver)
    root = base.project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_external_low_frequency_benchmark_20260818_v2"
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    discovery = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_external_proxy_spatial_gap_discovery_20260818_v1"
    )
    raw = discovery / "raw_unopened"
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "public": Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"),
        "hma_mask": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "mendeley_temperature": raw / "eastern_tibet_814y_temperature_reconstruction.xlsx",
        "figshare_precipitation": raw / "northwestern_qtp_845_2023_precipitation_REC.xlsx",
        "sun2026_temperature": raw / "sun2026_tibet_temperature_reconstruction.txt",
        "tibet2019_temperature": raw / "tibet2019_temperature_reconstructions.txt",
        "aweng_temperature": raw / "aweng_co_temperature.txt",
        "heihai_temperature": raw / "heihai_lake_temperature.txt",
        "ngamring_temperature": raw / "ngamring_co_d2h_maat.txt",
        "shenqi_precipitation": raw / "shenqi_cave_precipitation_index.txt",
        "contract": driver / "round122_external_low_frequency_benchmark_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "base_driver": driver / "run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1.py",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    shutil.copy2(paths["contract"], output / "ROUND_CONTRACT.md")
    before = {name: file_state(path) for name, path in paths.items()}

    allowed = np.concatenate([np.arange(1500, 1921), np.arange(1940, 2001)]).astype(int)
    grid_lat, grid_lon = common.state_coordinates()
    with h5py.File(paths["public"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).reshape(-1)
        indices = np.asarray([int(np.flatnonzero(source_years == year)[0]) for year in allowed])
        if np.any((source_years[indices] >= LOCKED[0]) & (source_years[indices] <= LOCKED[1])):
            raise RuntimeError("locked-current-field-access-attempt")
        source_lat = np.asarray(handle["lat"], dtype=float)
        source_lon = np.asarray(handle["lon"], dtype=float)
        fields = {}
        for variable, source_name in (
            ("temperature", "tas_ensemble_mean_rec_result"),
            ("precipitation", "pr_ensemble_mean_rec_result"),
        ):
            cube = common.source_grid(
                common.clean(np.asarray(handle[source_name][indices], dtype=float)),
                source_lat,
                source_lon,
                grid_lat,
                grid_lon,
            )
            state = common.cube_to_state(cube)
            baseline = (allowed >= 1961) & (allowed <= 1990)
            fields[variable] = state - np.nanmean(state[:, baseline], axis=1, keepdims=True)
    mask05 = np.load(paths["hma_mask"]).astype(float)
    mask2 = common.source_grid(mask05[None], source_lat, source_lon, grid_lat, grid_lon)[0] >= 0.5
    weights = common.region_weights(grid_lat, grid_lon, mask2)
    curves = {
        variable: {
            region: common.weighted_series(values, current_weights)
            for region, current_weights in weights.items()
        }
        for variable, values in fields.items()
    }

    benchmarks: dict[str, dict[str, Any]] = {}
    years, values = read_xlsx_ab(paths["mendeley_temperature"])
    benchmarks["eastern_tibet_814y_jjas_temperature"] = {
        "variable": "temperature", "region": "east_ge90", "years": years, "values": values
    }
    years, values = read_xlsx_ab(paths["figshare_precipitation"])
    benchmarks["northwestern_qtp_mjj_precipitation"] = {
        "variable": "precipitation", "region": "east_ge90", "years": years, "values": values
    }
    loaded = read_noaa_columns(paths["sun2026_temperature"], "age_AD", ["JJA_temp_rec"])
    benchmarks["sun2026_setp_jja_tmax"] = {
        "variable": "temperature", "region": "east_ge90", "years": loaded["JJA_temp_rec"][0], "values": loaded["JJA_temp_rec"][1]
    }
    loaded = read_noaa_columns(paths["tibet2019_temperature"], "age_CE", ["temp_NETP", "temp_SETP"])
    for column, name in (("temp_NETP", "tibet2019_netp_annual_temperature"), ("temp_SETP", "tibet2019_setp_jja_temperature")):
        benchmarks[name] = {"variable": "temperature", "region": "east_ge90", "years": loaded[column][0], "values": loaded[column][1]}
    for key, path_key, age_column, value_column, age_scale, region in (
        ("aweng_co_surface_temperature", "aweng_temperature", "age", "temperature", "BP1950", "central_80_90"),
        ("heihai_lake_summer_temperature", "heihai_temperature", "age", "temperature", "BP1950", "east_ge90"),
        ("ngamring_co_maat", "ngamring_temperature", "age_BP", "MAAT", "BP1950", "central_80_90"),
        ("shenqi_cave_precipitation_index", "shenqi_precipitation", "age_CE", "PI", "CE", "east_ge90"),
    ):
        loaded = read_noaa_columns(paths[path_key], age_column, [value_column], age_scale)
        benchmarks[key] = {
            "variable": "precipitation" if "precipitation" in key else "temperature",
            "region": region,
            "years": loaded[value_column][0],
            "values": loaded[value_column][1],
        }

    rows = []
    benchmark_rows = []
    for name, item in benchmarks.items():
        rows.extend(
            metric_rows(
                name,
                item["variable"],
                item["region"],
                item["years"],
                item["values"],
                allowed,
                curves[item["variable"]][item["region"]],
            )
        )
        for year, value in zip(item["years"], item["values"]):
            benchmark_rows.append(
                {"benchmark": name, "climate_variable": item["variable"], "region": item["region"], "year": year, "value": value}
            )
    metrics = pd.DataFrame(rows)
    benchmark_table = pd.DataFrame(benchmark_rows)
    metrics.to_csv(output / "external_benchmark_metrics.csv", index=False)
    benchmark_table.to_csv(output / "external_benchmark_allowed_values.csv", index=False)
    current_rows = []
    for variable, regional in curves.items():
        for region, values in regional.items():
            current_rows.extend(
                {"year": int(year), "climate_variable": variable, "region": region, "value": float(value)}
                for year, value in zip(allowed, values)
            )
    pd.DataFrame(current_rows).to_csv(output / "current_regional_curves_allowed_years.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    east_t = curves["temperature"]["east_ge90"]
    current_series = pd.Series(east_t, index=allowed).rolling(31, center=True, min_periods=21).mean()
    axes[0].plot(allowed, standardize(current_series.to_numpy()), color="#111111", linewidth=2.0, label="current east, 31y")
    for name, color in (
        ("eastern_tibet_814y_jjas_temperature", "#D55E00"),
        ("sun2026_setp_jja_tmax", "#0072B2"),
        ("tibet2019_netp_annual_temperature", "#009E73"),
        ("tibet2019_setp_jja_temperature", "#CC79A7"),
    ):
        item = benchmarks[name]
        series = pd.Series(item["values"], index=item["years"].astype(int)).groupby(level=0).mean()
        smooth = series.rolling(31, center=True, min_periods=21).mean()
        axes[0].plot(smooth.index, standardize(smooth.to_numpy()), color=color, linewidth=1.4, label=name)
    axes[0].set_ylabel("standardized temperature")
    axes[0].set_title("Eastern HMA annual benchmarks, 31-year low pass")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)

    for name, marker, color in (
        ("aweng_co_surface_temperature", "o", "#0072B2"),
        ("ngamring_co_maat", "s", "#009E73"),
        ("heihai_lake_summer_temperature", "^", "#D55E00"),
    ):
        item = benchmarks[name]
        axes[1].scatter(item["years"], standardize(item["values"]), marker=marker, color=color, s=32, label=name)
    axes[1].plot(allowed, standardize(curves["temperature"]["central_80_90"]), color="#777777", linewidth=1.0, alpha=0.8, label="current central")
    axes[1].set_ylabel("standardized temperature")
    axes[1].set_title("Sparse lake-temperature benchmarks")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)

    for name, color in (("northwestern_qtp_mjj_precipitation", "#0072B2"), ("shenqi_cave_precipitation_index", "#D55E00")):
        item = benchmarks[name]
        axes[2].plot(item["years"], standardize(item["values"]), color=color, linewidth=1.4, marker="o" if name.startswith("shenqi") else None, markersize=2.5, label=name)
    axes[2].plot(allowed, standardize(curves["precipitation"]["east_ge90"]), color="#111111", linewidth=1.2, label="current east precipitation")
    axes[2].set_ylabel("standardized precipitation")
    axes[2].set_xlabel("year CE")
    axes[2].set_title("Independent precipitation benchmarks")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.axvspan(1921, 1939, color="#DDDDDD", alpha=0.8)
        axis.axhline(0, color="#999999", linewidth=0.6)
        axis.set_xlim(1500, 2000)
    fig.savefig(output / "external_low_frequency_benchmark_comparison.png", dpi=220)
    plt.close(fig)

    after = {name: file_state(path) for name, path in paths.items()}
    unchanged = bool(before == after)
    if not unchanged:
        raise RuntimeError("source-changed-during-round122")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2), encoding="utf-8"
    )
    usable = metrics[metrics.sample_count >= 3].copy()
    annual_modern = usable[(usable.period == "modern_1940_2000") & usable.annual_resolution]
    prelock = usable[usable.period == "prelock_1500_1920"]
    summary = {
        "experiment_id": "round_122_external_low_frequency_benchmark_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "source_hashes_unchanged": unchanged,
        "benchmark_count": len(benchmarks),
        "annual_modern_median_raw_correlation": float(annual_modern.raw_phase_correlation.median()),
        "annual_modern_median_31y_correlation": float(annual_modern.lowpass_31y_correlation.median()),
        "prelock_trend_sign_agreement_fraction": float(prelock.trend_sign_agreement.mean()),
        "metrics": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ranked = usable.sort_values(["period", "raw_phase_correlation"], ascending=[True, False])
    lines = [
        "# Round 122 External Low-Frequency Benchmark",
        "",
        f"- Benchmarks assessed: `{len(benchmarks)}`.",
        f"- Modern annual median raw correlation: `{summary['annual_modern_median_raw_correlation']:.3f}`.",
        f"- Modern annual median 31-year correlation: `{summary['annual_modern_median_31y_correlation']:.3f}`.",
        f"- Pre-lock trend-sign agreement fraction: `{summary['prelock_trend_sign_agreement_fraction']:.3%}`.",
        "- Locked 1921-1939 accessed: `false`.",
        f"- Source hashes unchanged: `{str(unchanged).lower()}`.",
        "",
        "## Metrics",
        "",
    ]
    for _, row in ranked.iterrows():
        lines.append(
            f"- `{row.benchmark}` / `{row.period}`: n={int(row.sample_count)}, raw r={row.raw_phase_correlation:.3f}, 31y r={row.lowpass_31y_correlation:.3f}, trend-sign={str(bool(row.trend_sign_agreement)).lower()}."
        )
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

