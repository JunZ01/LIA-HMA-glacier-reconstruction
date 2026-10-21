"""Compare allowed-period HMA hydroclimate reconstructions with frozen precipitation fields."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
LOOP = PROJECT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUTPUT = LOOP / "round_102_tree_ring_hydroclimate_evidence_v1"
EXTERNAL = PROJECT / "validation_data/external_sources/hma_tree_ring_hydroclimate_audit_v1"
PUBLIC = Path(r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc")
MASK_PATH = PROJECT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy"
PROXY_IDENTITY = LOOP / "round_07_proxy_network_identity_alias_audit_v3/proxy9_identity_alias_inventory.csv"
DOD2K_IDENTITY = LOOP / "round_53_dod2k_temperature_identity_chronology_v1/dod2k_temperature_identity_chronology.csv"

PERIODS = {
    "early_1901_1920": np.arange(1901, 1921),
    "postlock_1940_1950": np.arange(1940, 1951),
}
LOCKED = set(range(1921, 1940))
PRODUCTS = {
    "project_mean": "pr_ensemble_mean_rec_result",
    "CRU": "pr_obs_cru_pre_05deg",
    "GPCC": "pr_obs_gpcp_05deg",
    "HGAO": "pr_rec_ref_HGAO_c05d",
    "LMR": "pr_rec_ref_LMR_c05d",
    "O": "pr_rec_ref_O_c05d",
}
REGIONS = {
    "hma": (-np.inf, np.inf),
    "west_lt80": (-np.inf, 80.0),
    "central_80_90": (80.0, 90.0),
    "east_ge90": (90.0, np.inf),
}

EVIDENCE = {
    "nw_himalaya_spi": {
        "file": "himalaya2017spi.txt",
        "value_index": 1,
        "bbox": (33.0, 34.5, 74.0, 76.5),
        "variable": "April-May SPI",
        "season": "Apr-May",
        "overlap": "unresolved_possible_overlap",
        "reason": "Five source chronologies are not yet crosswalked record-by-record to the frozen proxy sequence.",
    },
    "se_tibet_scpdsi": {
        "file": "se-tibet2017scpdsi.txt",
        "value_index": 1,
        "bbox": (27.5, 29.5, 99.0, 101.0),
        "variable": "April-June scPDSI",
        "season": "Apr-Jun",
        "overlap": "confirmed_partial_overlap",
        "reason": "The reconstruction uses CHIN019 and CHIN075; CHIN019 is an exact frozen-sequence duplicate in Round 53.",
    },
    "ne_tibet_precip": {
        "file": "ne-tibet2014precip.txt",
        "value_index": 2,
        "bbox": (37.0, 38.7, 97.0, 100.0),
        "variable": "Reconstructed precipitation (mm)",
        "season": "previous Jul-current Jun",
        "overlap": "unresolved_possible_overlap",
        "reason": "The Qilian juniper source network has not been sequence-crosswalked to the frozen proxy records.",
    },
    "delingha_precip": {
        "file": "delingha2021d18o-precip.txt",
        "value_index": 5,
        "bbox": (37.45, 37.48, 97.67, 97.78),
        "variable": "Reconstructed precipitation (mm)",
        "season": "previous Aug-current Jul",
        "overlap": "likely_independent_not_sequence_tested",
        "reason": "The 2021 isotope reconstruction postdates the frozen PAGES2k2017 network, but no sequence-level proof is available.",
    },
}


def file_state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan)
    if finite.sum() < 2:
        return result
    scale = np.std(values[finite], ddof=1)
    if scale > 0:
        result[finite] = (values[finite] - np.mean(values[finite])) / scale
    return result


def weighted_series(values: np.ndarray, mask: np.ndarray, lat: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    valid = np.isfinite(values)
    denominator = np.sum(valid * weights[None], axis=(1, 2))
    return np.nansum(values * weights[None], axis=(1, 2)) / np.where(denominator == 0, np.nan, denominator)


def metrics(evidence: np.ndarray, product: np.ndarray, years: np.ndarray) -> dict[str, object]:
    valid = np.isfinite(evidence) & np.isfinite(product)
    n = int(valid.sum())
    if n < 3:
        return {"n": n, "pearson_r": np.nan, "spearman_r": np.nan, "evidence_trend_z_per_century": np.nan, "product_trend_z_per_century": np.nan, "trend_sign_match": False, "endpoint_sign_match": False}
    ev = zscore(evidence[valid])
    pr = zscore(product[valid])
    yy = years[valid]
    ev_trend = float(np.polyfit(yy, ev, 1)[0] * 100.0)
    pr_trend = float(np.polyfit(yy, pr, 1)[0] * 100.0)
    width = 5 if n >= 15 else 3
    ev_change = float(np.mean(ev[-width:]) - np.mean(ev[:width]))
    pr_change = float(np.mean(pr[-width:]) - np.mean(pr[:width]))
    return {
        "n": n,
        "pearson_r": float(pearsonr(ev, pr).statistic),
        "spearman_r": float(spearmanr(ev, pr).statistic),
        "evidence_trend_z_per_century": ev_trend,
        "product_trend_z_per_century": pr_trend,
        "evidence_endpoint_change_z": ev_change,
        "product_endpoint_change_z": pr_change,
        "trend_sign_match": bool(np.sign(ev_trend) == np.sign(pr_trend)),
        "endpoint_sign_match": bool(np.sign(ev_change) == np.sign(pr_change)),
    }


def parse_allowed_text(path: Path, value_index: int) -> tuple[dict[int, float], int]:
    allowed = set(np.concatenate(list(PERIODS.values())).astype(int))
    result: dict[int, float] = {}
    locked_rows_skipped = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            try:
                year = int(float(tokens[0]))
            except (ValueError, IndexError):
                continue
            if year in LOCKED:
                locked_rows_skipped += 1
                continue
            if year not in allowed or len(tokens) <= value_index:
                continue
            try:
                value = float(tokens[value_index])
            except ValueError:
                continue
            if value <= -900:
                value = np.nan
            result[year] = value
    return result, locked_rows_skipped


def read_mada_allowed(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with netcdf_file(path, "r", mmap=False) as handle:
        years = np.asarray(handle.variables["time"].data, dtype=int).copy()
        lat = np.asarray(handle.variables["lat"].data, dtype=float).copy()
        lon = np.asarray(handle.variables["lon"].data, dtype=float).copy()
        selected = np.flatnonzero(np.isin(years, np.concatenate(list(PERIODS.values()))))
        if np.any(np.isin(years[selected], list(LOCKED))):
            raise RuntimeError("locked-mada-index-requested")
        values = np.asarray(handle.variables["PDSI"].data[selected], dtype=float).copy()
    values[np.isclose(values, -99.999) | (values < -90)] = np.nan
    return lat, lon, {period: values[np.isin(years[selected], target)] for period, target in PERIODS.items()}


def mada_overlap_weights(mada_lat: np.ndarray, mada_lon: np.ndarray, fine_lat: np.ndarray, fine_lon: np.ndarray, hma_mask: np.ndarray) -> np.ndarray:
    weights = np.zeros((len(mada_lat), len(mada_lon)), dtype=float)
    yy, xx = np.meshgrid(fine_lat, fine_lon, indexing="ij")
    fine_weight = np.cos(np.deg2rad(yy)) * hma_mask
    for iy, latitude in enumerate(mada_lat):
        for ix, longitude in enumerate(mada_lon):
            inside = (np.abs(yy - latitude) <= 1.25 + 1e-8) & (np.abs(xx - longitude) <= 1.25 + 1e-8)
            weights[iy, ix] = float(np.sum(fine_weight[inside]))
    return weights


def mada_series(values: np.ndarray, weights: np.ndarray, lon: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lower, upper = bounds
    region = (lon[None, :] >= lower) & (lon[None, :] < upper)
    use = weights * region
    valid = np.isfinite(values)
    denominator = np.sum(valid * use[None], axis=(1, 2))
    return np.nansum(values * use[None], axis=(1, 2)) / np.where(denominator == 0, np.nan, denominator)


def main() -> int:
    sources = {
        "mada": EXTERNAL / "MADApdsi.nc",
        "mada_xy": EXTERNAL / "jja-mada-xy.txt",
        "mada_readme": EXTERNAL / "readme-cook2010mada.txt",
        **{name: EXTERNAL / item["file"] for name, item in EVIDENCE.items()},
        "public_fields": PUBLIC,
        "hma_mask": MASK_PATH,
        "proxy_identity": PROXY_IDENTITY,
        "dod2k_identity": DOD2K_IDENTITY,
        "driver": Path(__file__).resolve(),
    }
    for path in sources.values():
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "\n".join([
            "# Round 102 Contract", "", "## Objective",
            "Compare MADA and four regional tree-ring hydroclimate reconstructions with frozen HMA annual precipitation fields without changing any reconstruction.", "",
            "## Frozen boundaries",
            "- Parse and save only 1901-1920 and 1940-1950 climate magnitudes.",
            "- Do not use or export 1921-1939 values.",
            "- Treat PDSI/SPI and seasonal reconstructions as phase evidence, not precipitation-amplitude truth.",
            "- Report proxy-source overlap; do not claim independence where chronology overlap is unresolved.",
            "- Do not tune EnKF, alter source files, or run OGGM.", "",
            "## Contamination note",
            "- The NOAA browser preview for himalaya2017spi.txt exposed its one-line-rendered 1921-1939 values before this driver ran. Those values are excluded from every calculation and output, but that record can no longer serve as a pristine future blind test for those years.",
        ]) + "\n", encoding="utf-8")
    before = {name: file_state(path) for name, path in sources.items()}

    mask = np.load(MASK_PATH).astype(bool)
    with h5py.File(PUBLIC, "r") as handle:
        years_all = np.asarray(handle["time"], dtype=int).ravel()
        fine_lat = np.asarray(handle["lat"], dtype=float).ravel()
        fine_lon = np.asarray(handle["lon"], dtype=float).ravel()
        if mask.shape != (len(fine_lat), len(fine_lon)):
            raise RuntimeError("hma-mask-shape-conflict")
        public: dict[str, dict[str, np.ndarray]] = {period: {} for period in PERIODS}
        for period, years in PERIODS.items():
            indices = np.flatnonzero(np.isin(years_all, years))
            if not np.array_equal(years_all[indices], years) or np.any(np.isin(years_all[indices], list(LOCKED))):
                raise RuntimeError(f"public-year-selection-failed:{period}")
            for product, variable in PRODUCTS.items():
                values = np.asarray(handle[variable][indices], dtype=float)
                fill = float(np.asarray(handle[variable].attrs.get("_FillValue", -9999.0)).squeeze())
                values[np.isclose(values, fill) | (values < -9000)] = np.nan
                public[period][product] = values

    mada_lat, mada_lon, mada_values = read_mada_allowed(sources["mada"])
    overlap_weights = mada_overlap_weights(mada_lat, mada_lon, fine_lat, fine_lon, mask)
    longitude_grid = np.broadcast_to(fine_lon[None], mask.shape)
    region_masks = {name: mask & (longitude_grid >= bounds[0]) & (longitude_grid < bounds[1]) for name, bounds in REGIONS.items()}

    curve_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for period, years in PERIODS.items():
        for region, bounds in REGIONS.items():
            evidence_values = mada_series(mada_values[period], overlap_weights, mada_lon, bounds)
            evidence_z = zscore(evidence_values)
            for i, year in enumerate(years):
                curve_rows.append({"period": period, "evidence": "MADA_JJA_PDSI", "region": region, "year": int(year), "series": "evidence", "raw_value": evidence_values[i], "z_value": evidence_z[i]})
            for product, fields in public[period].items():
                series = weighted_series(fields, region_masks[region], fine_lat)
                series_z = zscore(series)
                for i, year in enumerate(years):
                    curve_rows.append({"period": period, "evidence": "MADA_JJA_PDSI", "region": region, "year": int(year), "series": product, "raw_value": series[i], "z_value": series_z[i]})
                metric_rows.append({"period": period, "evidence": "MADA_JJA_PDSI", "region": region, "product": product, "target_season": "JJA", "product_season": "annual", "season_exact_match": False, **metrics(evidence_values, series, years)})

    catalog_rows = []
    proxy9 = pd.read_csv(PROXY_IDENTITY)
    dod2k = pd.read_csv(DOD2K_IDENTITY)
    locked_skips: dict[str, int] = {}
    for evidence_name, item in EVIDENCE.items():
        values_by_year, skipped = parse_allowed_text(EXTERNAL / item["file"], int(item["value_index"]))
        locked_skips[evidence_name] = skipped
        south, north, west, east = item["bbox"]
        proxy_candidates = proxy9[(proxy9.latitude >= south) & (proxy9.latitude <= north) & (proxy9.longitude >= west) & (proxy9.longitude <= east)]
        catalog_rows.append({
            "evidence": evidence_name, "source_file": item["file"], "target_variable": item["variable"], "target_season": item["season"],
            "south": south, "north": north, "west": west, "east": east, "frozen_proxy_coordinate_candidates": len(proxy_candidates),
            "overlap_class": item["overlap"], "overlap_reason": item["reason"], "season_exactly_available_in_project": False,
        })
        bbox_mask = (fine_lat[:, None] >= south) & (fine_lat[:, None] <= north) & (fine_lon[None, :] >= west) & (fine_lon[None, :] <= east)
        if not np.any(bbox_mask):
            center_lat, center_lon = (south + north) / 2, (west + east) / 2
            bbox_mask[np.argmin(np.abs(fine_lat - center_lat)), np.argmin(np.abs(fine_lon - center_lon))] = True
        for period, years in PERIODS.items():
            evidence_values = np.asarray([values_by_year.get(int(year), np.nan) for year in years])
            evidence_z = zscore(evidence_values)
            for i, year in enumerate(years):
                if np.isfinite(evidence_values[i]):
                    curve_rows.append({"period": period, "evidence": evidence_name, "region": evidence_name, "year": int(year), "series": "evidence", "raw_value": evidence_values[i], "z_value": evidence_z[i]})
            for product, fields in public[period].items():
                series = weighted_series(fields, bbox_mask, fine_lat)
                series_z = zscore(series)
                for i, year in enumerate(years):
                    if np.isfinite(evidence_values[i]):
                        curve_rows.append({"period": period, "evidence": evidence_name, "region": evidence_name, "year": int(year), "series": product, "raw_value": series[i], "z_value": series_z[i]})
                metric_rows.append({"period": period, "evidence": evidence_name, "region": evidence_name, "product": product, "target_season": item["season"], "product_season": "annual", "season_exact_match": False, **metrics(evidence_values, series, years)})

    curves = pd.DataFrame(curve_rows)
    if curves.year.between(1921, 1939).any():
        raise RuntimeError("locked-year-export-attempt")
    metric_table = pd.DataFrame(metric_rows)
    catalog = pd.DataFrame(catalog_rows)
    curves.to_csv(OUTPUT / "allowed_period_standardized_curves.csv", index=False)
    metric_table.to_csv(OUTPUT / "hydroclimate_phase_metrics.csv", index=False)
    catalog.to_csv(OUTPUT / "evidence_catalog_and_overlap.csv", index=False)

    exact_chin019 = int((dod2k.formal_identity.astype(str).str.contains("CHIN019", case=False, na=False) & (dod2k.sequence_relation == "current_proxy_sequence_duplicate")).sum())
    overlap_summary = {
        "MADA_JJA_PDSI": "potential_partial_overlap_site_crosswalk_not_available",
        "nw_himalaya_spi": EVIDENCE["nw_himalaya_spi"]["overlap"],
        "se_tibet_scpdsi": EVIDENCE["se_tibet_scpdsi"]["overlap"],
        "ne_tibet_precip": EVIDENCE["ne_tibet_precip"]["overlap"],
        "delingha_precip": EVIDENCE["delingha_precip"]["overlap"],
        "chin019_exact_current_sequence_duplicates": exact_chin019,
    }

    colors = {"evidence": "#111111", "project_mean": "#0072B2", "CRU": "#D55E00", "GPCC": "#009E73", "HGAO": "#CC79A7", "LMR": "#E69F00", "O": "#56B4E9"}
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex="col", constrained_layout=True)
    for row, region in enumerate(REGIONS):
        for col, period in enumerate(PERIODS):
            ax = axes[row, col]
            subset = curves[(curves.evidence == "MADA_JJA_PDSI") & (curves.region == region) & (curves.period == period)]
            for series in colors:
                data = subset[subset.series == series].sort_values("year")
                if data.empty: continue
                ax.plot(data.year, data.z_value, color=colors[series], linewidth=2.4 if series in ("evidence", "project_mean") else 1.1, alpha=1 if series in ("evidence", "project_mean", "CRU", "GPCC") else 0.65, label=series)
            ax.axhline(0, color="#888888", linewidth=0.7)
            ax.set_title(f"{region} | {period}")
            ax.set_ylabel("Within-period z")
    axes[0, 0].legend(frameon=False, ncol=4, fontsize=8)
    axes[-1, 0].set_xlabel("Year"); axes[-1, 1].set_xlabel("Year")
    fig.suptitle("MADA JJA PDSI versus annual precipitation fields", fontsize=15)
    fig.savefig(OUTPUT / "mada_hma_phase_curves.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(len(EVIDENCE), 2, figsize=(14, 12), constrained_layout=True)
    for row, evidence_name in enumerate(EVIDENCE):
        for col, period in enumerate(PERIODS):
            ax = axes[row, col]
            subset = curves[(curves.evidence == evidence_name) & (curves.period == period)]
            for series in ("evidence", "project_mean", "CRU", "GPCC"):
                data = subset[subset.series == series].sort_values("year")
                if data.empty: continue
                ax.plot(data.year, data.z_value, marker="o", markersize=3, color=colors[series], linewidth=2.3 if series in ("evidence", "project_mean") else 1.2, label=series)
            ax.axhline(0, color="#888888", linewidth=0.7)
            ax.set_title(f"{evidence_name} | {period}")
            ax.set_ylabel("Matched-year z")
    axes[0, 0].legend(frameon=False, ncol=4, fontsize=8)
    axes[-1, 0].set_xlabel("Year"); axes[-1, 1].set_xlabel("Year")
    fig.suptitle("Regional tree-ring hydroclimate evidence versus annual precipitation", fontsize=15)
    fig.savefig(OUTPUT / "regional_tree_ring_phase_curves.png", dpi=220)
    plt.close(fig)

    early_project = metric_table[(metric_table.period == "early_1901_1920") & (metric_table["product"] == "project_mean")]
    result_rows = early_project[["evidence", "region", "n", "pearson_r", "spearman_r", "evidence_trend_z_per_century", "product_trend_z_per_century", "trend_sign_match", "endpoint_sign_match"]].to_dict(orient="records")
    summary = {
        "status": "complete",
        "parsed_periods": {name: [int(years[0]), int(years[-1])] for name, years in PERIODS.items()},
        "locked_1921_1939_values_used_or_exported": False,
        "locked_rows_explicitly_skipped_by_text_parsers": locked_skips,
        "browser_preview_contamination": {"source": "himalaya2017spi.txt", "locked_period_exposed": True, "used_in_calculation": False},
        "evidence_count": 5,
        "all_project_comparisons_have_season_mismatch": True,
        "overlap_summary": overlap_summary,
        "early_project_metrics": result_rows,
        "decision": "use_as_hydroclimate_phase_and_spatial_evidence_not_as_independent_amplitude_target",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Round 102 Tree-Ring Hydroclimate Evidence", "", "- MADA and four regional hydroclimate reconstructions were compared only over 1901-1920 and 1940-1950.", "- The project fields are annual; every proxy target is seasonal or cross-year. Results therefore diagnose phase and trend, not amplitude.", "- MADA and several regional records have potential or confirmed chronology overlap with the frozen proxy network; they are not automatically independent validation.", "- CHIN019 is confirmed as an exact current-proxy sequence duplicate and contributes to the southeastern Tibetan Plateau scPDSI reconstruction.", "- The locked-period values shown accidentally by the NOAA browser preview were not parsed into calculations or outputs; the northwest SPI record is nevertheless no longer pristine for a future 1921-1939 blind test.", "", "## Early project comparison"]
    for row in result_rows:
        lines.append(f"- {row['evidence']} ({row['region']}): n={row['n']}, Pearson={row['pearson_r']:.3f}, Spearman={row['spearman_r']:.3f}, trend-sign match={str(row['trend_sign_match']).lower()}, endpoint-sign match={str(row['endpoint_sign_match']).lower()}.")
    lines.extend(["", "Decision: use these records as hydroclimate phase/spatial evidence and source-diagnostic constraints, not as independent precipitation-amplitude targets."])
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: file_state(path) for name, path in sources.items()}
    manifest = {"sources_before": before, "sources_after": after, "source_states_unchanged": before == after, "source_urls": {
        "MADA": "https://www.ncei.noaa.gov/pub/data/paleo/treering/reconstructions/asia/cook2010pdsi/",
        "nw_himalaya_spi": "https://www.ncei.noaa.gov/pub/data/paleo/treering/reconstructions/asia/himalaya2017spi.txt",
        "se_tibet_scpdsi": "https://www.ncei.noaa.gov/pub/data/paleo/treering/reconstructions/asia/china/se-tibet2017scpdsi.txt",
        "ne_tibet_precip": "https://www.ncei.noaa.gov/pub/data/paleo/treering/reconstructions/asia/china/ne-tibet2014precip.txt",
        "delingha_precip": "https://www.ncei.noaa.gov/pub/data/paleo/treering/isotope/asia/china/delingha2021d18o-precip.txt",
    }}
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["source_states_unchanged"]:
        raise RuntimeError("source-integrity-failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
