"""Audit two independent PMIP temperature simulations as low-frequency priors."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import requests
from netCDF4 import Dataset, num2date
from scipy.interpolate import RegularGridInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
RESULTS = PROJECT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = RESULTS / "climate_reconstruction_temperature_pmip_multimodel_prior_audit_v1"
DRIVER = Path(__file__).resolve()
GRID_SOURCE = RESULTS / (
    "climate_reconstruction_improvement_loop_v2/"
    "round_115_temperature_low_rank_oof_v2/temperature_low_rank_oof_fields.npz"
)
MODE_SIM = RESULTS / (
    "climate_reconstruction_improvement_loop_v2/"
    "round_74_mode_sim_allowed_temperature_acquisition_v3/"
    "mode_sim_temp2_allowed_hma_bbox.nc"
)
REFERENCE_CURVES = RESULTS / (
    "climate_reconstruction_temperature_hma_reference_anchored_low_frequency_v1/"
    "temperature_hma_reference_anchored_candidate_1000_2000.csv"
)
PRIOR_CURVES = RESULTS / (
    "climate_reconstruction_temperature_lia_regional_prior_update_decomposition_v1/"
    "annual_gtng15_prior_update_curves.csv"
)
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"

ESGF_API = "https://esgf-node.llnl.gov/esg-search/search"
LOCKED_START = 1901
LOCKED_END = 1939
LIA_START = 1500
LIA_END = 1899
BASELINE_START = 1961
BASELINE_END = 1990
WINDOWS = [(start, start + 49) for start in range(1500, 1900, 50)]

MODEL_CONFIG = {
    "MPI-ESM1-2-LR": {
        "past_experiment": "past2k",
        "past_member": "r1i1p1f1",
        "historical_member": "r1i1p1f1",
        "past_year_offset": -7000,
    },
    "MIROC-ES2L": {
        "past_experiment": "past1000",
        "past_member": "r1i1p1f2",
        "historical_member": "r1i1p1f2",
        "past_year_offset": 0,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def query_esgf(source: str, experiment: str, member: str, activity: str) -> list[dict[str, Any]]:
    params = {
        "project": "CMIP6",
        "activity_id": activity,
        "experiment_id": experiment,
        "source_id": source,
        "member_id": member,
        "variable_id": "tas",
        "table_id": "Amon",
        "type": "File",
        "format": "application/solr+json",
        "limit": 1000,
    }
    response = requests.get(ESGF_API, params=params, timeout=240)
    response.raise_for_status()
    payload = response.json()
    docs = payload["response"]["docs"]
    if not docs:
        raise RuntimeError(f"No ESGF files: {source} {experiment} {member}")
    return docs


def year_span(title: str, offset: int) -> tuple[int, int]:
    match = re.search(r"_(\d{6})-(\d{6})\.nc$", title)
    if match is None:
        raise RuntimeError(f"Cannot parse filename years: {title}")
    return int(match.group(1)[:4]) + offset, int(match.group(2)[:4]) + offset


def opendap_url(doc: dict[str, Any]) -> str:
    urls = doc.get("url", [])
    candidates = [entry.split("|")[0] for entry in urls if "OPENDAP" in entry]
    if not candidates:
        raise RuntimeError(f"No OPeNDAP service: {doc['title']}")
    url = candidates[0]
    if url.endswith(".html"):
        url = url[:-5]
    return url.replace("http://esgf-node.ornl.gov", "https://esgf-node.ornl.gov")


def select_remote_files(
    docs: list[dict[str, Any]], offset: int, allowed_ranges: list[tuple[int, int]]
) -> list[dict[str, Any]]:
    by_title: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        by_title.setdefault(doc["title"], []).append(doc)
    selected: list[dict[str, Any]] = []
    for title, replicas in by_title.items():
        start, end = year_span(title, offset)
        if not any(start <= allowed_end and end >= allowed_start for allowed_start, allowed_end in allowed_ranges):
            continue
        ornl = [doc for doc in replicas if doc.get("data_node") == "esgf-node.ornl.gov"]
        candidates = ornl or replicas
        doc = sorted(candidates, key=lambda item: int(item.get("version", 0)), reverse=True)[0]
        selected.append(
            {
                "title": title,
                "start_year_ce": start,
                "end_year_ce": end,
                "offset": offset,
                "version": int(doc.get("version", 0)),
                "data_node": doc.get("data_node"),
                "checksum": (doc.get("checksum") or [None])[0],
                "checksum_type": (doc.get("checksum_type") or [None])[0],
                "size": int(doc.get("size", 0)),
                "url": opendap_url(doc),
            }
        )
    selected.sort(key=lambda item: (item["start_year_ce"], item["title"]))
    if not selected:
        raise RuntimeError("No remote files overlap allowed ranges")
    return selected


def interpolate_monthly_hma(
    dataset: Dataset,
    time_indices: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    lat = np.asarray(dataset.variables["lat"][:], dtype=float)
    lon = np.mod(np.asarray(dataset.variables["lon"][:], dtype=float), 360.0)
    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    lat = lat[lat_order]
    lon = lon[lon_order]
    lat_pad = max(float(np.median(np.diff(lat))) * 1.5, 2.0)
    lon_pad = max(float(np.median(np.diff(lon))) * 1.5, 2.0)
    lat_keep = np.flatnonzero((lat >= target_lat.min() - lat_pad) & (lat <= target_lat.max() + lat_pad))
    lon_keep = np.flatnonzero((lon >= target_lon.min() - lon_pad) & (lon <= target_lon.max() + lon_pad))
    if len(lat_keep) < 2 or len(lon_keep) < 2:
        raise RuntimeError("Insufficient source-grid overlap with HMA")

    original_lat = lat_order[lat_keep]
    original_lon = lon_order[lon_keep]
    lat_slice = slice(int(original_lat.min()), int(original_lat.max()) + 1)
    lon_slice = slice(int(original_lon.min()), int(original_lon.max()) + 1)
    source_lat_raw = np.asarray(dataset.variables["lat"][lat_slice], dtype=float)
    source_lon_raw = np.mod(np.asarray(dataset.variables["lon"][lon_slice], dtype=float), 360.0)
    lat_sort = np.argsort(source_lat_raw)
    lon_sort = np.argsort(source_lon_raw)
    source_lat = source_lat_raw[lat_sort]
    source_lon = source_lon_raw[lon_sort]

    target_lat_2d, target_lon_2d = np.meshgrid(target_lat, target_lon, indexing="ij")
    points = np.column_stack([target_lat_2d[target_mask], target_lon_2d[target_mask]])
    weights = np.cos(np.deg2rad(target_lat_2d[target_mask]))
    tas = dataset.variables["tas"]
    output = np.full(len(time_indices), np.nan)
    for output_index, time_index in enumerate(time_indices):
        indexers: list[Any] = []
        kept_dims: list[str] = []
        for dimension in tas.dimensions:
            if dimension == "time":
                indexers.append(int(time_index))
            elif dimension == "lat":
                indexers.append(lat_slice)
                kept_dims.append("lat")
            elif dimension == "lon":
                indexers.append(lon_slice)
                kept_dims.append("lon")
            else:
                indexers.append(0)
        field = np.asarray(tas[tuple(indexers)], dtype=float).squeeze()
        if kept_dims != ["lat", "lon"]:
            raise RuntimeError(f"Unexpected tas dimensions: {tas.dimensions}")
        field = field[np.ix_(lat_sort, lon_sort)]
        interpolator = RegularGridInterpolator(
            (source_lat, source_lon), field, bounds_error=False, fill_value=np.nan
        )
        target_values = interpolator(points)
        valid = np.isfinite(target_values)
        if int(valid.sum()) < int(0.9 * len(target_values)):
            raise RuntimeError("Too few finite interpolated HMA cells")
        output[output_index] = float(np.average(target_values[valid], weights=weights[valid]))
    return output


def read_remote_series(
    files: list[dict[str, Any]],
    allowed_ranges: list[tuple[int, int]],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, float | int]] = []
    audit: list[dict[str, Any]] = []
    for item in files:
        with Dataset(item["url"]) as dataset:
            time = dataset.variables["time"]
            calendar = getattr(time, "calendar", "standard")
            dates = num2date(time[:], units=time.units, calendar=calendar, only_use_cftime_datetimes=True)
            model_years = np.asarray([int(date.year) for date in dates])
            ce_years = model_years + int(item["offset"])
            allowed = np.zeros(len(ce_years), dtype=bool)
            for start, end in allowed_ranges:
                allowed |= (ce_years >= start) & (ce_years <= end)
            time_indices = np.flatnonzero(allowed)
            if len(time_indices) == 0:
                continue
            if np.any((ce_years[time_indices] >= LOCKED_START) & (ce_years[time_indices] <= LOCKED_END)):
                raise RuntimeError("Locked climate values requested")
            values = interpolate_monthly_hma(
                dataset, time_indices, target_lat, target_lon, target_mask
            )
            for index, value in zip(time_indices, values, strict=True):
                rows.append(
                    {
                        "year": int(ce_years[index]),
                        "month": int(dates[index].month),
                        "tas_K": float(value),
                    }
                )
            audit.append(
                {
                    **item,
                    "requested_month_count": int(len(time_indices)),
                    "requested_first_year": int(ce_years[time_indices].min()),
                    "requested_last_year": int(ce_years[time_indices].max()),
                    "locked_climate_values_requested": 0,
                }
            )
    monthly = pd.DataFrame(rows).drop_duplicates(["year", "month"]).sort_values(["year", "month"])
    counts = monthly.groupby("year")["month"].nunique()
    incomplete = counts[counts != 12]
    if len(incomplete):
        raise RuntimeError(f"Incomplete model years: {incomplete.to_dict()}")
    annual = monthly.groupby("year", as_index=False)["tas_K"].mean()
    return annual, audit


def read_mode_sim(
    target_lat: np.ndarray, target_lon: np.ndarray, target_mask: np.ndarray
) -> pd.DataFrame:
    with Dataset(MODE_SIM) as dataset:
        years = np.asarray(dataset.variables["time"][:], dtype=int)
        allowed = ((years >= 1500) & (years <= 1900)) | ((years >= 1940) & (years <= 2000))
        indices = np.flatnonzero(allowed)
        if np.any((years[indices] >= LOCKED_START) & (years[indices] <= LOCKED_END)):
            raise RuntimeError("Locked ModE-Sim values requested")
        source_lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        source_lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        target_lat_2d, target_lon_2d = np.meshgrid(target_lat, target_lon, indexing="ij")
        points = np.column_stack([target_lat_2d[target_mask], target_lon_2d[target_mask]])
        weights = np.cos(np.deg2rad(target_lat_2d[target_mask]))
        values = []
        variable = dataset.variables["temp2_anomaly_1961_1990"]
        for index in indices:
            field = np.asarray(variable[int(index)], dtype=float)
            interpolator = RegularGridInterpolator(
                (source_lat, source_lon), field, bounds_error=False, fill_value=np.nan
            )
            target_values = interpolator(points)
            valid = np.isfinite(target_values)
            values.append(float(np.average(target_values[valid], weights=weights[valid])))
    return pd.DataFrame({"year": years[indices], "anomaly_degC": values})


def window_means(years: np.ndarray, values: np.ndarray) -> np.ndarray:
    means = []
    for start, end in WINDOWS:
        selected = (years >= start) & (years <= end)
        if int(selected.sum()) != 50:
            raise RuntimeError(f"Incomplete 50-year window: {start}-{end}")
        means.append(float(np.mean(values[selected])))
    return np.asarray(means)


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def running_mean(values: pd.Series, width: int = 31) -> pd.Series:
    return values.rolling(width, center=True, min_periods=width // 2 + 1).mean()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    contract = OUTPUT / "ROUND_CONTRACT.md"
    contract.write_text(
        "# PMIP Multi-Model Prior Audit Contract\n\n"
        "## Question\n\n"
        "Do two structurally independent official PMIP simulations agree that the current HMA "
        "background is too warm during 1500-1899?\n\n"
        "## Frozen design\n\n"
        "- Models: MPI-ESM1-2-LR past2k r1i1p1f1 and MIROC-ES2L past1000 r1i1p1f2, "
        "continued with the matching historical members.\n"
        "- Extract only tas on the frozen Round-115 HMA mask.\n"
        "- Use 1500-1900 and 1940-2000 only; never request climate values for 1901-1939.\n"
        "- Baseline every model to its own 1961-1990 HMA mean.\n"
        "- Compare frozen 50-year windows against the existing four-family anchor and current background.\n"
        "- Advance only if both new models are cooler than the current background in the LIA mean and "
        "both reduce 50-year RMSE against the anchor. This is a mechanism gate, not production authorization.\n\n"
        "## Read-only preview incident\n\n"
        "Before this contract was written, a fixed-line preview of kathmandu_temp-supp.txt accidentally "
        "displayed rows in 1901-1939. Those values are excluded from all calculations and conclusions here. "
        "No source file was modified. This driver requests zero locked-period climate values.\n",
        encoding="utf-8",
    )
    paths = {
        "grid_source": GRID_SOURCE,
        "mode_sim": MODE_SIM,
        "reference_curves": REFERENCE_CURVES,
        "prior_curves": PRIOR_CURVES,
        "facts": FACTS,
        "driver": DRIVER,
        "contract": contract,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    before = {name: sha256(path) for name, path in paths.items()}

    grid = np.load(GRID_SOURCE)
    target_lat = grid["grid_lat"].astype(float)
    target_lon = grid["grid_lon"].astype(float)
    target_mask = grid["hma_mask"].astype(bool)
    if int(target_mask.sum()) < 90:
        raise RuntimeError("Frozen HMA mask unexpectedly small")

    remote_catalog: list[dict[str, Any]] = []
    annual_models: dict[str, pd.DataFrame] = {}
    for source, config in MODEL_CONFIG.items():
        past_docs = query_esgf(source, config["past_experiment"], config["past_member"], "PMIP")
        historical_docs = query_esgf(source, "historical", config["historical_member"], "CMIP")
        past_files = select_remote_files(
            past_docs, int(config["past_year_offset"]), [(1500, 1849)]
        )
        historical_files = select_remote_files(
            historical_docs, 0, [(1850, 1900), (1940, 2000)]
        )
        past, past_audit = read_remote_series(
            past_files, [(1500, 1849)], target_lat, target_lon, target_mask
        )
        historical, historical_audit = read_remote_series(
            historical_files, [(1850, 1900), (1940, 2000)], target_lat, target_lon, target_mask
        )
        annual = pd.concat([past, historical], ignore_index=True).drop_duplicates("year").sort_values("year")
        baseline = annual.loc[
            annual["year"].between(BASELINE_START, BASELINE_END), "tas_K"
        ]
        if len(baseline) != 30:
            raise RuntimeError(f"Incomplete 1961-1990 baseline: {source}: {len(baseline)}")
        annual["anomaly_degC"] = annual["tas_K"] - float(baseline.mean())
        annual["model"] = source
        annual_models[source] = annual
        remote_catalog.extend(
            [{"model": source, "experiment_role": "past", **row} for row in past_audit]
            + [{"model": source, "experiment_role": "historical", **row} for row in historical_audit]
        )

    mode = read_mode_sim(target_lat, target_lon, target_mask)
    mode["model"] = "ModE-Sim"
    annual_models["ModE-Sim"] = mode

    references = pd.read_csv(REFERENCE_CURVES)
    prior = pd.read_csv(PRIOR_CURVES)
    prior_hma = prior[prior["domain"].eq("hma")].copy()
    merged = references[["year", "four_family_anchor_hma_degC", "project_hma_degC"]].merge(
        prior_hma[
            [
                "year",
                "background_six_member_degC",
                "posterior_six_member_degC",
            ]
        ],
        on="year",
        how="inner",
    )
    merged = merged[merged["year"].between(LIA_START, LIA_END)].copy()
    if len(merged) != 400:
        raise RuntimeError("Reference/prior LIA alignment failed")
    for model, frame in annual_models.items():
        values = frame.loc[frame["year"].between(LIA_START, LIA_END), ["year", "anomaly_degC"]]
        if len(values) != 400:
            raise RuntimeError(f"Incomplete model LIA series: {model}: {len(values)}")
        merged = merged.merge(values.rename(columns={"anomaly_degC": model}), on="year", how="left")
    merged.to_csv(OUTPUT / "annual_hma_temperature_curves_1500_1899.csv", index=False)

    anchor_windows = window_means(
        merged["year"].to_numpy(int), merged["four_family_anchor_hma_degC"].to_numpy(float)
    )
    background_windows = window_means(
        merged["year"].to_numpy(int), merged["background_six_member_degC"].to_numpy(float)
    )
    background_rmse = rmse(background_windows, anchor_windows)
    rows: list[dict[str, Any]] = []
    model_metrics: dict[str, dict[str, float | bool]] = {}
    for name in ["background_six_member_degC", "posterior_six_member_degC", *annual_models]:
        values = merged[name].to_numpy(float)
        means = window_means(merged["year"].to_numpy(int), values)
        current_rmse = rmse(means, anchor_windows)
        lia_gap = float(np.mean(values - merged["four_family_anchor_hma_degC"].to_numpy(float)))
        model_metrics[name] = {
            "lia_mean_anomaly_degC": float(np.mean(values)),
            "lia_mean_gap_vs_anchor_degC": lia_gap,
            "window_rmse_vs_anchor_degC": current_rmse,
            "window_rmse_improvement_vs_background": float(1.0 - current_rmse / background_rmse),
            "cooler_than_background_lia_mean": bool(
                np.mean(values) < np.mean(merged["background_six_member_degC"])
            ),
        }
        for index, (start, end) in enumerate(WINDOWS):
            rows.append(
                {
                    "series": name,
                    "start_year": start,
                    "end_year": end,
                    "mean_degC": float(means[index]),
                    "anchor_mean_degC": float(anchor_windows[index]),
                    "gap_vs_anchor_degC": float(means[index] - anchor_windows[index]),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT / "temperature_50year_window_comparison.csv", index=False)

    new_models = ["MPI-ESM1-2-LR", "MIROC-ES2L"]
    both_cooler = all(bool(model_metrics[name]["cooler_than_background_lia_mean"]) for name in new_models)
    both_improve = all(float(model_metrics[name]["window_rmse_improvement_vs_background"]) > 0 for name in new_models)
    advance = bool(both_cooler and both_improve)

    long_frames = []
    for name, frame in annual_models.items():
        long_frames.append(frame[["year", "anomaly_degC"]].assign(model=name))
    pd.concat(long_frames, ignore_index=True).to_csv(
        OUTPUT / "pmip_model_annual_hma_temperature_allowed_years.csv", index=False
    )
    (OUTPUT / "remote_source_catalog.json").write_text(
        json.dumps(remote_catalog, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    plot = merged.set_index("year")
    axes[0].plot(plot.index, running_mean(plot["four_family_anchor_hma_degC"]), color="#111111", linewidth=2.8, label="Four-family anchor")
    axes[0].plot(plot.index, running_mean(plot["background_six_member_degC"]), color="#D55E00", linewidth=2.2, label="Current background")
    colors = {"ModE-Sim": "#0072B2", "MPI-ESM1-2-LR": "#009E73", "MIROC-ES2L": "#CC79A7"}
    for name, color in colors.items():
        axes[0].plot(plot.index, running_mean(plot[name]), color=color, linewidth=1.8, label=name)
    axes[0].set(title="HMA low-frequency temperature: current background and independent forced simulations", ylabel="Temperature anomaly (degC)")
    axes[0].legend(frameon=False, ncol=3)

    window_table = pd.DataFrame(rows)
    centers = np.asarray([(start + end) / 2 for start, end in WINDOWS])
    for name, color in {"background_six_member_degC": "#D55E00", **colors}.items():
        subset = window_table[window_table["series"].eq(name)]
        axes[1].plot(centers, subset["gap_vs_anchor_degC"], marker="o", color=color, linewidth=1.8, label=name.replace("_six_member_degC", ""))
    axes[1].axhline(0, color="#111111", linewidth=1.0)
    axes[1].set(title="Frozen 50-year mean difference from four-family anchor", xlabel="Window center year", ylabel="Series minus anchor (degC)")
    axes[1].legend(frameon=False, ncol=4)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    figure = OUTPUT / "temperature_pmip_multimodel_prior_audit.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    after = {name: sha256(path) for name, path in paths.items()}
    unchanged = before == after
    summary = {
        "experiment_id": OUTPUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "frozen_hma_cell_count": int(target_mask.sum()),
        "locked_1901_1939_climate_values_requested": 0,
        "preview_incident_excluded_from_analysis": True,
        "source_hashes_unchanged": unchanged,
        "background_window_rmse_vs_anchor_degC": background_rmse,
        "series_metrics": model_metrics,
        "mechanism_gates": {
            "both_new_models_cooler_than_current_background": both_cooler,
            "both_new_models_reduce_50year_rmse_vs_anchor": both_improve,
        },
        "advance_multimodel_prior_candidate": advance,
        "decision": (
            "advance_multimodel_prior_candidate_design"
            if advance
            else "do_not_use_pmip_multimodel_mean_as_prior_correction"
        ),
        "interpretation_limit": (
            "The four-family anchor and PMIP simulations are not independent observations. "
            "This audit tests model-structural robustness of the low-frequency background diagnosis; "
            "it does not authorize a production correction."
        ),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# PMIP Multi-Model Temperature Prior Audit",
        "",
        f"- Current-background 50-year RMSE versus anchor: `{background_rmse:.3f} degC`.",
    ]
    for name in ["ModE-Sim", *new_models]:
        metrics = model_metrics[name]
        lines.append(
            f"- {name}: LIA gap `{float(metrics['lia_mean_gap_vs_anchor_degC']):+.3f} degC`; "
            f"50-year RMSE improvement `{100 * float(metrics['window_rmse_improvement_vs_background']):+.1f}%`."
        )
    lines.extend(
        [
            f"- Both new models cooler than current background: `{str(both_cooler).lower()}`.",
            f"- Both new models improve 50-year RMSE: `{str(both_improve).lower()}`.",
            "- Locked 1901-1939 climate values requested: `0`.",
            f"- Source hashes unchanged: `{str(unchanged).lower()}`.",
            "",
            f"Decision: `{summary['decision']}`.",
        ]
    )
    (OUTPUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not unchanged:
        raise RuntimeError("A source file changed")
    print(json.dumps(summary, indent=2))
    print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
