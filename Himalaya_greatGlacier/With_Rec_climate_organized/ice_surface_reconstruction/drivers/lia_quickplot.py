"""Fast, reusable LIA catalog and plotting commands.

This module reads saved geometry and result files only. It never initializes
OGGM or starts a simulation.
"""

from __future__ import annotations

import argparse
import json
import re
import types
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
V0_ROOT = PROJECT_ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
DEFAULT_LIA_SHP = PROJECT_ROOT / "Himalaya_greatGlacier" / "Him_10_lia_with_large.shp"
DEFAULT_TARGETS = V0_ROOT / "lia_design_b_targets_and_inputs_v1" / "lia_design_b_system_targets.csv"
DEFAULT_CATALOG = V0_ROOT / "quickplot_catalog_v1"
DEFAULT_RUNS = (
    V0_ROOT
    / "d_layer_lhs_D_v1"
    / "modern_mb_probe_v1"
    / "local_melt_search_v1_spinup1000"
)


def _ids(values: list[int] | None, catalog: pd.DataFrame) -> list[int]:
    if values:
        return [int(value) for value in values]
    return sorted(catalog["lia_unique_id"].astype(int).unique().tolist())


def _read_catalog(catalog_dir: Path) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    catalog_csv = catalog_dir / "catalog.csv"
    boundaries = catalog_dir / "boundaries.gpkg"
    if not catalog_csv.exists() or not boundaries.exists():
        raise FileNotFoundError(
            f"catalog-missing:{catalog_dir}; run `catalog` first"
        )
    return pd.read_csv(catalog_csv), gpd.read_file(boundaries, layer="lia_boundaries")


def build_catalog(lia_shp: Path, targets: Path, output: Path, runs_root: Path | None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    outlines = gpd.read_file(lia_shp)
    outlines["lia_unique_id"] = outlines["UniqueID"].astype(int)
    target_frame = pd.read_csv(targets)
    keep = [
        column
        for column in (
            "lia_unique_id",
            "rgi_id",
            "target_area_km2",
            "target_volume_km3",
            "target_main_flowline_length_m",
            "lia_target_year_ce",
        )
        if column in target_frame.columns
    ]
    catalog = target_frame[keep].copy()
    if "rgi_id" not in catalog:
        catalog["rgi_id"] = ""
    if "AllRGIs" in outlines.columns:
        members = outlines[["lia_unique_id", "AllRGIs"]].copy()
        members["rgi_member_count"] = members["AllRGIs"].fillna("").map(
            lambda value: len([part for part in re.split(r"[;,]", str(value)) if part.strip()])
        )
        catalog = catalog.merge(members[["lia_unique_id", "rgi_member_count"]], on="lia_unique_id", how="left")
    catalog["boundary_source"] = str(lia_shp)
    if runs_root is not None:
        run_rows = []
        for path in sorted(runs_root.rglob("timeseries.csv")):
            try:
                frame = pd.read_csv(path, nrows=1)
                uid = int(frame.iloc[0]["lia_unique_id"])
            except Exception:
                continue
            run_rows.append({"lia_unique_id": uid, "timeseries_path": str(path)})
        run_index = pd.DataFrame(run_rows).drop_duplicates("timeseries_path")
        if not run_index.empty:
            catalog = catalog.merge(
                run_index.groupby("lia_unique_id", as_index=False).agg(
                    available_timeseries=("timeseries_path", "count")
                ),
                on="lia_unique_id",
                how="left",
            )
    catalog.to_csv(output / "catalog.csv", index=False)
    outlines.to_file(output / "boundaries.gpkg", layer="lia_boundaries", driver="GPKG")
    manifest = {
        "status": "ok",
        "system_count": int(len(catalog)),
        "lia_source": str(lia_shp),
        "target_source": str(targets),
        "runs_root": None if runs_root is None else str(runs_root),
        "catalog_csv": "catalog.csv",
        "boundaries_gpkg": "boundaries.gpkg",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def plot_boundaries(catalog_dir: Path, ids: list[int] | None, output: Path) -> Path:
    catalog, boundaries = _read_catalog(catalog_dir)
    selected_ids = _ids(ids, catalog)
    selected = boundaries[boundaries["lia_unique_id"].astype(int).isin(selected_ids)].copy()
    if len(selected) != len(selected_ids):
        raise ValueError("boundary-id-missing")
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, len(selected_ids), height_ratios=[1, 1.05])
    overview = fig.add_subplot(grid[0, :])
    selected.plot(ax=overview, facecolor="#8ecae6", edgecolor="#023047", linewidth=1.1)
    for row in selected.itertuples():
        point = row.geometry.representative_point()
        overview.annotate(str(int(row.lia_unique_id)), (point.x, point.y), ha="center", va="center", fontsize=9)
    overview.set_title("Selected LIA system boundaries")
    overview.set_aspect("equal")
    overview.set_axis_off()
    details = [fig.add_subplot(grid[1, index]) for index in range(len(selected_ids))]
    for axis, uid in zip(details, selected_ids):
        row = selected[selected["lia_unique_id"].astype(int).eq(uid)].iloc[0]
        gpd.GeoSeries([row.geometry], crs=selected.crs).plot(
            ax=axis, facecolor="#8ecae6", edgecolor="#023047", linewidth=1.2
        )
        target = catalog[catalog["lia_unique_id"].astype(int).eq(uid)].iloc[0]
        area = target.get("target_area_km2", np.nan)
        length = target.get("target_main_flowline_length_m", np.nan)
        axis.set_title(f"LIA {uid}\narea={area:.1f} km²; length={length / 1000:.2f} km")
        axis.set_aspect("equal")
        axis.set_axis_off()
    fig.suptitle("LIA boundaries used by the calibration pilot", fontsize=15)
    path = output / ("lia_boundaries_" + "_".join(map(str, selected_ids)) + ".png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _load_run_series(runs_root: Path, uid: int) -> list[pd.DataFrame]:
    result = []
    for path in sorted(runs_root.rglob("timeseries.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "lia_unique_id" not in frame or not frame["lia_unique_id"].astype(int).eq(uid).any():
            continue
        frame = frame[frame["lia_unique_id"].astype(int).eq(uid)].copy()
        frame["source_path"] = str(path)
        result.append(frame)
    return result


def _display_year(frame: pd.DataFrame) -> np.ndarray:
    spin = frame[frame["phase"].eq("spinup")]
    forward = frame[frame["phase"].eq("forward")]
    if spin.empty or forward.empty:
        return frame["year"].to_numpy(float)
    spin_end = int(spin["year"].max())
    forward_start = int(forward["year"].min())
    values = frame["year"].to_numpy(float).copy()
    spin_mask = frame["phase"].eq("spinup").to_numpy()
    values[spin_mask] = forward_start - spin_end + values[spin_mask]
    return values


def plot_timeseries(catalog_dir: Path, runs_root: Path, ids: list[int] | None, output: Path) -> list[Path]:
    catalog, _ = _read_catalog(catalog_dir)
    selected_ids = _ids(ids, catalog)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for uid in selected_ids:
        series = _load_run_series(runs_root, uid)
        if not series:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
        specs = [("length_m", 1000.0, "Length (km)"), ("area_m2", 1e6, "Area (km²)"), ("volume_m3", 1e9, "Volume (km³)")]
        for frame in series:
            label = str(frame["melt_f"].iloc[0]) if "melt_f" in frame else Path(frame["source_path"].iloc[0]).parent.name
            x = _display_year(frame)
            for axis, (column, scale, ylabel) in zip(axes, specs):
                axis.plot(x, frame[column] / scale, marker="o", ms=2.5, lw=1.5, label=f"melt_f={label}")
                axis.set_ylabel(ylabel)
                axis.grid(alpha=0.2)
        target = catalog[catalog["lia_unique_id"].astype(int).eq(uid)]
        if not target.empty:
            lia_length = float(target.iloc[0]["target_main_flowline_length_m"])
            axes[0].axhline(lia_length / 1000.0, color="#444", ls="--", lw=1.0, label="LIA target")
        axes[0].set_title(f"LIA {uid} | length")
        axes[1].set_title(f"LIA {uid} | area")
        axes[2].set_title(f"LIA {uid} | volume")
        axes[0].set_xlabel("Display year")
        axes[1].set_xlabel("Display year")
        axes[2].set_xlabel("Display year")
        axes[0].legend(fontsize=8, frameon=False)
        path = output / f"lia_{uid}_timeseries.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    if not paths:
        raise FileNotFoundError(f"no-timeseries-for-ids:{selected_ids}")
    return paths


def run_diagnostic(
    system_id: int,
    chain_dir: Path,
    elevation_band_root: Path,
    grid_root: Path,
    reference_csv: Path | None,
    output: Path,
) -> dict:
    # Import lazily so boundary/time-series plots never load OGGM-related
    # diagnostic dependencies.
    import lia_glacier_diagnostic_bundle as diagnostic

    namespace = types.SimpleNamespace(
        system_id=system_id,
        chain_dir=chain_dir,
        elevation_band_root=elevation_band_root,
        grid_root=grid_root,
        reference_csv=reference_csv,
        output_dir=output,
    )
    return diagnostic.run(namespace)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast LIA plotting without OGGM startup")
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog", help="build the cached geometry/result catalog")
    catalog.add_argument("--lia-shp", type=Path, default=DEFAULT_LIA_SHP)
    catalog.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    catalog.add_argument("--output", type=Path, default=DEFAULT_CATALOG)
    catalog.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)

    boundaries = sub.add_parser("boundaries", help="plot LIA boundaries")
    boundaries.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    boundaries.add_argument("--ids", type=int, nargs="+")
    boundaries.add_argument("--output", type=Path, default=V0_ROOT / "quickplots" / "boundaries")

    timeseries = sub.add_parser("timeseries", help="plot saved area/volume/length series")
    timeseries.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    timeseries.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    timeseries.add_argument("--ids", type=int, nargs="+")
    timeseries.add_argument("--output", type=Path, default=V0_ROOT / "quickplots" / "timeseries")

    diagnostic = sub.add_parser("diagnostic", help="plot flowline, 2D thickness, and 3D surface")
    diagnostic.add_argument("--id", dest="system_id", required=True, type=int)
    diagnostic.add_argument("--chain-dir", type=Path, required=True)
    diagnostic.add_argument("--elevation-band-root", type=Path, required=True)
    diagnostic.add_argument("--grid-root", type=Path, required=True)
    diagnostic.add_argument("--reference-csv", type=Path)
    diagnostic.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "catalog":
        print(json.dumps(build_catalog(args.lia_shp, args.targets, args.output, args.runs_root), indent=2))
    elif args.command == "boundaries":
        print(plot_boundaries(args.catalog, args.ids, args.output))
    elif args.command == "timeseries":
        for path in plot_timeseries(args.catalog, args.runs_root, args.ids, args.output):
            print(path)
    elif args.command == "diagnostic":
        print(json.dumps(run_diagnostic(
            args.system_id,
            args.chain_dir,
            args.elevation_band_root,
            args.grid_root,
            args.reference_csv,
            args.output,
        ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
