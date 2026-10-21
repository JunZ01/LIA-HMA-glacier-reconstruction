"""Create a reusable diagnostic bundle for one LIA glacier system."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def _flowline_object(path: Path):
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if isinstance(payload, (list, tuple)):
        return payload[0]
    return payload


def _state_object(path: Path):
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if isinstance(payload, (list, tuple)):
        return payload[0]
    return payload


def _profile(obj) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = float(getattr(obj, "dx_meter", getattr(obj, "map_dx", 1.0)))
    bed = np.asarray(obj.bed_h, dtype=float)
    surface = np.asarray(obj.surface_h, dtype=float)
    thick = np.asarray(obj.thick, dtype=float)
    distance = np.arange(len(bed), dtype=float) * dx
    return distance, bed, surface, thick


def _reference(summary: pd.DataFrame, unique_id: int) -> dict:
    row = summary[summary["lia_unique_id"].astype(int).eq(unique_id)].iloc[0]
    volume = float(row["forward_2000_volume_m3"])  # overwritten below when possible
    # The spin-up relative error is defined against the initial LIA flowline.
    return {
        "area_m2": float(row["forward_2000_area_m2"]),
        "volume_m3": volume,
        "length_m": float(row["forward_2000_length_m"]),
    }


def _plot_timeseries(
    chain_dir: Path,
    unique_id: int,
    out: Path,
    reference_row: dict | None = None,
) -> None:
    spin = pd.read_csv(chain_dir / "selected_spinup_timeseries.csv")
    forward = pd.read_csv(chain_dir / "forward_1500_2000_timeseries.csv")
    spin = spin[spin["lia_unique_id"].astype(int).eq(unique_id)].copy()
    forward = forward[forward["lia_unique_id"].astype(int).eq(unique_id)].copy()
    reference_row = reference_row or {}
    # LIA area/length can be supplied from the observed LIA outline. Do not
    # infer a LIA volume line from the geometric inversion.
    lia = {
        "area_m2": reference_row.get("lia_area_m2"),
        "length_m": reference_row.get("lia_length_m"),
    }
    rgi = {
        "area_m2": reference_row.get("rgi_area_m2"),
        "volume_m3": reference_row.get("rgi_volume_m3"),
        "length_m": reference_row.get("rgi_length_m"),
    }
    length_mode = str(reference_row.get("length_reference_mode", "model flowline scale"))
    units = {"area_m2": (1e6, "Area (km$^2$)"), "volume_m3": (1e9, "Volume (km$^3$)"), "length_m": (1e3, "Length (km)")}
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for i, metric in enumerate(("area_m2", "volume_m3", "length_m")):
        scale, label = units[metric]
        for ax, frame, title in (
            (axes[i, 0], spin, "Spin-up"),
            (axes[i, 1], forward, "Forward 1500-2000"),
        ):
            if frame.empty:
                ax.set_visible(False)
                continue
            year = frame["year"].to_numpy(float)
            ax.plot(year, frame[metric].to_numpy(float) / scale, color="#2563eb", linewidth=1.4, label="simulation")
            if lia.get(metric) is not None and np.isfinite(float(lia[metric])):
                ax.axhline(float(lia[metric]) / scale, color="#111827", linestyle="--", linewidth=0.9, label="LIA reference")
            # RGI volume is only drawn when an independent reference is supplied.
            if rgi.get(metric) is not None and np.isfinite(float(rgi[metric])):
                ax.axhline(float(rgi[metric]) / scale, color="#dc2626", linestyle=":", linewidth=1.0, label="RGI reference")
            ax.set_ylabel(label)
            ax.set_title(title if metric != "length_m" else f"{title} ({length_mode})")
            ax.grid(alpha=0.2)
            if title == "Forward 1500-2000":
                ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"LIA {unique_id} | area, volume, length diagnostics")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_flowline(chain_dir: Path, elevation_band_root: Path, unique_id: int, out: Path) -> None:
    base = _flowline_object(
        elevation_band_root / str(unique_id) / "elevation_band_flowline_lia.pkl.gz"
    )
    candidates = list(chain_dir.glob(f"lia{unique_id}_*_1500_state.pkl.gz"))
    states = [("LIA geometry", base)]
    if candidates:
        states.append(("1500 state", _state_object(candidates[0])))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for title, obj in states:
        distance, bed, surface, thick = _profile(obj)
        ax1.plot(distance / 1000.0, surface, label=title)
        if title == "LIA geometry":
            ax1.plot(distance / 1000.0, bed, color="#4b5563", linestyle="--", label="bed")
        ax2.plot(distance / 1000.0, thick, label=title)
    ax1.set_ylabel("Elevation (m)")
    ax2.set_ylabel("Thickness (m)")
    ax2.set_xlabel("Distance from head (km)")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    ax1.grid(alpha=0.2)
    ax2.grid(alpha=0.2)
    fig.suptitle(f"LIA {unique_id} | flowline geometry and thickness")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _find_grid(grid_root: Path, unique_id: int) -> Path | None:
    matches = sorted(grid_root.rglob(f"lia_{unique_id}_*_system_grid_v1.nc"))
    return matches[0] if matches else None


def _plot_grid(netcdf: Path, unique_id: int, out2d: Path, out3d: Path) -> dict:
    with xr.open_dataset(netcdf) as ds:
        thickness = np.asarray(ds["thickness_m"].isel(member=0), dtype=float)
        surface = np.asarray(ds["surface_elevation_m"].isel(member=0), dtype=float)
        valid = np.asarray(ds["valid_mask"].isel(member=0), dtype=bool)
        x = np.asarray(ds["x"], dtype=float)
        y = np.asarray(ds["y"], dtype=float)
        shown = np.where(valid, thickness, np.nan)
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        image = ax.pcolormesh(x, y, shown, shading="auto", cmap="viridis")
        ax.contour(x, y, valid.astype(float), levels=[0.5], colors="black", linewidths=0.7)
        ax.set_aspect("equal")
        ax.set_title(f"LIA {unique_id} | 2D ice thickness (m)")
        fig.colorbar(image, ax=ax, label="Thickness (m)")
        fig.savefig(out2d, dpi=170)
        plt.close(fig)
        stride = max(1, int(max(surface.shape) / 100))
        xx, yy = np.meshgrid(x[::stride], y[::stride])
        zz = np.where(valid[::stride, ::stride], surface[::stride, ::stride], np.nan)
        cc = np.where(valid[::stride, ::stride], thickness[::stride, ::stride], np.nan)
        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        plot = ax.plot_surface(xx, yy, zz, facecolors=plt.cm.viridis(
            np.nan_to_num(cc / max(float(np.nanmax(cc)), 1.0))
        ), linewidth=0, antialiased=True, shade=False)
        ax.set_title(f"LIA {unique_id} | 3D surface colored by thickness")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("Surface elevation (m)")
        fig.savefig(out3d, dpi=160)
        plt.close(fig)
        return {
            "grid_file": str(netcdf),
            "valid_cells": int(valid.sum()),
            "volume_km3": float(np.nansum(shown) * abs(float(np.median(np.diff(x)))) ** 2 / 1e9),
        }


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_row = None
    if args.reference_csv is not None:
        references = pd.read_csv(args.reference_csv)
        matches = references[references["lia_unique_id"].astype(int).eq(args.system_id)]
        if len(matches) != 1:
            raise ValueError(f"reference-row-count:{args.system_id}:{len(matches)}")
        reference_row = matches.iloc[0].to_dict()
    _plot_timeseries(
        args.chain_dir,
        args.system_id,
        args.output_dir / f"lia_{args.system_id}_area_volume_length.png",
        reference_row=reference_row,
    )
    flowline_error = None
    try:
        _plot_flowline(
            args.chain_dir,
            args.elevation_band_root,
            args.system_id,
            args.output_dir / f"lia_{args.system_id}_flowline_profile.png",
        )
    except Exception as exc:  # optional: unpickling needs the OGGM runtime
        flowline_error = f"{type(exc).__name__}: {exc}"
    grid = _find_grid(args.grid_root, args.system_id)
    grid_metrics = None
    grid_error = None
    if grid is not None:
        try:
            grid_metrics = _plot_grid(
                grid,
                args.system_id,
                args.output_dir / f"lia_{args.system_id}_thickness_2d.png",
                args.output_dir / f"lia_{args.system_id}_thickness_3d.png",
            )
        except Exception as exc:  # optional: NetCDF backends vary by environment
            grid_error = f"{type(exc).__name__}: {exc}"
    manifest = {
        "status": "completed",
        "lia_unique_id": args.system_id,
        "timeseries_plot": f"lia_{args.system_id}_area_volume_length.png",
        "flowline_plot": None if flowline_error else f"lia_{args.system_id}_flowline_profile.png",
        "flowline_error": flowline_error,
        "grid_plot_2d": None if grid is None else f"lia_{args.system_id}_thickness_2d.png",
        "grid_plot_3d": None if grid is None else f"lia_{args.system_id}_thickness_3d.png",
        "grid_metrics": grid_metrics,
        "grid_error": grid_error,
        "dynamic_2d_note": "OGGM elevation-band run states are represented by the flowline profile; the 2D/3D raster is the LIA system-grid reconstruction.",
    }
    (args.output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-id", required=True, type=int)
    parser.add_argument("--chain-dir", required=True, type=Path)
    parser.add_argument("--elevation-band-root", required=True, type=Path)
    parser.add_argument("--grid-root", required=True, type=Path)
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
