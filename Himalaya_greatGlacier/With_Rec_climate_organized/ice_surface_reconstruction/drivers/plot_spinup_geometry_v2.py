"""Replay and plot the first years of a saved LIA spin-up geometry."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .lia_elevation_band_spinup import _configure_oggm, _read_flowline


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FLOWLINE_ROOT = (
    PROJECT_ROOT
    / "validation_data"
    / "results"
    / "paleo_surface_validation_v0"
    / "lia_elevation_band_core191_length_corrected_v1"
)


def _load_summary(chain_dir: Path) -> tuple[Path, dict]:
    summary_path = chain_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary-missing:{summary_path}")
    return summary_path, json.loads(summary_path.read_text(encoding="utf-8"))


def _climate_suffix(summary: dict) -> str:
    climate_file = Path(summary["climate"]["spinup"]["climate_file"])
    stem = climate_file.stem
    if not stem.startswith("gcm_data"):
        raise ValueError(f"unexpected-climate-file:{climate_file}")
    return stem[len("gcm_data"):]


def _netcdf_attr(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def _workdir_root(summary: dict) -> Path:
    return Path(summary["climate"]["spinup"]["climate_file"]).parents[4]


def _parse_years(text: str | None, spinup_years: int) -> list[int]:
    if text is None or not text.strip():
        years = [0, 1, 5, 10, 20, 40, 60, 80, spinup_years]
    else:
        years = [int(part.strip()) for part in text.split(",") if part.strip()]
    years = sorted(dict.fromkeys(years))
    if years[0] != 0:
        years.insert(0, 0)
    if years[-1] != spinup_years:
        years.append(spinup_years)
    return sorted(dict.fromkeys(years))


def _replay_spinup(
    *,
    summary_path: Path,
    summary: dict,
    flowline_root: Path,
    output_dir: Path,
    selected_years: list[int],
) -> tuple[xr.Dataset, Path, float]:
    _configure_oggm(_workdir_root(summary), output_dir)
    from oggm import cfg, workflow
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    cfg.PARAMS["temp_melt"] = float(summary["parameters"]["temp_melt_c"])
    rgi_id = str(summary["rgi_id"])
    system_id = int(summary["lia_unique_id"])
    flowline_path = flowline_root / str(system_id) / "elevation_band_flowline_lia.pkl.gz"
    flowline = _read_flowline(flowline_path)
    flowline_dx_m = float(flowline.dx_meter)
    gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]

    spinup_start, spinup_end = map(int, summary["spinup_window"])
    mb_model = MonthlyTIModel(
        gdir,
        filename="gcm_data",
        input_filesuffix=_climate_suffix(summary),
        melt_f=float(summary["parameters"]["melt_f"]),
        temp_bias=float(summary["parameters"]["base_temp_bias_c"])
        + float(summary["parameters"]["delta_t_c"]),
        prcp_fac=float(summary["parameters"]["prcp_fac"]),
        ys=spinup_start,
        ye=spinup_end,
        repeat=True,
        check_calib_params=False,
    )
    model = SemiImplicitModel(
        [copy.deepcopy(flowline)],
        mb_model=mb_model,
        y0=0.0,
        glen_a=float(summary["parameters"]["glen_a_pa_3_s_1"]),
        fs=0.0,
        inplace=False,
        check_for_boundaries=True,
        fixed_dt=1.25 * cfg.SEC_IN_DAY,
    )
    diag_ds, geom_ds = model.run_until_and_store(int(summary["spinup_years"]), geom_path=None)
    if geom_ds is None or len(geom_ds) != 1:
        raise RuntimeError(f"unexpected-geometry-output:{None if geom_ds is None else len(geom_ds)}")

    geom = geom_ds[0]
    x_km = np.arange(flowline.nx, dtype=float) * flowline_dx_m / 1000.0
    geom = geom.assign_coords(x_km=("x", x_km))
    geom["bed_h_m"] = ("x", np.asarray(flowline.bed_h, dtype=float))
    geom["initial_surface_h_m"] = ("x", np.asarray(flowline.surface_h, dtype=float))
    geom["thickness_m"] = xr.where(
        (geom["ts_section"] > 0) & (geom["ts_width_m"] > 0),
        geom["ts_section"] / geom["ts_width_m"],
        np.nan,
    )
    geom["surface_h_m"] = geom["bed_h_m"] + geom["thickness_m"]

    replay = diag_ds.copy()
    for name in geom.data_vars:
        replay[name] = geom[name]

    replay.attrs.update(
        {
            "description": "LIA spin-up geometry replay",
            "lia_unique_id": system_id,
            "rgi_id": rgi_id,
            "spinup_window": f"{spinup_start}-{spinup_end}",
            "spinup_years": int(summary["spinup_years"]),
            "source_summary": str(summary_path.resolve()),
            "selected_years": selected_years,
        }
    )
    replay.attrs = {key: _netcdf_attr(value) for key, value in replay.attrs.items()}
    return replay, flowline_path, flowline_dx_m


def _selected_year_rows(
    replay: xr.Dataset,
    years: list[int],
    flowline_dx_m: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for year in years:
        snapshot = replay.sel(time=year)
        thickness = np.asarray(snapshot["thickness_m"].values, dtype=float)
        positive = np.isfinite(thickness) & (thickness > 0)
        positive_indices = np.flatnonzero(positive)
        if positive_indices.size:
            last_ice_index = int(positive_indices[-1])
            terminal_thickness = float(thickness[last_ice_index])
            min_positive_thickness = float(np.min(thickness[positive]))
        else:
            last_ice_index = -1
            terminal_thickness = np.nan
            min_positive_thickness = np.nan
        rows.append(
            {
                "year": int(year),
                "area_m2": float(snapshot["area_m2"]),
                "volume_m3": float(snapshot["volume_m3"]),
                "length_m": float(snapshot["length_m"]),
                "ice_node_count": int(positive.sum()),
                "last_ice_index": last_ice_index,
                "length_gt_1m": float(np.count_nonzero(thickness > 1.0) * flowline_dx_m),
                "length_gt_5m": float(np.count_nonzero(thickness > 5.0) * flowline_dx_m),
                "length_gt_10m": float(np.count_nonzero(thickness > 10.0) * flowline_dx_m),
                "terminal_thickness_m": terminal_thickness,
                "minimum_positive_thickness_m": min_positive_thickness,
                "minimum_all_nodes_thickness_m": float(np.nanmin(thickness)),
                "maximum_thickness_m": float(np.nanmax(thickness)),
            }
        )
    return pd.DataFrame(rows)


def _plot_profiles(
    replay: xr.Dataset,
    years: list[int],
    output_path: Path,
) -> None:
    selected = replay.sel(time=years)
    max_length_km = float(np.nanmax(selected["length_m"].values)) / 1000.0
    x = np.asarray(replay["x_km"].values, dtype=float)
    x_mask = x <= (max_length_km + 2.0)
    x = x[x_mask]
    bed = np.asarray(replay["bed_h_m"].values, dtype=float)[x_mask]
    colors = plt.cm.cividis(np.linspace(0.08, 0.96, len(years)))

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), sharex=True, constrained_layout=True)
    axes[0].plot(x, bed / 1000.0, color="black", lw=1.4, ls="--", label="bed")

    final_year = years[-1]
    for color, year in zip(colors, years):
        snapshot = replay.sel(time=year)
        surface = np.asarray(snapshot["surface_h_m"].values, dtype=float)[x_mask]
        thickness = np.asarray(snapshot["thickness_m"].values, dtype=float)[x_mask]
        linewidth = 2.0 if year in {0, final_year} else 1.15
        axes[0].plot(x, surface / 1000.0, color=color, lw=linewidth, label=f"{year}")
        axes[1].plot(x, thickness, color=color, lw=linewidth, label=f"{year}")

    axes[0].set_ylabel("Elevation (km)")
    axes[1].set_ylabel("Thickness (m)")
    axes[1].set_xlabel("Distance along flowline (km)")
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    axes[0].legend(ncol=3, fontsize=8, frameon=False, title="year")

    l0 = float(replay["length_m"].sel(time=0))
    lf = float(replay["length_m"].sel(time=final_year))
    uid = int(replay.attrs["lia_unique_id"])
    window = replay.attrs["spinup_window"]
    axes[0].set_title(
        f"LIA {uid} | spin-up window {window} | length {l0 / 1000.0:.2f} -> {lf / 1000.0:.2f} km"
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-dir", required=True, type=Path)
    parser.add_argument("--flowline-root", type=Path, default=DEFAULT_FLOWLINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--years",
        default=None,
        help="Comma-separated years, e.g. 0,1,5,10,20,40,60,80,100",
    )
    args = parser.parse_args()

    chain_dir = args.chain_dir
    output_dir = args.output_dir or (chain_dir / "spinup_geometry_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path, summary = _load_summary(chain_dir)
    selected_years = _parse_years(args.years, int(summary["spinup_years"]))
    replay, flowline_path, flowline_dx_m = _replay_spinup(
        summary_path=summary_path,
        summary=summary,
        flowline_root=args.flowline_root,
        output_dir=output_dir,
        selected_years=selected_years,
    )
    year_table = _selected_year_rows(replay, selected_years, flowline_dx_m)

    geometry_path = output_dir / "spinup_geometry.nc"
    table_path = output_dir / "spinup_geometry_selected_years.csv"
    figure_path = output_dir / "spinup_geometry_profiles.png"
    manifest_path = output_dir / "spinup_geometry_manifest.json"

    replay.to_netcdf(geometry_path)
    year_table.to_csv(table_path, index=False)
    _plot_profiles(replay, selected_years, figure_path)

    manifest = {
        "status": "ok",
        "chain_dir": str(chain_dir),
        "summary_json": str(summary_path),
        "flowline_path": str(flowline_path),
        "output_dir": str(output_dir),
        "geometry_nc": str(geometry_path),
        "selected_years_csv": str(table_path),
        "figure_png": str(figure_path),
        "selected_years": selected_years,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
