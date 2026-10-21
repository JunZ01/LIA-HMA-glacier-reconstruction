"""Run annual 0-300 year spin-up relaxation diagnostics for saved E2 chains."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .lia_elevation_band_spinup import _configure_oggm, _read_flowline
from .lia_lhs_single_chain import _snapshot
from .plot_spinup_geometry_v2 import _climate_suffix, _workdir_root


PROJECT_ROOT = Path(__file__).resolve().parents[4]
V0_ROOT = PROJECT_ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
DEFAULT_CHAIN_ROOT = V0_ROOT / "identified_experiments_v1" / "e2_window_chronological_v2"
DEFAULT_FLOWLINE_ROOT = V0_ROOT / "lia_elevation_band_core191_length_corrected_v1"
DEFAULT_OUTPUT_DIR = (
    V0_ROOT
    / "identified_experiments_v1"
    / "e3_spinup_relaxation_annual_v1"
)
DEFAULT_ANCHORS = {1044: "0048", 1545: "0142", 3584: "0003"}
DEFAULT_WINDOWS = ("1000_1049", "1272_1321", "1787_1836")


def _read_summary(chain_dir: Path) -> dict:
    path = chain_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"summary-missing:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _chain_dir(chain_root: Path, system_id: int, anchor: str, window: str) -> Path:
    return chain_root / f"system_{system_id}" / f"anchor_point_{anchor}" / f"window_{window}"


def _run_case(
    *,
    chain_dir: Path,
    flowline_root: Path,
    output_dir: Path,
    max_year: int,
) -> tuple[pd.DataFrame, dict]:
    from oggm import cfg, workflow
    from oggm.core.flowline import SemiImplicitModel
    from oggm.core.massbalance import MonthlyTIModel

    started = time.perf_counter()
    summary = _read_summary(chain_dir)
    system_id = int(summary["lia_unique_id"])
    rgi_id = str(summary["rgi_id"])
    spinup_start, spinup_end = map(int, summary["spinup_window"])

    case_dir = output_dir / f"system_{system_id}" / f"window_{spinup_start}_{spinup_end}"
    case_dir.mkdir(parents=True, exist_ok=True)

    _configure_oggm(_workdir_root(summary), case_dir)
    cfg.PARAMS["temp_melt"] = float(summary["parameters"]["temp_melt_c"])
    gdir = workflow.init_glacier_directories([rgi_id], reset=False)[0]
    flowline_path = flowline_root / str(system_id) / "elevation_band_flowline_lia.pkl.gz"
    flowline = _read_flowline(flowline_path)

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

    rows = []
    for year in range(0, max_year + 1):
        if year:
            model.run_until(float(year))
        row = _snapshot(model, "spinup", year)
        row.update(
            {
                "lia_unique_id": system_id,
                "rgi_id": rgi_id,
                "anchor_point": int(Path(chain_dir).parent.name.split("_")[-1]),
                "window_start": spinup_start,
                "window_end": spinup_end,
                "delta_t_c": float(summary["parameters"]["delta_t_c"]),
                "melt_f": float(summary["parameters"]["melt_f"]),
                "prcp_fac": float(summary["parameters"]["prcp_fac"]),
                "glen_a_pa_3_s_1": float(summary["parameters"]["glen_a_pa_3_s_1"]),
                "temp_melt_c": float(summary["parameters"]["temp_melt_c"]),
                "source_chain_dir": str(chain_dir),
            }
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(case_dir / "annual_timeseries_0_300.csv", index=False)

    minimum = frame.loc[frame["length_m"].astype(float).idxmin()]
    initial = frame.iloc[0]
    final = frame.iloc[-1]
    case_summary = {
        "status": "ok",
        "python_executable": sys.executable,
        "lia_unique_id": system_id,
        "rgi_id": rgi_id,
        "window": [spinup_start, spinup_end],
        "max_year": int(max_year),
        "source_chain_dir": str(chain_dir),
        "flowline_path": str(flowline_path),
        "annual_timeseries_csv": str(case_dir / "annual_timeseries_0_300.csv"),
        "metrics": {
            "initial_length_km": float(initial["length_m"]) / 1000.0,
            "minimum_length_year": int(minimum["year"]),
            "minimum_length_km": float(minimum["length_m"]) / 1000.0,
            "minimum_terminal_thickness_m": float(minimum["terminal_thickness_m"]),
            "year_300_length_km": float(final["length_m"]) / 1000.0,
            "regrowth_after_min_km": (
                float(final["length_m"]) - float(minimum["length_m"])
            )
            / 1000.0,
            "initial_volume_km3": float(initial["volume_m3"]) / 1e9,
            "minimum_year_volume_km3": float(minimum["volume_m3"]) / 1e9,
            "year_300_volume_km3": float(final["volume_m3"]) / 1e9,
            "year_300_terminal_thickness_m": float(final["terminal_thickness_m"]),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (case_dir / "summary.json").write_text(
        json.dumps(case_summary, indent=2),
        encoding="utf-8",
    )
    return frame, case_summary


def _plot_combined(frame: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, constrained_layout=True)
    specs = [
        ("length_m", 1000.0, "Length (km)"),
        ("volume_m3", 1e9, "Volume (km$^3$)"),
        ("terminal_thickness_m", 1.0, "Terminus thickness (m)"),
    ]
    systems = sorted(frame["lia_unique_id"].astype(int).unique())
    colors = {"1000-1049": "#2563eb", "1272-1321": "#059669", "1787-1836": "#b45309"}
    for col, system_id in enumerate(systems):
        part = frame[frame["lia_unique_id"].astype(int).eq(system_id)]
        for row, (column, scale, label) in enumerate(specs):
            axis = axes[row, col]
            for window, group in part.groupby("window_label", sort=True):
                axis.plot(
                    group["year"],
                    group[column] / scale,
                    lw=1.6,
                    color=colors.get(str(window), None),
                    label=str(window),
                )
                min_row = group.loc[group["length_m"].astype(float).idxmin()]
                if column == "length_m":
                    axis.scatter(
                        [float(min_row["year"])],
                        [float(min_row[column]) / scale],
                        s=24,
                        color=colors.get(str(window), None),
                        zorder=4,
                    )
            axis.grid(alpha=0.22)
            if col == 0:
                axis.set_ylabel(label)
            if row == 0:
                axis.set_title(f"LIA {system_id}")
            if row == 2:
                axis.set_xlabel("Spin-up year")
            if row == 0 and col == 2:
                axis.legend(frameon=False, fontsize=8, title="window")
    path = output_dir / "annual_relaxation_0_300_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-root", type=Path, default=DEFAULT_CHAIN_ROOT)
    parser.add_argument("--flowline-root", type=Path, default=DEFAULT_FLOWLINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-year", type=int, default=300)
    parser.add_argument("--windows", nargs="+", default=list(DEFAULT_WINDOWS))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    summaries = []
    for system_id, anchor in DEFAULT_ANCHORS.items():
        for window in args.windows:
            chain_dir = _chain_dir(args.chain_root, system_id, anchor, window)
            frame, summary = _run_case(
                chain_dir=chain_dir,
                flowline_root=args.flowline_root,
                output_dir=args.output_dir,
                max_year=args.max_year,
            )
            frame["window_label"] = f"{int(frame['window_start'].iloc[0])}-{int(frame['window_end'].iloc[0])}"
            frames.append(frame)
            summaries.append(summary)
            metrics = summary["metrics"]
            print(
                f"done:{system_id}:{metrics['minimum_length_year']}:"
                f"{metrics['minimum_length_km']:.3f}:"
                f"{metrics['year_300_length_km']:.3f}"
            )

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output_dir / "annual_timeseries_0_300_combined.csv", index=False)
    summary_rows = []
    for item in summaries:
        row = {
            "lia_unique_id": item["lia_unique_id"],
            "rgi_id": item["rgi_id"],
            "window_start": item["window"][0],
            "window_end": item["window"][1],
            **item["metrics"],
            "annual_timeseries_csv": item["annual_timeseries_csv"],
        }
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(args.output_dir / "annual_relaxation_summary.csv", index=False)
    figure = _plot_combined(combined, args.output_dir)
    manifest = {
        "status": "ok",
        "question": (
            "For three representative climate windows per system, test whether "
            "the 0-300 year spin-up length minimum is an early relaxation feature."
        ),
        "systems": sorted(DEFAULT_ANCHORS),
        "windows": args.windows,
        "max_year": args.max_year,
        "combined_csv": str(args.output_dir / "annual_timeseries_0_300_combined.csv"),
        "summary_csv": str(args.output_dir / "annual_relaxation_summary.csv"),
        "figure_png": str(figure),
        "case_count": len(summary_rows),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
