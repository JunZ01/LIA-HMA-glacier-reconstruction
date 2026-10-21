"""One-command entry point for reusable LIA plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import lia_glacier_diagnostic_bundle as diagnostic
from . import lia_spinup_curve_plot as spinup


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = PROJECT_ROOT / "validation_data" / "results" / "paleo_surface_validation_v6"
V0_ROOT = PROJECT_ROOT / "validation_data" / "results" / "paleo_surface_validation_v0"
FIRST10_FORWARD = V0_ROOT / "lia_first10_forward_v1"


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _spinup(args: argparse.Namespace) -> None:
    output = _project_path(args.output_dir) if args.output_dir else RESULTS_ROOT / "spinup_curves"
    output.mkdir(parents=True, exist_ok=True)
    series_list = spinup._load_selected(
        _project_path(args.status_csv),
        _project_path(args.candidates_csv),
        base_dir=PROJECT_ROOT,
    )
    if args.lia_id is not None:
        series_list = [s for s in series_list if int(s.iloc[0]["lia_unique_id"]) == args.lia_id]
    if not series_list:
        raise ValueError("no-selected-spinup-series")
    for series in series_list:
        spinup._plot_one(
            series,
            output / f"lia_{int(series.iloc[0]['lia_unique_id'])}_spinup_area_volume_length.png",
        )
    spinup._plot_overview(series_list, output / "spinup_area_volume_length_overview.png")
    print(f"plotted:{len(series_list)} output:{output}")


def _diagnostic(args: argparse.Namespace) -> None:
    output = _project_path(args.output_dir) if args.output_dir else RESULTS_ROOT / "quicklook" / f"lia_{args.lia_id}"
    namespace = argparse.Namespace(
        system_id=args.lia_id,
        chain_dir=_project_path(args.chain_dir),
        elevation_band_root=_project_path(args.elevation_band_root),
        grid_root=_project_path(args.grid_root),
        reference_csv=None if args.reference_csv is None else _project_path(args.reference_csv),
        output_dir=output,
    )
    print(diagnostic.run(namespace))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast reusable LIA plots")
    sub = parser.add_subparsers(dest="command", required=True)
    spin = sub.add_parser("spinup", help="plot selected spin-up area, volume, and length")
    spin.add_argument("--lia-id", type=int)
    spin.add_argument("--status-csv", type=Path, default=RESULTS_ROOT / "iteration_status.csv")
    spin.add_argument("--candidates-csv", type=Path, default=RESULTS_ROOT / "iteration_candidates.csv")
    spin.add_argument("--output-dir", type=Path)
    spin.set_defaults(handler=_spinup)
    diag = sub.add_parser("diagnostic", help="plot one glacier geometry and thickness")
    diag.add_argument("--lia-id", required=True, type=int)
    diag.add_argument("--chain-dir", type=Path, default=FIRST10_FORWARD)
    diag.add_argument("--elevation-band-root", type=Path, default=V0_ROOT / "lia_elevation_band_first10_buffer200km_v1")
    diag.add_argument("--grid-root", type=Path, default=V0_ROOT / "core191_anchor_reroute_f19_v1")
    diag.add_argument("--reference-csv", type=Path, default=FIRST10_FORWARD / "lia_first10_reference_lines.csv")
    diag.add_argument("--output-dir", type=Path)
    diag.set_defaults(handler=_diagnostic)
    args = parser.parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
