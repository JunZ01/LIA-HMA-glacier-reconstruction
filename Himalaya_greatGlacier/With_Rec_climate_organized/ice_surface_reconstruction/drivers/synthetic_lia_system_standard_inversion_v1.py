"""Run the gated standard OGGM inversion for synthetic LIA system 3850."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..paths import resolve
from .lia_dynamic_matrix_pilot import load_source_state
from .lia_first10_spinup_batch import _write_case_climate
from .synthetic_lia_system_geometry_preflight_v1 import _gdir_path, _topology_metrics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure(output_root: Path, config: dict) -> str:
    import oggm
    from oggm import cfg

    cfg.initialize(logging_level="WARNING")
    cfg.PATHS["working_dir"] = str(output_root / "synthetic_gdir")
    cfg.PATHS["dl_cache_dir"] = str(output_root / "oggm_cache")
    cfg.PATHS["tmp_dir"] = str(output_root / "oggm_tmp")
    cfg.PARAMS["use_intersects"] = False
    cfg.PARAMS["use_tar_shapefiles"] = False
    cfg.PARAMS["use_multiple_flowlines"] = True
    cfg.PARAMS["filter_min_slope"] = True
    cfg.PARAMS["min_slope_flowline_filter"] = 1.5
    cfg.PARAMS["flowline_dx"] = float(config["flowline_dx"])
    cfg.PARAMS["temp_melt"] = float(config["stage2"]["temp_melt_c"])
    cfg.PARAMS["use_multiprocessing"] = False
    cfg.PARAMS["auto_skip_task"] = False
    cfg.PARAMS["continue_on_error"] = False
    return str(oggm.__version__)


def _copy_geometry(config: dict, output_root: Path) -> Path:
    source_base = resolve(config["geometry_gdir_base"])
    target_base = output_root / "synthetic_gdir"
    if target_base.exists():
        raise FileExistsError(f"output-gdir-already-exists:{target_base}")
    shutil.copytree(source_base, target_base)
    return target_base


def _hypsometry_error(source, flowlines, bin_m: float = 100.0) -> float:
    source_mask = (
        np.asarray(source.valid_mask, dtype=bool)
        & np.asarray(source.lia_mask, dtype=bool)
        & (np.asarray(source.thickness_m, dtype=float) > 1e-6)
    )
    source_z = np.asarray(source.surface_h_m, dtype=float)[source_mask]
    source_w = np.full(source_z.shape, float(source.cell_area_m2), dtype=float)
    model_z = []
    model_w = []
    for flowline in flowlines:
        thick = np.asarray(flowline.thick, dtype=float)
        active = thick > 0
        if not np.any(active):
            continue
        model_z.append(np.asarray(flowline.surface_h, dtype=float)[active])
        model_w.append(
            np.asarray(flowline.widths_m, dtype=float)[active]
            * float(flowline.dx_meter)
        )
    model_z = np.concatenate(model_z)
    model_w = np.concatenate(model_w)
    low = np.floor(min(float(source_z.min()), float(model_z.min())) / bin_m) * bin_m
    high = np.ceil(max(float(source_z.max()), float(model_z.max())) / bin_m) * bin_m
    edges = np.arange(low, high + 2 * bin_m, bin_m)
    source_hist = np.histogram(source_z, bins=edges, weights=source_w)[0]
    model_hist = np.histogram(model_z, bins=edges, weights=model_w)[0]
    source_cdf = np.cumsum(source_hist) / np.sum(source_hist)
    model_cdf = np.cumsum(model_hist) / np.sum(model_hist)
    return float(np.max(np.abs(source_cdf - model_cdf)))


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "config_snapshot.json", config)
    summary = {
        "status": "inversion-technical-fail",
        "experiment": config["experiment"],
        "system_id": int(config["system_id"]),
        "geometry_gate_verified": False,
        "apparent_mb_run": False,
        "inversion_run": False,
        "model_flowlines_initialized": False,
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
    }
    try:
        geometry_summary_path = resolve(config["geometry_summary"])
        geometry_summary = json.loads(
            geometry_summary_path.read_text(encoding="utf-8")
        )
        if geometry_summary.get("status") != "geometry-pass":
            raise RuntimeError("registered-geometry-not-pass")
        summary["geometry_gate_verified"] = True
        source_hashes = {
            str(geometry_summary_path): _sha256(geometry_summary_path),
            str(resolve(config["paired_grid"])): _sha256(resolve(config["paired_grid"])),
        }
        target_base = _copy_geometry(config, output_root)
        oggm_version = _configure(output_root, config)
        summary["oggm_version"] = oggm_version

        from oggm import cfg, tasks, utils
        from oggm.core.massbalance import MonthlyTIModel

        gdir = utils.GlacierDirectory(
            str(config["synthetic_rgi_id"]), base_dir=target_base, reset=False
        )
        source_gdir = _gdir_path(
            resolve(config["source_gdir_root"]), str(config["main_rgi_id"])
        )
        climate_source = source_gdir / "climate_historical.nc"
        climate_target = Path(gdir.get_filepath("climate_historical"))
        shutil.copy2(climate_source, climate_target)

        pixels = pd.read_csv(resolve(config["climate_pixels"]))
        rows = pixels[
            pixels["rgi_id"].astype(str).eq(str(config["main_rgi_id"]))
            & pixels["climate_id"].astype(str).eq(str(config["climate_id"]))
        ]
        if len(rows) != 1:
            raise ValueError(f"climate-pixel-row-count:{len(rows)}")
        start_year, end_year = [int(value) for value in config["spinup_window"]]
        climate_suffix, climate_audit = _write_case_climate(
            gdir,
            Path(config["reconstruction_climate"]),
            rows.iloc[0],
            "off",
            start_year=start_year,
            end_year=end_year,
        )
        _write_json(output_root / "climate_audit.json", climate_audit)

        stage2 = config["stage2"]
        total_temp_bias = float(stage2["base_temp_bias_c"]) + float(
            stage2["delta_t_c"]
        )
        mb_model = MonthlyTIModel(
            gdir,
            filename="gcm_data",
            input_filesuffix=climate_suffix,
            melt_f=float(stage2["melt_f"]),
            prcp_fac=float(stage2["prcp_fac"]),
            temp_bias=total_temp_bias,
            ys=start_year,
            ye=end_year,
            repeat=True,
            check_calib_params=False,
        )
        mb_years = np.arange(start_year, end_year + 1, dtype=int)
        tasks.apparent_mb_from_any_mb(gdir, mb_model=mb_model, mb_years=mb_years)
        summary["apparent_mb_run"] = True
        tasks.prepare_for_inversion(gdir)
        volume_m3 = tasks.mass_conservation_inversion(
            gdir,
            glen_a=float(stage2["glen_a"]),
            fs=float(stage2["fs"]),
        )
        tasks.filter_inversion_output(gdir)
        summary["inversion_run"] = True
        tasks.init_present_time_glacier(gdir)
        summary["model_flowlines_initialized"] = True

        model_flowlines = gdir.read_pickle("model_flowlines")
        topology = _topology_metrics(
            model_flowlines,
            abs(float(gdir.grid.dx)),
            float(config["gates"]["reverse_slope_tolerance_m"]),
        )
        sections = [np.asarray(fl.section, dtype=float) for fl in model_flowlines]
        thickness = [np.asarray(fl.thick, dtype=float) for fl in model_flowlines]
        widths = [np.asarray(fl.widths_m, dtype=float) for fl in model_flowlines]
        model_area_m2 = float(sum(float(fl.area_m2) for fl in model_flowlines))
        model_volume_m3 = float(sum(float(fl.volume_m3) for fl in model_flowlines))
        root = model_flowlines[int(topology["root_index"])]
        positive_root = np.flatnonzero(np.asarray(root.thick) > 0)
        root_length_m = float(positive_root.size * root.dx_meter)
        free_tail_m = float((root.nx - positive_root.size) * root.dx_meter)
        paired = load_source_state(resolve(config["paired_grid"]), "farinotti2019")
        paired_mask = (
            np.asarray(paired.valid_mask, dtype=bool)
            & np.asarray(paired.lia_mask, dtype=bool)
            & (np.asarray(paired.thickness_m, dtype=float) > 1e-6)
        )
        paired_area_m2 = float(np.sum(paired_mask) * paired.cell_area_m2)
        paired_volume_m3 = float(
            np.sum(np.asarray(paired.thickness_m)[paired_mask]) * paired.cell_area_m2
        )
        outline_area_m2 = float(geometry_summary["outline_area_m2"])
        flags = []
        if not topology["topology_valid"]:
            flags.append("invalid-topology")
        if topology["within_branch_reverse_slope_count"]:
            flags.append("within-branch-reverse-slope")
        if topology["junction_reverse_slope_count"]:
            flags.append("junction-reverse-slope")
        if any(not np.isfinite(values).all() for values in sections + thickness + widths):
            flags.append("nonfinite-model-geometry")
        if any(np.any(values < 0) for values in sections + thickness):
            flags.append("negative-section-or-thickness")
        if any(np.any(w[t > 0] <= 0) for w, t in zip(widths, thickness)):
            flags.append("nonpositive-active-width")
        if any(float(t[-1]) > 0 for t in thickness):
            flags.append("terminal-ice-present")
        if abs(model_area_m2 - outline_area_m2) / outline_area_m2 > float(
            config["gates"]["model_area_relative_error_max"]
        ):
            flags.append("model-outline-area-mismatch")
        if abs(root_length_m - float(config["registered_lia_length_m"])) / float(
            config["registered_lia_length_m"]
        ) > float(config["gates"]["root_length_relative_error_max"]):
            flags.append("model-root-length-mismatch")
        if free_tail_m < float(config["gates"]["minimum_free_downstream_m"]):
            flags.append("model-free-tail-too-short")

        current_hashes = {
            str(geometry_summary_path): _sha256(geometry_summary_path),
            str(resolve(config["paired_grid"])): _sha256(resolve(config["paired_grid"])),
        }
        if current_hashes != source_hashes:
            flags.append("source-mutation")
        diagnostics = gdir.get_diagnostics()
        summary.update(
            {
                "status": "inversion-pass" if not flags else "inversion-static-fail",
                "review_flags": flags,
                "climate": climate_audit,
                "spinup_window": [start_year, end_year],
                "stage2": {**stage2, "total_temp_bias_c": total_temp_bias},
                "apparent_mb_residual": diagnostics.get(
                    "apparent_mb_from_any_mb_residual"
                ),
                "mass_conservation_inversion_return_m3": float(volume_m3),
                "model_area_m2": model_area_m2,
                "model_volume_m3": model_volume_m3,
                "outline_area_m2": outline_area_m2,
                "model_outline_area_relative_error": abs(
                    model_area_m2 - outline_area_m2
                ) / outline_area_m2,
                "paired_area_m2": paired_area_m2,
                "paired_volume_m3": paired_volume_m3,
                "model_paired_area_relative_difference": (
                    model_area_m2 - paired_area_m2
                ) / paired_area_m2,
                "model_paired_volume_relative_difference": (
                    model_volume_m3 - paired_volume_m3
                ) / paired_volume_m3,
                "paired_model_hypsometry_cdf_error_100m": _hypsometry_error(
                    paired, model_flowlines
                ),
                "model_root_length_m": root_length_m,
                "model_free_tail_m": free_tail_m,
                "maximum_terminal_thickness_m": float(
                    max(float(t[-1]) for t in thickness)
                ),
                **topology,
            }
        )
        pd.DataFrame(
            [
                {
                    "branch_index": i,
                    "node_count": int(fl.nx),
                    "area_m2": float(fl.area_m2),
                    "volume_m3": float(fl.volume_m3),
                    "length_m": float(fl.length_m),
                    "terminal_thickness_m": float(fl.thick[-1]),
                    "minimum_section_m2": float(np.min(fl.section)),
                    "minimum_thickness_m": float(np.min(fl.thick)),
                }
                for i, fl in enumerate(model_flowlines)
            ]
        ).to_csv(output_root / "model_flowline_inventory.csv", index=False)
    except Exception as exc:
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()
    _write_json(output_root / "inversion_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.config.resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
