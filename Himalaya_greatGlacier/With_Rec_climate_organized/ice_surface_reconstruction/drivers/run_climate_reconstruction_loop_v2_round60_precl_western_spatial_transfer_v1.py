"""Test spatial transfer of provisional western candidates against NOAA PREC/L."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YEARS = np.arange(1948, 2001)
BLOCKS = ((1952, 1963), (1964, 1975), (1976, 1987), (1988, 2000))


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def load_round58_helpers(path: Path):
    spec = importlib.util.spec_from_file_location("round58_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("round58-helper-import-spec-failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def main() -> int:
    project = project_root()
    loop = project / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_60_precl_western_spatial_transfer_v1"
    round57 = loop / "round_57_noaa_precl_allowed_subset_acquisition_v3"
    round59 = loop / "round_59_western_four_product_candidate_consensus_v1"
    driver58 = Path(__file__).resolve().with_name(
        "run_climate_reconstruction_loop_v2_round58_noaa_precl_western_adjudication_v1.py"
    )
    chunks = ((1948, 1957), (1958, 1967), (1968, 1977), (1978, 1987), (1988, 1997), (1998, 2000))
    precl_paths = [round57 / f"precl_{start}_{end}.nc" for start, end in chunks]
    paths = {
        **{f"precl_{start}_{end}": path for (start, end), path in zip(chunks, precl_paths)},
        "round59_summary": round59 / "summary.json",
        "udel": loop / "round_44_udel_allowed_subset_acquisition_v2/udel_precip_v501_1940_2000_hma_bbox.nc",
        "public_reference_source": Path(
            r"E:\3spyderProject\HMA_500\Down_scale_new\out_data\climate_anomaly_05deg_with_ensemble_1000-2000.nc"
        ),
        "hma_mask": project
        / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_input_spatial_audit_v3/hma_mask_on_netcdf_grid.npy",
        "round58_driver": driver58,
        "contract": output / "ROUND_CONTRACT.md",
        "driver": Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    before = {name: state(path) for name, path in paths.items()}
    summary59 = json.loads(paths["round59_summary"].read_text(encoding="utf-8"))
    candidate_names = list(summary59["provisional_multi_product_consensus_candidates"])
    if candidate_names != ["O", "local_enkf"]:
        raise RuntimeError(f"unexpected-round59-provisional-candidates:{candidate_names}")
    helpers = load_round58_helpers(driver58)

    mask = np.load(paths["hma_mask"]).astype(bool)
    with h5py.File(paths["public_reference_source"], "r") as handle:
        source_years = np.asarray(handle["time"], dtype=int).ravel()
        indices = np.flatnonzero((source_years >= 1948) & (source_years <= 2000))
        if not np.array_equal(source_years[indices], YEARS):
            raise RuntimeError("public-year-axis-failed")
        if np.any((source_years[indices] >= 1921) & (source_years[indices] <= 1939)):
            raise RuntimeError("locked-period-index-requested")
        selection = slice(int(indices[0]), int(indices[-1]) + 1)
        latitudes = np.asarray(handle["lat"], dtype=float).ravel()
        longitudes = np.asarray(handle["lon"], dtype=float).ravel()
        observations = {
            name: helpers.recenter(helpers.clean(handle[variable], selection))
            for name, variable in helpers.OBSERVATIONS.items()
        }
        members = {
            name: helpers.recenter(helpers.clean(handle[variable], selection))
            for name, variable in helpers.MEMBERS.items()
        }
    observations["UDel"] = helpers.recenter(
        helpers.annualize_udel(paths["udel"], latitudes, longitudes)
    )
    observations["PREC/L"] = helpers.recenter(
        helpers.annualize_precl(precl_paths, latitudes, longitudes)
    )
    candidates = {
        "O": members["O"],
        "local_enkf": helpers.finite_mean([members["HGAO"], members["O"]]),
    }
    west = mask & (np.broadcast_to(longitudes[None], mask.shape) < 80.0)
    common = west.copy()
    for values in observations.values():
        common &= np.all(np.isfinite(values), axis=0)
    if int(common.sum()) != int(summary59["common_western_cell_count"]):
        raise RuntimeError(f"common-support-closure-failed:{int(common.sum())}")
    target = observations["PREC/L"]
    area_weights = np.cos(np.deg2rad(latitudes))[:, None] * common

    rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    maps: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, object]] = {}
    for candidate, estimate in candidates.items():
        pixel_rmse = np.sqrt(np.mean((estimate - target) ** 2, axis=0))
        zero_pixel_rmse = np.sqrt(np.mean(target**2, axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            improvement_map = 1.0 - pixel_rmse / zero_pixel_rmse
        improvement_map[~common] = np.nan
        maps[candidate] = improvement_map
        valid_cells = common & np.isfinite(improvement_map)
        full_weighted_candidate_rmse = weighted_mean(pixel_rmse, area_weights)
        full_weighted_zero_rmse = weighted_mean(zero_pixel_rmse, area_weights)
        full_weighted_improvement = 1.0 - full_weighted_candidate_rmse / full_weighted_zero_rmse
        median_cell_improvement = float(np.nanmedian(improvement_map[valid_cells]))
        improving_cell_fraction = float(np.mean(improvement_map[valid_cells] > 0))
        block_improvements: list[float] = []
        for fold, (start, end) in enumerate(BLOCKS, start=2):
            selected = (YEARS >= start) & (YEARS <= end)
            block_pixel_rmse = np.sqrt(np.mean((estimate[selected] - target[selected]) ** 2, axis=0))
            block_zero_pixel_rmse = np.sqrt(np.mean(target[selected] ** 2, axis=0))
            weighted_candidate = weighted_mean(block_pixel_rmse, area_weights)
            weighted_zero = weighted_mean(block_zero_pixel_rmse, area_weights)
            improvement = 1.0 - weighted_candidate / weighted_zero
            block_improvements.append(improvement)
            block_rows.append(
                {
                    "candidate": candidate,
                    "fold": fold,
                    "start_year": start,
                    "end_year": end,
                    "weighted_pixel_rmse_mm_day": weighted_candidate,
                    "weighted_zero_pixel_rmse_mm_day": weighted_zero,
                    "weighted_pixel_rmse_improvement": improvement,
                }
            )
        gates = {
            "full_weighted_pixel_rmse_improvement_nonnegative": full_weighted_improvement >= 0,
            "median_cellwise_rmse_improvement_nonnegative": median_cell_improvement >= 0,
            "more_than_half_cells_improve": improving_cell_fraction > 0.5,
            "at_least_3_of_4_blocks_improve": int(np.sum(np.asarray(block_improvements) > 0)) >= 3,
            "maximum_block_regression_at_most_5pct": float(np.max(-np.asarray(block_improvements))) <= 0.05,
        }
        result = {
            "common_valid_cell_count": int(valid_cells.sum()),
            "full_weighted_pixel_rmse_improvement": full_weighted_improvement,
            "median_cellwise_rmse_improvement": median_cell_improvement,
            "improving_cell_fraction": improving_cell_fraction,
            "improving_block_count": int(np.sum(np.asarray(block_improvements) > 0)),
            "maximum_block_regression": float(np.max(-np.asarray(block_improvements))),
            "gates": gates,
            "passes_precl_spatial_transfer": bool(all(gates.values())),
        }
        results[candidate] = result
        rows.append({"candidate": candidate, **{key: value for key, value in result.items() if key != "gates"}})

    pd.DataFrame(rows).to_csv(output / "precl_spatial_transfer_summary.csv", index=False)
    pd.DataFrame(block_rows).to_csv(output / "precl_spatial_transfer_block_scores.csv", index=False)
    passing = [candidate for candidate, result in results.items() if result["passes_precl_spatial_transfer"]]
    summary = {
        "status": "complete",
        "locked_1921_1939_accessed": False,
        "common_western_cell_count": int(common.sum()),
        "candidate_results": results,
        "passing_spatial_transfer_candidates": passing,
        "decision": (
            "retain_spatially_supported_provisional_candidate"
            if passing
            else "regional_consensus_does_not_transfer_spatially"
        ),
        "production_field_selected": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.9), constrained_layout=True, sharex=True, sharey=True)
    for axis, candidate in zip(axes, candidate_names):
        image = axis.pcolormesh(longitudes, latitudes, maps[candidate] * 100.0, shading="auto", cmap="RdYlGn", vmin=-50, vmax=50)
        axis.set(title=f"{candidate}: cellwise RMSE improvement", xlabel="Longitude")
        axis.set_ylabel("Latitude")
    figure.colorbar(image, ax=axes, label="Improvement over zero (%)", shrink=0.9)
    figure.savefig(output / "precl_western_spatial_transfer_maps.png", dpi=180)
    plt.close(figure)

    blocks_frame = pd.DataFrame(block_rows)
    figure, axis = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    pivot = blocks_frame.pivot(index="fold", columns="candidate", values="weighted_pixel_rmse_improvement")
    pivot.mul(100).plot.bar(ax=axis, color=["#0072b2", "#d55e00"])
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.axhline(-5, color="#b2182b", linestyle="--", linewidth=1.0)
    axis.set(title="PREC/L western spatial skill by frozen block", xlabel="Frozen block", ylabel="Weighted pixel-RMSE improvement (%)")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(frameon=False)
    figure.savefig(output / "precl_western_spatial_transfer_blocks.png", dpi=180)
    plt.close(figure)

    lines = ["# PREC/L Western Spatial Transfer", ""]
    for candidate, result in results.items():
        lines.append(
            f"- {candidate}: weighted pixel skill `{result['full_weighted_pixel_rmse_improvement']:.2%}`, "
            f"median cell skill `{result['median_cellwise_rmse_improvement']:.2%}`, "
            f"improving cells `{result['improving_cell_fraction']:.2%}`, "
            f"improving blocks `{result['improving_block_count']}/4`, "
            f"passes `{str(result['passes_precl_spatial_transfer']).lower()}`."
        )
    lines.extend(["", f"Passing candidates: `{passing}`.", f"Decision: `{summary['decision']}`."])
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    after = {name: state(path) for name, path in paths.items()}
    unchanged = before == after
    (output / "source_manifest.json").write_text(
        json.dumps({"sources_before": before, "sources_after": after, "source_states_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    if not unchanged:
        raise RuntimeError("source-state-changed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
