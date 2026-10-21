"""Adapt v0.1 LIA system grids to OGGM flowline initial-state packages.

This module does not run climate dynamics. It samples the reconstructed 2-D
LIA thickness and bed onto existing OGGM model flowlines, writes copied
flowline pickles under an adapter output directory, and records area/volume
audits. Source glacier directories and v0.1 NetCDF files are read-only.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer

from ..paths import resolve


SCENARIOS = ("farinotti2019", "millan2022")


@dataclass(frozen=True)
class SampledFlowlineAudit:
    rgi_id: str
    source_flowline_policy: str
    coordinate_transform_policy: str
    flowline_count: int
    node_count: int
    finite_node_count: int
    invalid_node_count: int
    bed_fallback_node_count: int
    positive_node_count: int
    minimum_sampled_thickness_m: float
    maximum_sampled_thickness_m: float
    flowline_area_km2: float
    flowline_volume_km3: float
    output_pickle: str


def gdir_path(root: Path, rgi_id: str) -> Path:
    return root / rgi_id[:-6] / rgi_id[:-3] / rgi_id


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_flowlines(path: Path) -> list:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with gzip.open(path, "rb") as source:
            return pickle.load(source)


def read_pickle(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with gzip.open(path, "rb") as source:
            return pickle.load(source)


def write_flowlines(path: Path, flowlines: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb") as sink:
        pickle.dump(flowlines, sink, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


_OGGM_INITIALIZED = False


def ensure_oggm_initialized() -> None:
    global _OGGM_INITIALIZED
    if _OGGM_INITIALIZED:
        return
    from oggm import cfg

    cfg.initialize(logging_level="WARNING")
    _OGGM_INITIALIZED = True


def build_model_flowlines_from_inversion(gdir_dir: Path) -> list:
    """Construct OGGM model flowlines from inversion files without writing gdir."""
    ensure_oggm_initialized()
    from oggm import cfg
    from oggm.core.flowline import MixedBedFlowline, line_order

    grid = read_json(gdir_dir / "glacier_grid.json")
    map_dx = abs(float(grid["dxdy"][0]))
    default_lambda = cfg.PARAMS["trapezoid_lambdas"]
    downstream_shape = cfg.PARAMS["downstream_line_shape"]
    centerlines = read_pickle(gdir_dir / "inversion_flowlines.pkl")
    inversions = read_pickle(gdir_dir / "inversion_output.pkl")
    new_flowlines = []
    flows_to_ids = []
    for centerline, inversion in zip(centerlines, inversions):
        line = centerline.line
        surface_h = np.asarray(centerline.surface_h, dtype=float)
        widths_m = np.asarray(centerline.widths, dtype=float) * map_dx
        section = np.asarray(inversion["volume"], dtype=float) / (
            float(centerline.dx) * map_dx
        )
        thick = np.asarray(inversion["thick"], dtype=float)
        bed_h = surface_h - thick
        bed_shape = 4.0 * thick / widths_m**2
        lambdas = thick * np.nan
        lambdas[np.asarray(inversion["is_trapezoid"], dtype=bool)] = default_lambda
        lambdas[np.asarray(inversion["is_rectangular"], dtype=bool)] = 0.0
        lambdas[bed_shape == 0] = default_lambda
        if bool(inversion["is_last"]):
            downstream = read_pickle(gdir_dir / "downstream_line.pkl")
            if downstream_shape == "parabola":
                bed_shape = np.append(bed_shape, downstream["bedshapes"])
                lambdas = np.append(lambdas, np.asarray(downstream["bedshapes"]) * np.nan)
                widths_m = np.append(widths_m, np.asarray(downstream["bedshapes"]) * 0.0)
            elif downstream_shape == "trapezoidal":
                bed_shape = np.append(
                    bed_shape,
                    np.asarray(downstream["bedshapes"], dtype=float) * np.nan,
                )
                lambdas = np.append(
                    lambdas,
                    np.ones(len(downstream["w0s"]), dtype=float) * default_lambda,
                )
                widths_m = np.append(widths_m, downstream["w0s"])
            else:
                raise ValueError(f"unsupported-downstream-line-shape:{downstream_shape}")
            section = np.append(section, np.asarray(downstream["bedshapes"]) * 0.0)
            surface_h = np.append(surface_h, downstream["surface_h"])
            bed_h = np.append(bed_h, downstream["surface_h"])
            line = downstream["full_line"]
        model_flowline = MixedBedFlowline(
            line=line,
            dx=centerline.dx,
            map_dx=map_dx,
            surface_h=surface_h,
            bed_h=bed_h,
            section=section,
            bed_shape=bed_shape,
            is_trapezoid=np.isfinite(lambdas),
            lambdas=lambdas,
            widths_m=widths_m,
            rgi_id=centerline.rgi_id,
        )
        flows_to_ids.append(
            centerlines.index(centerline.flows_to)
            if getattr(centerline, "flows_to", None) is not None
            else None
        )
        new_flowlines.append(model_flowline)
    for flowline, flow_to_id in zip(new_flowlines, flows_to_ids):
        if flow_to_id is not None:
            flowline.set_flows_to(new_flowlines[flow_to_id])
    for flowline in new_flowlines:
        flowline.order = line_order(flowline)
    return new_flowlines


def grid_xy_from_flowline(flowline, glacier_grid: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert OGGM line coords from (col, row) grid units to projected metres."""
    coords = np.asarray(flowline.line.coords, dtype=float)
    x0, y0 = map(float, glacier_grid["x0y0"])
    dx, dy = map(float, glacier_grid["dxdy"])
    return x0 + coords[:, 0] * dx, y0 + coords[:, 1] * dy


class NearestGridSampler:
    """Nearest-neighbour sampler for regular x/y coordinate vectors."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        arrays: dict[str, np.ndarray],
        crs: CRS | None = None,
    ):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("system-grid-coordinates-must-be-1d")
        if len(self.x) < 2 or len(self.y) < 2:
            raise ValueError("system-grid-too-small")
        self.dx = float(np.median(np.diff(self.x)))
        self.dy = float(np.median(np.diff(self.y)))
        self.crs = crs
        if self.dx == 0 or self.dy == 0:
            raise ValueError("system-grid-has-zero-spacing")
        self.arrays = {key: np.asarray(value) for key, value in arrays.items()}
        expected = (len(self.y), len(self.x))
        for key, value in self.arrays.items():
            if value.shape != expected:
                raise ValueError(f"array-shape-mismatch:{key}:{value.shape}!={expected}")

    def indices(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        col = np.rint((np.asarray(x, dtype=float) - self.x[0]) / self.dx).astype(int)
        row = np.rint((np.asarray(y, dtype=float) - self.y[0]) / self.dy).astype(int)
        inside = (col >= 0) & (col < len(self.x)) & (row >= 0) & (row < len(self.y))
        return row, col, inside

    def sample(self, x: np.ndarray, y: np.ndarray, key: str, fill_value=np.nan) -> np.ndarray:
        row, col, inside = self.indices(x, y)
        out = np.full(row.shape, fill_value, dtype=float)
        out[inside] = self.arrays[key][row[inside], col[inside]]
        return out


def member_index(dataset: xr.Dataset, scenario: str) -> int:
    if "member_id" not in dataset.coords:
        raise ValueError("system-netcdf-missing-member_id-coordinate")
    members = [str(value) for value in dataset["member_id"].values]
    if scenario not in members:
        raise ValueError(f"scenario-not-in-system-netcdf:{scenario}")
    index = members.index(scenario)
    status = str(dataset["member_status"].isel(member=index).values)
    if status != "success":
        raise ValueError(f"scenario-not-success:{scenario}:{status}")
    return index


def load_sampler(dataset: xr.Dataset, scenario: str) -> tuple[NearestGridSampler, dict]:
    index = member_index(dataset, scenario)
    arrays = {
        "thickness_m": dataset["thickness_m"].isel(member=index).values,
        "bed_elevation_m": dataset["bed_elevation_m"].isel(member=index).values,
        "valid_mask": dataset["valid_mask"].isel(member=index).values.astype(float),
    }
    crs = None
    if "spatial_ref" in dataset:
        attrs = dataset["spatial_ref"].attrs
        crs_text = attrs.get("crs_wkt") or attrs.get("spatial_ref")
        if crs_text:
            crs = CRS.from_user_input(crs_text)
    sampler = NearestGridSampler(dataset["x"].values, dataset["y"].values, arrays, crs=crs)
    dx = abs(float(np.median(np.diff(dataset["x"].values))))
    dy = abs(float(np.median(np.diff(dataset["y"].values))))
    source_valid = arrays["valid_mask"] > 0.5
    source_h = np.asarray(arrays["thickness_m"], dtype=float)
    source_positive = source_valid & np.isfinite(source_h) & (source_h > 0)
    audit = {
        "source_valid_area_km2": float(source_valid.sum() * dx * dy / 1e6),
        "source_positive_area_km2": float(source_positive.sum() * dx * dy / 1e6),
        "source_volume_km3": float(np.nansum(np.where(source_valid, source_h, 0.0)) * dx * dy / 1e9),
        "source_grid_dx_m": dx,
        "source_grid_dy_m": dy,
        "source_valid_cell_count": int(source_valid.sum()),
    }
    return sampler, audit


def flowline_volume_m3(flowline) -> float:
    if hasattr(flowline, "volume_m3"):
        return float(flowline.volume_m3)
    return float(np.sum(np.asarray(flowline.section, dtype=float)) * float(flowline.dx_meter))


def flowline_area_m2(flowline) -> float:
    if hasattr(flowline, "bin_area_m2"):
        return float(np.sum(flowline.bin_area_m2))
    return float(
        np.sum(
            np.where(
                np.asarray(flowline.thick, dtype=float) > 0,
                np.asarray(flowline.widths_m, dtype=float),
                0.0,
            )
        )
        * float(flowline.dx_meter)
    )


def flowline_stats(flowlines: list) -> dict:
    thickness_parts = [np.asarray(flowline.thick, dtype=float) for flowline in flowlines]
    thickness = np.concatenate(thickness_parts) if thickness_parts else np.array([], dtype=float)
    return {
        "node_count": int(thickness.size),
        "positive_node_count": int((thickness > 0).sum()),
        "minimum_sampled_thickness_m": float(np.min(thickness)) if thickness.size else 0.0,
        "maximum_sampled_thickness_m": float(np.max(thickness)) if thickness.size else 0.0,
        "flowline_area_km2": float(sum(flowline_area_m2(fl) for fl in flowlines) / 1e6),
        "flowline_volume_km3": float(sum(flowline_volume_m3(fl) for fl in flowlines) / 1e9),
    }


def total_flowline_volume_m3(flowlines_by_rgi: dict[str, list]) -> float:
    return float(
        sum(flowline_volume_m3(flowline) for flowlines in flowlines_by_rgi.values() for flowline in flowlines)
    )


def total_flowline_area_m2(flowlines_by_rgi: dict[str, list]) -> float:
    return float(
        sum(flowline_area_m2(flowline) for flowlines in flowlines_by_rgi.values() for flowline in flowlines)
    )


def apply_width_factor(flowlines_by_rgi: dict[str, list], factor: float) -> None:
    """Scale MixedBedFlowline width geometry without changing thickness."""
    factor = float(factor)
    if factor <= 0 or not np.isfinite(factor):
        raise ValueError(f"invalid-width-factor:{factor}")
    for flowlines in flowlines_by_rgi.values():
        for flowline in flowlines:
            if not all(hasattr(flowline, name) for name in ("bed_shape", "_sqrt_bed")):
                raise TypeError("width-scaling-requires-mixed-bed-flowline")
            is_trapezoid = np.asarray(getattr(flowline, "is_trapezoid", np.zeros_like(flowline.thick, dtype=bool)), dtype=bool)
            parabolic = ~is_trapezoid
            if np.any(parabolic):
                flowline.bed_shape[parabolic] = flowline.bed_shape[parabolic] / factor**2
                flowline._sqrt_bed[parabolic] = np.sqrt(flowline.bed_shape[parabolic])
            if np.any(is_trapezoid):
                flowline._w0_m[is_trapezoid] = flowline._w0_m[is_trapezoid] * factor
                flowline._lambdas[is_trapezoid] = flowline._lambdas[is_trapezoid] * factor


def scale_flowlines_to_volume(
    flowlines_by_rgi: dict[str, list],
    target_volume_m3: float,
) -> float:
    """Uniformly scale sampled thickness until OGGM section volume matches target."""
    originals = {
        rgi_id: [np.asarray(flowline.thick, dtype=float).copy() for flowline in flowlines]
        for rgi_id, flowlines in flowlines_by_rgi.items()
    }

    def apply_factor(factor: float) -> float:
        for rgi_id, flowlines in flowlines_by_rgi.items():
            for flowline, original in zip(flowlines, originals[rgi_id]):
                flowline.thick = original * factor
        return total_flowline_volume_m3(flowlines_by_rgi)

    if target_volume_m3 <= 0:
        apply_factor(0.0)
        return 0.0
    high = 1.0
    high_volume = apply_factor(high)
    while high_volume < target_volume_m3:
        high *= 2.0
        high_volume = apply_factor(high)
        if high > 10.0:
            raise RuntimeError("volume-scaling-factor-exceeded-maximum")
    low = 0.0
    for _ in range(48):
        mid = 0.5 * (low + high)
        mid_volume = apply_factor(mid)
        if mid_volume < target_volume_m3:
            low = mid
        else:
            high = mid
    factor = 0.5 * (low + high)
    apply_factor(factor)
    return float(factor)


def scale_flowlines_to_area_and_volume(
    flowlines_by_rgi: dict[str, list],
    target_area_m2: float,
    target_volume_m3: float,
) -> tuple[float, float]:
    """Scale thickness and width geometry to match target area and volume."""
    target_area_m2 = float(target_area_m2)
    target_volume_m3 = float(target_volume_m3)
    if target_area_m2 <= 0 or target_volume_m3 <= 0:
        scale_flowlines_to_volume(flowlines_by_rgi, 0.0)
        return 0.0, 1.0
    originals = {
        rgi_id: [np.asarray(flowline.thick, dtype=float).copy() for flowline in flowlines]
        for rgi_id, flowlines in flowlines_by_rgi.items()
    }

    def apply_thickness_factor(factor: float) -> tuple[float, float]:
        for rgi_id, flowlines in flowlines_by_rgi.items():
            for flowline, original in zip(flowlines, originals[rgi_id]):
                flowline.thick = original * factor
        return total_flowline_area_m2(flowlines_by_rgi), total_flowline_volume_m3(flowlines_by_rgi)

    target_mean_thickness_m = target_volume_m3 / target_area_m2

    def mean_after_factor(factor: float) -> float:
        area, volume = apply_thickness_factor(factor)
        return volume / area if area > 0 else 0.0

    low = 0.0
    high = 1.0
    while mean_after_factor(high) < target_mean_thickness_m:
        high *= 2.0
        if high > 100.0:
            raise RuntimeError("area-volume-thickness-factor-exceeded-maximum")
    for _ in range(56):
        mid = 0.5 * (low + high)
        if mean_after_factor(mid) < target_mean_thickness_m:
            low = mid
        else:
            high = mid
    thickness_factor = 0.5 * (low + high)
    area, _ = apply_thickness_factor(thickness_factor)
    if area <= 0:
        raise RuntimeError("area-volume-scaling-produced-zero-area")
    width_factor = target_area_m2 / area
    apply_width_factor(flowlines_by_rgi, width_factor)
    return float(thickness_factor), float(width_factor)


def adapt_one_gdir(
    *,
    gdir_dir: Path,
    rgi_id: str,
    sampler: NearestGridSampler,
    scenario: str,
    output_dir: Path,
    input_pickle_name: str = "model_flowlines.pkl",
) -> tuple[SampledFlowlineAudit, list]:
    source_pickle = gdir_dir / input_pickle_name
    glacier_grid = read_json(gdir_dir / "glacier_grid.json")
    source_crs = CRS.from_user_input(glacier_grid["proj"])
    target_crs = sampler.crs
    transformer = None
    coordinate_transform_policy = "source-crs-equals-system-crs"
    if target_crs is not None and not source_crs.equals(target_crs):
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        coordinate_transform_policy = "pyproj-source-gdir-crs-to-system-grid-crs"
    if source_pickle.exists():
        flowlines = copy.deepcopy(read_flowlines(source_pickle))
        source_flowline_policy = "read-existing-model_flowlines"
    elif (gdir_dir / "inversion_flowlines.pkl").exists():
        flowlines = build_model_flowlines_from_inversion(gdir_dir)
        source_flowline_policy = "built-from-inversion-files-read-only"
    else:
        raise FileNotFoundError(f"missing-model-or-inversion-flowlines:{gdir_dir}")
    invalid_nodes = 0
    bed_fallback_nodes = 0
    finite_nodes = 0
    positive_nodes = 0
    min_h = math.inf
    max_h = -math.inf
    for flowline in flowlines:
        x, y = grid_xy_from_flowline(flowline, glacier_grid)
        if transformer is not None:
            x, y = transformer.transform(x, y)
        sampled_h = sampler.sample(x, y, "thickness_m")
        sampled_b = sampler.sample(x, y, "bed_elevation_m")
        sampled_valid = sampler.sample(x, y, "valid_mask", fill_value=0.0) > 0.5
        sampled_h = np.where(sampled_valid, sampled_h, 0.0)
        finite_bed = np.isfinite(sampled_b)
        missing_bed_with_ice = (~finite_bed) & (sampled_h > 0)
        if missing_bed_with_ice.any():
            raise ValueError(
                f"flowline-samples-missing-bed:{rgi_id}:"
                f"cell_count={int(missing_bed_with_ice.sum())}"
            )
        bed_fallback_nodes += int((~finite_bed).sum())
        sampled_b = np.where(finite_bed, sampled_b, np.asarray(flowline.bed_h, dtype=float))
        finite_h = np.isfinite(sampled_h)
        if not finite_h.all():
            raise ValueError(
                f"flowline-samples-missing-thickness:{rgi_id}:"
                f"cell_count={int((~finite_h).sum())}"
            )
        if np.any(sampled_h < -1e-6):
            raise ValueError(
                f"flowline-samples-negative-thickness:{rgi_id}:"
                f"minimum={float(np.min(sampled_h))}"
            )
        sampled_h = np.where(sampled_h < 0, 0.0, sampled_h)
        flowline.bed_h = sampled_b.astype(float)
        flowline.thick = sampled_h.astype(float)
        invalid_nodes += int((~sampled_valid).sum())
        finite_nodes += int(finite_h.sum())
        positive_nodes += int((sampled_h > 0).sum())
        min_h = min(min_h, float(np.min(sampled_h)))
        max_h = max(max_h, float(np.max(sampled_h)))
    out_pickle = output_dir / rgi_id / f"model_flowlines_lia_{scenario}.pkl"
    write_flowlines(out_pickle, flowlines)
    stats = flowline_stats(flowlines)
    audit = SampledFlowlineAudit(
        rgi_id=rgi_id,
        source_flowline_policy=source_flowline_policy,
        coordinate_transform_policy=coordinate_transform_policy,
        flowline_count=len(flowlines),
        node_count=stats["node_count"],
        finite_node_count=finite_nodes,
        invalid_node_count=invalid_nodes,
        bed_fallback_node_count=bed_fallback_nodes,
        positive_node_count=positive_nodes,
        minimum_sampled_thickness_m=0.0 if min_h == math.inf else min_h,
        maximum_sampled_thickness_m=0.0 if max_h == -math.inf else max_h,
        flowline_area_km2=stats["flowline_area_km2"],
        flowline_volume_km3=stats["flowline_volume_km3"],
        output_pickle=str(out_pickle),
    )
    return audit, flowlines


def member_ids_for_system(manifest: pd.DataFrame, lia_unique_id: int, main_rgi_id: str) -> list[str]:
    row = manifest.loc[manifest["lia_unique_id"].astype(int) == int(lia_unique_id)]
    if row.empty:
        return [main_rgi_id]
    raw = str(row.iloc[0].get("download_member_rgi_ids", ""))
    ids = [item for item in raw.split(";") if item and item != "nan"]
    return list(dict.fromkeys([main_rgi_id, *[item for item in ids if item != main_rgi_id]]))


def adapt_system(
    *,
    system_netcdf: Path,
    gdir_root: Path,
    output_root: Path,
    scenarios: tuple[str, ...] = SCENARIOS,
    system_manifest: Path | None = None,
    volume_policy: str = "scale-thickness-to-source-volume",
) -> dict:
    with xr.open_dataset(system_netcdf) as dataset:
        attrs = dict(dataset.attrs)
        lia_unique_id = int(attrs["lia_unique_id"])
        main_rgi_id = str(attrs["rgi_id"])
        if system_manifest is not None:
            manifest = pd.read_csv(system_manifest)
            rgi_ids = member_ids_for_system(manifest, lia_unique_id, main_rgi_id)
        else:
            rgi_ids = [main_rgi_id]
        system_out = output_root / str(lia_unique_id)
        scenario_rows = []
        for scenario in scenarios:
            try:
                sampler, source_audit = load_sampler(dataset, scenario)
            except ValueError as exc:
                scenario_rows.append(
                    {
                        "bed_scenario": scenario,
                        "status": "skipped",
                        "skip_reason": str(exc),
                    }
                )
                continue
            member_rows = []
            flowlines_by_rgi = {}
            output_by_rgi = {}
            for rgi_id in rgi_ids:
                audit, flowlines = adapt_one_gdir(
                    gdir_dir=gdir_path(gdir_root, rgi_id),
                    rgi_id=rgi_id,
                    sampler=sampler,
                    scenario=scenario,
                    output_dir=system_out / scenario,
                )
                member_rows.append(audit.__dict__)
                flowlines_by_rgi[rgi_id] = flowlines
                output_by_rgi[rgi_id] = Path(audit.output_pickle)
            pre_scale_volume = float(sum(row["flowline_volume_km3"] for row in member_rows))
            thickness_scale_factor = 1.0
            width_scale_factor = 1.0
            if volume_policy == "scale-thickness-to-source-volume":
                thickness_scale_factor = scale_flowlines_to_volume(
                    flowlines_by_rgi,
                    float(source_audit["source_volume_km3"]) * 1e9,
                )
            elif volume_policy == "scale-thickness-and-width-to-source-area-volume":
                thickness_scale_factor, width_scale_factor = scale_flowlines_to_area_and_volume(
                    flowlines_by_rgi,
                    float(source_audit["source_positive_area_km2"]) * 1e6,
                    float(source_audit["source_volume_km3"]) * 1e9,
                )
            if volume_policy in {
                "scale-thickness-to-source-volume",
                "scale-thickness-and-width-to-source-area-volume",
            }:
                refreshed_rows = []
                for row in member_rows:
                    rgi_id = str(row["rgi_id"])
                    write_flowlines(output_by_rgi[rgi_id], flowlines_by_rgi[rgi_id])
                    updated = dict(row)
                    updated.update(flowline_stats(flowlines_by_rgi[rgi_id]))
                    refreshed_rows.append(updated)
                member_rows = refreshed_rows
            elif volume_policy != "audit-only":
                raise ValueError(f"unsupported-volume-policy:{volume_policy}")
            flowline_volume = float(sum(row["flowline_volume_km3"] for row in member_rows))
            flowline_area = float(sum(row["flowline_area_km2"] for row in member_rows))
            source_volume = float(source_audit["source_volume_km3"])
            source_area = float(source_audit["source_positive_area_km2"])
            scenario_rows.append(
                {
                    "bed_scenario": scenario,
                    "status": "adapted",
                    "volume_policy": volume_policy,
                    "pre_scale_flowline_volume_km3": pre_scale_volume,
                    "thickness_scale_factor": thickness_scale_factor,
                    "width_scale_factor": width_scale_factor,
                    "source": source_audit,
                    "member_flowlines": member_rows,
                    "aggregate_flowline_area_km2": flowline_area,
                    "aggregate_flowline_volume_km3": flowline_volume,
                    "source_positive_area_km2": source_area,
                    "source_volume_km3": source_volume,
                    "flowline_minus_source_area_km2": flowline_area - source_area,
                    "flowline_minus_source_volume_km3": flowline_volume - source_volume,
                    "flowline_minus_source_volume_fraction": (
                        (flowline_volume - source_volume) / source_volume
                        if source_volume > 0
                        else None
                    ),
                }
            )
    summary = {
        "status": "adapted-lia-v01-system-grid-to-oggm-flowline-package",
        "method": "nearest-node-sampling-preserve-oggm-flowline-widths-and-topology",
        "system_netcdf": str(system_netcdf),
        "lia_unique_id": lia_unique_id,
        "main_rgi_id": main_rgi_id,
        "member_rgi_ids": rgi_ids,
        "scenario_count": len(scenario_rows),
        "scenarios": scenario_rows,
        "notes": [
            "No climate dynamics were run.",
            "Source NetCDF and source glacier directories were not modified.",
            "Scaling policy is explicit per system and bed scenario.",
            "No source flowline or source grid is modified.",
        ],
    }
    write_json(
        output_root / str(lia_unique_id) / "initial_state_adapter_audit.json",
        summary,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-netcdf", required=True, type=Path)
    parser.add_argument("--gdir-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--system-manifest")
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    parser.add_argument(
        "--volume-policy",
        choices=(
            "scale-thickness-to-source-volume",
            "scale-thickness-and-width-to-source-area-volume",
            "audit-only",
        ),
        default="scale-thickness-to-source-volume",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = tuple(args.scenario) if args.scenario else SCENARIOS
    summary = adapt_system(
        system_netcdf=args.system_netcdf.resolve(),
        gdir_root=resolve(args.gdir_root),
        output_root=resolve(args.output_root),
        scenarios=scenarios,
        system_manifest=resolve(args.system_manifest) if args.system_manifest else None,
        volume_policy=args.volume_policy,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
