"""Static gate for deterministic conservative native-network trunk collapse.

All active nodes of an already legal OGGM native multi-flowline state are
ordered by source surface elevation and fractionally accumulated into the
unchanged native root axis.  This preserves source area and volume exactly,
retains the registered root length and safety tail, and removes tributary
junctions without discarding tributary ice.  No dynamics or climate is run.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .v3_flowline_geometry_audit import (
    maximum_cumulative_hypsometry_error_fraction,
)


SCHEMA_VERSION = 1


def _write_json(path: Path, payload: dict) -> None:
    def safe(value):
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(safe(payload), indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_flowlines(path: Path) -> list:
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def _source_elements(flowlines: list) -> dict[str, np.ndarray]:
    surface = []
    area = []
    volume = []
    branch = []
    node = []
    for branch_index, fl in enumerate(flowlines):
        dx_m = float(fl.dx * fl.map_dx)
        widths_m = np.asarray(fl.widths_m, float)
        section = np.asarray(fl.section, float)
        thick = np.asarray(fl.thick, float)
        active = np.isfinite(thick) & (thick > 1e-6) & np.isfinite(section)
        indices = np.flatnonzero(active)
        for index in indices.tolist():
            surface.append(float(fl.surface_h[index]))
            area.append(float(widths_m[index] * dx_m))
            volume.append(float(section[index] * dx_m))
            branch.append(int(branch_index))
            node.append(int(index))
    values = {
        "surface_m": np.asarray(surface, float),
        "area_m2": np.asarray(area, float),
        "volume_m3": np.asarray(volume, float),
        "branch_index": np.asarray(branch, int),
        "node_index": np.asarray(node, int),
    }
    if not len(values["surface_m"]):
        raise RuntimeError("source-network-has-no-active-ice")
    if np.any(values["area_m2"] <= 0) or np.any(values["volume_m3"] < 0):
        raise RuntimeError("source-network-has-illegal-area-or-volume")
    return values


def _accumulate_equal_area_bins(
    surface_m: np.ndarray,
    area_m2: np.ndarray,
    volume_m3: np.ndarray,
    bin_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split source control volumes fractionally into equal-area elevation bins."""
    order = np.argsort(-np.asarray(surface_m, float), kind="mergesort")
    z = np.asarray(surface_m, float)[order]
    a = np.asarray(area_m2, float)[order]
    v = np.asarray(volume_m3, float)[order]
    target_area = float(a.sum() / bin_count)
    out_a = np.zeros(bin_count, dtype=float)
    out_v = np.zeros(bin_count, dtype=float)
    out_za = np.zeros(bin_count, dtype=float)
    source_index = 0
    remaining_area = float(a[0])
    remaining_volume = float(v[0])
    for target_index in range(bin_count):
        need = target_area if target_index < bin_count - 1 else float(a.sum() - out_a.sum())
        while need > max(1e-9, 1e-12 * target_area):
            if source_index >= len(a):
                raise RuntimeError("area-bin-accumulation-exhausted-source")
            take = min(need, remaining_area)
            fraction = take / remaining_area
            take_volume = remaining_volume * fraction
            out_a[target_index] += take
            out_v[target_index] += take_volume
            out_za[target_index] += take * z[source_index]
            need -= take
            remaining_area -= take
            remaining_volume -= take_volume
            if remaining_area <= max(1e-9, 1e-12 * target_area):
                source_index += 1
                if source_index < len(a):
                    remaining_area = float(a[source_index])
                    remaining_volume = float(v[source_index])
    if np.any(out_a <= 0):
        raise RuntimeError("empty-trunk-active-bin")
    return out_za / out_a, out_a, out_v


def run(config_path: Path, project_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = _resolve(project_root, config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    source_path = _resolve(project_root, config["source_model_flowlines"])
    before_hash = _sha256(source_path)
    flowlines = _load_flowlines(source_path)
    roots = [(index, fl) for index, fl in enumerate(flowlines) if fl.flows_to is None]
    if len(roots) != 1:
        raise RuntimeError(f"source-root-count-not-one:{len(roots)}")
    root_index, source_root = roots[0]
    source = _source_elements(flowlines)

    spacing_m = float(source_root.dx * source_root.map_dx)
    registered_length = float(config["registered_lia_length_m"])
    active_count = int(round(registered_length / spacing_m))
    active_count = max(1, min(active_count, int(source_root.nx) - 1))
    observed_length = float(active_count * spacing_m)
    free_tail = float((int(source_root.nx) - active_count) * spacing_m)

    surface_active, area_active, volume_active = _accumulate_equal_area_bins(
        source["surface_m"], source["area_m2"], source["volume_m3"], active_count
    )
    thickness_active = volume_active / area_active
    bed_active = surface_active - thickness_active
    widths_m_active = area_active / spacing_m
    if np.any(np.diff(surface_active) > float(config["gates"]["reverse_slope_tolerance_m"])):
        raise RuntimeError("elevation-bin-order-created-reverse-surface")

    nx = int(source_root.nx)
    surface = np.asarray(source_root.bed_h, float).copy()
    bed = np.asarray(source_root.bed_h, float).copy()
    widths_m = np.asarray(source_root.widths_m, float).copy()
    surface[:active_count] = surface_active
    bed[:active_count] = bed_active
    widths_m[:active_count] = widths_m_active
    widths_m[active_count:] = np.maximum(widths_m[active_count:], 1.0)

    from oggm import cfg
    from oggm.core.flowline import RectangularBedFlowline, ScalarMassBalance, SemiImplicitModel

    cfg.initialize(logging_level="WARNING")
    trunk = RectangularBedFlowline(
        line=copy.deepcopy(source_root.line),
        dx=float(source_root.dx),
        map_dx=float(source_root.map_dx),
        surface_h=surface,
        bed_h=bed,
        widths=widths_m / float(source_root.map_dx),
        rgi_id=str(config["synthetic_rgi_id"]),
    )
    model = SemiImplicitModel(
        [copy.deepcopy(trunk)],
        mb_model=ScalarMassBalance(0.0),
        y0=0.0,
        glen_a=float(config["stage2"]["glen_a"]),
        fs=float(config["stage2"]["fs"]),
    )

    source_area = float(source["area_m2"].sum())
    source_volume = float(source["volume_m3"].sum())
    model_area = float(model.area_m2)
    model_volume = float(model.volume_m3)
    area_error = abs(model_area - source_area) / source_area
    volume_error = abs(model_volume - source_volume) / source_volume
    hypsometry_error = maximum_cumulative_hypsometry_error_fraction(
        source_surface_m=source["surface_m"],
        source_area_m2=source["area_m2"],
        node_surface_m=surface_active,
        model_area_m2=area_active,
        bin_size_m=100.0,
    )
    length_error = abs(observed_length - registered_length) / registered_length
    thick = np.asarray(trunk.thick, float)
    section = np.asarray(trunk.section, float)
    reverse_count = int(
        np.sum(np.diff(surface_active) > float(config["gates"]["reverse_slope_tolerance_m"]))
    )
    terminal_thickness = float(thick[-1])

    gates = config["gates"]
    flags = {
        "single_root": len(model.fls) == 1 and model.fls[0].flows_to is None,
        "area_conservation": area_error <= float(gates["area_relative_error_max"]),
        "volume_conservation": volume_error <= float(gates["volume_relative_error_max"]),
        "hypsometry": hypsometry_error <= float(gates["hypsometry_cdf_error_max"]),
        "registered_length": length_error <= float(gates["length_relative_error_max"]),
        "free_downstream": free_tail >= float(gates["minimum_free_downstream_m"]),
        "monotonic_active_surface": reverse_count == 0,
        "finite_legal_state": bool(
            np.isfinite(surface).all()
            and np.isfinite(bed).all()
            and np.isfinite(widths_m).all()
            and np.isfinite(thick).all()
            and np.isfinite(section).all()
            and np.all(widths_m > 0)
            and np.all(thick >= 0)
            and np.all(section >= 0)
        ),
        "zero_ice_safety_tail": bool(
            np.all(thick[active_count:] <= float(gates["terminal_thickness_m_max"]))
            and terminal_thickness <= float(gates["terminal_thickness_m_max"])
        ),
        "stage2_supported": float(config["stage2"]["fs"]) == 0.0,
    }

    with gzip.open(output_root / "conservative_trunk_flowline.pkl.gz", "wb") as stream:
        pickle.dump([trunk], stream, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame({
        "node": np.arange(nx, dtype=int),
        "surface_h_m": surface,
        "bed_h_m": bed,
        "thickness_m": thick,
        "width_m": widths_m,
        "section_m2": section,
        "active_registered_prefix": np.arange(nx) < active_count,
    }).to_csv(output_root / "trunk_static_inventory.csv", index=False)

    after_hash = _sha256(source_path)
    source_unchanged = before_hash == after_hash
    flags["source_immutable"] = source_unchanged
    passed = bool(all(flags.values()))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "static-pass" if passed else "static-fail",
        "experiment": "native-network-conservative-trunk-condensation-v1",
        "stage": "static-preflight",
        "system_id": int(config["system_id"]),
        "source_state": "legal native standard-inversion multi-flowline network",
        "source_branch_count": len(flowlines),
        "source_root_index": int(root_index),
        "source_active_element_count": len(source["area_m2"]),
        "trunk_count": 1,
        "trunk_node_count": nx,
        "trunk_active_node_count": active_count,
        "node_spacing_m": spacing_m,
        "source_area_m2": source_area,
        "trunk_area_m2": model_area,
        "area_relative_error": area_error,
        "source_volume_m3": source_volume,
        "trunk_volume_m3": model_volume,
        "volume_relative_error": volume_error,
        "hypsometry_cdf_error_100m": hypsometry_error,
        "registered_lia_length_m": registered_length,
        "trunk_initial_length_m": observed_length,
        "registered_length_relative_error": length_error,
        "free_downstream_m": free_tail,
        "active_reverse_slope_count": reverse_count,
        "terminal_thickness_m": terminal_thickness,
        "stage2_frozen": config["stage2"],
        "gate_flags": flags,
        "source_sha256_before": before_hash,
        "source_sha256_after": after_hash,
        "source_unchanged": source_unchanged,
        "solver_instantiated": "SemiImplicitModel",
        "zero_mb_run": False,
        "steady_50y_run": False,
        "historical_run": False,
        "next_authorized_stage": "zero-mb" if passed else None,
    }
    _write_json(output_root / "static_summary.json", summary)
    _write_json(output_root / "config_snapshot.json", config)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run(args.config.resolve(), args.project_root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
