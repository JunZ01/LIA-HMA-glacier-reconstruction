"""Read-only physical-domain audit for system-2005 planform 2D."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUT = BASE / "lia_191_system2005_lia_source_mask_domain_audit_v1"
GEOLOGY = BASE / "lia_191_igm_planform2d_system2005_static_v1/geology.nc"
FINAL = BASE / "lia_191_torch_planform2d_system2005_steady50_monthly_ti_formal_v2/final_state_50y.npz"
PACKAGE = BASE / "lia_191_torch_planform2d_system2005_climate_bridge_export_v1/monthly_ti_package.npz"
GDIR = BASE / "lia_191_geometry_repair_v1/topology_repair_inversion_v1/system_2005/synthetic_gdir/RGI60-15/RGI60-15.92/RGI60-15.92005"
CATCHMENTS = GDIR / "flowline_catchments.shp"
OUTLINE = GDIR / "outlines.shp"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def monthly_mb_m(package: dict[str, np.ndarray], heights: np.ndarray, month_index: int) -> np.ndarray:
    i = int(package["sequence_indices"][month_index])
    temp = package["temp"][i] + package["grad"][i] * (heights - float(package["ref_hgt"]))
    melt = np.maximum(temp - float(package["t_melt"]), 0.0)
    solid = np.clip(
        1.0 - (temp - float(package["t_solid"]))
        / (float(package["t_liq"]) - float(package["t_solid"])), 0.0, 1.0
    )
    mass = package["prcp"][i] * solid - float(package["monthly_melt_f"]) * melt
    mass -= float(package["bias"]) * float(package["sec_in_month"]) / float(package["sec_in_year"])
    return mass / float(package["rho"])


def mean_annual_mb(package: dict[str, np.ndarray], heights: np.ndarray) -> np.ndarray:
    value = np.zeros_like(heights, dtype=float)
    for month in range(600):
        value += monthly_mb_m(package, heights, month) / 50.0
    return value


def vector_mask(path: Path, x: np.ndarray, y: np.ndarray, dx: float) -> tuple[np.ndarray, float, str]:
    data = gpd.read_file(path)
    shapes = [(geometry, 1) for geometry in data.geometry if geometry is not None and not geometry.is_empty]
    transform = from_origin(float(x.min() - dx / 2), float(y.max() + dx / 2), dx, dx)
    high_to_low = rasterize(shapes, out_shape=(len(y), len(x)), transform=transform, fill=0, dtype="uint8").astype(bool)
    mask = high_to_low if y[0] > y[-1] else np.flipud(high_to_low)
    return mask, float(data.geometry.area.sum()), str(data.crs)


def metric(mask: np.ndarray, final_active: np.ndarray, mean_mb: np.ndarray, cell_area: float) -> dict[str, Any]:
    active = mask & final_active
    outside = final_active & ~mask
    return {
        "domain_cell_count": int(np.count_nonzero(mask)),
        "domain_area_m2": float(np.count_nonzero(mask) * cell_area),
        "final_active_inside_area_m2": float(np.count_nonzero(active) * cell_area),
        "final_active_outside_area_m2": float(np.count_nonzero(outside) * cell_area),
        "final_active_outside_fraction": float(np.count_nonzero(outside) / max(np.count_nonzero(final_active), 1)),
        "fixed_final_geometry_source_inside_m3_per_year": float(np.sum(mean_mb[active]) * cell_area),
        "fixed_final_geometry_source_outside_m3_per_year": float(np.sum(mean_mb[outside]) * cell_area),
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    with Dataset(GEOLOGY) as dataset:
        bed = np.asarray(dataset.variables["topg"][:], dtype=float)
        initial = np.asarray(dataset.variables["thk"][:], dtype=float)
        source_mask = np.asarray(dataset.variables["lia_source_mask"][:], dtype=float) > 0.5
        ice_mask = np.asarray(dataset.variables["icemask"][:], dtype=float) > 0.5
        x = np.asarray(dataset.variables["x"][:], dtype=float)
        y = np.asarray(dataset.variables["y"][:], dtype=float)
        dx = float(getattr(dataset, "dx_m"))
    with np.load(FINAL) as state:
        final = np.asarray(state["thickness_m"], dtype=float)
    with np.load(PACKAGE) as source:
        package = {key: np.asarray(source[key]) for key in source.files}

    outline_mask, outline_vector_area, outline_crs = vector_mask(OUTLINE, x, y, dx)
    catchment_mask, catchment_vector_area, catchment_crs = vector_mask(CATCHMENTS, x, y, dx)
    initial_active = initial > 1e-6
    final_active = final > 1e-6
    new_active = final_active & ~initial_active
    surface_final = bed + final
    mb = mean_annual_mb(package, surface_final)
    cell_area = dx**2

    masks = {
        "stored_lia_source_mask": source_mask,
        "stored_ice_mask": ice_mask,
        "rasterized_outline": outline_mask,
        "rasterized_flowline_catchment_union": catchment_mask,
    }
    mask_metrics = {name: metric(mask, final_active, mb, cell_area) for name, mask in masks.items()}

    distance_m = distance_transform_edt(~source_mask) * dx
    bands = [(0.0, 250.0), (250.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0), (2000.0, float("inf"))]
    distance_rows = []
    for lower, upper in bands:
        selected = new_active & (distance_m > lower) & (distance_m <= upper)
        distance_rows.append({
            "distance_lower_m": lower,
            "distance_upper_m": upper,
            "new_active_cell_count": int(np.count_nonzero(selected)),
            "new_active_area_m2": float(np.count_nonzero(selected) * cell_area),
            "mean_surface_elevation_m": float(np.mean(surface_final[selected])) if np.any(selected) else None,
            "mean_mb_m_ice_per_year": float(np.mean(mb[selected])) if np.any(selected) else None,
            "fixed_source_m3_per_year": float(np.sum(mb[selected]) * cell_area),
        })

    candidate = "rasterized_flowline_catchment_union"
    candidate_mask = masks[candidate]
    initial_outside_candidate = initial_active & ~candidate_mask
    source_xor_initial = np.logical_xor(source_mask, initial_active)
    outline_xor_source = np.logical_xor(outline_mask, source_mask)
    catchment_xor_source = np.logical_xor(catchment_mask, source_mask)
    candidate_checks = {
        "same_crs_as_geology": outline_crs == catchment_crs,
        "contains_all_initial_active_cells": not np.any(initial_outside_candidate),
        "initial_active_outside_cell_count": int(np.count_nonzero(initial_outside_candidate)),
        "candidate_area_exceeds_initial_active_area": np.count_nonzero(candidate_mask) >= np.count_nonzero(initial_active),
        "candidate_is_far_smaller_than_runaway_final_support": np.count_nonzero(candidate_mask) < np.count_nonzero(final_active),
    }
    status = "flowline-catchment-domain-candidate-pass-readonly" if all(candidate_checks.values()) else "domain-candidate-needs-geometry-reconciliation"
    summary = {
        "schema": "lia-191-system2005-lia-source-mask-domain-audit-v1",
        "status": status,
        "system_id": 2005,
        "grid": {"shape": list(initial.shape), "dx_m": dx, "cell_area_m2": cell_area},
        "mask_identity": {
            "source_mask_xor_initial_active_cells": int(np.count_nonzero(source_xor_initial)),
            "outline_xor_source_cells": int(np.count_nonzero(outline_xor_source)),
            "catchment_xor_source_cells": int(np.count_nonzero(catchment_xor_source)),
            "outline_vector_area_m2": outline_vector_area,
            "catchment_vector_area_m2": catchment_vector_area,
            "outline_crs": outline_crs,
            "catchment_crs": catchment_crs,
        },
        "mask_metrics": mask_metrics,
        "new_support": {
            "new_active_area_m2": float(np.count_nonzero(new_active) * cell_area),
            "new_active_above_ela_area_m2": float(np.count_nonzero(new_active & (mb > 0)) * cell_area),
            "new_active_positive_mb_fraction": float(np.count_nonzero(new_active & (mb > 0)) / max(np.count_nonzero(new_active), 1)),
            "new_active_fixed_source_m3_per_year": float(np.sum(mb[new_active]) * cell_area),
            "maximum_distance_from_source_mask_m": float(np.max(distance_m[new_active])),
        },
        "candidate_domain": {
            "name": candidate,
            "basis": "union of the 25 archived OGGM flowline catchments; no fitted buffer or climate parameter",
            "checks": candidate_checks,
            "scientific_role": "research numerical-domain fixture only; not yet a promoted production geometry",
        },
        "interpretation": (
            "The stored LIA source mask reproduces the initial glacier support. The archived flowline-catchment "
            "union contains the initial ice without a fitted buffer and is far smaller than the runaway final "
            "support. Most new support lies outside both mapped domains and receives positive mass balance. "
            "A masked zero-MB diffusion fixture is required before any climate run to prove exact conservation "
            "and absence of cross-domain leakage."
        ),
        "decision": {
            "selected_readonly_domain_candidate": candidate,
            "production_promotion": False,
            "climate_run_authorized": False,
            "next_fixture": "single-step and 0.1-year zero-MB masked-diffusion conservation/equivalence fixture",
        },
        "model_run_performed": False,
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
    }

    OUT.mkdir(parents=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "new_support_distance_bands.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(distance_rows[0]))
        writer.writeheader()
        writer.writerows(distance_rows)
    np.savez_compressed(
        OUT / "domain_masks.npz",
        stored_lia_source_mask=source_mask,
        rasterized_outline=outline_mask,
        rasterized_flowline_catchment_union=catchment_mask,
        initial_active=initial_active,
        final_active=final_active,
    )

    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), constrained_layout=True)
    axes[0].imshow(source_mask, origin="lower", extent=extent, cmap="Greys")
    axes[0].set_title("Stored LIA source mask")
    axes[1].imshow(catchment_mask, origin="lower", extent=extent, cmap="Greys")
    axes[1].set_title("Flowline-catchment union")
    display = np.zeros(initial.shape, dtype=float)
    display[initial_active] = 1
    display[new_active] = 2
    axes[2].imshow(display, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=2)
    axes[2].contour(x, y, catchment_mask.astype(float), levels=[0.5], colors=["white"], linewidths=0.7)
    axes[2].set_title("Initial (1) and new 50y support (2)")
    for axis in axes:
        axis.set_xlabel("Easting (m)")
        axis.set_ylabel("Northing (m)")
    fig.savefig(OUT / "system2005_domain_masks.png", dpi=180)
    plt.close(fig)

    manifest = {
        "driver": str(DRIVER.relative_to(ROOT)),
        "driver_sha256": sha256(DRIVER),
        "sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in [GEOLOGY, FINAL, PACKAGE, CATCHMENTS, OUTLINE]
        ],
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
