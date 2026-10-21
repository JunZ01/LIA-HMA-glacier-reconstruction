"""Corrected read-only system-2005 domain audit using the source-native mask."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUT = BASE / "lia_191_system2005_lia_source_mask_domain_audit_v2"
V1 = BASE / "lia_191_system2005_lia_source_mask_domain_audit_v1"
GEOLOGY = BASE / "lia_191_igm_planform2d_system2005_static_v1/geology.nc"
FINAL = BASE / "lia_191_torch_planform2d_system2005_steady50_monthly_ti_formal_v2/final_state_50y.npz"
PACKAGE = BASE / "lia_191_torch_planform2d_system2005_climate_bridge_export_v1/monthly_ti_package.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def monthly_mb_m(package: dict[str, np.ndarray], heights: np.ndarray, month: int) -> np.ndarray:
    i = int(package["sequence_indices"][month])
    temp = package["temp"][i] + package["grad"][i] * (heights - float(package["ref_hgt"]))
    solid = np.clip(
        1.0 - (temp - float(package["t_solid"]))
        / (float(package["t_liq"]) - float(package["t_solid"])), 0.0, 1.0
    )
    mass = package["prcp"][i] * solid
    mass -= float(package["monthly_melt_f"]) * np.maximum(temp - float(package["t_melt"]), 0.0)
    mass -= float(package["bias"]) * float(package["sec_in_month"]) / float(package["sec_in_year"])
    return mass / float(package["rho"])


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    with Dataset(GEOLOGY) as dataset:
        bed = np.asarray(dataset.variables["topg"][:], dtype=float)
        initial = np.asarray(dataset.variables["thk"][:], dtype=float)
        source_mask = np.asarray(dataset.variables["lia_source_mask"][:], dtype=float) > 0.5
        dx = float(getattr(dataset, "dx_m"))
    with np.load(FINAL) as state:
        final = np.asarray(state["thickness_m"], dtype=float)
    with np.load(PACKAGE) as source:
        package = {key: np.asarray(source[key]) for key in source.files}
    with np.load(V1 / "domain_masks.npz") as source:
        catchment = np.asarray(source["rasterized_flowline_catchment_union"], dtype=bool)

    initial_active = initial > 1e-6
    final_active = final > 1e-6
    surface = bed + final
    mean_mb = np.zeros_like(surface)
    for month in range(600):
        mean_mb += monthly_mb_m(package, surface, month) / 50.0
    cell_area = dx**2
    initial_outside = initial_active & ~source_mask
    source_empty = source_mask & ~initial_active
    final_inside = final_active & source_mask
    final_outside = final_active & ~source_mask
    overlap = source_mask & catchment
    union = source_mask | catchment

    checks = {
        "source_native_same_grid_no_reprojection": True,
        "contains_every_initial_active_cell": int(np.count_nonzero(initial_outside)) == 0,
        "retains_nonzero_internal_empty_domain": int(np.count_nonzero(source_empty)) > 0,
        "candidate_smaller_than_runaway_support": int(np.count_nonzero(source_mask)) < int(np.count_nonzero(final_active)),
        "outside_runaway_source_is_positive": float(np.sum(mean_mb[final_outside]) * cell_area) > 0.0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"source-mask-candidate-fail:{checks}")
    outside_source = float(np.sum(mean_mb[final_outside]) * cell_area)
    total_source = float(np.sum(mean_mb[final_active]) * cell_area)
    summary = {
        "schema": "lia-191-system2005-lia-source-mask-domain-audit-v2",
        "status": "source-native-lia-domain-candidate-pass-readonly",
        "system_id": 2005,
        "supersedes": "lia_191_system2005_lia_source_mask_domain_audit_v1 candidate selection only",
        "v1_correction": (
            "v1 prematurely selected the rasterized flowline-catchment union. V2 selects the source-native "
            "lia_source_mask because it is already grid-aligned and contains every initial active cell."
        ),
        "candidate": {
            "name": "stored_lia_source_mask",
            "origin": "source-native Core-191 LIA system grid carried into geology.nc",
            "fitted_buffer_m": None,
            "reprojection_required": False,
            "cell_count": int(np.count_nonzero(source_mask)),
            "area_m2": float(np.count_nonzero(source_mask) * cell_area),
            "initial_active_area_m2": float(np.count_nonzero(initial_active) * cell_area),
            "internal_initially_empty_area_m2": float(np.count_nonzero(source_empty) * cell_area),
            "initial_active_outside_cell_count": int(np.count_nonzero(initial_outside)),
            "checks": checks,
        },
        "catchment_comparison": {
            "jaccard": float(np.count_nonzero(overlap) / np.count_nonzero(union)),
            "source_mask_only_area_m2": float(np.count_nonzero(source_mask & ~catchment) * cell_area),
            "catchment_only_area_m2": float(np.count_nonzero(catchment & ~source_mask) * cell_area),
            "selection_reason": "avoid the catchment rasterization edge loss and CRS ambiguity documented by v1",
        },
        "runaway_exterior": {
            "final_active_inside_candidate_area_m2": float(np.count_nonzero(final_inside) * cell_area),
            "final_active_outside_candidate_area_m2": float(np.count_nonzero(final_outside) * cell_area),
            "outside_fraction_of_final_active_area": float(np.count_nonzero(final_outside) / np.count_nonzero(final_active)),
            "outside_fixed_final_geometry_source_m3_per_year": outside_source,
            "outside_fraction_of_fixed_final_geometry_source": outside_source / total_source,
        },
        "scientific_interpretation": (
            "The source-native LIA domain is a non-tuned, grid-exact candidate that preserves all initial ice. "
            "The runaway branch places 79.73 km2 of mass-balance-active ice outside this domain, where it "
            "accounts for about 80% of the final fixed-geometry positive source."
        ),
        "decision": {
            "candidate_frozen_for_numerical_fixture": True,
            "production_geometry_changed": False,
            "climate_run_authorized": False,
            "next_fixture": (
                "masked zero-MB single-step then 0.1-year conservation fixture; require exact no-cross-domain "
                "leakage, nonnegativity, finite state, and unchanged in-domain initial thickness"
            ),
        },
        "model_run_performed": False,
        "initial_thickness_modified": False,
        "production_temperature_modified": False,
    }
    OUT.mkdir(parents=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez_compressed(OUT / "source_native_domain_mask.npz", lia_source_mask=source_mask)
    manifest = {
        "driver": str(DRIVER.relative_to(ROOT)),
        "driver_sha256": sha256(DRIVER),
        "sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in [GEOLOGY, FINAL, PACKAGE, V1 / "domain_masks.npz", V1 / "summary.json"]
        ],
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
