"""Full-chain modern replay on the three frozen controls (NumPy exports only).

Runs the production surface chain on the MODERN footprint, where a reference
thickness exists, so the whole chain -- tau calibration, flowline network
solve, raster constraints, 2-D extension, volume -- can be compared with it:

  step 0  harness check: replay the legacy tau calibration (1-D inversion bed,
          independent profiles anchored at the stored reference surface,
          native grid, 132 m exclusion) with the production ``_evaluate`` and
          require the archived legacy tau to be reproduced.
  step 1  production-style modern chain on the saved LIA grid: unified 2-D bed
          B = topo - H_ref sampled bilinearly on the projected flowlines,
          explicit connectors (alpha = beta = 0, avoid existing cells),
          16x integration refinement, network solve, production constraint
          collection, 100 m raster boundary exclusion, then the legacy harmonic
          thickness operator and operator A.
  step 2  the full tau curve (40-260 kPa grid) for both outlet conventions, and
          a tau refit on the unified chain with the legacy loss form.

Factors, all reported, none silently chosen:
  tau source        legacy (archived, 1-D bed) | refit (unified 2-D bed, same chain)
  outlet convention H0 (production: outlet starts at bed) |
                    Href (outlet starts at modern DEM surface)
  operator          legacy harmonic thickness | A harmonic surface minus bed

Disclosed departures from production: raster (not vector) distance for the
100 m exclusion; modern footprint and modern OGGM flowlines instead of LIA
ones. The reference is an OGGM redistribution calibrated to F19 total volume,
not observation, so this is a self-consistency replay, not validation, and it
does not test temporal transfer to the LIA.

Usage (repo root):  python replay_modern_fullchain_20261021.py NPZ_DIR OUT_DIR
"""

from __future__ import annotations

import ast
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
PKG_PARENT = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized"
sys.path.insert(0, str(PKG_PARENT))
try:  # gridded_bed imports xarray at module level; it is not used here.
    import xarray  # noqa: F401
except ImportError:
    sys.modules["xarray"] = types.ModuleType("xarray")

from ice_surface_reconstruction.core import (  # noqa: E402
    SurfaceScenario,
    harmonic_extend_thickness,
    refine_flowline_network,
    sample_network_solution_at_original_nodes,
    solve_flowline_network,
    surface_extend_thickness,
)
from ice_surface_reconstruction.drivers.gridded_bed import GriddedBed  # noqa: E402
from ice_surface_reconstruction.drivers.junction_connector import ConnectorCostWeights  # noqa: E402
from ice_surface_reconstruction.drivers.modern_tau_calibration import (  # noqa: E402
    _evaluate,
    _interior_truth_points,
)
from ice_surface_reconstruction.drivers.oggm_network_adapter import (  # noqa: E402
    EXPLICIT_CONNECTOR,
    FlowlineInput,
    build_network,
)

SYSTEMS = (3632, 3131, 2906)
TAU_GRID_KPA = [40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0, 240.0, 260.0]
OUTLETS = ("H0", "Href")
LEGACY_TOLERANCE_KPA = 0.05


def _production_function(relative_path: str, name: str):
    """Exec one function verbatim from a production module whose other imports
    (geopandas, shapely, xarray) are unavailable here."""
    path = PKG_PARENT / "ice_surface_reconstruction" / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np, "annotations": None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


collect_network_constraints = _production_function(
    "drivers/lia_3d_zero_boundary_pilot.py", "collect_network_constraints")


def load(npz_dir: Path, uid: int) -> dict:
    z = np.load(npz_dir / f"fullchain_inputs_{uid}.npz", allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["config"] = json.loads(str(d["legacy_config_json"]))
    d["uid"] = uid
    return d


def flowlines(d: dict, frame: str, bed_key: str, surface_key: str) -> list[FlowlineInput]:
    off = d["flowline_offsets"]
    xy = d[f"flowline_xy_{frame}"]
    map_dx = float(d["native_dx_m"] if frame == "native" else d["dx_m"])
    out = []
    for i in range(len(off) - 1):
        s = slice(off[i], off[i + 1])
        parent = int(d["flowline_flows_to"][i])
        join = int(d["flowline_flows_to_indice"][i])
        out.append(FlowlineInput(
            flowline_index=int(d["flowline_index"][i]),
            flows_to=None if parent < 0 else parent,
            flows_to_indice=None if join < 0 else join,
            bed_elevation_m=np.asarray(d[bed_key][s], float),
            reference_surface_m=np.asarray(d[surface_key][s], float),
            dx_meter=float(d["flowline_dx_m"][i]),
            map_dx_m=map_dx,
            coords_xy=np.asarray(xy[s], float),
        ))
    return out


# ---------------------------------------------------------------- step 0
def legacy_calibration(d: dict) -> dict:
    """Mirror calibrate_system's search, calling the production _evaluate."""
    cal = d["config"]["modern_tau_calibration"]
    off = d["flowline_offsets"]
    inputs = flowlines(d, "native", "flowline_inversion_bed_m", "flowline_surface_oggm_m")
    truth = {item.flowline_index: (np.asarray(d["flowline_surface_oggm_m"][off[i]:off[i + 1]], float),
                                   np.asarray(d["flowline_inversion_H_m"][off[i]:off[i + 1]], float))
             for i, item in enumerate(inputs)}
    coords = {item.flowline_index: item.coords_xy for item in inputs}
    mask = np.asarray(d["native_modern_mask"], bool)
    dx = float(d["native_dx_m"])
    cache: dict[float, dict] = {}

    def evaluate(value):
        value = round(float(value), 6)
        if value not in cache:
            cache[value] = _evaluate(value, inputs, truth, mask, dx, cal, coords)
        return cache[value]

    for value in cal["coarse_grid_kpa"]:
        evaluate(value)
    valid = [r for r in cache.values() if np.isfinite(r.get("loss", np.nan))]
    best = min(valid, key=lambda r: r["loss"])
    values = sorted(r["tau_kpa"] for r in valid)
    lo = max(min(values), best["tau_kpa"] - cal["refinement_half_width_kpa"])
    hi = min(max(values), best["tau_kpa"] + cal["refinement_half_width_kpa"])
    if hi > lo:
        res = minimize_scalar(lambda v: evaluate(v).get("loss", np.inf), bounds=(lo, hi),
                              method="bounded", options={"xatol": cal["refinement_xatol_kpa"]})
        evaluate(float(res.x))
    valid = [r for r in cache.values() if np.isfinite(r.get("loss", np.nan))]
    best = min(valid, key=lambda r: r["loss"])
    return {"tau_kpa": best["tau_kpa"], "loss": best["loss"],
            "thickness_rmse_m": best["thickness_rmse_m"],
            "usable_point_count": best["usable_point_count"]}


# ---------------------------------------------------------------- step 1
class ModernChain:
    """Production network chain on the saved LIA grid with the unified bed."""

    def __init__(self, d: dict):
        self.d = d
        cfg = d["config"]
        self.factor = int(cfg["lia_3d_refined_flowline_pilot_v1"]["flowline_integration_refinement_factor"])
        self.shape = float(d["shape_factor"])
        self.dx = float(d["dx_m"])
        self.mask = np.asarray(d["modern_mask"], bool)
        self.bed2d = np.asarray(d["bed_m"], float)
        self.topo = np.asarray(d["topo_m"], float)
        self.href = np.where(self.mask, np.asarray(d["reference_H_m"], float), 0.0)
        gy, gx = np.gradient(self.bed2d, self.dx, self.dx)
        self.gridded = GriddedBed(bed_m=self.bed2d, glacier_mask=self.mask,
                                  slope=np.arctan(np.hypot(gx, gy)), map_dx_m=self.dx,
                                  bed_scenario="farinotti2019-unified-2d")
        self.inputs = flowlines(d, "lia", "flowline_bed_lia_bilinear_m", "flowline_surface_dem_lia_m")
        self.build = build_network(
            self.inputs, junction_transfer_policy=EXPLICIT_CONNECTOR, gridded_bed=self.gridded,
            connector_weights=ConnectorCostWeights(alpha=0.0, beta=0.0),
            avoid_existing_network_cells=bool(cfg.get("production_connector", {}).get(
                "avoid_existing_network_cells", True)))
        off = d["flowline_offsets"]
        self.truth = {item.flowline_index: (np.asarray(d["flowline_surface_dem_lia_m"][off[i]:off[i + 1]], float),
                                            np.asarray(d["flowline_reference_H_lia_m"][off[i]:off[i + 1]], float))
                      for i, item in enumerate(self.inputs)}
        self.coords = {item.flowline_index: item.coords_xy for item in self.inputs}
        trunk = next(i for i, item in enumerate(self.inputs) if item.flows_to is None)
        self.outlet_surface_m = float(d["flowline_surface_dem_lia_m"][off[trunk + 1] - 1])
        self.outlet_reference_H_m = float(d["flowline_reference_H_lia_m"][off[trunk + 1] - 1])
        boundary = self.mask & ~binary_erosion(self.mask, border_value=0)
        self.boundary = boundary
        self.boundary_distance = distance_transform_edt(~boundary) * self.dx
        frozen = cfg["lia_3d_refined_flowline_pilot_v1"]
        self.exclusion_m = max(float(frozen["head_exclusion_minimum_m"]),
                               float(frozen["head_exclusion_grid_multiplier"]) * self.dx)
        pilot = cfg["lia_3d_zero_boundary_pilot"]
        self.dup_tol = float(pilot["duplicate_surface_tolerance_m"])
        self.cross_tol = max(self.dup_tol, float(pilot.get("cross_branch_surface_tolerance_grid_fraction", 0.0)) * self.dx)
        self.ref_volume_km3 = float(self.href[self.mask].sum() * self.dx ** 2 / 1e9)
        self.inner = self.mask & ~boundary

    def solve(self, tau_kpa: float, outlet: str):
        scenario = SurfaceScenario("modern-replay", tau_kpa * 1000.0, self.shape)
        outlet_surface = ({node: self.outlet_surface_m for node in self.build.outlet_node_ids}
                          if outlet == "Href" else None)
        network = solve_flowline_network(refine_flowline_network(self.build.branches, self.factor),
                                         scenario, outlet_node_ids=self.build.outlet_node_ids,
                                         outlet_surface_m=outlet_surface)
        if not network.success:
            return None, {"status": f"network-failed:{network.status}"}
        network = sample_network_solution_at_original_nodes(self.build.branches, network, self.factor, scenario)
        return network, {"status": "success"}

    def flowline_fit(self, network) -> dict:
        cal = self.d["config"]["modern_tau_calibration"]
        model, observed, meta = _interior_truth_points(
            self.build, network, self.truth, self.mask, self.dx,
            float(cal["boundary_exclusion_m"]), float(cal["minimum_thickness_m"]), self.coords)
        if model.size < int(cal["minimum_points"]):
            return {"loss": np.nan, "flowline_rmse_m": np.nan, "flowline_bias_m": np.nan, **meta}
        rmse = float(np.sqrt(np.mean((model - observed) ** 2)))
        return {"loss": rmse / max(float(np.median(observed)), 1.0), "flowline_rmse_m": rmse,
                "flowline_bias_m": float(np.mean(model - observed)), **meta}

    def extend(self, network) -> dict:
        rows, cols, thick, surf, cmeta = collect_network_constraints(
            self.inputs, self.build, network, duplicate_surface_tolerance_m=self.dup_tol,
            cross_branch_surface_tolerance_m=self.cross_tol)
        inside = self.mask[rows, cols]
        rows, cols, thick, surf = rows[inside], cols[inside], thick[inside], surf[inside]
        keep = ~self.boundary[rows, cols] & (self.boundary_distance[rows, cols] >= self.exclusion_m)
        rows, cols, thick, surf = rows[keep], cols[keep], thick[keep], surf[keep]
        out = {"constraints": int(len(rows)), "constraints_outside_modern_mask": int((~inside).sum())}
        legacy, meta = harmonic_extend_thickness(
            self.mask, rows, cols, thick, centerline_surface_m=surf,
            duplicate_tolerance_m=self.dup_tol, boundary_tolerance_m=1e-3,
            max_iterations=10000, tolerance_m=1e-3, method="conjugate-gradient",
            cg_relative_tolerance=1e-10)
        assert meta["status"] == "converged", meta
        # The network integrates on the bilinear bed at the flowline points; a
        # constraint cell has its own raster bed. Give A the same thickness the
        # legacy operator receives (S_cell = B_cell + H) so both operators share
        # identical constraints, and record the bed mismatch this absorbs.
        bed_gap = surf - thick - self.bed2d[rows, cols]
        out["constraint_bed_bilinear_minus_cell_rmse_m"] = float(np.sqrt(np.mean(bed_gap ** 2)))
        out["constraint_bed_bilinear_minus_cell_max_abs_m"] = float(np.max(np.abs(bed_gap)))
        a, ameta = surface_extend_thickness(self.mask, self.bed2d, rows, cols,
                                            self.bed2d[rows, cols] + thick,
                                            duplicate_tolerance_m=np.inf)
        for name, h, s in (("legacy", legacy, self.bed2d + legacy), ("A", a, ameta["surface_m"])):
            v = float(h[self.mask].sum() * self.dx ** 2 / 1e9)
            e = s[self.inner] - self.topo[self.inner]
            out[f"{name}_volume_km3"] = v
            out[f"{name}_volume_bias_percent"] = 100 * (v / self.ref_volume_km3 - 1)
            out[f"{name}_thickness_rmse_m"] = float(np.sqrt(np.mean((h[self.mask] - self.href[self.mask]) ** 2)))
            out[f"{name}_surface_rmse_vs_dem_m"] = float(np.sqrt(np.mean(e ** 2)))
            out[f"{name}_surface_mean_error_m"] = float(np.mean(e))
        out["A_negative_raw_cells_percent"] = 100 * ameta["negative_raw_thickness_fraction"]
        out["_fields"] = {"legacy": legacy, "A": a}
        return out


def run(npz_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    harness, curves, summary = [], [], []
    for uid in SYSTEMS:
        d = load(npz_dir, uid)
        # step 0
        legacy = legacy_calibration(d)
        archived = float(d["legacy_tau_kpa"])
        ok = abs(legacy["tau_kpa"] - archived) <= LEGACY_TOLERANCE_KPA
        harness.append({"lia_unique_id": uid, "archived_legacy_tau_kpa": archived,
                        "replayed_legacy_tau_kpa": legacy["tau_kpa"],
                        "replayed_thickness_rmse_m": legacy["thickness_rmse_m"],
                        "replayed_points": legacy["usable_point_count"], "reproduced": ok})
        print(json.dumps(harness[-1]), flush=True)
        if not ok:
            raise RuntimeError(f"legacy tau not reproduced for {uid}; replay stopped before any new choice")
        # steps 1-2
        chain = ModernChain(d)
        for outlet in OUTLETS:
            for tau in TAU_GRID_KPA:
                network, status = chain.solve(tau, outlet)
                row = {"lia_unique_id": uid, "outlet": outlet, "tau_kpa": tau, **status}
                if network is not None:
                    row.update(chain.flowline_fit(network))
                    ext = chain.extend(network)
                    ext.pop("_fields")
                    row.update(ext)
                curves.append(row)
            # refit on the unified chain, legacy loss form and search
            cal = d["config"]["modern_tau_calibration"]
            cache = {}

            def loss(v):
                v = round(float(v), 6)
                if v not in cache:
                    net, _ = chain.solve(v, outlet)
                    cache[v] = chain.flowline_fit(net)["loss"] if net is not None else np.inf
                return cache[v] if np.isfinite(cache[v]) else np.inf

            grid = [r for r in curves if r["lia_unique_id"] == uid and r["outlet"] == outlet
                    and np.isfinite(r.get("loss", np.nan))]
            best = min(grid, key=lambda r: r["loss"])["tau_kpa"]
            lo, hi = max(TAU_GRID_KPA[0], best - 20.0), min(TAU_GRID_KPA[-1], best + 20.0)
            refit = float(minimize_scalar(loss, bounds=(lo, hi), method="bounded",
                                          options={"xatol": cal["refinement_xatol_kpa"]}).x)
            for source, tau in (("legacy", archived), ("refit", refit)):
                network, status = chain.solve(tau, outlet)
                row = {"lia_unique_id": uid, "region": str(d["region"]), "outlet": outlet,
                       "tau_source": source, "tau_kpa": tau,
                       "outlet_reference_H_m": chain.outlet_reference_H_m,
                       "reference_volume_km3": chain.ref_volume_km3, **status}
                if network is not None:
                    row.update(chain.flowline_fit(network))
                    ext = chain.extend(network)
                    fields = ext.pop("_fields")
                    row.update(ext)
                    np.savez_compressed(out_dir / f"fields_{uid}_{outlet}_{source}.npz",
                                        legacy_H_m=fields["legacy"].astype("float32"),
                                        A_H_m=fields["A"].astype("float32"))
                summary.append(row)
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                                  if k in ("lia_unique_id", "outlet", "tau_source", "tau_kpa", "status",
                                           "flowline_rmse_m", "legacy_volume_bias_percent",
                                           "A_volume_bias_percent")}), flush=True)
    pd.DataFrame(harness).to_csv(out_dir / "legacy_tau_harness_check.csv", index=False)
    pd.DataFrame(curves).to_csv(out_dir / "tau_curves.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "fullchain_summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "status": "completed", "elapsed_seconds": time.perf_counter() - started,
        "systems": list(SYSTEMS), "tau_grid_kpa": TAU_GRID_KPA, "outlets": list(OUTLETS),
        "legacy_tau_reproduced": all(r["reproduced"] for r in harness),
        "frame": "saved LIA grid; unified bed B = topo - H_ref; projected modern OGGM flowlines",
        "departures_from_production": ["100 m exclusion uses raster distance, not the vector polygon",
                                       "modern footprint and modern flowlines instead of LIA ones",
                                       "Href outlet uses network-solver outlet_surface_m (new optional argument)"],
        "scope": "self-consistency replay against an OGGM reference; not validation; no LIA, climate or dynamic run",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
