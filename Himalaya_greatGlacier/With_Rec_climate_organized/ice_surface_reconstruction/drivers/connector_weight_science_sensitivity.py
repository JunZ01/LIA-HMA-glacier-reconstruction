"""Science-quantity sensitivity of the connector weights (step 6).

Runs ONLY the representative (alpha, beta) weights (from the calibration's
`representative_alphas`: control / knee / high) through the full explicit-connector
network on the 12 modern pilots x 2 beds, and asks the decision question: does the
connector weight change the reconstructed science quantities more or less than the
Farinotti-vs-Millan bed choice already does?

For each science quantity Q it reports, per glacier:

* ``alpha_effect`` = max |Q(alpha_i) - Q(alpha_j)| across the representative alphas
  (same bed), i.e. how much the weight moves Q;
* ``bed_effect``   = |Q(farinotti) - Q(millan)| at the knee alpha, i.e. how much the
  bed product already moves Q;
* ``effect_ratio`` = alpha_effect / bed_effect.

Reported quantities (modern Level B PROXIES — NOT 3-D volume): network solve rate,
modern surface RMSE, modern thickness RMSE, tributary-start thickness, and a
flowline thickness/volume proxy (sum of thickness x dx over solved OGGM points).
The true LIA 3-D thickness/volume sensitivity must be redone once the LIA
polygon/flowline entry is wired; this modern proxy must not stand in for it.

Decision rule: if ``effect_ratio`` is small (weight moves Q less than the bed does)
a single point weight is unnecessary — keep the small ensemble in production. If it
is large, the weight is not closed and must not enter the 191-system run.

Outputs (new names; never overwrites ``full_network_*_v0.csv``):
``connector_weight_science_sensitivity_v0.csv`` (per glacier x weight x bed) and
``connector_weight_science_effect_ratio_v0.csv`` (per-quantity effect ratios with
per-glacier median + P90).
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import solve_flowline_network
from ..paths import resolve
from .connector_weight_calibration import representative_alphas
from .full_network_validation import (
    _THICK_SUFFIX,
    _real_bed_inputs,
    _rmse,
    _scenario,
    _unique_point_errors,
)
from .gridded_bed import load_gridded_bed
from .junction_connector import ConnectorCostWeights
from .oggm_flowline_io import load_flowline_inputs
from .oggm_network_adapter import EXPLICIT_CONNECTOR, build_network


def _science_quantities(build, solution, truth) -> dict:
    """Modern Level B proxies for one solved network."""
    surf_err, thick_err, n_pts = _unique_point_errors(build, solution, truth)
    # Flowline thickness/volume proxy: sum thickness * dx over solved OGGM points.
    prov = {p.branch_id: p for p in build.provenance}
    vol_proxy = 0.0
    trib_starts = []
    for branch_id, bs in solution.branch_solutions.items():
        p = prov.get(branch_id)
        if p is None:
            continue
        dx = 132.0  # OGGM inversion flowline dx (2 px * 66 m); proxy scale only
        vol_proxy += float(np.nansum(bs.thickness_m) * dx)
        # Tributary-start thickness = thickness at the upstream end of a leaf/trib.
        if p.source_flowline_index != 0 and len(bs.thickness_m):
            trib_starts.append(float(bs.thickness_m[-1]))
    return {
        "surface_rmse_m": _rmse(surf_err),
        "thickness_rmse_m": _rmse(thick_err),
        "n_unique_points": n_pts,
        "flowline_volume_proxy_m2": vol_proxy,
        "mean_tributary_start_thickness_m": float(np.mean(trib_starts)) if trib_starts else np.nan,
    }


def run(config: dict, weights: list[dict] | None = None) -> dict:
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    root = resolve(config["stage0_gdir_root"])
    pilot = pd.read_csv(resolve(config["pilot_selection_path"]))
    rgi_ids = sorted(set(pilot["main_rgi_id"].astype(str)))

    # Representative weights: from the calibration scan unless supplied.
    if weights is None:
        cal_csv = output_dir / "connector_weight_calibration_v0.csv"
        reps = representative_alphas(pd.read_csv(cal_csv)) if cal_csv.exists() else {}
        knee = reps.get("alpha_knee") or 30.0
        high = reps.get("alpha_high") or 100.0
        weights = [
            {"label": "control", "alpha": 0.0, "beta": 0.0},
            {"label": "knee", "alpha": float(knee), "beta": 0.0},
            {"label": "high", "alpha": float(high), "beta": 0.0},
        ]

    # Use the central stress scenario for the science comparison.
    central = next(s for s in config["scenarios"] if s["scenario_id"] == "central")
    scenario = _scenario(config, central)

    rows: list[dict] = []
    for rgi_id in rgi_ids:
        gdir_dir = root / rgi_id[:-6] / rgi_id[:-3] / rgi_id
        try:
            inputs, raw_fls = load_flowline_inputs(gdir_dir)
        except FileNotFoundError:
            continue
        for bed_scenario in config["bed_scenarios"]:
            inv_path = gdir_dir / f"inversion_output_{_THICK_SUFFIX[bed_scenario]}.pkl"
            if not inv_path.exists():
                continue
            with gzip.open(inv_path, "rb") as fh:
                inversion = pickle.load(fh)
            rebuilt, truth = _real_bed_inputs(inputs, raw_fls, inversion)
            try:
                bed2d = load_gridded_bed(gdir_dir, bed_scenario)
            except (FileNotFoundError, ValueError):
                continue
            for w in weights:
                cw = ConnectorCostWeights(alpha=w["alpha"], beta=w["beta"])
                try:
                    build = build_network(
                        rebuilt, junction_transfer_policy=EXPLICIT_CONNECTOR,
                        gridded_bed=bed2d, connector_weights=cw,
                    )
                    sol = solve_flowline_network(
                        build.branches, scenario, outlet_node_ids=build.outlet_node_ids
                    )
                except ValueError as exc:
                    rows.append({"rgi_id": rgi_id, "bed_scenario": bed_scenario,
                                 "weight_label": w["label"], "alpha": w["alpha"],
                                 "beta": w["beta"], "status": f"adapter-error:{exc}"})
                    continue
                row = {"rgi_id": rgi_id, "bed_scenario": bed_scenario,
                       "weight_label": w["label"], "alpha": w["alpha"], "beta": w["beta"],
                       "status": "success" if sol.success else f"failed:{sol.status}",
                       "solved": bool(sol.success)}
                if sol.success:
                    row.update(_science_quantities(build, sol, truth))
                rows.append(row)

    sens = pd.DataFrame(rows)
    sens.to_csv(output_dir / "connector_weight_science_sensitivity_v0.csv", index=False)
    effect = _effect_ratios(sens)
    effect.to_csv(output_dir / "connector_weight_science_effect_ratio_v0.csv", index=False)
    return {"sensitivity": sens, "effect_ratio": effect, "weights": weights}


_QUANTITIES = [
    "surface_rmse_m", "thickness_rmse_m",
    "flowline_volume_proxy_m2", "mean_tributary_start_thickness_m",
]


def _effect_ratios(sens: pd.DataFrame) -> pd.DataFrame:
    """Per-quantity alpha_effect / bed_effect with per-glacier median + P90."""
    if sens.empty or "surface_rmse_m" not in sens:
        return pd.DataFrame()
    solved = sens[sens.get("solved") == True]  # noqa: E712
    out = []
    for q in _QUANTITIES:
        per_glacier_ratio = []
        for rgi_id, g in solved.groupby("rgi_id"):
            # alpha effect: spread across weights within a bed, then max over beds.
            alpha_effects = []
            for bed, gb in g.groupby("bed_scenario"):
                vals = gb.dropna(subset=[q])[q]
                if len(vals) >= 2:
                    alpha_effects.append(float(vals.max() - vals.min()))
            if not alpha_effects:
                continue
            alpha_effect = max(alpha_effects)
            # bed effect at the knee weight.
            knee = g[g["weight_label"] == "knee"].dropna(subset=[q])
            if knee["bed_scenario"].nunique() >= 2:
                bed_effect = float(knee.groupby("bed_scenario")[q].first().max()
                                   - knee.groupby("bed_scenario")[q].first().min())
            else:
                bed_effect = np.nan
            if bed_effect and np.isfinite(bed_effect) and bed_effect > 0:
                per_glacier_ratio.append(alpha_effect / bed_effect)
        if per_glacier_ratio:
            arr = np.array(per_glacier_ratio)
            out.append({
                "quantity": q,
                "n_glaciers": len(arr),
                "effect_ratio_median": float(np.median(arr)),
                "effect_ratio_p90": float(np.percentile(arr, 90)),
                "alpha_dominates_fraction": float(np.mean(arr > 1.0)),
            })
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run(config)
    print("weights:", result["weights"])
    print("effect ratios (alpha vs bed):")
    print(result["effect_ratio"].to_string(index=False))


if __name__ == "__main__":
    main()
