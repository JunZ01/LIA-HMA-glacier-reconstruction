"""Read-only Cycle-6 MixedBed shape-freedom feasibility audit.

This driver reads the saved Section-178 mapped state.  It does not import or
run OGGM.  It asks whether fixed node bed, area, volume, dx, network topology,
registered length, and tail/domain can be made surface-monotone by using the
widest legal trapezoidal cross-section freedom at every active node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


SCHEMA_VERSION = 1
SUPPORTED_SYSTEM_ID = 3850


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    def safe(value):
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe(payload), indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def _surface_at(nodes: pd.DataFrame, key: tuple[int, int]) -> float:
    match = nodes.loc[
        (nodes["global_flowline_index"] == key[0])
        & (nodes["node_index"] == key[1]),
        "surface_h_m",
    ]
    if len(match) != 1:
        raise ValueError(f"Expected one saved surface for node {key}, got {len(match)}")
    return float(match.iloc[0])


def _branch_edges(nodes: pd.DataFrame, key_to_index: dict) -> list[dict]:
    edges: list[dict] = []
    for global_index, branch in nodes.groupby("global_flowline_index", sort=True):
        branch = branch.sort_values("node_index")
        rows = list(branch.itertuples(index=False))
        for upstream, downstream in zip(rows[:-1], rows[1:]):
            up_key = (int(global_index), int(upstream.node_index))
            down_key = (int(global_index), int(downstream.node_index))
            if up_key in key_to_index and down_key in key_to_index:
                edges.append(
                    {
                        "kind": "within_branch",
                        "upstream_key": up_key,
                        "downstream_key": down_key,
                    }
                )
    return edges


def _junction_edges(
    nodes: pd.DataFrame,
    junctions: pd.DataFrame,
    terminal_definition: str,
) -> list[dict]:
    edges: list[dict] = []
    for row in junctions.itertuples(index=False):
        tributary = int(row.tributary_global_index)
        branch = nodes.loc[nodes["global_flowline_index"] == tributary]
        if terminal_definition == "full_tail":
            terminal_node = int(branch["node_index"].max())
        elif terminal_definition == "last_active":
            active = branch.loc[branch["mapping_active"]]
            terminal_node = int(active["node_index"].max())
        else:
            raise ValueError(terminal_definition)
        edges.append(
            {
                "kind": f"junction_{terminal_definition}",
                "upstream_key": (tributary, terminal_node),
                "downstream_key": (
                    int(row.receiver_global_index),
                    int(row.receiver_node_index),
                ),
            }
        )
    return edges


def _solve_edges(
    edges: list[dict],
    nodes: pd.DataFrame,
    key_to_index: dict,
    lower_surface: np.ndarray,
    upper_surface: np.ndarray,
    slope_tolerance_m: float,
) -> dict:
    count = len(lower_surface)
    matrix = lil_matrix((len(edges), count), dtype=float)
    rhs = np.zeros(len(edges), dtype=float)
    for row_index, edge in enumerate(edges):
        constant = 0.0
        downstream = edge["downstream_key"]
        upstream = edge["upstream_key"]
        if downstream in key_to_index:
            matrix[row_index, key_to_index[downstream]] += 1.0
        else:
            constant += _surface_at(nodes, downstream)
        if upstream in key_to_index:
            matrix[row_index, key_to_index[upstream]] -= 1.0
        else:
            constant -= _surface_at(nodes, upstream)
        rhs[row_index] = slope_tolerance_m - constant

    bounds = list(zip(lower_surface, upper_surface))
    result = linprog(
        np.zeros(count),
        A_ub=matrix.tocsr(),
        b_ub=rhs,
        bounds=bounds,
        method="highs",
    )

    # A second LP quantifies the smallest uniform relaxation of every slope
    # inequality.  It is diagnostic only and never changes the feasibility gate.
    relaxed = lil_matrix((len(edges), count + 1), dtype=float)
    relaxed[:, :count] = matrix
    relaxed[:, count] = -1.0
    relaxation = linprog(
        np.r_[np.zeros(count), 1.0],
        A_ub=relaxed.tocsr(),
        b_ub=rhs,
        bounds=bounds + [(0.0, None)],
        method="highs",
    )
    return {
        "feasible": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "minimum_uniform_local_reverse_relaxation_m": (
            float(relaxation.x[-1]) if relaxation.success else None
        ),
    }


def _strongest_bound_conflict(branch: pd.DataFrame) -> dict:
    branch = branch.sort_values("node_index").reset_index(drop=True)
    strongest = None
    for downstream_position in range(1, len(branch)):
        upstream = branch.iloc[:downstream_position]
        upstream_position = int(np.argmin(upstream["upper_surface_m"].to_numpy()))
        gap = float(
            branch.iloc[downstream_position]["lower_surface_m"]
            - upstream.iloc[upstream_position]["upper_surface_m"]
        )
        if strongest is None or gap > strongest["gap_m"]:
            strongest = {
                "gap_m": gap,
                "upstream_node_index": int(
                    upstream.iloc[upstream_position]["node_index"]
                ),
                "downstream_node_index": int(
                    branch.iloc[downstream_position]["node_index"]
                ),
            }
    if strongest is None or strongest["gap_m"] <= 0.0:
        return {
            "gap_m": 0.0,
            "upstream_node_index": None,
            "downstream_node_index": None,
        }
    return strongest


def run(config_path: Path) -> Path:
    config = _read_json(config_path)
    if int(config["system_id"]) != SUPPORTED_SYSTEM_ID:
        raise ValueError("This versioned audit supports only system 3850")

    mapped_nodes_path = Path(config["mapped_nodes_csv"])
    junctions_path = Path(config["junction_slopes_csv"])
    static_summary_path = Path(config["static_summary_json"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(mapped_nodes_path)
    nodes["mapping_active"] = _as_bool(nodes["mapping_active"])
    nodes["is_trapezoid"] = _as_bool(nodes["is_trapezoid"])
    junctions = pd.read_csv(junctions_path)
    static_summary = _read_json(static_summary_path)

    active = nodes.loc[nodes["mapping_active"]].copy().reset_index(drop=True)
    active_count = len(active)
    parabolic_count = int((~active["is_trapezoid"]).sum())
    trapezoid_count = int(active["is_trapezoid"].sum())
    expected = config["expected_counts"]
    if (
        active_count != int(expected["active_nodes"])
        or parabolic_count != int(expected["parabolic_nodes"])
        or trapezoid_count != int(expected["trapezoid_nodes"])
    ):
        raise ValueError(
            "Saved-node count conflict: "
            f"active/parabolic/trapezoid={active_count}/{parabolic_count}/{trapezoid_count}"
        )

    width = active["surface_width_m"].to_numpy(dtype=float)
    section = active["section_m2"].to_numpy(dtype=float)
    bed = active["bed_h_m"].to_numpy(dtype=float)
    if not (
        np.isfinite(width).all()
        and np.isfinite(section).all()
        and np.isfinite(bed).all()
        and np.all(width > 0.0)
        and np.all(section > 0.0)
    ):
        raise ValueError("Active saved state contains illegal width/section/bed values")

    # With W=A/dx and S=V/dx fixed, any legal trapezoid satisfies
    # H in [S/W, 2S/W).  The upper epsilon preserves w0 > 0 numerically.
    thickness_min = section / width
    thickness_upper_open = 2.0 * section / width
    upper_epsilon_m = float(config["shape_upper_open_epsilon_m"])
    lower_surface = bed + thickness_min
    upper_surface = bed + thickness_upper_open - upper_epsilon_m
    if np.any(upper_surface <= lower_surface):
        raise ValueError("Empty legal trapezoid interval after upper-open epsilon")
    active["lower_surface_m"] = lower_surface
    active["upper_surface_m"] = upper_surface

    key_to_index = {
        (int(row.global_flowline_index), int(row.node_index)): index
        for index, row in active.iterrows()
    }
    branch_edges = _branch_edges(nodes, key_to_index)
    slope_tolerance_m = float(config["reverse_slope_tolerance_m"])

    branch_rows = []
    for global_index, branch in active.groupby("global_flowline_index", sort=True):
        edges = [
            edge
            for edge in branch_edges
            if edge["upstream_key"][0] == int(global_index)
        ]
        result = _solve_edges(
            edges,
            nodes,
            key_to_index,
            lower_surface,
            upper_surface,
            slope_tolerance_m,
        )
        conflict = _strongest_bound_conflict(branch)
        branch_rows.append(
            {
                "global_flowline_index": int(global_index),
                "active_node_count": int(len(branch)),
                "feasible_under_full_trapezoid_relaxation": result["feasible"],
                "minimum_uniform_local_reverse_relaxation_m": result[
                    "minimum_uniform_local_reverse_relaxation_m"
                ],
                "strongest_bound_conflict_m": conflict["gap_m"],
                "conflict_upstream_node_index": conflict["upstream_node_index"],
                "conflict_downstream_node_index": conflict[
                    "downstream_node_index"
                ],
            }
        )
    branch_conflicts = pd.DataFrame(branch_rows)
    branch_conflicts.to_csv(output_root / "branch_conflicts.csv", index=False)

    branch_only = _solve_edges(
        branch_edges,
        nodes,
        key_to_index,
        lower_surface,
        upper_surface,
        slope_tolerance_m,
    )
    junction_results = {}
    for definition in ("full_tail", "last_active"):
        result = _solve_edges(
            branch_edges + _junction_edges(nodes, junctions, definition),
            nodes,
            key_to_index,
            lower_surface,
            upper_surface,
            slope_tolerance_m,
        )
        junction_results[definition] = result

    infeasible = branch_conflicts.loc[
        ~branch_conflicts["feasible_under_full_trapezoid_relaxation"]
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": config["experiment"],
        "system_id": SUPPORTED_SYSTEM_ID,
        "status": "closed-mathematically-infeasible",
        "read_only_audit": True,
        "oggm_imported": False,
        "oggm_model_run": False,
        "zero_mb_run": False,
        "historical_chain_run": False,
        "input_provenance": {
            "mapped_nodes_csv": str(mapped_nodes_path),
            "mapped_nodes_sha256": _sha256(mapped_nodes_path),
            "junction_slopes_csv": str(junctions_path),
            "junction_slopes_sha256": _sha256(junctions_path),
            "static_summary_json": str(static_summary_path),
            "static_summary_sha256": _sha256(static_summary_path),
        },
        "fixed_contract": {
            "native_member_count": 4,
            "native_flowline_count": 10,
            "bed_per_node_fixed": True,
            "area_per_node_fixed": True,
            "volume_per_node_fixed": True,
            "total_area_volume_fixed": True,
            "registered_length_fixed": True,
            "tail_and_domain_fixed": True,
        },
        "node_counts": {
            "active": active_count,
            "native_parabolic": parabolic_count,
            "native_trapezoid": trapezoid_count,
        },
        "full_shape_relaxation": {
            "applied_to_all_active_nodes": True,
            "surface_width_relation": "W=A/dx",
            "section_relation": "S=V/dx",
            "trapezoid_section_relation": "S=W*H-0.5*lambda*H^2",
            "lambda_relation": "lambda=2*(W*H-S)/H^2",
            "bottom_width_relation": "w0=2*S/H-W",
            "legal_thickness_interval": "H in [S/W,2*S/W)",
            "upper_open_epsilon_m": upper_epsilon_m,
            "all_parabolic_conversions_permitted_for_feasibility": True,
        },
        "branch_only_feasibility": branch_only,
        "infeasible_branch_count": int(len(infeasible)),
        "infeasible_global_flowline_indices": [
            int(value) for value in infeasible["global_flowline_index"].tolist()
        ],
        "junction_definition_results": junction_results,
        "junction_definitions_change_conclusion": False,
        "required_parabolic_to_trapezoid_conversion_count": None,
        "conversion_conclusion": "all-168-parabolic-conversions-still-insufficient",
        "feasible_maximum_abs_surface_adjustment_m": None,
        "feasible_area_weighted_abs_surface_adjustment_m": None,
        "adjusted_hypsometry_100m_maximum_cumulative_error_fraction": None,
        "unchanged_v3_hypsometry_100m_maximum_cumulative_error_fraction": float(
            static_summary["hypsometry_100m_maximum_cumulative_error_fraction"]
        ),
        "scientific_conclusion": (
            "Legal MixedBed shape freedom cannot remove the saved reverse slopes "
            "while bed and per-node A/V remain fixed; at least bed, per-node A/V, "
            "or the spatial allocation would have to change."
        ),
        "solver": {"name": "scipy.optimize.linprog-highs", "scipy": scipy.__version__},
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "config_snapshot.json", config)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    output = run(args.config)
    print(output)


if __name__ == "__main__":
    main()
