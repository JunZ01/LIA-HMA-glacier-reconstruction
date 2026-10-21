"""Read-only promotion audit for the system-3660 root-coarsen2x mechanism.

This script does not run OGGM dynamics and does not write any candidate
flowlines.  It screens the v9.6 remaining population for systems with the
same structural failure as 3660: a formal zero-MB CFL failure inside the
unique 50 m root.  For those systems it evaluates the exact conditioned
width-weighted 2x root construction in memory and applies the registered
static gates needed before a future representative run can be authorized.
"""

from __future__ import annotations

import copy
import csv
import gzip
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np

from .build_zero_mb_3660_root_coarsen2x_v5 import _conditioned_build_root


PROJECT = Path(__file__).resolve().parents[4]
ROUTE = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_current_route_v9_6/remaining_137_current_route.csv"
)
OUTPUT = (
    PROJECT
    / "validation_data/results/paleo_surface_validation_v0"
    / "lia_191_root_coarsen2x_promotion_class_v1"
)
REFERENCE_SYSTEM = 3660
LENGTH_TOLERANCE = 0.10
FREE_TAIL_MIN_M = 5000.0
ROOT_DX_M = 50.0


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT / path


def _find_model_flowlines(inversion_root: Path) -> Path:
    matches = sorted(inversion_root.glob("synthetic_gdir/**/model_flowlines.pkl"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected-one-model-flowlines:{inversion_root}:{len(matches)}"
        )
    return matches[0]


def _active_reverse_count(flowlines) -> int:
    count = 0
    for flowline in flowlines:
        surface = np.asarray(flowline.surface_h, dtype=float)
        active = np.asarray(flowline.thick, dtype=float) > 0.0
        count += int(
            np.count_nonzero(active[:-1] & active[1:] & (np.diff(surface) > 1e-9))
        )
    return count


def _active_junction_reverse_count(flowlines, replacement_root=None) -> int:
    count = 0
    root = replacement_root
    for branch in flowlines:
        if branch.flows_to is None:
            continue
        active = np.flatnonzero(np.asarray(branch.thick, dtype=float) > 0.0)
        if not len(active):
            continue
        terminal = int(active[-1])
        if root is not None and branch.flows_to.flows_to is None:
            old_index = int(branch.flows_to_indice)
            receiver = root
            receiver_node = int(math.ceil(old_index / 2.0))
        else:
            receiver = branch.flows_to
            receiver_node = int(branch.flows_to_indice)
        step = float(receiver.surface_h[receiver_node] - branch.surface_h[terminal])
        count += int(step > 1e-9)
    return count


def _parse_cfl(summary: dict) -> tuple[int | None, int | None, float | None]:
    error = str(summary.get("runtime_error") or "")
    match = re.search(
        r"required time step smaller than the minimum allowed:\s*"
        r"([0-9.]+)s.*?fl_id\s+(\d+),\s*bin_id\s+(\d+)",
        error,
    )
    if match is None:
        return None, None, None
    return int(match.group(2)), int(match.group(3)), float(match.group(1))


def _source_inversion_root(route_row: dict, evidence: Path) -> Path:
    if evidence.name == "inversion_summary_formal.json":
        return evidence.parent
    config = evidence.parent / "config_snapshot.json"
    if not config.exists():
        raise RuntimeError(f"missing-zero-config:{config}")
    payload = _read_json(config)
    root = payload.get("inversion_root")
    if not isinstance(root, dict) or root.get("anchor") != "project":
        raise RuntimeError(f"unsupported-inversion-root:{config}")
    return PROJECT / str(root["sub"])


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["lia_unique_id"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _audit_row(route_row: dict) -> dict:
    system_id = int(route_row["lia_unique_id"])
    evidence = _resolve(route_row["latest_evidence"])
    latest = _read_json(evidence)
    inversion_root = _source_inversion_root(route_row, evidence)
    inversion_summary_path = inversion_root / "inversion_summary_formal.json"
    inversion = _read_json(inversion_summary_path)
    model_path = _find_model_flowlines(inversion_root)
    with gzip.open(model_path, "rb") as stream:
        flowlines = pickle.load(stream)
    roots = [index for index, flowline in enumerate(flowlines) if flowline.flows_to is None]
    if len(roots) != 1:
        raise RuntimeError(f"nonunique-root:{system_id}:{roots}")
    root_index = roots[0]
    root = flowlines[root_index]

    cfl_flowline, cfl_bin, cfl_required_s = _parse_cfl(latest)
    explicit_cfl = cfl_flowline is not None
    root_internal_cfl = bool(
        explicit_cfl
        and cfl_flowline == root_index
        and cfl_bin is not None
        and 0 <= cfl_bin < int(root.nx)
    )
    parent_reverse = _active_reverse_count(flowlines)
    parent_junction_reverse = _active_junction_reverse_count(flowlines)
    parent_no_reverse = parent_reverse == 0 and parent_junction_reverse == 0
    root_dx = float(root.dx_meter)
    uniform_50m_root = abs(root_dx - ROOT_DX_M) <= 1e-9
    special_text = "|".join(
        str(route_row.get(key, ""))
        for key in (
            "current_route",
            "current_gate",
            "latest_flags",
            "mechanism_class",
            "next_legal_action",
        )
    ).lower()
    non_special_spatial_domain = "special" not in special_text

    record = {
        "lia_unique_id": system_id,
        "registered_length_m": float(route_row["lia_registered_length_m"]),
        "model_area_m2": float(inversion.get("model_area_m2", np.nan)),
        "model_volume_m3": float(inversion.get("model_volume_m3", np.nan)),
        "latest_evidence": str(evidence.relative_to(PROJECT)),
        "inversion_summary": str(inversion_summary_path.relative_to(PROJECT)),
        "source_model_flowlines": str(model_path.relative_to(PROJECT)),
        "root_index": root_index,
        "root_nodes_before": int(root.nx),
        "root_dx_before_m": root_dx,
        "root_active_nodes_before": int(
            np.count_nonzero(np.asarray(root.thick, dtype=float) > 0.0)
        ),
        "explicit_cfl_evidence": explicit_cfl,
        "cfl_flowline": cfl_flowline if cfl_flowline is not None else "",
        "cfl_bin": cfl_bin if cfl_bin is not None else "",
        "cfl_required_timestep_s": (
            cfl_required_s if cfl_required_s is not None else ""
        ),
        "root_internal_cfl": root_internal_cfl,
        "uniform_50m_root": uniform_50m_root,
        "parent_active_within_reverse_count": parent_reverse,
        "parent_active_junction_reverse_count": parent_junction_reverse,
        "parent_no_active_reverse": parent_no_reverse,
        "non_special_spatial_domain": non_special_spatial_domain,
        "in_memory_coarsen_evaluated": False,
        "coarsen_error": "",
        "eligible": False,
        "exclusion_reasons": "",
    }

    prerequisites = [
        explicit_cfl,
        root_internal_cfl,
        uniform_50m_root,
        parent_no_reverse,
        non_special_spatial_domain,
        int(root.nx) % 2 == 1,
    ]
    if all(prerequisites):
        try:
            new_root, shape = _conditioned_build_root(root)
            active_after = np.asarray(new_root.thick, dtype=float) > 0.0
            root_length_after = float(new_root.length_m)
            registered_length = float(route_row["lia_registered_length_m"])
            length_error = abs(root_length_after - registered_length) / registered_length
            free_tail_after = float(
                (int(new_root.nx) - int(np.count_nonzero(active_after)))
                * float(new_root.dx_meter)
            )
            coarse_root_reverse = int(
                np.count_nonzero(
                    active_after[:-1]
                    & active_after[1:]
                    & (np.diff(np.asarray(new_root.surface_h, dtype=float)) > 1e-9)
                )
            )
            coarse_junction_reverse = _active_junction_reverse_count(
                flowlines, replacement_root=new_root
            )
            direct_receivers = [
                int(branch.flows_to_indice)
                for branch in flowlines
                if branch.flows_to is root
            ]
            receiver_error = max(
                [
                    abs(
                        math.ceil(index / 2.0) * float(new_root.dx_meter)
                        - index * root_dx
                    )
                    for index in direct_receivers
                ]
                or [0.0]
            )
            test = copy.deepcopy(new_root)
            section0 = np.asarray(new_root.section, dtype=float).copy()
            volume0 = float(new_root.volume_m3)
            test.section = section0.copy()
            setter_volume_error = abs(float(test.volume_m3) - volume0)
            setter_section_error = float(
                np.max(np.abs(np.asarray(test.section, dtype=float) - section0))
            )
            root_nodes_enough = bool(
                int(new_root.nx) >= 3
                and int(np.count_nonzero(active_after)) >= 2
                and free_tail_after >= FREE_TAIL_MIN_M
            )
            record.update(
                {
                    "in_memory_coarsen_evaluated": True,
                    "root_nodes_after": int(new_root.nx),
                    "root_dx_after_m": float(new_root.dx_meter),
                    "root_active_nodes_after": int(np.count_nonzero(active_after)),
                    "root_nodes_enough": root_nodes_enough,
                    "root_length_after_m": root_length_after,
                    "registered_length_relative_error_after": length_error,
                    "registered_length_after_within_10pct": length_error
                    <= LENGTH_TOLERANCE,
                    "free_tail_after_m": free_tail_after,
                    "free_tail_after_at_least_5km": free_tail_after
                    >= FREE_TAIL_MIN_M,
                    "coarse_active_within_reverse_count": coarse_root_reverse,
                    "coarse_active_junction_reverse_count": coarse_junction_reverse,
                    "coarse_no_active_reverse": coarse_root_reverse == 0
                    and coarse_junction_reverse == 0,
                    "maximum_receiver_mapping_error_m": receiver_error,
                    "receiver_mapping_within_half_cell": receiver_error <= 50.0,
                    "raw_minimum_lambda": shape["raw_minimum_lambda"],
                    "minimum_origin_width_m": shape["minimum_origin_width_m"],
                    "setter_volume_error_m3": setter_volume_error,
                    "setter_max_section_error_m2": setter_section_error,
                    "setter_roundtrip_pass": setter_volume_error <= 1e-5
                    and setter_section_error <= 1e-6,
                }
            )
        except Exception as exc:  # record structural infeasibility, never hide it
            record["coarsen_error"] = f"{type(exc).__name__}:{exc}"

    gates = {
        "explicit-root-internal-cfl": bool(record["root_internal_cfl"]),
        "uniform-50m-root": bool(record["uniform_50m_root"]),
        "parent-no-active-reverse": bool(record["parent_no_active_reverse"]),
        "non-special-spatial-domain": bool(record["non_special_spatial_domain"]),
        "odd-root-node-count": int(record["root_nodes_before"]) % 2 == 1,
        "coarsen-evaluated": bool(record["in_memory_coarsen_evaluated"]),
        "root-nodes-enough": bool(record.get("root_nodes_enough", False)),
        "registered-length-after-within-10pct": bool(
            record.get("registered_length_after_within_10pct", False)
        ),
        "coarse-no-active-reverse": bool(
            record.get("coarse_no_active_reverse", False)
        ),
        "receiver-mapping-within-half-cell": bool(
            record.get("receiver_mapping_within_half_cell", False)
        ),
        "setter-roundtrip-pass": bool(record.get("setter_roundtrip_pass", False)),
    }
    record["eligible"] = all(gates.values()) and system_id != REFERENCE_SYSTEM
    record["exclusion_reasons"] = "|".join(
        name for name, passed in gates.items() if not passed
    )
    if system_id == REFERENCE_SYSTEM:
        record["exclusion_reasons"] = "reference-system-3660"
    return record


def _select_representatives(eligible: list[dict]) -> list[dict]:
    if not eligible:
        return []
    ordered = sorted(eligible, key=lambda row: float(row["model_area_m2"]))
    indices = [0] if len(ordered) == 1 else [0, len(ordered) // 2, len(ordered) - 1]
    labels = ["small", "medium", "large"] if len(ordered) > 1 else ["only"]
    selected = []
    seen = set()
    for label, index in zip(labels, indices):
        row = ordered[index]
        system_id = int(row["lia_unique_id"])
        if system_id in seen:
            continue
        seen.add(system_id)
        selected.append(
            {
                "size_class": label,
                "lia_unique_id": system_id,
                "model_area_m2": float(row["model_area_m2"]),
                "model_volume_m3": float(row["model_volume_m3"]),
                "registered_length_m": float(row["registered_length_m"]),
                "root_nodes_before": int(row["root_nodes_before"]),
                "root_nodes_after": int(row["root_nodes_after"]),
                "registered_length_relative_error_after": float(
                    row["registered_length_relative_error_after"]
                ),
                "cfl_required_timestep_s": float(row["cfl_required_timestep_s"]),
                "source_model_flowlines": row["source_model_flowlines"],
            }
        )
    return selected


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with ROUTE.open("r", encoding="utf-8-sig", newline="") as stream:
        route_rows = [
            row for row in csv.DictReader(stream) if row["current_route"] == "zero-mb-blocked"
        ]
    audit = []
    for route_row in route_rows:
        try:
            audit.append(_audit_row(route_row))
        except Exception as exc:
            audit.append(
                {
                    "lia_unique_id": int(route_row["lia_unique_id"]),
                    "registered_length_m": float(route_row["lia_registered_length_m"]),
                    "latest_evidence": route_row["latest_evidence"],
                    "eligible": False,
                    "exclusion_reasons": f"audit-error:{type(exc).__name__}:{exc}",
                }
            )
    eligible = [row for row in audit if bool(row.get("eligible"))]
    selected = _select_representatives(eligible)
    _write_csv(OUTPUT / "candidate_audit.csv", audit)
    _write_csv(OUTPUT / "eligible_candidates.csv", eligible)
    _write_csv(OUTPUT / "selected_confirmation_representatives.csv", selected)
    payload = {
        "experiment": "lia-191-root-coarsen2x-promotion-class-read-only-v1",
        "read_only": True,
        "route_source": str(ROUTE.relative_to(PROJECT)),
        "reference_system": REFERENCE_SYSTEM,
        "mechanism": "uniform-root-2x-width-weighted-conditioned-mixedbed-control-volumes",
        "screened_zero_mb_blocked_system_count": len(route_rows),
        "explicit_root_internal_cfl_system_count": sum(
            bool(row.get("root_internal_cfl")) for row in audit
        ),
        "eligible_promotion_candidate_count": len(eligible),
        "eligible_systems": [int(row["lia_unique_id"]) for row in eligible],
        "selected_confirmation_representatives": selected,
        "selection_metric": "model_area_m2 small/median/large",
        "gates": {
            "formal_failure_class": "CFL inside unique root",
            "root_dx_m": ROOT_DX_M,
            "parent_active_reverse_count": 0,
            "root_node_support": "odd source node count, >=2 coarse active nodes, >=5 km coarse free tail",
            "registered_length_relative_error_after_max": LENGTH_TOLERANCE,
            "spatial_contract": "exclude any route explicitly marked special",
            "tentative_static": "coarse active slopes and junctions non-reverse, receiver shift <=50 m, MixedBed setter roundtrip",
        },
        "dynamics_run": False,
        "next_action": "wait-for-system-3660-steady-and-historical-gates-before-running-any-selected-representative",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
