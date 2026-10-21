"""Classify saved multi-flowline states without advancing any glacier model.

The audit separates main-line gaps, tributary internal separations, normal
terminal retreat, and vanished tributaries.  It deliberately does not promote
any result: tributary separation remains a conditional-review class until the
project contract defines an impact threshold.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np


STATE_NAMES = {
    "lia_equilibrium_state_50y.pkl.gz",
    "model_flowlines_5y.pkl.gz",
    "model_flowlines_0p1y.pkl.gz",
}
SUMMARY_NAMES = (
    "summary.json",
    "steady_50y_summary.json",
    "steady_50y_summary_formal.json",
    "zero_mb_summary_formal.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_flowlines(path: Path):
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if hasattr(payload, "fls"):
        payload = payload.fls
    if isinstance(payload, dict):
        for key in ("flowlines", "model_flowlines", "fls"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, (list, tuple)):
        raise TypeError(f"unsupported-state-payload:{type(payload).__name__}")
    return list(payload)


def summary_near(path: Path) -> tuple[dict, Path | None]:
    for name in SUMMARY_NAMES:
        candidate = path.parent / name
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8")), candidate
            except Exception:
                continue
    return {}, None


def system_id_from(path: Path, summary: dict) -> int | None:
    for key in ("system_id", "lia_unique_id", "lia_id"):
        value = summary.get(key)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    for part in reversed(path.parts):
        matches = re.findall(r"(?:system[_-]?|steady|lia[_-]?)(\d{1,4})(?:\D|$)", part.lower())
        if matches:
            return int(matches[-1])
    return None


def contiguous_components(indices: np.ndarray) -> list[np.ndarray]:
    if len(indices) == 0:
        return []
    cuts = np.flatnonzero(np.diff(indices) > 1) + 1
    return [chunk for chunk in np.split(indices, cuts) if len(chunk)]


def dx_m(flowline) -> float:
    if hasattr(flowline, "dx_meter"):
        return float(flowline.dx_meter)
    return float(flowline.dx) * float(flowline.map_dx)


def state_connectivity(flowlines, active_tolerance_m: float) -> dict:
    roots = [i for i, flowline in enumerate(flowlines) if flowline.flows_to is None]
    if len(roots) != 1:
        raise ValueError(f"root-count:{len(roots)}")
    root = roots[0]
    total_volume = sum(float(np.sum(np.asarray(fl.section, dtype=float)) * dx_m(fl)) for fl in flowlines)
    total_area = sum(float(np.sum(np.asarray(fl.widths_m, dtype=float)) * dx_m(fl)) for fl in flowlines)
    branches = []
    root_gap_count = 0
    tributary_gap_count = 0
    detached_volume = 0.0
    detached_area = 0.0
    for branch, flowline in enumerate(flowlines):
        thick = np.asarray(flowline.thick, dtype=float)
        section = np.asarray(flowline.section, dtype=float)
        widths = np.asarray(flowline.widths_m, dtype=float)
        active = np.flatnonzero(np.isfinite(thick) & (thick > active_tolerance_m))
        components = contiguous_components(active)
        is_root = branch == root
        branch_dx = dx_m(flowline)
        component_rows = []
        for component_index, nodes in enumerate(components):
            volume = float(np.sum(section[nodes]) * branch_dx)
            area = float(np.sum(widths[nodes]) * branch_dx)
            component_rows.append({
                "component": component_index,
                "first_node": int(nodes[0]),
                "last_node": int(nodes[-1]),
                "node_count": int(len(nodes)),
                "volume_m3": volume,
                "area_m2": area,
            })
        has_internal_gap = len(components) > 1
        if has_internal_gap:
            if is_root:
                root_gap_count += 1
            else:
                tributary_gap_count += 1
                for row in component_rows[:-1]:
                    detached_volume += row["volume_m3"]
                    detached_area += row["area_m2"]
        if not components:
            structural_class = "root-vanished-hard-fail" if is_root else "tributary-vanished"
        elif has_internal_gap:
            structural_class = "mainline-internal-gap-hard-fail" if is_root else "tributary-internal-separation-review"
        elif not is_root and int(active[-1]) < len(thick) - 1:
            structural_class = "tributary-terminal-retreat"
        elif is_root and int(active[-1]) < len(thick) - 1:
            structural_class = "mainline-terminal-retreat"
        else:
            structural_class = "continuous"
        branches.append({
            "branch": branch,
            "is_root": is_root,
            "flows_to": flowline.flows_to is not None,
            "node_count": len(thick),
            "active_node_count": int(len(active)),
            "component_count": len(components),
            "internal_gap_count": max(0, len(components) - 1),
            "first_active": None if not len(active) else int(active[0]),
            "last_active": None if not len(active) else int(active[-1]),
            "structural_class": structural_class,
            "components": component_rows,
        })
    if root_gap_count:
        system_class = "hard-fail-mainline-internal-gap"
    elif tributary_gap_count:
        system_class = "conditional-review-tributary-separation"
    else:
        system_class = "no-internal-gap"
    return {
        "root_branch": root,
        "flowline_count": len(flowlines),
        "system_class": system_class,
        "root_gap_branch_count": root_gap_count,
        "tributary_gap_branch_count": tributary_gap_count,
        "detached_head_volume_m3": detached_volume,
        "detached_head_area_m2": detached_area,
        "detached_head_volume_fraction_system": detached_volume / total_volume if total_volume > 0 else None,
        "detached_head_area_fraction_system": detached_area / total_area if total_area > 0 else None,
        "system_volume_m3": total_volume,
        "system_area_m2": total_area,
        "branches": branches,
    }


def discover(scan_root: Path) -> list[Path]:
    return sorted(path for path in scan_root.rglob("*.pkl.gz") if path.name in STATE_NAMES)


def run(scan_root: Path, output_root: Path, tolerance: float) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    branch_rows = []
    seen_hashes = set()
    for path in discover(scan_root):
        digest = sha256(path)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        summary, summary_path = summary_near(path)
        system_id = system_id_from(path, summary)
        try:
            result = state_connectivity(load_flowlines(path), tolerance)
            error = None
        except Exception as exc:
            result = None
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "system_id": system_id,
            "state_path": str(path.resolve()),
            "state_sha256": digest,
            "summary_path": None if summary_path is None else str(summary_path.resolve()),
            "saved_status": summary.get("status"),
            "error": error,
        }
        if result is not None:
            row.update({key: value for key, value in result.items() if key != "branches"})
            for branch in result["branches"]:
                branch_rows.append({
                    "system_id": system_id,
                    "state_sha256": digest,
                    "state_path": str(path.resolve()),
                    **{key: value for key, value in branch.items() if key != "components"},
                })
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with (output_root / "saved_state_connectivity.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    branch_fields = sorted({key for row in branch_rows for key in row})
    with (output_root / "saved_state_branch_connectivity.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=branch_fields)
        writer.writeheader()
        writer.writerows(branch_rows)
    counts = {}
    for row in rows:
        label = row.get("system_class") or "unreadable"
        counts[label] = counts.get(label, 0) + 1
    systems = sorted({int(row["system_id"]) for row in rows if row.get("system_id") is not None})
    decision = {
        "schema": "lia-191-saved-state-connectivity-audit-v1",
        "status": "read-only-classification-complete-no-acceptance-change",
        "active_thickness_tolerance_m": tolerance,
        "scan_root": str(scan_root.resolve()),
        "unique_state_count": len(rows),
        "identified_system_count": len(systems),
        "identified_system_ids": systems,
        "class_counts_by_state": counts,
        "policy": {
            "mainline_internal_gap": "hard-fail",
            "tributary_internal_separation": "conditional-review-only",
            "terminal_retreat": "structurally-allowed-subject-to-existing-gates",
            "vanished_tributary": "structurally-allowed-subject-to-existing-gates",
            "acceptance_change_authorized": False,
            "impact_threshold_registered": False,
        },
        "limitations": [
            "Only exact saved flowline states are classified.",
            "The inventory is state-level and can contain multiple states per system.",
            "No 191-system acceptance registry is modified.",
        ],
    }
    (output_root / "summary.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--active-thickness-tolerance-m", type=float, default=1e-6)
    args = parser.parse_args()
    print(json.dumps(run(args.scan_root, args.output_root, args.active_thickness_tolerance_m), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
