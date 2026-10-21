"""Read-only historical/reproduction integrity audit after accepted-54."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _truth(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _norm(path) -> str:
    return str(Path(str(path)).resolve()).lower()


def run(project_root: Path, accepted_csv: Path, route_csv: Path,
        collector_csv: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    accepted = pd.read_csv(accepted_csv)
    accepted_ids = set(pd.to_numeric(accepted["system_id"]).astype(int))
    route = pd.read_csv(route_csv)
    route["lia_unique_id"] = pd.to_numeric(route["lia_unique_id"]).astype(int)
    remaining = route.loc[~route["lia_unique_id"].isin(accepted_ids)].copy()
    collector = pd.read_csv(collector_csv)
    collector["system_id"] = pd.to_numeric(collector["system_id"], errors="coerce")
    collector["candidate_key"] = collector["candidate_root"].map(_norm)

    formal = collector["formal_pass"].map(_truth)
    collector_accepted = collector["system_id"].isin(accepted_ids)
    new_formal = collector.loc[formal & ~collector_accepted].copy()
    status_pass_unregistered = collector.loc[
        collector["historical_status"].eq("historical-pass") & ~collector_accepted
    ].copy()
    collector.to_csv(output_root / "all_historical_rescore.csv", index=False)
    new_formal.to_csv(output_root / "unregistered_formal_historical_passes.csv", index=False)
    status_pass_unregistered.to_csv(
        output_root / "unregistered_saved_status_historical_passes.csv", index=False
    )

    summary_paths = sorted(project_root.glob("w*/historical_summary.json"))
    scored_keys = set(collector["candidate_key"])
    unscored = []
    for path in summary_paths:
        root = path.parent.resolve()
        if _norm(root) not in scored_keys:
            config = _read_json(root / "config_snapshot.json") or {}
            unscored.append({
                "candidate_root": str(root),
                "historical_summary": str(path.resolve()),
                "system_id": config.get("system_id"),
                "historical_status": (_read_json(path) or {}).get("status"),
                "accepted_member": config.get("system_id") in accepted_ids,
                "reason": "collector-incompatible-config-schema",
            })
    pd.DataFrame(unscored).to_csv(
        output_root / "unscored_historical_summaries.csv", index=False
    )

    collector_by_root = {
        row.candidate_key: row for row in collector.itertuples(index=False)
    }
    reproduction_rows = []
    evidence_root = project_root / "validation_data/results/paleo_surface_validation_v0"
    for path in sorted(evidence_root.rglob("*.json")):
        if "repro" not in path.name.lower():
            continue
        payload = _read_json(path)
        if payload is None or payload.get("status") != "reproduction-pass":
            continue
        original_value = payload.get("original")
        reproduction_value = payload.get("reproduction")
        if not original_value or not reproduction_value:
            continue
        original = Path(str(original_value)).resolve()
        reproduction = Path(str(reproduction_value)).resolve()
        historical = (
            (original / "historical_summary.json").is_file()
            and (reproduction / "historical_summary.json").is_file()
        )
        config = _read_json(original / "config_snapshot.json") or {}
        system_id = config.get("system_id")
        if system_id is None:
            config = _read_json(reproduction / "config_snapshot.json") or {}
            system_id = config.get("system_id")
        try:
            system_id = int(system_id)
        except (TypeError, ValueError):
            system_id = None
        candidate = collector_by_root.get(_norm(original))
        original_formal_pass = bool(candidate is not None and _truth(candidate.formal_pass))
        reproduction_rows.append({
            "evidence_json": str(path.resolve()),
            "system_id": system_id,
            "historical_reproduction": historical,
            "original_root": str(original),
            "reproduction_root": str(reproduction),
            "original_formal_pass": original_formal_pass,
            "accepted_member": system_id in accepted_ids if system_id is not None else False,
            "unregistered_reproduction_pass": bool(
                historical and original_formal_pass
                and system_id is not None and system_id not in accepted_ids
            ),
        })
    reproduction = pd.DataFrame(reproduction_rows)
    reproduction.to_csv(output_root / "reproduction_evidence_audit.csv", index=False)
    if reproduction.empty:
        unregistered_reproduction = reproduction.copy()
    else:
        unregistered_reproduction = reproduction.loc[
            reproduction["unregistered_reproduction_pass"]
        ].copy()
    unregistered_reproduction.to_csv(
        output_root / "unregistered_reproduction_passes.csv", index=False
    )

    categories = []
    for item in remaining.itertuples(index=False):
        system_id = int(item.lia_unique_id)
        override = None
        evidence = str(item.latest_evidence or "")
        if system_id == 2898:
            category = "mechanism_closed"
            override = "post-v9.1 MultiPolygon support repair passed but static slope gate failed"
            evidence = "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/multipolygon_2898_component_audit_v2/pilot_progression_v2.json"
        elif system_id == 3915:
            category = "mechanism_closed"
            override = "historical endpoints cancelled by formal LIA peak hard bound; not an infrastructure failure"
            evidence = "validation_data/results/paleo_surface_validation_v0/lia_191_geometry_repair_v1/geometry_repair_progression_v3.json"
        elif system_id == 3144:
            category = "technical_pending_retry"
            override = "registered endpoint +0.9375 K is currently active; await final summary"
            evidence = "w746hist3144_endp0p9375_v2/config_snapshot.json"
        elif str(item.current_route) == "endpoint-order-blocked":
            category = "mathematically_unreachable"
        elif str(item.controller_state) == "active":
            category = "technical_pending_retry"
        elif str(item.current_route) == "historical-output-incomplete-hold":
            category = "technical_pending_retry"
        elif str(item.controller_state) in {"queued", "actionable"}:
            category = "actionable"
        else:
            category = "mechanism_closed"
        categories.append({
            "system_id": system_id,
            "rgi_id": item.rgi_id,
            "category": category,
            "current_route_v9_1": item.current_route,
            "controller_state_v9_1": item.controller_state,
            "override_reason": override,
            "evidence": evidence,
        })
    classes = pd.DataFrame(categories).sort_values("system_id")
    classes.to_csv(output_root / "remaining137_mutually_exclusive_classification.csv", index=False)
    counts = classes["category"].value_counts().to_dict()
    for name in ("mathematically_unreachable", "mechanism_closed",
                 "technical_pending_retry", "actionable"):
        counts.setdefault(name, 0)
    if sum(counts.values()) != 137:
        raise RuntimeError(f"remaining-count-mismatch:{counts}")

    result = {
        "schema_version": 1,
        "status": "global-historical-integrity-audit-complete",
        "accepted_registry_count": len(accepted_ids),
        "remaining_count": len(remaining),
        "historical_summary_inventory_count": len(summary_paths),
        "collector_scored_complete_count": int(
            collector["collection_state"].eq("complete").sum()
        ),
        "collector_pending_count": int(
            collector["collection_state"].eq("pending").sum()
        ),
        "collector_incompatible_summary_count": len(unscored),
        "formal_historical_pass_not_accepted_count": len(new_formal),
        "saved_status_historical_pass_not_accepted_count": len(status_pass_unregistered),
        "historical_reproduction_pass_evidence_count": int(
            reproduction.get("historical_reproduction", pd.Series(dtype=bool)).sum()
        ),
        "reproduction_pass_not_registered_count": len(unregistered_reproduction),
        "registry_omission_found": bool(len(new_formal) or len(unregistered_reproduction)),
        "remaining137_counts": counts,
        "active_systems_observed_during_audit": [3144],
        "oggm_run": False,
        "source_modified": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--accepted-csv", type=Path, required=True)
    parser.add_argument("--route-csv", type=Path, required=True)
    parser.add_argument("--collector-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.project_root.resolve(), args.accepted_csv.resolve(),
                 args.route_csv.resolve(), args.collector_csv.resolve(),
                 args.output_root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
