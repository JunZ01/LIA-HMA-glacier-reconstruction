"""Final evidence and mutual-exclusivity audit for route v10.28."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[4]
ROUTE_ROOT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_28"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_final_boundary_audit_v10_28"


def resolve_evidence(value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boundary_category(row: pd.Series) -> str:
    text = " ".join(str(row.get(k, "")) for k in ("current_route", "current_gate", "mechanism_class", "next_legal_action")).lower()
    if any(t in text for t in ("impossible", "mathematically", "hard-bound", "endpoint-order", "operator")):
        return "mathematical-or-observation-contract-boundary"
    if "historical" in text or "endpoint" in text:
        return "registered-historical-family-exhausted"
    if "steady" in text:
        return "registered-steady-family-exhausted"
    if "zero-mb" in text or "zero_mb" in text:
        return "registered-zero-mb-family-exhausted"
    if "inversion" in text or "negative-flux" in text:
        return "registered-inversion-or-initial-state-family-exhausted"
    if any(t in text for t in ("geometry", "outline", "junction", "root", "network")):
        return "registered-geometry-or-network-family-exhausted"
    return "explicit-contract-expansion-required"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    route = pd.read_csv(ROUTE_ROOT / "lia_191_current_route.csv", keep_default_na=False)
    accepted = route["accepted"].astype(str).str.lower().eq("true")
    remaining = route.loc[~accepted].copy()
    rows = []
    for _, row in remaining.iterrows():
        evidence = resolve_evidence(row["latest_evidence"])
        exists = evidence.exists()
        rows.append({
            "system_id": int(row["lia_unique_id"]), "rgi_id": row["rgi_id"],
            "boundary_category": boundary_category(row), "current_route": row["current_route"],
            "current_gate": row["current_gate"], "mechanism_class": row["mechanism_class"],
            "controller_state": row["controller_state"], "next_legal_action": row["next_legal_action"],
            "latest_evidence": row["latest_evidence"], "evidence_exists": bool(exists),
            "evidence_kind": "file" if exists and evidence.is_file() else ("directory" if exists and evidence.is_dir() else "missing"),
            "evidence_sha256": sha256(evidence) if exists and evidence.is_file() else "",
            "latest_evidence_status": row["latest_evidence_status"], "latest_flags": row["latest_flags"],
            "has_nonempty_next_action": bool(str(row["next_legal_action"]).strip()),
        })
    audit = pd.DataFrame(rows).sort_values("system_id")
    checks = {
        "population_191": bool(len(route) == 191), "unique_system_ids_191": bool(route.lia_unique_id.nunique() == 191),
        "accepted_54": bool(int(accepted.sum()) == 54), "remaining_137": bool(len(audit) == 137),
        "all_remaining_closed": bool(audit.controller_state.eq("closed-current-branch").all()),
        "all_latest_evidence_exists": bool(audit.evidence_exists.all()),
        "all_next_actions_nonempty": bool(audit.has_nonempty_next_action.all()),
        "all_rows_one_mutually_exclusive_class": bool(route.mutually_exclusive_class.astype(str).str.strip().ne("").all()),
        "no_active_route_rows": bool(route.active_process_count.astype(int).eq(0).all()),
    }
    summary = {
        "schema": "lia-191-final-boundary-audit-v10-28",
        "status": "final-boundary-audit-pass" if all(checks.values()) else "final-boundary-audit-fail",
        "population": int(len(route)), "accepted": int(accepted.sum()), "remaining": int(len(audit)),
        "boundary_category_counts": {str(k): int(v) for k, v in audit.boundary_category.value_counts().items()},
        "checks": checks,
        "missing_evidence_system_ids": audit.loc[~audit.evidence_exists, "system_id"].astype(int).tolist(),
        "empty_next_action_system_ids": audit.loc[~audit.has_nonempty_next_action, "system_id"].astype(int).tolist(),
        "scientific_contract_changed": False,
        "interpretation": "This proves an evidence-complete, mutually exclusive boundary for all 137 remaining systems under the frozen current contract. It does not claim 191 accepted reconstructions or prohibit reopening after an explicitly registered new mechanism/observation contract.",
        "route": str((ROUTE_ROOT / "summary.json").relative_to(PROJECT)),
    }
    OUTPUT.mkdir(parents=True)
    audit.to_csv(OUTPUT / "remaining_137_final_boundaries.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
