"""Build auditable evidence for the 22 blank-evidence rows in route v10.27.

This is read-only with respect to all source/model outputs.  It distinguishes
mathematical/contract boundaries from five invalid outlines that still have a
concrete deterministic repair action and therefore must not be called final.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.validation import explain_validity


PROJECT = Path(__file__).resolve().parents[4]
ROUTE = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_current_route_v10_27/lia_191_current_route.csv"
FACTS = PROJECT / "LIA_PROJECT_FACTS.md"
OUTLINE = PROJECT / "Himalaya_greatGlacier/Him_10_lia_with_large.shp"
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_legacy_missing22_boundary_evidence_v1"

INVALID_IDS = {516, 2919, 4637, 5353, 5398}
ENDPOINT_ORDER_IDS = {1038, 1412, 1425, 1545, 2155, 2353, 2387, 2528, 2790, 2925, 3053, 3058, 4489}
SPECIAL_IDS = {843, 1044, 2711, 3584}
EXPECTED_IDS = INVALID_IDS | ENDPOINT_ORDER_IDS | SPECIAL_IDS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


SPECIAL = {
    843: {
        "boundary_class": "observation-domain-contract-conflict",
        "adjudication": "closed-under-current-contract",
        "reason": (
            "The standard-Core parent scoring domain and the registered spatial-downstream "
            "observation domain are inconsistent. Facts Sections 161/163 explicitly withhold "
            "843 from the 25 m result; scalar/split-forward follow-ups retain an endpoint tradeoff."
        ),
        "facts_sections": "161,163",
        "supporting_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_continuous_class_controller_v1/cycle1/Q3_rgi_short_25m_historical_v1",
        "reopen_condition": "register one authoritative observation domain and rescore without mixing domains",
    },
    1044: {
        "boundary_class": "observation-domain-overflow-and-closed-scalar-family",
        "adjudication": "closed-under-current-contract",
        "reason": (
            "Facts Section 159 rejects the old provisional endpoint candidate because it first "
            "exceeds the current observation domain in year 1800; later registered scalar/physical "
            "blocks do not provide a current-domain formal solution."
        ),
        "facts_sections": "159",
        "supporting_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_continuous_class_controller_v1/q4_preflight_audit_v1",
        "reopen_condition": "a new domain-valid structural state/geometry mechanism or an explicit observation-domain contract change",
    },
    2711: {
        "boundary_class": "no-convergence-bracket",
        "adjudication": "closed-under-current-contract",
        "reason": (
            "Facts Section 159 records no convergence bracket in ten delta-T chains; every chain "
            "remains nonconverged/out-of-range at 1500 years with 14.77-17.31% preceding-200-year "
            "volume growth. Scalar delta-T refinement and simple spin-up extension are closed."
        ),
        "facts_sections": "159",
        "supporting_evidence": "validation_data/results/paleo_surface_validation_v0/lia_191_continuous_class_controller_v1/q4_preflight_audit_v1",
        "reopen_condition": "a genuinely new convergence/state/geometry mechanism, not a longer scalar spin-up",
    },
    3584: {
        "boundary_class": "named-level-amplitude-tradeoff",
        "adjudication": "closed-under-current-contract",
        "reason": (
            "The named B-window configuration can approach the length level only with a persistent "
            "stability/amplitude or modern-MB mismatch. Facts close melt_f, glen_a, temp_melt, and "
            "prcp_fac scalar shape blocks for the current geometry/window/forcing contract."
        ),
        "facts_sections": "selected-five and D-layer blocks; physical-family closure near Facts lines 1290-1366",
        "supporting_evidence": "validation_data/results/paleo_surface_validation_v0/selected_five_targeted_round2_v1/system_3584_dt0p75_spin1000/summary.json",
        "reopen_condition": "new forcing, state/window design, observation-shape contract, or structural geometry; no scalar micro-search",
    },
}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"versioned-output-already-exists:{OUTPUT}")
    route = pd.read_csv(ROUTE)
    subset = route.loc[route["lia_unique_id"].astype(int).isin(EXPECTED_IDS)].copy()
    found = set(subset["lia_unique_id"].astype(int))
    if found != EXPECTED_IDS:
        raise RuntimeError(f"route-id-mismatch:missing={sorted(EXPECTED_IDS-found)} extra={sorted(found-EXPECTED_IDS)}")

    outlines = gpd.read_file(OUTLINE)
    if "UniqueID" not in outlines.columns:
        raise KeyError("outline-UniqueID-column-missing")
    outline_by_id = {int(row.UniqueID): row.geometry for _, row in outlines.iterrows() if int(row.UniqueID) in INVALID_IDS}

    rows: list[dict] = []
    for _, src in subset.sort_values("lia_unique_id").iterrows():
        sid = int(src["lia_unique_id"])
        base = {
            "system_id": sid,
            "rgi_id": str(src["rgi_id"]),
            "lia_registered_length_m": float(src["lia_registered_length_m"]),
            "rgi_registered_length_m": float(src["rgi_registered_length_m"]),
            "source_route": str(src["current_route"]),
        }
        if sid in ENDPOINT_ORDER_IDS:
            lia = base["lia_registered_length_m"]
            rgi = base["rgi_registered_length_m"]
            delta = rgi - lia
            if not delta > 0:
                raise RuntimeError(f"endpoint-order-assumption-failed:{sid}:{lia}:{rgi}")
            final_lower = rgi - 0.2 * delta
            rows.append({
                **base,
                "boundary_class": "endpoint-order-operator-impossibility",
                "adjudication": "closed-under-current-operator",
                "reason": (
                    "Because L_RGI>L_LIA, passing the RGI separation gate requires "
                    "L_2000>=L_LIA+0.8D. Since L_peak>=L_2000, the LIA separation error "
                    "is at least 0.8, exceeding the fixed 0.2 gate."
                ),
                "endpoint_difference_m": delta,
                "rgi_gate_final_lower_bound_m": final_lower,
                "minimum_possible_lia_separation_error": 0.8,
                "geometry_type": "",
                "geometry_is_valid": "",
                "geometry_validity_reason": "",
                "facts_sections": "formal peak/final separation operator",
                "supporting_evidence": str(ROUTE.relative_to(PROJECT)),
                "reopen_condition": "explicitly change the endpoint-direction/observation operator contract; no dynamics can solve the current operator",
            })
        elif sid in INVALID_IDS:
            geometry = outline_by_id.get(sid)
            if geometry is None:
                raise RuntimeError(f"outline-row-missing:{sid}")
            rows.append({
                **base,
                "boundary_class": "invalid-source-outline-actionable",
                "adjudication": "actionable-not-final-boundary",
                "reason": "The authoritative source is an invalid Polygon with a ring self-intersection; deterministic topology repair has not been adjudicated for this system in the synthetic-system route.",
                "endpoint_difference_m": base["rgi_registered_length_m"] - base["lia_registered_length_m"],
                "rgi_gate_final_lower_bound_m": "",
                "minimum_possible_lia_separation_error": "",
                "geometry_type": geometry.geom_type,
                "geometry_is_valid": bool(geometry.is_valid),
                "geometry_validity_reason": explain_validity(geometry),
                "facts_sections": "canonical source-outline contract",
                "supporting_evidence": str(OUTLINE.relative_to(PROJECT)),
                "reopen_condition": "run deterministic make_valid/topology QC and unchanged static geometry gates",
            })
        else:
            info = SPECIAL[sid]
            support = PROJECT / info["supporting_evidence"]
            if not support.exists():
                raise FileNotFoundError(f"supporting-evidence-missing:{sid}:{support}")
            rows.append({
                **base,
                **info,
                "endpoint_difference_m": base["rgi_registered_length_m"] - base["lia_registered_length_m"],
                "rgi_gate_final_lower_bound_m": "",
                "minimum_possible_lia_separation_error": "",
                "geometry_type": "",
                "geometry_is_valid": "",
                "geometry_validity_reason": "",
            })

    audit = pd.DataFrame(rows).sort_values("system_id")
    invalid = audit.loc[audit["adjudication"].eq("actionable-not-final-boundary"), "system_id"].astype(int).tolist()
    closed = audit.loc[~audit["adjudication"].eq("actionable-not-final-boundary"), "system_id"].astype(int).tolist()
    summary = {
        "schema": "lia-191-legacy-missing22-boundary-evidence-v1",
        "status": "evidence-built-five-actionable",
        "system_count": int(len(audit)),
        "closed_under_current_contract_count": int(len(closed)),
        "closed_under_current_contract_system_ids": closed,
        "actionable_invalid_outline_count": int(len(invalid)),
        "actionable_invalid_outline_system_ids": invalid,
        "endpoint_operator_impossible_count": int((audit.boundary_class == "endpoint-order-operator-impossibility").sum()),
        "special_or_legacy_contract_count": int(len(SPECIAL_IDS)),
        "source_hashes": {
            "route_v10_27_csv": sha256(ROUTE),
            "facts": sha256(FACTS),
            **{p.name: sha256(p) for p in sorted(OUTLINE.parent.glob(OUTLINE.stem + ".*")) if p.is_file()},
        },
        "scientific_contract_changed": False,
        "interpretation": (
            "Seventeen rows now have a rigorous mathematical or registered-contract boundary. "
            "Five source outlines are confirmed invalid but remain actionable; they cannot be "
            "counted as final scientific boundaries before deterministic topology repair and "
            "unchanged static gates are run."
        ),
    }
    OUTPUT.mkdir(parents=True)
    audit.to_csv(OUTPUT / "legacy_missing22_evidence.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
