"""Build the accepted-38 registry and a complete 191-system route audit."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


NEW_ACCEPTED = {
    3297: ("w632hist_batch35_39_3297_end0p5_v1", "w642repro3297_linear_end0p5_v1"),
    3897: ("w641power3897_e0775_g025_v1", "w642repro3897_power_end0p775_g025_v1"),
    5146: ("w640power5146_e0375_g2_v1", "w642repro5146_power_end0p375_g2_v1"),
}


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    validation = root / "validation_data/results/paleo_surface_validation_v0"
    refs = pd.read_csv(validation / "lia_191_controller_v1/core191_length_references.csv")
    refs["lia_unique_id"] = pd.to_numeric(refs["lia_unique_id"]).astype(int)
    refs = refs.set_index("lia_unique_id", drop=False)

    old = pd.read_csv(validation / "lia_191_accepted_35_curve_gallery_v1/accepted_35_curve_metrics.csv")
    old["system_id"] = pd.to_numeric(old["system_id"]).astype(int)
    accepted_rows = old.to_dict(orient="records")
    for system_id, (history_root, reproduction_root) in NEW_ACCEPTED.items():
        history = load_json(root / history_root / "historical_summary.json")
        reproduction = load_json(root / reproduction_root / "reproduction_comparison.json")
        if history is None or history.get("status") != "historical-pass":
            raise RuntimeError(f"historical-pass-missing:{system_id}")
        if reproduction is None or reproduction.get("status") != "reproduction-pass":
            raise RuntimeError(f"reproduction-pass-missing:{system_id}")
        row = refs.loc[system_id]
        lia_m = float(row["lia_registered_length_m"])
        rgi_m = float(row["rgi_registered_length_m"])
        separation = abs(lia_m - rgi_m)
        accepted_rows.append({
            "system_id": system_id,
            "rgi_id": str(row["rgi_id"]),
            "route": "synthetic-system",
            "lia_km": lia_m / 1000.0,
            "rgi_km": rgi_m / 1000.0,
            "lia_tol_km": min(0.2 * separation, 0.1 * lia_m) / 1000.0,
            "rgi_tol_km": min(0.2 * separation, 0.1 * rgi_m) / 1000.0,
            "e_lia": float(history["e_lia_separation"]),
            "e_rgi": float(history["e_rgi_separation"]),
            "history_root": history_root,
            "reproduction_root": reproduction_root,
        })
    accepted = pd.DataFrame(accepted_rows).sort_values("system_id")
    if len(accepted) != 38 or accepted["system_id"].nunique() != 38:
        raise RuntimeError(f"accepted-registry-size:{len(accepted)}/{accepted['system_id'].nunique()}")
    registry_root = validation / "lia_191_accepted_38_registry_v1"
    registry_root.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(registry_root / "accepted_38_curve_metrics.csv", index=False)
    accepted_ids = set(accepted["system_id"].astype(int))

    config_root = root / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    attempted: set[int] = set()
    for path in config_root.glob("geometry_*.json"):
        payload = load_json(path)
        if payload is not None and "system_id" in payload:
            attempted.add(int(payload["system_id"]))

    outlines = gpd.read_file(root / "Himalaya_greatGlacier/Him_10_lia_with_large.shp")
    outlines["UniqueID"] = pd.to_numeric(outlines["UniqueID"]).astype(int)
    polygon_state: dict[int, str] = {}
    for system_id, group in outlines.groupby("UniqueID"):
        if len(group) != 1:
            polygon_state[int(system_id)] = "multiple-outline-features"
            continue
        geom = group.iloc[0].geometry
        if geom is None or geom.is_empty:
            polygon_state[int(system_id)] = "empty-outline"
        elif geom.geom_type != "Polygon":
            polygon_state[int(system_id)] = f"non-polygon:{geom.geom_type}"
        elif not geom.is_valid:
            polygon_state[int(system_id)] = "invalid-polygon"
        else:
            polygon_state[int(system_id)] = "valid-single-polygon"

    stage_records: dict[str, dict[int, list[tuple[str, dict]]]] = {
        name: {} for name in ("geometry", "inversion", "zero_mb", "steady", "historical")
    }
    for work in root.glob("w*"):
        if not work.is_dir():
            continue
        candidates = [
            ("geometry", work / "geometry_summary.json"),
            ("inversion", work / "inversion_summary_formal.json"),
            ("zero_mb", work / "zero_mb_summary_formal.json"),
            ("steady", work / "steady_50y_summary.json"),
            ("historical", work / "historical_summary.json"),
        ]
        for system_dir in work.glob("system_*"):
            candidates.append(("geometry", system_dir / "geometry_summary_formal.json"))
        for stage, path in candidates:
            payload = load_json(path)
            if payload is None or "system_id" not in payload:
                continue
            system_id = int(payload["system_id"])
            stage_records[stage].setdefault(system_id, []).append((str(path.relative_to(root)), payload))

    def any_status(stage: str, system_id: int, status: str) -> bool:
        return any(payload.get("status") == status for _, payload in stage_records[stage].get(system_id, []))

    def latest_evidence(stage: str, system_id: int) -> tuple[str, str, str]:
        rows = stage_records[stage].get(system_id, [])
        if not rows:
            return "", "", ""
        path, payload = rows[-1]
        flags = payload.get("review_flags", [])
        return path, str(payload.get("status", "")), "|".join(map(str, flags))

    audit_rows = []
    for system_id, ref in refs.sort_index().iterrows():
        system_id = int(system_id)
        if system_id in accepted_ids:
            route = "accepted"
            gate = "historical-reproduction-pass"
        elif system_id not in attempted:
            if system_id in {843, 1044}:
                route = "special-contract-hold"
                gate = "registered-special-case"
            elif polygon_state.get(system_id) != "valid-single-polygon":
                route = "input-geometry-blocked"
                gate = polygon_state.get(system_id, "outline-missing")
            elif float(ref["lia_registered_length_m"]) <= float(ref["rgi_registered_length_m"]):
                route = "endpoint-order-blocked"
                gate = "lia-not-longer-than-rgi"
            else:
                route = "not-routed"
                gate = "unexpected-unattempted"
        elif any_status("historical", system_id, "historical-pass"):
            route = "historical-pass-unregistered"
            gate = "requires-reproduction"
        elif any_status("steady", system_id, "steady-50y-pass"):
            route = "historical-endpoint-blocked"
            gate = "no-formal-historical-pass"
        elif any_status("zero_mb", system_id, "zero-mb-pass"):
            route = "steady-state-blocked"
            gate = "no-50y-equilibrium-pass"
        elif any_status("inversion", system_id, "inversion-pass"):
            route = "zero-mb-blocked"
            gate = "no-formal-zero-mb-pass"
        elif any_status("geometry", system_id, "geometry-pass"):
            route = "inversion-blocked"
            gate = "no-formal-inversion-pass"
        else:
            route = "geometry-blocked"
            gate = "no-formal-geometry-pass"
        evidence_stage = next((stage for stage in ("historical", "steady", "zero_mb", "inversion", "geometry") if stage_records[stage].get(system_id)), "")
        evidence, evidence_status, flags = latest_evidence(evidence_stage, system_id) if evidence_stage else ("", "", "")
        audit_rows.append({
            "lia_unique_id": system_id,
            "rgi_id": str(ref["rgi_id"]),
            "lia_registered_length_m": float(ref["lia_registered_length_m"]),
            "rgi_registered_length_m": float(ref["rgi_registered_length_m"]),
            "accepted": system_id in accepted_ids,
            "synthetic_route_attempted": system_id in attempted,
            "outline_state": polygon_state.get(system_id, "outline-missing"),
            "current_route": route,
            "current_gate": gate,
            "latest_evidence_stage": evidence_stage,
            "latest_evidence_status": evidence_status,
            "latest_flags": flags,
            "latest_evidence": evidence,
        })
    audit = pd.DataFrame(audit_rows)
    if len(audit) != 191 or audit["lia_unique_id"].nunique() != 191:
        raise RuntimeError(f"audit-size:{len(audit)}/{audit['lia_unique_id'].nunique()}")
    route_root = validation / "lia_191_remaining_route_audit_v1"
    route_root.mkdir(parents=True, exist_ok=True)
    audit.to_csv(route_root / "lia_191_route_status.csv", index=False)
    counts = audit.groupby("current_route").size().sort_values(ascending=False).to_dict()
    summary = {
        "population": 191,
        "accepted": int(audit["accepted"].sum()),
        "remaining": int((~audit["accepted"]).sum()),
        "synthetic_route_attempted": int(audit["synthetic_route_attempted"].sum()),
        "route_counts": {str(key): int(value) for key, value in counts.items()},
        "accepted_registry": str((registry_root / "accepted_38_curve_metrics.csv").resolve()),
        "route_table": str((route_root / "lia_191_route_status.csv").resolve()),
    }
    (registry_root / "summary.json").write_text(json.dumps({"accepted_count": 38, "systems": accepted["system_id"].astype(int).tolist()}, indent=2) + "\n", encoding="utf-8")
    (route_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
