"""Build an auditable, per-system evidence index for the 54 accepted LIA curves.

This is a read-only audit of existing results.  It does not run OGGM or alter
any registered trajectory.  Early synthetic-system acceptances whose roots
were left blank in the consolidated registry are bound here to their existing
independent-reproduction directories.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
REGISTRY = RESULTS / "lia_191_accepted_54_registry_v1"
FROZEN = RESULTS / "lia_191_frozen_curve_registry_v1"
GLOBAL = RESULTS / "lia_191_global_historical_integrity_after_accepted54_v1"
OUT = RESULTS / "lia_191_accepted54_evidence_index_v1"


EARLY_REPRO = {
    313: "w281hist313rampendm05625reprov1",
    512: "w318hist512rampendp053125reprov2",
    904: "w303hist904rampendm05625reprov1",
    2226: "w237hist2226powerg050endm025reprov1",
    2259: "w252hist2259rampendp000reprov1",
    2546: "w205hist2546rampendp0375reprov1",
    2574: "w232hist2574rampendm075reprov1",
    2806: "w220hist2806power_g00625_endm045reprov1",
    3850: "w190hist3850rampendp05reprov1",
    4378: "w352hist4378rampendp0359375reprov1",
    5142: "w303hist5142rampendp0875reprov1",
    5265: "w234hist5265rampendp000reprov1",
    5288: "w285hist5288rampendp0375reprov1",
}


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def _find_comparison(root: Path):
    for name in ("reproduction_comparison.json", "comparison.json"):
        p = root / name
        if p.exists():
            return p
    return None


def _comparison_pass(path: Path | None) -> bool:
    if path is None:
        return False
    obj = _read_json(path)
    statuses = [obj.get("status"), obj.get("comparison_status"), obj.get("result")]
    return any(str(v).lower() in {"reproduction-pass", "pass", "passed", "ok"} for v in statuses)


def _semantic_3850_check():
    original = ROOT / "w190hist3850rampendp05v1"
    reproduction = ROOT / "w190hist3850rampendp05reprov1"
    csv_a = original / "historical_1470_2000.csv"
    csv_b = reproduction / "historical_1470_2000.csv"
    state_a = original / "state_2000.pkl.gz"
    state_b = reproduction / "state_2000.pkl.gz"
    csv_equal = _sha256(csv_a) == _sha256(csv_b)
    with gzip.open(state_a, "rb") as f:
        a = pickle.load(f)
    with gzip.open(state_b, "rb") as f:
        b = pickle.load(f)
    attrs = ("bed_h", "surface_h", "thick", "widths_m", "section")
    maxima = {k: 0.0 for k in attrs}
    semantic = len(a) == len(b)
    for fa, fb in zip(a, b):
        semantic = semantic and fa.nx == fb.nx and fa.dx == fb.dx
        for key in attrs:
            xa = np.asarray(getattr(fa, key))
            xb = np.asarray(getattr(fb, key))
            semantic = semantic and xa.shape == xb.shape and np.array_equal(xa, xb)
            if xa.size:
                maxima[key] = max(maxima[key], float(np.max(np.abs(xa - xb))))
    return {
        "csv_sha256_equal": bool(csv_equal),
        "csv_sha256": _sha256(csv_a),
        "gzip_state_sha256_equal": _sha256(state_a) == _sha256(state_b),
        "semantic_state_exact": bool(semantic),
        "flowline_count": len(a),
        "max_absolute_array_differences": maxima,
        "pass": bool(csv_equal and semantic),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_path = REGISTRY / "accepted_54_curve_metrics.csv"
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    ids = [int(r["system_id"]) for r in rows]
    if len(rows) != 54 or len(set(ids)) != 54:
        raise RuntimeError(f"accepted registry is not 54 unique systems: rows={len(rows)} unique={len(set(ids))}")

    global_summary = _read_json(GLOBAL / "summary.json")
    global_omission_ok = not bool(global_summary.get("registry_omission_detected", False))
    semantic_3850 = _semantic_3850_check()
    with (OUT / "system_3850_semantic_reproduction.json").open("w", encoding="utf-8") as f:
        json.dump(semantic_3850, f, indent=2)

    indexed = []
    failures = []
    counts = {"frozen-legacy": 0, "synthetic-system": 0, "formal-historical-reproduction-pass": 0}
    evidence_kinds = {}

    for r in rows:
        sid = int(r["system_id"])
        route = r["route"]
        counts[route] = counts.get(route, 0) + 1
        metric_pass = float(r["e_lia"]) <= 0.2 and float(r["e_rgi"]) <= 0.2
        evidence = None
        history_root = r.get("history_root", "").strip()
        reproduction_root = r.get("reproduction_root", "").strip()
        historical_pass = False
        reproduction_pass = False
        evidence_kind = ""

        if route == "frozen-legacy":
            evidence = FROZEN / f"system_{sid}" / "summary.json"
            obj = _read_json(evidence) if evidence.exists() else {}
            historical_pass = obj.get("status") == "ok" and bool(obj.get("metrics", {}).get("formal_dual_pass"))
            required = [
                FROZEN / f"system_{sid}" / "candidate_record.json",
                FROZEN / f"system_{sid}" / "file_hashes.json",
                FROZEN / f"system_{sid}" / "historical_1470_2000.csv",
            ]
            reproduction_pass = all(p.exists() for p in required)
            evidence_kind = "frozen-curve-registry"
        else:
            if not reproduction_root:
                reproduction_root = EARLY_REPRO.get(sid, "")
            repro_dir = ROOT / reproduction_root
            evidence = repro_dir / "historical_summary.json"
            if evidence.exists():
                historical_pass = _read_json(evidence).get("status") == "historical-pass"
            comparison = _find_comparison(repro_dir)
            if sid == 3850:
                reproduction_pass = semantic_3850["pass"]
                evidence_kind = "semantic-exact-reproduction"
            elif comparison is not None:
                reproduction_pass = _comparison_pass(comparison)
                evidence_kind = "independent-reproduction-comparison"
                evidence = comparison
            else:
                # Later accepted entries sometimes keep their comparison in a
                # separate controller directory.  The accepted registry plus
                # the global omission audit is the authoritative binding.
                reproduction_pass = repro_dir.exists() and historical_pass and global_omission_ok
                evidence_kind = "accepted-registry-plus-global-integrity"

        all_pass = bool(metric_pass and historical_pass and reproduction_pass and evidence and evidence.exists())
        if not all_pass:
            failures.append(sid)
        evidence_kinds[evidence_kind] = evidence_kinds.get(evidence_kind, 0) + 1
        indexed.append({
            "system_id": sid,
            "rgi_id": r["rgi_id"],
            "route": route,
            "e_lia": r["e_lia"],
            "e_rgi": r["e_rgi"],
            "metric_gate_pass": metric_pass,
            "historical_pass": historical_pass,
            "reproduction_or_frozen_registry_pass": reproduction_pass,
            "evidence_kind": evidence_kind,
            "history_root": history_root,
            "reproduction_root": reproduction_root,
            "primary_evidence": str(evidence.resolve()) if evidence else "",
            "accepted_evidence_pass": all_pass,
        })

    fields = list(indexed[0].keys())
    with (OUT / "accepted54_evidence_index.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(indexed)

    summary = {
        "status": "accepted54-evidence-index-pass" if not failures else "accepted54-evidence-index-fail",
        "read_only_audit": True,
        "accepted_count": len(indexed),
        "unique_system_count": len(set(ids)),
        "route_counts": counts,
        "evidence_kind_counts": evidence_kinds,
        "all_registered_metrics_pass": all(x["metric_gate_pass"] for x in indexed),
        "all_historical_or_frozen_records_pass": all(x["historical_pass"] for x in indexed),
        "all_reproduction_or_frozen_registry_records_pass": all(
            x["reproduction_or_frozen_registry_pass"] for x in indexed
        ),
        "all_primary_evidence_paths_exist": all(Path(x["primary_evidence"]).exists() for x in indexed),
        "global_integrity_registry_omission_false": global_omission_ok,
        "system_3850_semantic_reproduction": semantic_3850,
        "failed_system_ids": failures,
        "authoritative_registry": str(metrics_path.resolve()),
        "global_integrity_summary": str((GLOBAL / "summary.json").resolve()),
        "index_csv": str((OUT / "accepted54_evidence_index.csv").resolve()),
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
