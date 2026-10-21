"""Build one automatic steady-state bisection/extension round for 19 systems."""

from __future__ import annotations

import json
from pathlib import Path


def _read_eval(root: Path, case: dict) -> dict:
    out = root / str(case["output_root"])
    summary = json.loads((out / "steady_50y_summary.json").read_text(encoding="utf-8"))
    cfg = json.loads((out / "config_snapshot.json").read_text(encoding="utf-8"))
    return {
        "system_id": int(case["system_id"]),
        "delta": float(cfg["stage2"]["delta_t_c"]),
        "dv": float(summary["volume_change_fraction_50y"]),
        "status": str(summary["status"]),
        "config": cfg,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    cfg_root = root / (
        "Himalaya_greatGlacier/With_Rec_climate_organized/"
        "ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    )
    manifests = [
        json.loads((root / "w644steady_conservative_batch35_39_manifest_v1.json").read_text(encoding="utf-8")),
        json.loads((root / "w645steady_conservative_batch35_39_cold05_manifest_v1.json").read_text(encoding="utf-8")),
    ]
    grouped: dict[int, list[dict]] = {}
    for manifest in manifests:
        for case in manifest["cases"]:
            ev = _read_eval(root, case)
            grouped.setdefault(ev["system_id"], []).append(ev)

    written = []
    for system_id, evaluations in sorted(grouped.items()):
        if any(ev["status"] == "steady-50y-pass" for ev in evaluations):
            continue
        neg = sorted((ev for ev in evaluations if ev["dv"] < 0), key=lambda x: abs(x["dv"]))
        pos = sorted((ev for ev in evaluations if ev["dv"] >= 0), key=lambda x: abs(x["dv"]))
        if neg and pos:
            new_delta = 0.5 * (neg[0]["delta"] + pos[0]["delta"])
            action = "bisect-sign-bracket"
        elif neg:
            coldest = min(evaluations, key=lambda x: x["delta"])
            new_delta = coldest["delta"] - 0.5
            action = "extend-colder"
        else:
            warmest = max(evaluations, key=lambda x: x["delta"])
            new_delta = warmest["delta"] + 0.5
            action = "extend-warmer"
        if any(abs(ev["delta"] - new_delta) < 1e-10 for ev in evaluations):
            raise RuntimeError(f"duplicate delta for {system_id}: {new_delta}")
        template = dict(evaluations[-1]["config"])
        template["stage"] = "batch-35-39-formal-conservative-steady-bisection-round1"
        template["stage2"] = dict(template["stage2"])
        template["stage2"]["delta_t_c"] = new_delta
        output_name = f"w646steady_conservative_batch35_39_{system_id}_round1_v1"
        template["output_root"] = {"anchor": "project", "sub": output_name}
        target = cfg_root / f"steady_conservative_batch35_39_{system_id}_round1_v1.json"
        target.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append({
            "system_id": system_id,
            "delta_t_c": new_delta,
            "action": action,
            "config": str(target),
            "output_root": output_name,
        })
    out = root / "w646steady_conservative_batch35_39_round1_manifest_v1.json"
    out.write_text(json.dumps({"count": len(written), "cases": written}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(written), "manifest": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
