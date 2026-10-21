"""Build a reusable automatic steady-state bisection/extension round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", action="append", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--round", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[4]
    cfg_root = root / (
        "Himalaya_greatGlacier/With_Rec_climate_organized/"
        "ice_surface_reconstruction/config/synthetic_lia_system_standard_inversion_v1"
    )
    grouped: dict[int, list[dict]] = {}
    for manifest_name in args.input_manifest:
        manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            out = root / str(case["output_root"])
            summary_path = out / "steady_50y_summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            cfg = json.loads((out / "config_snapshot.json").read_text(encoding="utf-8"))
            sid = int(case["system_id"])
            grouped.setdefault(sid, []).append({
                "delta": float(cfg["stage2"]["delta_t_c"]),
                "dv": float(summary["volume_change_fraction_50y"]),
                "status": str(summary["status"]),
                "config": cfg,
            })
    written = []
    for sid, evaluations in sorted(grouped.items()):
        if any(ev["status"] == "steady-50y-pass" for ev in evaluations):
            continue
        negative = sorted((ev for ev in evaluations if ev["dv"] < 0), key=lambda ev: abs(ev["dv"]))
        positive = sorted((ev for ev in evaluations if ev["dv"] >= 0), key=lambda ev: abs(ev["dv"]))
        if negative and positive:
            delta = 0.5 * (negative[0]["delta"] + positive[0]["delta"])
            action = "bisect-sign-bracket"
        elif negative:
            delta = min(ev["delta"] for ev in evaluations) - 0.5
            action = "extend-colder"
        else:
            delta = max(ev["delta"] for ev in evaluations) + 0.5
            action = "extend-warmer"
        if any(abs(ev["delta"] - delta) < 1e-10 for ev in evaluations):
            raise RuntimeError(f"duplicate delta for system {sid}: {delta}")
        cfg = dict(evaluations[-1]["config"])
        cfg["stage"] = f"batch-35-39-formal-conservative-steady-bisection-round{args.round}"
        cfg["stage2"] = dict(cfg["stage2"])
        cfg["stage2"]["delta_t_c"] = delta
        output_name = f"{args.output_prefix}_{sid}_v1"
        cfg["output_root"] = {"anchor": "project", "sub": output_name}
        target = cfg_root / f"steady_conservative_batch35_39_{sid}_round{args.round}_v1.json"
        target.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append({"system_id": sid, "delta_t_c": delta, "action": action, "config": str(target), "output_root": output_name})
    manifest_path = root / f"{args.output_prefix}_manifest_v1.json"
    manifest_path.write_text(json.dumps({"count": len(written), "cases": written}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(written), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
