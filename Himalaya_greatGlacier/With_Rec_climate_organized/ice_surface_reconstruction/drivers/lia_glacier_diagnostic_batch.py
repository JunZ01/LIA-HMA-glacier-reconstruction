"""Generate multiple glacier diagnostic bundles in parallel."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

try:
    from .lia_glacier_diagnostic_bundle import run as run_bundle
except ImportError:  # allow direct CLI execution from the project root
    from lia_glacier_diagnostic_bundle import run as run_bundle


def _run_one(payload: dict) -> dict:
    args = argparse.Namespace(
        system_id=int(payload["system_id"]),
        chain_dir=Path(payload["chain_dir"]),
        elevation_band_root=Path(payload["elevation_band_root"]),
        grid_root=Path(payload["grid_root"]),
        reference_csv=None if payload.get("reference_csv") is None else Path(payload["reference_csv"]),
        output_dir=Path(payload["output_root"]) / f"lia_{payload['system_id']}",
    )
    return run_bundle(args)


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [
        {
            "system_id": unique_id,
            "chain_dir": str(args.chain_dir.resolve()),
            "elevation_band_root": str(args.elevation_band_root.resolve()),
            "grid_root": str(args.grid_root.resolve()),
            "reference_csv": None if args.reference_csv is None else str(args.reference_csv.resolve()),
            "output_root": str(args.output_dir.resolve()),
        }
        for unique_id in args.system_id
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_one, payload): payload for payload in payloads}
        for future in as_completed(futures):
            payload = futures[future]
            try:
                result = future.result()
                rows.append({
                    "lia_unique_id": int(payload["system_id"]),
                    "status": result["status"],
                    "output_dir": str(Path(args.output_dir) / f"lia_{payload['system_id']}"),
                })
            except Exception as exc:
                rows.append({
                    "lia_unique_id": int(payload["system_id"]),
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                })
    frame = pd.DataFrame(rows).sort_values("lia_unique_id")
    frame.to_csv(args.output_dir / "diagnostic_batch_status.csv", index=False)
    summary = {
        "status": "completed" if frame["status"].eq("completed").all() else "completed-with-failures",
        "workers": int(args.workers),
        "completed_count": int(frame["status"].eq("completed").sum()),
        "failed_count": int(frame["status"].ne("completed").sum()),
    }
    (args.output_dir / "diagnostic_batch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-id", action="append", type=int, required=True)
    parser.add_argument("--chain-dir", required=True, type=Path)
    parser.add_argument("--elevation-band-root", required=True, type=Path)
    parser.add_argument("--grid-root", required=True, type=Path)
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
