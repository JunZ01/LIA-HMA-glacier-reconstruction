from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    output_default = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_increment_attribution_v1"
    )
    parser.add_argument("--output-dir", type=Path, default=output_default)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    summary_path = output / "summary.json"
    first_pass = output / "proxy_increment_first_pass_summary.json"
    if summary_path.exists() and not first_pass.exists():
        shutil.copy2(summary_path, first_pass)

    audit = json.loads(
        (output / "spatial_export_chain_audit.json").read_text(encoding="utf-8")
    )
    preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    authoritative = {
        "experiment_id": audit["experiment_id"],
        "authoritative_result": "FINAL_RESULT.md",
        "status": "spatial export geometry conflict resolved",
        "decision": (
            "Quarantine the current public 0.5-degree rec_result products for "
            "local glacier forcing. Regenerate them with a verified cell-center "
            "affine transform before any further climate or glacier calibration."
        ),
        "proxy_increment_scope": (
            "Exact only for fixed OEnKF-proxy6 experiment 1 on the correctly "
            "located 2-degree replay grid."
        ),
        "conservation_preflight": preflight,
        "spatial_export_audit": audit,
        "superseded_summary": first_pass.name,
    }
    summary_path.write_text(
        json.dumps(authoritative, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(authoritative, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
