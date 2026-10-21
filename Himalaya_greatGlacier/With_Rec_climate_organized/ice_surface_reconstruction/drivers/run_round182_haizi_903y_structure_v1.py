from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_structure_driver(path: Path):
    spec = importlib.util.spec_from_file_location("round179_structure", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    loop = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2"
    )
    source = loop / (
        "round_181_haizi_903y_temperature_supplement_v2/raw_unopened/"
        "41598_2024_79096_MOESM1_ESM.xlsx"
    )
    output = loop / "round_182_haizi_903y_structure_v1"
    output.mkdir(parents=True, exist_ok=False)
    before = sha256(source)
    round179 = load_structure_driver(
        drivers / "run_round179_mendeley_temperature_structure_identity_v1.py"
    )
    structure = round179.workbook_structure(source)
    after = sha256(source)
    if before != after:
        raise RuntimeError("source-hash-changed")
    (output / "workbook_structure.json").write_text(
        json.dumps(structure, indent=2, ensure_ascii=True), encoding="ascii"
    )
    summary = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source_sha256_before": before,
        "source_sha256_after": after,
        "sheet_count": structure["sheet_count"],
        "sheet_titles": [item["title"] for item in structure["sheets"]],
        "numeric_spreadsheet_values_read": False,
        "locked_1921_1939_values_accessed": False,
        "jionglacuo_opened": False,
        "enkf_run": False,
        "oggm_run": False,
        "decision": "contract_targeted_modern_only_parser",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="ascii"
    )
    (output / "FINAL_RESULT.md").write_text(
        "# Round 182 Haizi 903-Year Workbook Structure\n\n"
        f"- Sheets: `{structure['sheet_count']}`.\n"
        "- Numeric spreadsheet values read: `false`.\n"
        "- Locked-period values accessed: `false`.\n\n"
        "Decision: `contract_targeted_modern_only_parser`.\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
