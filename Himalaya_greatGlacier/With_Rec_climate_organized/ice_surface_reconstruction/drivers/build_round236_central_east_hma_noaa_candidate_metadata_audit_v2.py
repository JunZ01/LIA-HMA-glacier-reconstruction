from __future__ import annotations

from pathlib import Path


def main() -> int:
    directory = Path(__file__).resolve().parent
    source = directory / "run_round236_central_east_hma_noaa_candidate_metadata_audit_v1.py"
    destination = directory / "run_round236_central_east_hma_noaa_candidate_metadata_audit_v2.py"
    if destination.exists():
        raise FileExistsError(destination)
    text = source.read_text(encoding="ascii")
    replacements = {
        'output = loop / "round_236_central_east_hma_noaa_candidate_metadata_audit_v1"':
            'output = loop / "round_236_central_east_hma_noaa_candidate_metadata_audit_v2"',
        'metadata_dir = output / "noaa_full_study_metadata_unmodified"':
            'metadata_dir = output / "raw"',
        '"round236_central_east_hma_noaa_candidate_metadata_audit_v1_contract.md"':
            '"round236_central_east_hma_noaa_candidate_metadata_audit_v2_contract.md"',
        'f"{label.lower()}_noaa_{noaa_id}_xmlid_{xml_id}.json"':
            'f"{label.lower()}.json"',
    }
    for old, new in replacements.items():
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"unexpected-replacement-count:{count}:{old}")
        text = text.replace(old, new)
    destination.write_text(text, encoding="ascii")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

