from __future__ import annotations

from pathlib import Path


def main() -> int:
    directory = Path(__file__).resolve().parent
    source = directory / "run_round234_hengduan_individual_site_modern_oof_v2.py"
    destination = directory / "run_round234_hengduan_individual_site_modern_oof_v3.py"
    if destination.exists():
        raise FileExistsError(destination)
    text = source.read_text(encoding="ascii")
    replacements = {
        'output = loop / "round_234_hengduan_individual_site_modern_oof_v2"':
            'output = loop / "round_234_hengduan_individual_site_modern_oof_v3"',
        '"contract": drivers / "round234_hengduan_individual_site_modern_oof_v2_contract.md"':
            '"contract": drivers / "round234_hengduan_individual_site_modern_oof_v3_contract.md"',
        '            frame.at[year, sample] = value\n    return frame, {\n        "sample_count": int(frame.shape[1]),':
            '            frame.at[year, sample] = value\n'
            '    source_sample_count = int(frame.shape[1])\n'
            '    frame = frame.dropna(axis=1, how="all")\n'
            '    return frame, {\n'
            '        "source_sample_count": source_sample_count,\n'
            '        "modern_nonempty_sample_count": int(frame.shape[1]),\n'
            '        "modern_empty_sample_count_skipped": source_sample_count - int(frame.shape[1]),',
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

