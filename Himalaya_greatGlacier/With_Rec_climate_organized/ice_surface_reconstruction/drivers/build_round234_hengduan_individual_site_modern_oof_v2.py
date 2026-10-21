from __future__ import annotations

from pathlib import Path


def main() -> int:
    directory = Path(__file__).resolve().parent
    source = directory / "run_round234_hengduan_individual_site_modern_oof_v1.py"
    destination = directory / "run_round234_hengduan_individual_site_modern_oof_v2.py"
    if destination.exists():
        raise FileExistsError(destination)
    text = source.read_text(encoding="ascii")
    replacements = {
        'output = loop / "round_234_hengduan_individual_site_modern_oof_v1"':
            'output = loop / "round_234_hengduan_individual_site_modern_oof_v2"',
        '"contract": drivers / "round234_hengduan_individual_site_modern_oof_v1_contract.md"':
            '"contract": drivers / "round234_hengduan_individual_site_modern_oof_v2_contract.md"',
        'r118.aggregate_temperature(monthly[product], [tuple(range(5, 10))])':
            'r118.aggregate_temperature(monthly[product], [tuple(range(5, 10))] * len(SITES))',
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

