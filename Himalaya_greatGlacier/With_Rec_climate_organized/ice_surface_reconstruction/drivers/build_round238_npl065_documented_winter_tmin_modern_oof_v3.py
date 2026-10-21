from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "run_round238_npl065_documented_winter_tmin_modern_oof_v2.py"
TARGET = HERE / "run_round238_npl065_documented_winter_tmin_modern_oof_v3.py"

text = SOURCE.read_text(encoding="utf-8")
replacements = [
    (
        'round238_npl065_documented_winter_tmin_modern_oof_v2_contract.md',
        'round238_npl065_documented_winter_tmin_modern_oof_v3_contract.md',
    ),
    (
        'round_238_npl065_documented_winter_tmin_modern_oof_v2',
        'round_238_npl065_documented_winter_tmin_modern_oof_v3',
    ),
    (
        'if not year_text.lstrip(b"-").isdigit():\n                raise RuntimeError(f"invalid-rwl-year:{line_number}:{year_text}")',
        'if not year_text.lstrip(b"-").isdigit():\n                continue',
    ),
]
for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"replacement-count:{count}:{old}")
    text = text.replace(old, new)
if TARGET.exists():
    raise FileExistsError(TARGET)
TARGET.write_text(text, encoding="utf-8")
print(TARGET)
