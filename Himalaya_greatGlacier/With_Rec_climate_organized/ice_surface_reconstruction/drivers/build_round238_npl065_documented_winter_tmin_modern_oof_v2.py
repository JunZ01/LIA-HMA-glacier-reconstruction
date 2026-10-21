from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "run_round238_npl065_documented_winter_tmin_modern_oof_v1.py"
TARGET = HERE / "run_round238_npl065_documented_winter_tmin_modern_oof_v2.py"

text = SOURCE.read_text(encoding="utf-8")
replacements = [
    (
        'round238_npl065_documented_winter_tmin_modern_oof_v1_contract.md',
        'round238_npl065_documented_winter_tmin_modern_oof_v2_contract.md',
    ),
    (
        'round_238_npl065_documented_winter_tmin_modern_oof_v1',
        'round_238_npl065_documented_winter_tmin_modern_oof_v2',
    ),
    (
        'with path.open("r", encoding="ascii", errors="strict") as handle:',
        'with path.open("rb") as handle:',
    ),
    ('line = raw.rstrip("\\r\\n")', 'line = raw.rstrip(b"\\r\\n")'),
    ('line == "\\x1a"', 'line == b"\\x1a"'),
    ('sample = line[:8].strip()', 'sample = line[:8].strip().decode("ascii", errors="strict")'),
    ('year_text.lstrip("-").isdigit()', 'year_text.lstrip(b"-").isdigit()'),
    (
        'for offset, token in enumerate(line[12:].split()):\n                if token in marker_counts:\n                    marker_counts[token] += 1',
        'for offset, token in enumerate(line[12:].split()):\n                token_text = token.decode("ascii", errors="strict")\n                if token_text in marker_counts:\n                    marker_counts[token_text] += 1',
    ),
    ('value = int(token)', 'value = int(token_text)'),
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
