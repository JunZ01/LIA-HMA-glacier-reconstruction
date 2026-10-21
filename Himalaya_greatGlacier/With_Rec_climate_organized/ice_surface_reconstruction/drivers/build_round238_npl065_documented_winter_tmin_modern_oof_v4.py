from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "run_round238_npl065_documented_winter_tmin_modern_oof_v3.py"
TARGET = HERE / "run_round238_npl065_documented_winter_tmin_modern_oof_v4.py"

text = SOURCE.read_text(encoding="utf-8")
helper = '''def detrend_with_linear_fallback(dplpy: Any, raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    columns: list[pd.Series] = []
    fallback: list[dict[str, str]] = []
    for sample in raw.columns:
        try:
            series = dplpy.detrend(raw[sample], fit="ModNegEx", method="residual", plot=False)
        except RuntimeError as error:
            series = dplpy.detrend(raw[sample], fit="linear", method="residual", plot=False)
            fallback.append({"sample": str(sample), "exception_type": type(error).__name__})
        columns.append(series.rename(sample))
    return pd.concat(columns, axis=1).reindex(raw.index), fallback


'''
replacements = [
    (
        'round238_npl065_documented_winter_tmin_modern_oof_v3_contract.md',
        'round238_npl065_documented_winter_tmin_modern_oof_v4_contract.md',
    ),
    (
        'round_238_npl065_documented_winter_tmin_modern_oof_v3',
        'round_238_npl065_documented_winter_tmin_modern_oof_v4',
    ),
    ('def main() -> int:\n', helper + 'def main() -> int:\n'),
    (
        'rwi = dplpy.detrend(raw, fit="ModNegEx", method="residual", plot=False)',
        'rwi, detrending_fallbacks = detrend_with_linear_fallback(dplpy, raw)',
    ),
    (
        '"published_chronology_equivalence_claimed": False,',
        '"published_chronology_equivalence_claimed": False,\n            "linear_fallback_count": len(detrending_fallbacks),\n            "linear_fallbacks": detrending_fallbacks,',
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
