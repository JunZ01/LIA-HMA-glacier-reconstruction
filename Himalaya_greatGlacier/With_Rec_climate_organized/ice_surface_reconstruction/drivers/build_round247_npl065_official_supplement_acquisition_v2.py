from pathlib import Path


HERE = Path(__file__).resolve().parent
source = HERE / "acquire_round247_npl065_official_supplement_v1.py"
target = HERE / "acquire_round247_npl065_official_supplement_v2.py"
text = source.read_text(encoding="utf-8")
text = text.replace(
    'BASE = "https://ars.els-cdn.com/content/image/1-s2.0-"',
    'BASE = "https://ars.els-cdn.com/content/image/"',
)
text = text.replace(
    'output = loop / "round_247_npl065_official_supplement_acquisition_v1"',
    'output = loop / "round_247_npl065_official_supplement_acquisition_v2"',
)
if target.exists():
    raise FileExistsError(target)
target.write_text(text, encoding="utf-8")
print(target)
