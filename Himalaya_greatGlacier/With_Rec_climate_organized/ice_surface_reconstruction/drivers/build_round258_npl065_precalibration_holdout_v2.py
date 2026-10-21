from pathlib import Path


HERE = Path(__file__).resolve().parent
source = HERE / "run_round258_npl065_precalibration_holdout_v1.py"
target = HERE / "run_round258_npl065_precalibration_holdout_v2.py"
text = source.read_text(encoding="utf-8")
text = text.replace(
    "YEARS = np.arange(1952, 2014, dtype=int)\n",
    "MATRIX_YEARS = np.arange(1941, 2014, dtype=int)\nYEARS = np.arange(1952, 2014, dtype=int)\n",
)
text = text.replace(
    'output = loop / "round_258_npl065_precalibration_holdout_v1"',
    'output = loop / "round_258_npl065_precalibration_holdout_v2"',
)
if target.exists():
    raise FileExistsError(target)
target.write_text(text, encoding="utf-8")
print(target)
