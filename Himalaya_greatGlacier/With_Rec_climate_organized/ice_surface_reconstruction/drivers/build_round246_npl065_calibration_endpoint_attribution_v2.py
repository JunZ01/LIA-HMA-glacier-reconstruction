from pathlib import Path


HERE = Path(__file__).resolve().parent
source = HERE / "run_round246_npl065_calibration_endpoint_attribution_v1.py"
target = HERE / "run_round246_npl065_calibration_endpoint_attribution_v2.py"
text = source.read_text(encoding="utf-8")
text = text.replace(
    'output = loop / "round_246_npl065_calibration_endpoint_attribution_v1"',
    'output = loop / "round_246_npl065_calibration_endpoint_attribution_v2"',
)
text = text.replace(
    '"oof_1960_2000": np.where(short_mask, short_prediction, np.nan),',
    '"oof_1960_2000": np.r_[short_prediction, np.full(len(FULL_YEARS) - len(short_prediction), np.nan)],',
)
if target.exists():
    raise FileExistsError(target)
target.write_text(text, encoding="utf-8")
print(target)
