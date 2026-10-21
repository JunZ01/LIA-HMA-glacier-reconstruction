from pathlib import Path


HERE = Path(__file__).resolve().parent
source = HERE / "run_round252_dplr_ads_python_port_validation_v1.py"
target = HERE / "run_round252_dplr_ads_python_port_validation_v2.py"
text = source.read_text(encoding="utf-8")
text = text.replace(
    'output = loop / "round_252_dplr_ads_python_port_validation_v1"',
    'output = loop / "round_252_dplr_ads_python_port_validation_v2"',
)
text = text.replace(
    'raise RuntimeError("ads-matrix-not-positive-definite")',
    'raise RuntimeError(f"ads-matrix-not-positive-definite:i={i}:j={j}:total={total:.17g}:stored={a[i,j]:.17g}:rn={rn:.17g}")',
)
if target.exists():
    raise FileExistsError(target)
target.write_text(text, encoding="utf-8")
print(target)
