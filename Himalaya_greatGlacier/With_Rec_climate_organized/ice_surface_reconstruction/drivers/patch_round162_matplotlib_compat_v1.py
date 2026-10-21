"""Apply the one-line Matplotlib compatibility correction to the Round 162 driver."""

from pathlib import Path


path = Path(__file__).with_name("run_round162_temperature_twin_information_geometry_v1.py")
text = path.read_text(encoding="utf-8")
old = '        tick_labels=["norm ratio", "cosine", "oracle scale"],'
new = '        labels=["norm ratio", "cosine", "oracle scale"],'
if text.count(old) != 1:
    raise RuntimeError(f"expected-one-match:{text.count(old)}")
path.write_text(text.replace(old, new), encoding="utf-8")
print(path)
