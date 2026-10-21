"""Correct the single Round 166 string-literal typo."""

from pathlib import Path


path = Path(__file__).with_name("run_round166_temperature_family_decision_synthesis_v1.py")
text = path.read_text(encoding="utf-8")
old = '    next_contract = """# Next Temperature Evidence Route\\n\\n"'
new = '    next_contract = "# Next Temperature Evidence Route\\n\\n"'
if text.count(old) != 1:
    raise RuntimeError(f"expected-one-match:{text.count(old)}")
path.write_text(text.replace(old, new), encoding="utf-8")
print(path)
