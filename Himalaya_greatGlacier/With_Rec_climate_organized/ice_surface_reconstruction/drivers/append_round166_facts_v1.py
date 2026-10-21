"""Append the verified Round 166 terminal decision to the facts file once."""

import shutil
from pathlib import Path


driver = Path(__file__).resolve().parent
for parent in driver.parents:
    if (parent / "LIA_PROJECT_FACTS.md").exists():
        root = parent
        break
else:
    raise RuntimeError("project-root-not-found")

facts = root / "LIA_PROJECT_FACTS.md"
addendum = driver / "LIA_PROJECT_FACTS_ADDENDUM_302.md"
text = facts.read_text(encoding="utf-8")
section = addendum.read_text(encoding="utf-8").strip()
if "## 302. Archived temperature-family terminal decision" in text:
    raise RuntimeError("section-302-already-present")
if "## 301. Perfect-model joint-network fusion closure" not in text:
    raise RuntimeError("section-301-missing")
facts.write_text(text.rstrip() + "\n\n\n" + section + "\n", encoding="utf-8")
output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_166_temperature_family_decision_synthesis_v1"
shutil.copy2(addendum, output / "LIA_PROJECT_FACTS_ADDENDUM_302.md")
print(facts)
