"""Append the verified Round 162 closure to the canonical facts file once."""

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
addendum = driver / "LIA_PROJECT_FACTS_ADDENDUM_298.md"
text = facts.read_text(encoding="utf-8")
section = addendum.read_text(encoding="utf-8").strip()
if "## 298. Temperature twin information-geometry closure" in text:
    raise RuntimeError("section-298-already-present")
if "## 297. Temperature perfect-model twin" not in text:
    raise RuntimeError("section-297-missing")
facts.write_text(text.rstrip() + "\n\n\n" + section + "\n", encoding="utf-8")
output = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_162_temperature_twin_information_geometry_v1"
shutil.copy2(addendum, output / "LIA_PROJECT_FACTS_ADDENDUM_298.md")
print(facts)
