"""One-time ACL-safe repair for the split-forward freeze driver import list."""

from pathlib import Path


path = Path(
    "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/"
    "drivers/freeze_lia_191_split_forward_delta_passes_v1.py"
)
text = path.read_text(encoding="utf-8")
needle = "import json\nimport shutil\n"
replacement = "import json\nimport math\nimport shutil\n"
if needle in text:
    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
elif "import math\n" not in text:
    raise RuntimeError("expected import block not found")
