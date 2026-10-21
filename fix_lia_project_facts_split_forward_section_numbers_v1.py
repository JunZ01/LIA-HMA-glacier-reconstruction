"""Resolve concurrent section-number additions without changing content."""

from pathlib import Path


path = Path("LIA_PROJECT_FACTS.md")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "## 49. GPCC-CRU cross-product PSM stability (2026-08-15)",
    "## 50. GPCC-CRU cross-product PSM stability (2026-08-15)",
    1,
)
text = text.replace(
    "## 50. Saved-state split forward-delta representative pilot (2026-08-15)",
    "## 51. Saved-state split forward-delta representative pilot (2026-08-15)",
    1,
)
path.write_text(text, encoding="utf-8")
