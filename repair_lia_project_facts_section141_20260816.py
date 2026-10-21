"""Resolve the concurrent facts-section number collision without changing content."""

from pathlib import Path


FACTS = Path(__file__).resolve().parent / "LIA_PROJECT_FACTS.md"
OLD_TITLE = "## 137. Alternate-state split-forward family closure (2026-08-16)"
NEW_TITLE = "## 141. Alternate-state split-forward family closure (2026-08-16)"
OLD_HEADER = "Last updated: 2026-08-16 (alternate-state split-forward family closed; system 1524 frozen)"
NEW_HEADER = "Last updated: 2026-08-16 (post-final-test climate boundary; alternate-state split-forward family closed; system 1524 frozen)"


def main() -> None:
    text = FACTS.read_text(encoding="utf-8")
    if text.count(OLD_TITLE) != 1:
        raise RuntimeError(f"unexpected-old-title-count:{text.count(OLD_TITLE)}")
    if NEW_TITLE in text:
        raise RuntimeError("new-title-already-present")
    if text.count("## 140. Post-final-test climate-reconstruction boundary (2026-08-16)") != 1:
        raise RuntimeError("missing-or-duplicated-section-140")
    if text.count(OLD_HEADER) != 1:
        raise RuntimeError("unexpected-header-state")
    text = text.replace(OLD_TITLE, NEW_TITLE, 1)
    text = text.replace(OLD_HEADER, NEW_HEADER, 1)
    FACTS.write_text(text, encoding="utf-8")
    print("repaired", FACTS)


if __name__ == "__main__":
    main()
