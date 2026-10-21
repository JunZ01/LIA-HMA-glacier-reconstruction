from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve()
    while root.name != "1_LIA_himalaya_oggm":
        root = root.parent
    facts = root / "LIA_PROJECT_FACTS.md"
    addendum = Path(__file__).with_name("LIA_PROJECT_FACTS_ADDENDUM_307_308.md")
    marker = "## 307. Huang 2025 Ranwu source-family refinement"
    existing = facts.read_text(encoding="utf-8")
    if marker in existing:
        raise RuntimeError("facts-sections-307-308-already-present")
    payload = addendum.read_text(encoding="utf-8")
    with facts.open("a", encoding="utf-8", newline="") as stream:
        stream.write("\n\n" + payload.rstrip() + "\n")
    print("APPENDED_FACTS_307_308")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
