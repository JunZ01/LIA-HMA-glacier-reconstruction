"""Resolve the concurrent duplicate Section 38 heading without changing facts."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8-sig")
    old = "## 38. No-formal-pass N2 glen-a promotion (2026-08-15)"
    new = "## 39. No-formal-pass N2 glen-a promotion (2026-08-15)"
    if new in text:
        print("heading already renumbered")
        return
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one N2 heading, found {text.count(old)}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("renumbered N2 facts section 38 -> 39")


if __name__ == "__main__":
    main()
