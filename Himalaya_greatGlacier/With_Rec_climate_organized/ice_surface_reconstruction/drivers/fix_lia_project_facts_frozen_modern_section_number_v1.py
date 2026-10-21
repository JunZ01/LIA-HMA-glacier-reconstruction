"""Resolve the concurrent project-facts section-number collision."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


OLD = "## 46. Eleven frozen-curve 2000-2020 validation (2026-08-15)"
NEW = "## 48. Eleven frozen-curve 2000-2020 validation (2026-08-15)"


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8")
    if text.count(OLD) != 1 or NEW in text:
        raise RuntimeError("unexpected-facts-section-number-state")
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
