"""Update only the project-facts last-updated label after verified batch 3."""

from __future__ import annotations

from .build_lia_191_iterative_loop import project_root


def main() -> None:
    path = project_root() / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for index, line in enumerate(lines[:20]):
        if line.startswith("Last updated:"):
            lines[index] = (
                "Last updated: 2026-08-15 (formal delta-T batch 3 complete; "
                "nine curves frozen)"
            )
            break
    else:
        raise ValueError("project facts last-updated header not found")
    suffix = "\n" if text.endswith("\n") else ""
    path.write_text("\n".join(lines) + suffix, encoding="utf-8")
    print(lines[index])


if __name__ == "__main__":
    main()
