"""Renumber Round 155 after a concurrent Section 288 addition."""

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1 or new in text:
        raise RuntimeError(f"renumber-precondition-failed:{path}:{text.count(old)}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    old_heading = "## 288. Existing temperature prior/PSM/operator family closure"
    new_heading = "## 289. Existing temperature prior/PSM/operator family closure"
    replace_once(root / "LIA_PROJECT_FACTS.md", old_heading, new_heading)
    replace_once(
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_temperature_existing_operator_family_closure_v1/FACTS_APPEND.md",
        old_heading,
        new_heading,
    )
    updater = driver / "update_lia_project_facts_round155_temperature_existing_operator_family_closure_v1.py"
    replace_once(
        updater,
        'heading = "## 288. Existing temperature prior/PSM/operator family closure"',
        'heading = "## 289. Existing temperature prior/PSM/operator family closure"',
    )
    print("ROUND155_RENUMBERED_288_TO_289")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
