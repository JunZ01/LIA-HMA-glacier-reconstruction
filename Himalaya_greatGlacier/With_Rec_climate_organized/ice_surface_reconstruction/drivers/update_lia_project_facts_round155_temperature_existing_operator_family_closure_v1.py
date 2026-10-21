"""Append the verified Round 155 facts exactly once."""

from pathlib import Path


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    facts = root / "LIA_PROJECT_FACTS.md"
    addition = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_temperature_existing_operator_family_closure_v1/"
        "FACTS_APPEND.md"
    )
    heading = "## 289. Existing temperature prior/PSM/operator family closure"
    original = facts.read_text(encoding="utf-8")
    if heading in original:
        raise RuntimeError("round155-facts-already-present")
    text = addition.read_text(encoding="utf-8").strip()
    if not text.startswith(heading):
        raise RuntimeError("round155-facts-heading-mismatch")
    facts.write_text(original.rstrip() + "\n\n" + text + "\n", encoding="utf-8")
    print("ROUND155_FACTS_APPENDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
