"""Append the verified Round 154 facts exactly once."""

from pathlib import Path


def main() -> int:
    driver = Path(__file__).resolve().parent
    root = driver.parents[3]
    facts = root / "LIA_PROJECT_FACTS.md"
    addition = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_temperature_adaptive_background_spatial_balance_oof_v1/"
        "FACTS_APPEND.md"
    )
    heading = "## 287. Spatially balanced adaptive-background OOF closure"
    original = facts.read_text(encoding="utf-8")
    if heading in original:
        raise RuntimeError("round154-facts-already-present")
    text = addition.read_text(encoding="utf-8").strip()
    if not text.startswith(heading):
        raise RuntimeError("round154-facts-heading-mismatch")
    facts.write_text(original.rstrip() + "\n\n" + text + "\n", encoding="utf-8")
    print("ROUND154_FACTS_APPENDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
