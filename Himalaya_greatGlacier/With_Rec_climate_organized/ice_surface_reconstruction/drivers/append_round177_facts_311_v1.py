from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def main() -> int:
    root = project_root()
    facts = root / "LIA_PROJECT_FACTS.md"
    addendum = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_improvement_loop_v2/"
        "round_177_noaa_non_tree_temperature_closure_v1/"
        "LIA_PROJECT_FACTS_ADDENDUM_311.md"
    )
    marker = b"## 311. NOAA non-tree HMA temperature-candidate closure"
    payload = facts.read_bytes()
    if marker in payload:
        print("FACTS_311_ALREADY_PRESENT")
        return 0
    addition = addendum.read_bytes()
    facts.write_bytes(payload.rstrip() + b"\n" + addition.rstrip() + b"\n")
    if marker not in facts.read_bytes():
        raise RuntimeError("facts-append-verification-failed")
    print("FACTS_311_APPENDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
