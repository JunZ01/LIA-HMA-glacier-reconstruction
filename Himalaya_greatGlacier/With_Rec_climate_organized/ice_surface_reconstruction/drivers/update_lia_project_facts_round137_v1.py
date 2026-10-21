"""Append the verified Round 137 result to the canonical project facts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
HEADER = "## 269. NETP split-calibration temporal stability (2026-08-19)"
TEXT = f"""

{HEADER}

- Round 137 constructed four paleo branches using disjoint modern calibration halves: GPCC and CRU each fit on 1940-1969 or 1970-2000. Proxy smoothing, standardization, sign orientation, and scalar regression were fit inside each block; the Round 134 paleo curves, withheld proxy 15, and QT-Prec remained validation-only.
- Split-branch fit correlations range from 0.7383 to 0.8788. Each branch remains close to its same-product full-calibration curve: the minimum correlation is 0.9355 over 1515-1900 and 0.9436 over 1800-1900.
- Early- versus late-calibrated curves are also stable within each product. The minimum early/late correlation is 0.9464 over 1515-1900 and 0.9586 over 1800-1900.
- All four branches preserve the positive 1800-1900 trend and all four remain closer to withheld proxy 15 than QT-Prec by the frozen 0.10 correlation-difference rule.
- Decision: the Round 134 temporal trajectory is not an artifact of choosing the full 1940-2000 calibration period. Advance to a structural sensitivity envelope combining product, calibration-half, and spatial-group omission branches; the envelope must not be represented as a probabilistic confidence interval.
- Round 137 changed no source file, read no 1921-1939 value, did not open the frozen external validation reserve, and ran neither EnKF nor OGGM. All tracked source fingerprints were unchanged.
"""


def main() -> int:
    content = FACTS.read_text(encoding="utf-8")
    if HEADER in content:
        print("ROUND137_FACTS_ALREADY_PRESENT")
        return 0
    with FACTS.open("a", encoding="utf-8", newline="") as handle:
        handle.write(TEXT)
    print("ROUND137_FACTS_UPDATE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
