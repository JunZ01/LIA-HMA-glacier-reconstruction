"""Append the verified Round 136 result to the canonical project facts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
HEADER = "## 268. NETP proxy-network leave-one-group-out robustness (2026-08-19)"
TEXT = f"""

{HEADER}

- Round 136 reproduced the full Round 134 GPCC, CRU, and product-marginalized paleo curves before testing omissions. Maximum absolute closure error is 2.22e-16 mm/day for each curve.
- Exactly 15 branches then omitted each complete 0.25-degree proxy spatial group in turn while retaining the Round 134 season, smoothing, equal group weighting, modern orientation, products, and scalar calibration design. The minimum product-specific modern full-fit correlation across all branches is 0.8295.
- No omitted group materially changes the low-frequency trajectory. The minimum branch-versus-full correlation is 0.9888 over 1515-1900 and 0.9912 over 1800-1900. The most influential omission is the 35.0 N, 100.0 E group, but it remains far inside the predeclared 0.90 gate.
- Every leave-one-group-out branch preserves the positive 1800-1900 trend. All 15 branches also remain closer to the completely withheld proxy 15 than to QT-Prec by the frozen correlation-difference rule.
- Decision: the Round 134 low-frequency trajectory is distributed across the fixed NETP proxy network and is not a one-site artifact. Advance the temporal candidate to calibration-method and uncertainty validation; this does not yet authorize spatialization, EnKF assimilation, production replacement, or OGGM.
- Round 136 changed no source file, read no 1921-1939 value, did not open the frozen external validation reserve, and ran neither EnKF nor OGGM. All tracked source fingerprints were unchanged.
"""


def main() -> int:
    content = FACTS.read_text(encoding="utf-8")
    if HEADER in content:
        print("ROUND136_FACTS_ALREADY_PRESENT")
        return 0
    with FACTS.open("a", encoding="utf-8", newline="") as handle:
        handle.write(TEXT)
    print("ROUND136_FACTS_UPDATE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
