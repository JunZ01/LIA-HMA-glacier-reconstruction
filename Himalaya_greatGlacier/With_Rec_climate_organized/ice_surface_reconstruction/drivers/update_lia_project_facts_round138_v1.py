"""Append the verified Round 138 result to the canonical project facts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
HEADER = "## 270. NETP MJJ structural sensitivity envelope (2026-08-19)"
TEXT = f"""

{HEADER}

- Round 138 packages exactly 18 product-marginalized 31-year sensitivity members: the full Round 134 candidate, 15 Round 136 leave-one-group-out curves, and product means of the Round 137 early- and late-calibration curves. No member was selected, reweighted, rescaled, or tuned.
- Every member is finite over 1515-1900. The minimum member-versus-full correlation is 0.9447 over 1515-1900 and 0.9547 over 1800-1900.
- All 18 members retain positive 1800-1900 MJJ precipitation trends. Their trend range is +0.1132 to +0.1824 mm/day/century, with median +0.1749. The mean annual 5-95% structural range over that period is 0.02479 mm/day and the maximum full member range is 0.07061 mm/day.
- The structural median remains independently supported by withheld proxy 15. Correlations are +0.4541 over 1515-1900 and +0.9590 over 1800-1900, while the corresponding QT-Prec correlations are -0.3130 and -0.5659.
- Decision: retain a structurally robust NETP MJJ temporal candidate with an explicit sensitivity envelope. The percentile bands summarize tested structural sensitivity and are not probability confidence or credible intervals. This milestone validates temporal phase robustness for the NETP prototype but does not yet establish a spatial HMA field or authorize production replacement, EnKF, or OGGM.
- Round 138 changed no source file, read no 1921-1939 value, did not open the frozen external validation reserve, and ran neither EnKF nor OGGM. All tracked source fingerprints were unchanged.
"""


def main() -> int:
    content = FACTS.read_text(encoding="utf-8")
    if HEADER in content:
        print("ROUND138_FACTS_ALREADY_PRESENT")
        return 0
    with FACTS.open("a", encoding="utf-8", newline="") as handle:
        handle.write(TEXT)
    print("ROUND138_FACTS_UPDATE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
