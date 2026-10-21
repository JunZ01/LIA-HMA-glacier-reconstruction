"""Append the verified Round 135 result to the canonical project facts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
HEADER = "## 267. Withheld NETP proxy arbitration of the QT-Prec conflict (2026-08-19)"
TEXT = f"""

{HEADER}

- Round 135 used proxy6 index 15 at 37.93 N, 101.53 E, which was excluded before Round 134. Metadata-only identity closure maps it uniquely to `Asi-GHEGAN.PAGES2k.2013`. It is an older eastern-Qilian tree-ring record and is not the QT-Prec sequence; QT-Prec uses newly sampled Langya/Pingqiang material. The two records are regionally related but physically independent evidence.
- The withheld proxy passes the predeclared modern informativeness gate without using QT-Prec or the Round 134 paleo curve for orientation. Its 11-year relation is positive for both GPCC and CRU over the effective 1945-1995 interval, with correlations 0.5424 and 0.6535. Three of four within-block signs agree with that orientation.
- Over the exact common 1515-1900 interval, the oriented proxy correlation is -0.3130 with QT-Prec but +0.4526 with the Round 134 product-marginalized composite. The frozen correlation-difference statistic is -0.7656 and favors the Round 134 composite.
- The conflict-period result is much stronger. Over 1800-1900, proxy 15 correlates -0.5659 with QT-Prec and +0.9571 with the Round 134 composite; the difference is -1.5231. Proxy 15 and Round 134 also share a positive trend, whereas QT-Prec has a negative trend.
- Decision: the independent local record supports the Round 134 low-frequency trajectory. QT-Prec can no longer be used alone to reject that candidate; the remaining discrepancy is a regional-representativeness or chronology/method disagreement. The next admissible test is spatial-group leave-one-out robustness of the fixed Round 134 proxy network, not tuning toward QT-Prec.
- Round 135 changed no source file, read no 1921-1939 value, did not open the frozen external validation reserve, and ran neither EnKF nor OGGM. All tracked source fingerprints were unchanged.
"""


def main() -> int:
    content = FACTS.read_text(encoding="utf-8")
    if HEADER in content:
        print("ROUND135_FACTS_ALREADY_PRESENT")
        return 0
    with FACTS.open("a", encoding="utf-8", newline="") as handle:
        handle.write(TEXT)
    print("ROUND135_FACTS_UPDATE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
