"""Append verified Round 134 prior and reconstruction results to project facts."""

from pathlib import Path


SECTION = r'''

## 266. NETP exact MJJ prior and low-frequency reconstruction prototype (2026-08-19)

- Round 134 first exported a May-July precipitation prior through the original MATLAB R2024b `load_S_720360` production path using CESM-LME010, RNG seed 4236, member years 1001-1848, and the unchanged 24-51 N / 62-110 E state domain. Annual precipitation closes with zero changed values against both the recovered production prior and Round 129 V3.
- Prior-QA V1 stopped at an incorrectly specified requirement that all 336 cells have nonzero temporal variance. Read-only diagnosis showed exactly three constant southwest-corner cells at 25 N and 63/65/67 E; the already verified annual and JJA production fields have the identical mask. V2 therefore required exact zero-variance-mask closure and variable support in all 32 NETP cells. All corrected gates pass; the NETP minimum MJJ cell standard deviation is 0.16768 mm/day.
- NETP MJJ is not interchangeable with the existing seasonal or annual state. Its area-mean correlations with annual and JJA precipitation across the exact prior are only 0.6792 and 0.7001. The exact MJJ prior is therefore retained for any future seasonal spatial experiment.
- The frozen regional prototype excluded proxy index 15 at 37.93 N, 101.53 E because it is nearest the independent QT-Prec sampling area. The other 19 NETP proxy6 tree records were combined into 15 equal-weight 0.25-degree spatial groups. Proxy standardization and sign alignment were fit-only within two disjoint modern blocks; 11-year smoothing never crossed the 1969/1970 boundary. No proxy weight, regularization parameter, record subset, or window was tuned.
- Modern low-frequency performance is high and product-robust. Concatenated blocked-OOF correlation/skill are 0.8843/0.9078 for GPCC and 0.8751/0.9063 for CRU. All four fold correlations are positive, GPCC-versus-CRU OOF prediction correlation is 0.9886, and their paleo 31-year curves correlate 0.9904.
- This modern success does not transfer to the independent QT-Prec reconstruction. The product-marginalized correlation is 0.0846 over the effective 1515-1900 comparison, versus the Round 133 annual baseline 0.2370 (change -0.1524). Over the effective 1800-1900 comparison it is -0.3782 versus baseline 0.1401 (change -0.5183), and the reconstructed trend is positive while QT-Prec is negative.
- The V2 coverage gate initially counted 1916-1920 as failures even though a full centered 11-year window cannot use future locked years. The corrected raw support is 1500-1915 with all 15 groups active. A V4 final report also corrected the first effective QT comparison year to 1515; these reporting corrections changed no model value, correlation, trend, or decision.
- Decision: close the fixed equal-weight NETP temporal-composite design and do not spatialize it with the MJJ prior. The result demonstrates a stationarity/low-frequency transfer failure: excellent 1940-2000 blocked skill is insufficient evidence for the preinstrumental trajectory. The next admissible diagnostic is validation against the completely withheld nearby proxy 15, not model tuning.
- Across all Round 134 stages, source fingerprints remained unchanged; no 1921-1939 value or frozen-reserve value was read; no EnKF or OGGM run occurred.
'''


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    facts = root / "LIA_PROJECT_FACTS.md"
    text = facts.read_text(encoding="utf-8")
    marker = "## 266. NETP exact MJJ prior and low-frequency reconstruction prototype"
    if marker in text:
        raise RuntimeError("round134-section-already-present")
    if "## 265. Independent precipitation low-frequency validation" not in text:
        raise RuntimeError("round133-anchor-missing")
    facts.write_text(text.rstrip() + SECTION + "\n", encoding="utf-8")
    print("ROUND134_FACTS_UPDATE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
