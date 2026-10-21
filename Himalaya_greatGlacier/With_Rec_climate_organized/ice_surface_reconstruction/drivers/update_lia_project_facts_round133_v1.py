"""Append the verified Round 133 result to the canonical project facts."""

from pathlib import Path


SECTION = r'''

## 265. Independent precipitation low-frequency validation of Xinpu/Keylong branches (2026-08-18)

- Round 133 corrected the spatial interpretation of the public QT-Prec benchmark before testing the Round 130/132 branches. The source paper defines the northeastern Qinghai-Tibet Plateau model-comparison area as 35-42 degrees N, 92-108 degrees E. Round 122's preliminary `east_ge90` comparison covered a much broader part of HMA; its reported 1500-1920 31-year correlation of 0.598 is therefore a coarse screening value and is superseded for formal NETP validation by the paper-matched Round 133 mask.
- The frozen comparison used baseline, combined Xinpu/Keylong BEA, combined Xinpu/Keylong CRU, and product-marginalized posterior fields. QT-Prec was area-weighted over 32 state cells in the published NETP box. The independent Shenqi Cave precipitation index was evaluated only as secondary isotope evidence against the nearest state cell to the published 28.93 degrees N, 103.10 degrees E cave location, namely 29 degrees N, 103 degrees E.
- Because the posterior state is annual precipitation while QT-Prec is May-July total precipitation and Shenqi is a monsoon precipitation index, the formal test used standardized low-frequency phase and trend direction rather than raw amplitude. It used only 1500-1920 and 1800-1920; no external series was interpolated.
- For QT-Prec over 1500-1920, baseline 31-year correlation is 0.2370. The BEA, CRU, and product-marginalized branches are 0.2406, 0.2408, and 0.2409. Over 1800-1920, baseline is 0.1401 while the candidates are lower at 0.1387, 0.1346, and 0.1368. Mean correlation changes relative to baseline are only +0.00110, -0.00082, and +0.00026, all far below the frozen +0.05 support gate.
- Shenqi correlations likewise change negligibly. The mean changes across the two periods are +0.00046, +0.00080, and +0.00069. All four branches preserve the same trend-sign pattern: agreement for both external records over 1500-1920 and disagreement for both over 1800-1920. Thus none repairs the nineteenth-to-early-twentieth-century low-frequency direction.
- Decision: close the current annual-Gaussian precipitation-assimilation route for Xinpu and Keylong. The records remain scientifically valid low-frequency constraints, but the tested operators do not provide independent evidence of improved precipitation reconstruction. Any future use requires an explicitly temporal/seasonal low-frequency operator and a new validation contract, not further innovation clipping, R inflation, or product selection on these branches.
- Round 133 changed no source file, read no 1921-1939 value, did not open the frozen external validation reserve, and ran no OGGM simulation. All source fingerprints were unchanged.
'''


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    facts = root / "LIA_PROJECT_FACTS.md"
    text = facts.read_text(encoding="utf-8")
    marker = "## 265. Independent precipitation low-frequency validation"
    if marker in text:
        raise RuntimeError("round133-section-already-present")
    if "## 264. New d18O seasonal operators and frozen EnKF addition result" not in text:
        raise RuntimeError("expected-round264-anchor-missing")
    facts.write_text(text.rstrip() + SECTION + "\n", encoding="utf-8")
    print("ROUND133_FACTS_UPDATE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
