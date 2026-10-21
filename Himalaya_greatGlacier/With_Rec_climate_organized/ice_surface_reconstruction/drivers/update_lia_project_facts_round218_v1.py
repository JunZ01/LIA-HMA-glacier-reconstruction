from __future__ import annotations

import hashlib
from pathlib import Path


FACTS = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm\LIA_PROJECT_FACTS.md")
EXPECTED_SHA256 = "2328e67f29467e1652f51f5a10f26ce4a5b7ecb1b0984951ce367267cb9ff916"
APPENDIX = r"""

## 337. NETP MJJ fixed-prior spatialization closure (2026-08-22)

- Round 218 tested whether the validated NETP MJJ temporal composite can be
  spatialized over the 32-cell published NETP box using only fixed loadings
  from the exact 336 x 848 CESM-LME MJJ prior covariance. The reciprocal
  1940-1969/1970-2000 folds, 19 proxies in 15 equal spatial groups, centered
  11-year smoothing, GPCC/CRU targets, and all gates were frozen in advance.
  No observational cell-specific slope was fitted.
- V1 completed all scientific calculations but stopped while plotting because
  pandas resolved the `product` column name as a DataFrame method. Independent
  v2 changed only that selection to bracket notation and reproduced the same
  frozen calculations in a separate output directory.
- Regional and cell-amplitude skill is substantial. GPCC/CRU pooled field
  skills are `0.226/0.545`, median cell skills are `0.453/0.538`, positive-cell
  fractions are `78.1%/90.6%`, all four fold field skills are positive, and
  the two predicted anomaly fields correlate `0.994`.
- The spatial-pattern gate nevertheless fails for GPCC. Median yearly spatial
  anomaly-pattern correlations are `0.179` for GPCC and `0.413` for CRU,
  against the prospectively frozen `0.30` requirement for both products.
- Decision:
  `close_fixed_netp_mjj_prior_covariance_spatialization_retain_temporal_curve`.
  Retain the structurally robust NETP MJJ regional temporal curve and envelope,
  but do not publish this fixed-prior branch as a spatial field. Do not tune
  loadings, masks, regions, or thresholds after seeing the GPCC failure.
- Round 218 used only 1940-2000 modern values, read no paleo proxy value,
  no 1901-1939 or Jionglacuo value, changed no source, and ran no EnKF,
  production reconstruction, or OGGM.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    before = sha256(FACTS)
    if before != EXPECTED_SHA256:
        raise RuntimeError(f"Facts hash changed: {before}")
    text = FACTS.read_text(encoding="utf-8")
    if "## 337. NETP MJJ fixed-prior spatialization closure" in text:
        raise RuntimeError("Round 218 facts already appended")
    FACTS.write_text(text.rstrip() + APPENDIX + "\n", encoding="utf-8")
    print(f"before={before}")
    print(f"after={sha256(FACTS)}")


if __name__ == "__main__":
    main()
