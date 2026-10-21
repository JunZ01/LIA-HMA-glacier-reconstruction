from __future__ import annotations

import hashlib
from pathlib import Path


FACTS = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm\LIA_PROJECT_FACTS.md")
EXPECTED_SHA256 = "2075a16b83aebc0120d951107354b58764f2e2a70ca8be390d2dc6fee44f9cbd"
OLD_HEADER = "Last updated: 2026-08-22 (Rounds 211-213: independent western-Sichuan evidence rejects a uniform warm correction; early-20th-century phase conflict remains unresolved)"
NEW_HEADER = "Last updated: 2026-08-22 (Rounds 219-222: an independent prior and REGEN confirm that NETP MJJ rank-1 spatialization is not production-ready)"
APPENDIX = r"""

## 338. Independent CCSM4 NETP MJJ prior recovery (2026-08-22)

- Round 219 recovered a second physically independent prior covariance from
  the local CMIP5 CCSM4 past1000 monthly precipitation simulation, covering
  850-1850. It was reduced to the same 32 NETP MJJ target cells used in Round
  218 without opening any proxy or observational reconstruction value.
- The resulting prior is complete and physically valid: 1001 years x 32
  cells, no missing or negative MJJ precipitation, regional mean
  `2.56055 mm/day`, interannual regional standard deviation `0.33311 mm/day`,
  and area-weighted loading closure exactly `1.0`. Cell loadings range from
  `0.1385` to `2.4484`.
- Decision: `admit_ccsm4_as_independent_netp_mjj_prior_covariance`. Admission
  means it is suitable for a structural sensitivity test, not that it is a
  verified historical precipitation field.

## 339. Two-prior NETP MJJ spatialization closure (2026-08-22)

- Round 220 repeated the unchanged Round-218 reciprocal OOF experiment with
  CESM-LME010 loadings, CCSM4 past1000 loadings, and their fixed 50/50 mean.
  CESM and CCSM4 loadings correlate `0.9534` with RMSE `0.1940`, showing that
  two independent model priors imply nearly the same rank-1 spatial pattern.
- The equal-prior branch retains useful regional/amplitude skill: GPCC/CRU
  pooled field skills are `0.2286/0.5589`. Its median yearly spatial-pattern
  correlations are only `0.1709/0.4209`; the frozen GPCC threshold of `0.30`
  still fails. The CESM and CCSM4 branches fail the same GPCC gate at
  `0.1786` and `0.1528`, respectively.
- Decision:
  `close_two_prior_netp_mjj_spatialization_retain_temporal_curve`. The failure
  is not specific to the CESM-LME prior. No tested fixed prior loading makes
  the rank-1 spatial field production-eligible.

## 340. Official REGEN precipitation recovery boundary (2026-08-22)

- Round 221 recovered REGEN All Stations v1-2019 from the official NCI
  THREDDS service, DOI `10.25914/5ca4c380b0d44`, under CC BY-NC-SA 4.0.
  Fifty-one annual NETP subsets for 1950-2000, the official quality mask,
  catalog, README, and license were stored unmodified and individually hashed.
- Bilinear interpolation to the 32 Round-218 targets yields 21 cells for which
  all four source-grid neighbors pass the official high-quality mask. Every
  eligible cell-year has all 92 finite MJJ daily values.
- REGEN uses an independent gridding method but is not fully source-independent:
  its station collection overlaps GHCN/GPCC and its absolute daily scale uses
  GPCC Full Data Monthly v2018. It is admitted only as a source-correlated
  gridding-method diagnostic and cannot override the frozen production gate.

## 341. REGEN adjudication of the NETP MJJ spatial failure (2026-08-22)

- Round 222 compared 31 common 11-year-smoothed centers in 1955-1964 and
  1975-1995 across the 21 official high-quality REGEN cells. Smoothing was
  performed separately over 1950-1969 and 1970-2000, and each product used its
  own finite 1961-1990 cell climatology.
- REGEN agrees strongly with the observed GPCC and CRU spatial anomaly fields:
  median annual spatial-pattern correlations are `0.9006` and `0.8412`.
  REGEN is therefore not exposing a GPCC-only interpolation failure.
- In contrast, REGEN correlates `-0.1115` with both equal-prior OOF predicted
  fields. The equality is expected because GPCC- and CRU-calibrated rank-1
  predictions differ in temporal amplitude but share the same fixed loading
  pattern; spatial correlation is invariant to that positive scalar.
- Decision:
  `rank1_spatialization_failure_supported_by_two_of_three_gridding_methods`.
  REGEN and GPCC directly agree on the observed spatial pattern while both
  reject the fixed-loading prediction. Retain the validated NETP regional MJJ
  temporal curve, but close rank-1 spatialization as a production route. A
  future field requires at least one additional independently estimable spatial
  degree of freedom with adequate proxy support; lowering the current spatial
  gate or tuning a fixed loading after this result is unauthorized.
- Rounds 219-222 changed no read-only source, accessed no numerical value for
  1921-1939 or Jionglacuo, opened no paleo proxy value, and ran no EnKF,
  production reconstruction, climate correction, or OGGM simulation.
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
    if text.count(OLD_HEADER) != 1:
        raise RuntimeError("Expected facts header not unique")
    if "## 338. Independent CCSM4 NETP MJJ prior recovery" in text:
        raise RuntimeError("Rounds 219-222 facts already appended")
    updated = text.replace(OLD_HEADER, NEW_HEADER, 1).rstrip() + APPENDIX + "\n"
    FACTS.write_text(updated, encoding="utf-8")
    print(f"before={before}")
    print(f"after={sha256(FACTS)}")


if __name__ == "__main__":
    main()
