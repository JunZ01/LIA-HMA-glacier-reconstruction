from __future__ import annotations

import hashlib
from pathlib import Path


FACTS = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm\LIA_PROJECT_FACTS.md")
EXPECTED_SHA256 = "071202d7fde9302933fdfb8c6ca0f8bafc7f85ac47f657d6591e8f7239981e4f"
APPENDIX = r"""

## 333. Official Haizi annual-temperature evidence recovery (2026-08-22)

- Round 214 recovered the official supplementary workbook for Xu et al.
  (2024), Scientific Reports, DOI `10.1038/s41598-024-79096-6`, from the
  Springer static-content archive. The workbook is 45,409 bytes with SHA256
  `35302c0ba1912b015ee8d8b1a5c05df9aa82dd44d4467f7b07278e723611d3eb`.
- It publishes a previous-October to current-September mean-temperature
  reconstruction for the southeastern Tibetan Plateau. The chronology is
  reliable over 1115-2017 and the reported Daocheng calibration spans
  1959-2017 with `r=0.75` and 57% adjusted explained variance.
- The study-area audit centroid is 29.58 N, 100.65 E. The nearest current
  project tree group is `P9_0048` / `proxy6_049`, formal identity
  `Asia-DAOSIC::Asia_040`, 64.499 km away. Round 214 read workbook structure
  and project-ledger identity metadata only; no numerical temperature or
  locked-period value was read.

## 334. Haizi-versus-Daocheng sequence identity closure (2026-08-22)

- Round 215 v3 compared the official Haizi standard chronology with the exact
  NOAA/PAGES2k source `Asi-CHIN018.Li.2010.txt` used by the nearest project
  identity. Both parsers decoded year first and skipped all 19 rows for
  1921-1939 before chronology-value access.
- Across 448 allowed common years in 1540-2006, raw, first-difference, and
  31-year correlations are `0.2705`, `0.5050`, and `-0.1196`. An affine fit
  leaves residual RMSE equal to 96.2% of the Haizi chronology standard
  deviation.
- Decision: `sequence_independent_for_annual_temperature_gate`. The new
  chronology is not the existing Daocheng sequence in disguise and is
  admissible as independent annual-temperature evidence.

## 335. Independent Haizi annual-temperature comparison (2026-08-22)

- Round 216 compared the admitted Haizi record with the frozen six-member
  project annual field and four-family annual reference in a 1-degree box
  centered on 29.58 N, 100.65 E. All anomalies use 1961-1990; smoothing was
  performed separately on each side of the sealed 1921-1939 gap.
- In eight 50-year windows over 1500-1899, Haizi is below baseline in 0/8,
  the project local annual curve in 1/8, and the four-family local reference
  in 8/8. Their median anomalies are `+0.260`, `+0.122`, and `-0.192 degC`.
  The project-minus-Haizi median difference is `-0.138 degC`; thus this annual
  record does not support the frozen local project warm-bias gate.
- Over 1850-1900, Haizi and the project both cool at `-0.172` and
  `-0.190 degC/century`. Over 1901-1920, both warm at `+2.539` and
  `+0.253 degC/century`. The annual evidence therefore does not reproduce the
  early cooling sign found in the two seasonal reconstructions.
- No production file changed, no locked project/reference raster band was
  read, and no Jionglacuo, EnKF, climate correction, or OGGM run occurred.

## 336. Three-record temperature bias-versus-phase decision (2026-08-22)

- Round 217 synthesized only locked-safe derived outputs for Huang winter
  Tmin, Baiyu June-July temperature, and Haizi annual temperature. All three
  independent records have positive 1500-1899 median anomalies; zero records
  pass the frozen uniform-cooling gate, including the annual record.
- Against the three external historical medians, the project local curves
  have RMSE `0.089 degC`; the four-family local reference has RMSE
  `0.396 degC`. The project-to-reference RMSE ratio is `0.225`. The apparent
  project warm offset relative to that persistently cold reference is not a
  validated project bias; in these three regions the project level is much
  closer to independent evidence.
- Over 1850-1900, project and external trend signs agree for 3/3 records. Over
  1940-2000 they agree for 2/3. For 1901-1920, both seasonal records cool, but
  the independent annual record and project annual curve warm. The remaining
  early conflict is season-dependent rather than a common annual reversal.
- Decision:
  `uniform_historical_warm_bias_rejected_early20_conflict_is_season_dependent`.
  Do not apply a uniform temperature translation or a common 1901-1920 phase
  correction, and do not rerun production reconstruction or OGGM from this
  branch. Future improvement should target reconstruction skill, spatial and
  seasonal PSM structure, and uncertainty rather than forcing agreement with
  the four-family reference mean.
- Rounds 214-217 changed no read-only source, accessed no numerical value for
  1921-1939, and accessed no Jionglacuo value.
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
    if "## 333. Official Haizi annual-temperature evidence recovery" in text:
        raise RuntimeError("Round 214-217 facts already appended")
    FACTS.write_text(text.rstrip() + APPENDIX + "\n", encoding="utf-8")
    print(f"before={before}")
    print(f"after={sha256(FACTS)}")


if __name__ == "__main__":
    main()
