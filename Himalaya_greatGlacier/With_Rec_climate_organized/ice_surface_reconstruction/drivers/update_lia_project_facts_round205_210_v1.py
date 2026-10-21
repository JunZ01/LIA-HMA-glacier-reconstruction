from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
EXPECTED_SHA256 = "7d1429539564347cd4e1a632a2169cc4911dd1c350a0395738ea97d86fca1f38"
OLD_HEADER = (
    "Last updated: 2026-08-22 (Rounds 197-204: new gridded fields closed or "
    "restricted to comparison-only use)"
)
NEW_HEADER = (
    "Last updated: 2026-08-22 (Rounds 197-210: southeastern-Tibetan winter "
    "evidence audited; uniform warm correction remains unauthorized)"
)


APPENDIX = r"""

## 326. Official southeastern-Tibetan winter-temperature evidence boundary (2026-08-22)

- Round 205 recovered the official CASEarth dataset for Huang et al. (2019),
  article DOI `10.1007/s00382-019-04695-3`, dataset DOI
  `10.11888/Geogra.tpdc.270139`. The public CSV `Changdu winter Tmin.csv` is
  14,081 bytes with SHA256
  `9957b1717ac11f0ecafac739963ae77c7e96a47c65c9068b372ef17cfe4924d6`.
- The reconstruction targets previous-November to current-February minimum
  temperature over 94.5-97.5 E, 29.5-31.5 N for 1340-2007 and is calibrated
  to CRU TS4.01 over 1960-2007. It is seasonal, CRU-calibrated evidence, not an
  independent absolute annual-HMA temperature scale.
- Official Springer Supplement Table S1 identifies 12 high-elevation tree-ring
  sites. The current proxy ledger has one probable/related chronology candidate,
  eight nearby but identity-unclosed candidates, and three sites without a
  candidate within 75 km. No exact overlap was promoted from coordinates and
  chronology bounds alone.
- Round 205 opened only Supplement Table S1 metadata and held the annual CSV
  unopened. No source file changed, and no locked-period or Jionglacuo value was
  accessed.

## 327. Southeastern-Tibetan 1500-1899 winter comparison (2026-08-22)

- Round 206 v3 is authoritative; v1/v2 used an unsuitable smoothed-SD display
  and are superseded for plotting. The v3 figure uses 31-year temperature
  anomalies relative to 1961-1990 and performs smoothing separately on each
  side of the locked gap.
- In eight 50-year windows over 1500-1899, Huang winter Tmin is below its
  1961-1990 mean in only 2/8 windows, the frozen six-member project SETP annual
  curve in 0/8, and the four-family SETP annual reference in 8/8. Median
  anomalies are `+0.127`, `+0.182`, and `-0.249 degC`, respectively.
- The external 50-year shape correlates `0.607` with the project SETP curve and
  `0.799` with the four-family SETP reference; both have 6/7 adjacent-window
  sign agreements. Thus the external record is close to the project in level
  but closer to the reference in low-frequency shape.
- Decision: this record does not confirm a uniform 1500-1899 project warm
  offset and cannot set an absolute correction amplitude. Do not translate or
  rescale the current reconstruction from this comparison.
- The parser decoded each year first and discarded all 19 rows for 1921-1939
  before temperature conversion. Project raster bands for those years were not
  read, no smoothing crossed the gap, and no correction, EnKF, production, or
  OGGM run occurred.

## 328. Early-twentieth-century phase interpretation correction (2026-08-22)

- Round 207 v2 found that all six frozen project members have positive
  1901-1920 trends while Huang winter Tmin is negative. This establishes a
  project-common disagreement with Huang, not a project-common error. Round
  208 supersedes any stronger `wrong-sign` interpretation.
- Round 208 downloaded only the official CRU TS4.01 Tmin decade files for
  1901-1920 and 1941-2000. The 1921-1930 and 1931-1940 files were never
  requested, opened, or hashed. Exact previous-November to current-February
  CRU Tmin has only `20.50%` station-supported box cell-months in 1902-1920,
  versus `100%` in 1960-2000.
- Over 1902-1920, the project SETP annual trend is
  `+2.824 degC/century`; all five gridded annual products are positive
  (`+2.131` to `+4.222`), and exact CRU TS4.01 winter Tmin is `+7.142`.
  Huang winter Tmin alone is negative at `-1.613`. Therefore this interval is
  an early proxy-versus-gridded-product conflict under weak direct-station
  support, not evidence that the project warming sign is erroneous.
- Over 1942-2000, annual products remain mixed. Project/CRU annual/NOAA trends
  are `-0.441/-0.487/-0.956 degC/century`, while Berkeley Earth, GISTEMP, and
  HadCRUT5 are `+0.248/+0.404/+0.345`. Exact CRU winter Tmin and Huang winter
  Tmin are positive at `+0.486/+0.584`. The winter-versus-annual target
  difference is material, but the annual product disagreement prevents a
  unique modern correction target.
- Decision: retain `not_resolved_mixed_instrumental_products`. Do not call
  1901-1920 a validated project error, and do not use Huang or CRU alone to
  tune the reconstruction.

## 329. Southeastern-Tibetan direct-station route closure (2026-08-22)

- Round 209 found two GHCN-M QCU stations inside the Huang box (DENGQEN and
  QAMDO), four within 100 km, 12 within 200 km, and 23 within 300 km. The
  value file remained unopened in that metadata round.
- Round 210 inspected availability and QC flags only for the four stations
  within 100 km over 1940-2000. DENGQEN, QAMDO, and SOG_XIAN have 40, 49, and
  39 complete years and pass the 30-year station gate; NYINGCHI has 29 and
  fails. At least two eligible stations are complete in only `70.49%` of the
  years, below the prospectively frozen `75%` gate.
- Decision: close direct GHCN-M phase-value adjudication on coverage. No station
  temperature magnitude was converted or retained, and the threshold must not
  be relaxed after seeing the near miss.
- Current main-line state: keep the existing temperature reconstruction
  unchanged. A uniform 1500-1899 warm correction, member reweighting, and an
  early-twentieth-century phase correction remain unauthorized. Reopening this
  regional branch requires genuinely stronger direct evidence such as
  author-supplied CHTDS/CMA station series or an independently documented
  source-native chronology with adequate modern and historical transport.
- Rounds 205-210 changed no read-only source, accessed no Jionglacuo value, ran
  no EnKF, production reconstruction, climate correction, or OGGM, and read no
  numeric value for 1921-1939.
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if digest(FACTS) != EXPECTED_SHA256:
        raise RuntimeError(f"Facts hash conflict: {digest(FACTS)}")
    text = FACTS.read_text(encoding="utf-8")
    if OLD_HEADER not in text:
        raise RuntimeError("Expected facts header not found")
    if "## 326. Official southeastern-Tibetan winter-temperature evidence boundary" in text:
        raise RuntimeError("Rounds 205-210 facts already appended")
    updated = text.replace(OLD_HEADER, NEW_HEADER, 1).rstrip() + APPENDIX + "\n"
    FACTS.write_text(updated, encoding="utf-8")
    if "## 329. Southeastern-Tibetan direct-station route closure" not in FACTS.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("Facts append verification failed")


if __name__ == "__main__":
    main()
