from __future__ import annotations

import hashlib
from pathlib import Path


FACTS = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm\LIA_PROJECT_FACTS.md")
EXPECTED_SHA256 = "0e4bfbc4bf8afb374b576a430cd5381ea7302e52966712760cd1179277cafd7"
OLD_HEADER = (
    "Last updated: 2026-08-22 (Rounds 197-210: southeastern-Tibetan winter "
    "evidence audited; uniform warm correction remains unauthorized)"
)
NEW_HEADER = (
    "Last updated: 2026-08-22 (Rounds 211-213: independent western-Sichuan "
    "evidence rejects a uniform warm correction; early-20th-century phase "
    "conflict remains unresolved)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


ADDENDUM = r"""

## 330. Official western-Sichuan summer-temperature data recovery (2026-08-22)

- Round 211 v2 recovered the official GeoDOI dataset
  `10.3974/geodb.2022.02.03.V1`, associated with Zhang et al. (2022), paper
  DOI `10.1002/joc.7336`. The public archive
  `TemReconWSCP_1383-2005.rar` is 2,072,378 bytes with SHA256
  `d9a16a56a837ba185ae47839c71a2dc20ef256237b1af3dbe389e2dd34610fb8`.
- The archive contains nine files. The numerical workbook
  `Temp_Reconstruction&Meteorological.xlsx` has SHA256
  `286e9338e1cfeddb2b51f6805241d1e0989c891af3adbdd81691f1918b794d68`
  and publishes June-July mean-temperature reconstruction for 1383-2005 and
  a seven-station instrumental target for 1962-2005.
- The Picea balfouriana chronology was sampled at Baiyu, 31.15 N, 99.81 E,
  in October 2017. The nearest current project tree evidence group is 111.355
  km away and no group is within 35 km. Together with the post-PAGES2k2017
  sampling date, this supports classification as source-independent of the
  current assimilated tree network.
- Round 211 read file and ledger metadata only. It read no annual temperature
  value, locked-period value, or Jionglacuo value.

## 331. Independent Baiyu temperature comparison (2026-08-22)

- Round 212 decoded each workbook year first and skipped all 19 rows for
  1921-1939 before accessing the temperature cell. Project and reference
  locked raster bands remained unread, and no smoothing crossed the gap.
- The official Baiyu reconstruction reproduces its seven-station June-July
  target over 1962-2000 at `r=0.7449`, consistent with the published 55.4%
  explained variance. The project annual local curve correlates `0.4414` with
  the seasonal instrumental target; the four-family local annual reference
  correlates `0.3501`. These annual-seasonal correlations are diagnostic, not
  absolute validation gates.
- Over eight 50-year windows in 1500-1899, Baiyu is below its 1961-1990
  baseline in only 2/8 windows, the project source-centered annual curve in
  1/8, and the four-family local annual reference in 7/8. Median anomalies are
  `+0.191`, `+0.152`, and `-0.160 degC`, respectively.
- Decision:
  `independent_wscp_record_does_not_support_uniform_warm_correction`. This
  source-independent record rejects using the persistently cold four-family
  local reference as sufficient evidence for translating the project field.
- A separate phase disagreement remains: over 1901-1920 Baiyu trends
  `-5.234 degC/century`, while the project local annual curve trends
  `+0.353 degC/century`. Seasonal versus annual targets prevent direct
  correction from this difference alone.

## 332. Temperature level-versus-phase synthesis (2026-08-22)

- Round 213 used only locked-safe derived outputs from Rounds 206, 208, and
  212. It opened no original climate, proxy, station, raster, MAT, NetCDF, or
  workbook source.
- Neither available eastern-HMA seasonal reconstruction supports uniform
  1500-1899 cooling: Huang winter Tmin is negative in 2/8 50-year windows and
  Baiyu June-July temperature is negative in 2/8. Both have positive median
  historical anomalies relative to 1961-1990. The uniform warm-bias
  hypothesis is therefore rejected for the current evidence base.
- The evidence does strengthen a distinct 1901-1920 phase conflict. Both
  seasonal proxies cool and both corresponding project-local annual curves
  warm. In contrast, all five annual gridded products and exact CRU winter
  Tmin warm, while only 20.50% of early CRU box cell-months are
  station-supported.
- Decision:
  `uniform_warm_correction_rejected_early20_phase_conflict_strengthened_but_unresolved`.
  Keep the current reconstruction unchanged. Do not authorize a uniform
  translation, member reweighting, early-phase correction, production rerun,
  or OGGM run from this evidence.
- Reopen the early-phase branch only with direct historical station values or
  another source-independent chronology that can arbitrate proxy-versus-grid
  sign without relying on the same sparse early instrumental field.
- Rounds 211-213 changed no read-only source, accessed no Jionglacuo value,
  read no numeric value for 1921-1939, and ran no EnKF, production
  reconstruction, climate correction, or OGGM simulation.
"""


def main() -> None:
    actual = sha256(FACTS)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"Facts hash changed: {actual}")
    text = FACTS.read_text(encoding="utf-8")
    if "## 330. Official western-Sichuan summer-temperature data recovery" in text:
        raise RuntimeError("Round 211-213 addendum already present")
    if text.count(OLD_HEADER) != 1:
        raise RuntimeError("Expected facts header is absent or duplicated")
    text = text.replace(OLD_HEADER, NEW_HEADER, 1)
    FACTS.write_text(text.rstrip() + ADDENDUM + "\n", encoding="utf-8")
    print(sha256(FACTS))


if __name__ == "__main__":
    main()
