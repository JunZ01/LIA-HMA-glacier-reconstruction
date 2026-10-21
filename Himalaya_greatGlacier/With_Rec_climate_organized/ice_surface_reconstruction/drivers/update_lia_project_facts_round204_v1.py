from __future__ import annotations

import hashlib
import os
from pathlib import Path


FACTS = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm\LIA_PROJECT_FACTS.md")
EXPECTED_SHA256 = "E89E563833BFB24E153545CBF344426BDD927B46CAD91165B836030A3BA86D12"
MARKER = "## 325. SETP PPR temperature-field comparison boundary (2026-08-22)"
OLD_HEADER = (
    "Last updated: 2026-08-22 (Rounds 197-203: two new gridded fields closed; "
    "Zhu 2011 exact download remains login-blocked)"
)
NEW_HEADER = (
    "Last updated: 2026-08-22 (Rounds 197-204: new gridded fields closed or "
    "restricted to comparison-only use)"
)

SECTION = """
## 325. SETP PPR temperature-field comparison boundary (2026-08-22)

- Round 204 audited Li et al. (2022), DOI `10.2113/2022/9648384`, as a
  possible eastern-HMA temperature constraint. The product reconstructs
  May-June mean temperature at 28 station targets over 25-35 N and 90-104 E,
  calibrated against the Chinese Homogeneous Temperature Data Set for
  1961-2004. CRU TS 3.20 is described for spatial-correlation analysis rather
  than as the station reconstruction target.
- The longest single station records span 1480-2002, but only `ZaD` and `DaR`
  begin by 1500. Eleven stations begin by 1520, five begin only in 1730, and
  the complete common field spans 1730-1998. This is not a complete regional
  1500-1899 validation field.
- The 53-chronology predictor network contains 37 ITRDB records explicitly
  associated by the paper with the existing PAGES2k Asia reconstruction. The
  current project ledger already contains 110 tree-record representations in
  the same SETP domain, covering 56 evidence groups; 106 representations are
  from PAGES2k2017. Source-family overlap is therefore confirmed, while exact
  record overlap cannot be closed without the paper's 53-record identity table.
- The article states that its numerical dataset is available only by request.
  Crossref, DOAJ, and OpenAlex expose no numeric attachment. Automated access
  to the publisher returned a Cloudflare challenge rather than a PDF, and the
  Lanzhou University repository endpoints returned HTTP 503.
- Decision: retain this study as
  `comparison_only_pending_numeric_data`. If the authors provide the
  28-station field, 53 chronology identities, and predictor membership, run a
  locked-safe comparison over 1480-1920 and 1940-2002, with 1921-1939 rows
  skipped before value conversion. Do not represent it as independent absolute
  truth and do not use it alone to decide the full-HMA 1500-1899 warm bias.
- No production reconstruction, correction, EnKF, OGGM, locked-period annual
  value access, or Jionglacuo access occurred. Evidence is in
  `round_204_setp_ppr_temperature_field_metadata_v1/`.
""".strip()


def main() -> None:
    raw = FACTS.read_bytes()
    actual = hashlib.sha256(raw).hexdigest().upper()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"facts hash changed: expected {EXPECTED_SHA256}, got {actual}")

    had_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    if MARKER in text:
        raise RuntimeError("Round 204 section already exists")
    if text.count(OLD_HEADER) != 1:
        raise RuntimeError("facts header did not match exactly once")

    newline = "\r\n" if text.count("\r\n") > text.count("\n") / 2 else "\n"
    updated = text.replace(OLD_HEADER, NEW_HEADER, 1).rstrip("\r\n")
    updated += newline * 2 + SECTION.replace("\n", newline) + newline
    encoded = updated.encode("utf-8-sig" if had_bom else "utf-8")

    tmp = FACTS.with_name(FACTS.name + ".round204.tmp")
    tmp.write_bytes(encoded)
    os.replace(tmp, FACTS)


if __name__ == "__main__":
    main()
