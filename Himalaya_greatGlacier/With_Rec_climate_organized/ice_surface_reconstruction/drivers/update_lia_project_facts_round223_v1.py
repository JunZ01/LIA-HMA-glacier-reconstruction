from __future__ import annotations

import hashlib
from pathlib import Path


FACTS = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm\LIA_PROJECT_FACTS.md")
EXPECTED_SHA256 = "cc62f678c4366cd459a0caeaec40f0c05954b7a840d02a044ffc05afe334d148"
OLD_HEADER = "Last updated: 2026-08-22 (Rounds 219-222: an independent prior and REGEN confirm that NETP MJJ rank-1 spatialization is not production-ready)"
NEW_HEADER = "Last updated: 2026-08-22 (Round 223: keep current production climate; retain NETP MJJ only as a regional temporal research product)"
APPENDIX = r"""

## 342. Climate-reconstruction production eligibility synthesis (2026-08-22)

- Round 223 synthesized only locked-safe derived summaries from Rounds 138,
  217, and 218-222 plus proxy-coordinate metadata. It read no original climate
  field or proxy value.
- A uniform temperature translation is rejected, and a common 1901-1920 phase
  correction remains unauthorized because the remaining disagreement is
  season-dependent. The existing production temperature field therefore stays
  unchanged.
- The NETP MJJ regional temporal curve and structural envelope remain useful as
  a regional research product. Fixed rank-1 spatialization is rejected by the
  two-prior and REGEN tests and is not a gridded replacement candidate.
- A second NETP spatial mode is not presently estimable from the fixed training
  network. Its 15 spatial groups occupy only `97.00-100.25 E` and
  `35.00-38.75 N` inside a `93-107 E`, `35-41 N` target; only two groups are
  at or east of 100 E. This is a geometry-of-evidence limit, not a reason to
  lower the spatial-pattern gate.
- Decision:
  `retain_current_production_climate_and_regional_netp_temporal_research_product_only`.
  No production reconstruction, EnKF, climate correction, or OGGM rerun is
  authorized by the current branch. Reopening a gridded replacement requires
  new independent, spatially distributed central/eastern-HMA evidence capable
  of identifying an additional large-scale spatial degree of freedom.
- Round 223 changed no source file, accessed no numerical value for 1921-1939
  or Jionglacuo, and ran no EnKF, production reconstruction, correction, or
  OGGM simulation.
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
    if "## 342. Climate-reconstruction production eligibility synthesis" in text:
        raise RuntimeError("Round 223 facts already appended")
    updated = text.replace(OLD_HEADER, NEW_HEADER, 1).rstrip() + APPENDIX + "\n"
    FACTS.write_text(updated, encoding="utf-8")
    print(f"before={before}")
    print(f"after={sha256(FACTS)}")


if __name__ == "__main__":
    main()
