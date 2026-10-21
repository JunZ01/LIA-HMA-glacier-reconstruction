from pathlib import Path


PATH = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm\LIA_PROJECT_FACTS.md")
OLD_HEADER = "Last updated: 2026-08-22 (Rounds 197-201: Zenodo East Asia field closed as an independent absolute-temperature target)"
NEW_HEADER = "Last updated: 2026-08-22 (Rounds 197-203: two new gridded fields closed; Zhu 2011 exact download remains login-blocked)"
HEADING = "## 324. Zhu 2011 exact-download and Yangtze-field closures (2026-08-22)"
SECTION = r'''

## 324. Zhu 2011 exact-download and Yangtze-field closures (2026-08-22)

- Round 202 recovered the exact indexed ResearchGate endpoint for
  `Zhuetal2011P3.xls`, but an anonymous request returned HTTP 403 with a
  20,200-byte HTML response rather than an OLE or ZIP spreadsheet. The response
  is retained as failed-access evidence. Automatic anonymous recovery is
  closed; the only remaining access path is an authenticated browser download
  or direct author delivery.
- Round 203 closed the 2026 Yangtze River Basin millennial summer-temperature
  field (`10.1038/s41597-026-06959-0`, dataset
  `10.5281/zenodo.18724777`) at metadata level. The final field integrates
  ModE, PHYDA, Shi, Cook, and nine CMIP5/CMIP6 simulations, while CRU TS is used
  for GCM bias correction, grid weights, and 1901-2005 evaluation. It therefore
  overlaps the project's existing comparison families and absolute scale, and
  covers the Yangtze basin rather than the full HMA.
- Decision: do not download the Yangtze field as an independent warm-bias
  target. Zhu 2011 remains the more valuable unresolved eastern long record,
  but no tuning or production work may proceed without the actual file and its
  source-lineage audit.
- No spreadsheet cell, annual climate/proxy value, locked-period value, or
  Jionglacuo value was opened, and no reconstruction, EnKF, correction, or OGGM
  run occurred. Evidence is in
  `round_202_zhu2011_researchgate_exact_download_v1/` and
  `round_203_yangtze_millennial_temperature_source_overlap_v1/`.
'''


def main() -> None:
    raw = PATH.read_bytes()
    text = raw.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    normalized = text.replace("\r\n", "\n")
    if HEADING in normalized:
        raise RuntimeError("Round 202-203 facts section already exists")
    if normalized.count(OLD_HEADER) != 1:
        raise RuntimeError("Expected prior facts header exactly once")
    normalized = normalized.replace(OLD_HEADER, NEW_HEADER, 1)
    normalized = normalized.rstrip() + SECTION + "\n"
    PATH.write_bytes(normalized.replace("\n", newline).encode("utf-8"))
    print(f"updated {PATH}")


if __name__ == "__main__":
    main()
