from pathlib import Path


PATH = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm\LIA_PROJECT_FACTS.md")
OLD_HEADER = "Last updated: 2026-08-21 (Rounds 190-196: dual index closed; exact four-site chronology awaits authorized SigFree reproduction)"
NEW_HEADER = "Last updated: 2026-08-22 (Rounds 197-201: Zenodo East Asia field closed as an independent absolute-temperature target)"
HEADING = "## 323. Zenodo East Asia temperature-field admission closure (2026-08-22)"
SECTION = r'''

## 323. Zenodo East Asia temperature-field admission closure (2026-08-22)

- Rounds 197-198 audited Zenodo `10.5281/zenodo.18477496` without downloading
  its 1.2 GB `Data.zip` or 215 MB `Results.zip`. The deposited field covers
  East Asia at annual 1-degree resolution over 1400-2000. Its distinct improved
  analog method selects fields from 37 CMIP6/PMIP4 model files using a
  423-record proxy ensemble.
- The field is not an independent absolute-temperature scale. The deposited
  code converts selected standardized model states to degrees Celsius using
  CRU TS4.04 cellwise means and standard deviations over 1902-1977.
- Round 199 extracted only the proxy workbook member and decoded physical rows
  1-14 as metadata. Of 423 external proxies, 219 have strong metadata matches
  to the current project ledger. Of 114 HMA records spanning both 1500 and
  1940, 102 (`89.47%`) strongly match. The product therefore supplies a new
  method family but not an independent HMA long-proxy evidence family.
- MATLAB R2024b metadata closure used a synthetic workbook containing the
  unchanged metadata and only year labels 1-2019. Numeric metadata close with
  maximum error zero and all 423 IDs match; two raw type mismatches are only
  trailing spaces. However, the deposited expression `num(15:end,1)` yields
  2018 rows spanning 2-2019, while the following Step-1 loop requests 2019
  rows. The public code/input pair is therefore not reproducible as deposited.
- Decision: close this field as an independent absolute-temperature validation
  target for the HMA 1500-1899 warm-bias magnitude. Do not download or inspect
  `Results.zip` on the current mainline. It may be reconsidered only as
  comparison-only evidence after corrected-code and source-lineage
  clarification.
- No annual climate, proxy, model, reconstruction, 1921-1939, or Jionglacuo
  value was read, and no reconstruction, correction, EnKF, or OGGM run
  occurred. Evidence is in
  `round_197_east_asia_temperature_metadata_v2/`,
  `round_198_east_asia_remote_data_inventory_v1/`,
  `round_199_east_asia_proxy_identity_metadata_v3/`,
  `round_200_east_asia_xlsread_metadata_closure_v4/`, and
  `round_201_east_asia_temperature_field_admission_v1/`.
'''


def main() -> None:
    raw = PATH.read_bytes()
    text = raw.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    normalized = text.replace("\r\n", "\n")
    if HEADING in normalized:
        raise RuntimeError("Round 197-201 facts section already exists")
    if normalized.count(OLD_HEADER) != 1:
        raise RuntimeError("Expected prior facts header exactly once")
    normalized = normalized.replace(OLD_HEADER, NEW_HEADER, 1)
    normalized = normalized.rstrip() + SECTION + "\n"
    PATH.write_bytes(normalized.replace("\n", newline).encode("utf-8"))
    print(f"updated {PATH}")


if __name__ == "__main__":
    main()
