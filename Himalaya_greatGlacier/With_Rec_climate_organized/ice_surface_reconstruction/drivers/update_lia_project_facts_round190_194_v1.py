from __future__ import annotations

from pathlib import Path


SECTION = r'''

## 318. Wang 2015 source-overlap boundary (2026-08-21)

- Round 190 transcribed the 12 Wang et al. (2015) ETP JJA component sites and
  froze their evidential role before further operator testing. The publication
  declares four old-PAGES-Asia2k overlaps: Gongshan, Litang, Shangrila, and
  Wulan. Refined source-native metadata confirms current-network overlap for at
  least Litang, Shangrila, and Wulan; Gongshan requires the lineage correction
  in Section 321.
- Therefore Wang 2015 is not a fully independent validation set for the current
  reconstruction. It may be used only as a partially source-overlapping regional
  trajectory benchmark for shape and phase, not to estimate the HMA warm-bias
  amplitude or authorize production correction.
- Haizi is 92.46 km from the nearest Wang component with no known shared source.
  The new Hengduan DLL site is within the same Gongshan sampling region and
  requires sample-level lineage review.
- Round 190 decoded no annual value, accessed no locked-period or Jionglacuo
  value, and ran no EnKF or OGGM.

## 319. CHIN041 sample-identity audit and scope (2026-08-21)

- Round 191 recovered the official NOAA ITRDB `chin041.rwl` unchanged
  (SHA256 `9495d84f64cab0d76abd9831b6e46af02c6d3990a7bf64b222d1c4971064e082`).
  Prefix-only parsing found 35 CHIN041 sample IDs and 168 IDs in the five new
  Hengduan files, with zero exact normalized ID overlap.
- No ring-width token was read or converted. Lines whose decades intersect
  1921-1939 were identified from their prefix and their value fields remained
  unread. No Jionglacuo value was opened or hashed.
- Section 321 narrows the scientific role: this result excludes direct reuse of
  CHIN041 samples only. CHIN041 is not Wang's Gongshan chronology and therefore
  Round 191 does not resolve Wang-Gongshan versus new-DLL independence.

## 320. Dual eastern-HMA thermal-index transport result (2026-08-21)

- Round 192 prospectively combined the Hengduan and Haizi standard chronologies
  with equal weights after training-fold standardization. Their documented
  instrumental targets were likewise combined as a dimensionless local thermal
  index; it was never represented as a degrees-C HMA field.
- The five-block 1940-1999 OOF test passed every frozen modern gate. Berkeley
  Earth correlation/skill were `0.306/0.093`; CRU were `0.438/0.188`; positive
  fold counts were `5/5` and `4/5`, all ten fitted slopes were positive, and the
  two product predictions correlated `0.990`.
- Round 193 then applied the unchanged equal-weight index to 1500-1899 and
  compared only low-frequency shape with the partially source-overlapping Wang
  2015 trajectory benchmark. The dual index failed every historical gate:
  centered-31-year correlation `-0.111`, eight-window 50-year correlation
  `0.066`, and adjacent-window sign agreement `3/7`.
- The failure is diagnostic. Hengduan alone has 50-year correlation `0.573` and
  sign agreement `5/7` with Wang, while Haizi is opposite at `-0.342` and `2/7`.
  Combining them improves modern instrumental skill but cancels historical
  low-frequency shape. This is direct evidence that modern common skill does
  not guarantee paleo low-frequency transport across differently standardized,
  differently seasonal regional chronologies.
- Decision: close the equal-weight dual index. Do not tune weights, seasons,
  signs, lags, or thresholds against Wang. Its absolute temperature amplitude
  was never validated, and no EnKF, production correction, or OGGM is
  authorized. Both parsers matched their modern Round 180/183 exports exactly,
  skipped all 19 locked rows before value conversion, and left sources unchanged.

## 321. Wang Gongshan--CHIN041 lineage correction (2026-08-21)

- Round 194 establishes that Wang et al. (2015) Gongshan and NOAA CHIN041 are
  different collections. Wang Gongshan is the Fan et al. (2010) `Larix
  speciosa` DL1+DL2 regional chronology with 68 series and a reliable period of
  1585-2005. NOAA CHIN041 is the `Tsuga dumosa` `22DL_T_r` collection with 35
  dated series over 1530-2005.
- This corrects the inference path used after Round 190: the zero CHIN041 sample
  overlap in Round 191 cannot establish independence between Wang Gongshan and
  the new `Larix speciosa` DLL site. DLL lies in the same sampling region; the
  relationship remains unresolved until Fan 2010 DL1/DL2 raw identities are
  recovered or DLL is prospectively excluded and the chronology rebuilt.
- Round 193 remains scientifically valid as a failed comparison against a
  potentially locally source-overlapping benchmark. Because it fails rather
  than promotes the operator, the unresolved overlap cannot create a false
  acceptance. The dual index remains closed.
- No annual value, locked-period value, or Jionglacuo value was read in Round
  194, and no EnKF or OGGM run occurred. Evidence is in
  `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_194_wang_gongshan_chin041_lineage_correction_v1/`.
'''


def main() -> int:
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "LIA_PROJECT_FACTS.md").exists())
    path = root / "LIA_PROJECT_FACTS.md"
    text = path.read_text(encoding="utf-8")
    if "## 318. Wang 2015 source-overlap boundary" in text:
        raise RuntimeError("round190-194-facts-already-present")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Last updated:"):
            lines[index] = "Last updated: 2026-08-21 (Rounds 190-194: Wang overlap corrected; dual thermal index closed on paleo transport)"
            break
    updated = "\n".join(lines).rstrip() + SECTION + "\n"
    path.write_text(updated, encoding="utf-8")
    print("UPDATED_LIA_PROJECT_FACTS_ROUNDS_190_194")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

