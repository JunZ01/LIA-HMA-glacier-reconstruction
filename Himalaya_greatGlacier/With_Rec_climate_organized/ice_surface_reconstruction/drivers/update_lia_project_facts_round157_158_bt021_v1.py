"""Append verified BT021 acquisition, identity, and OOF results."""

from pathlib import Path


SECTIONS = """

## 292. Public BT021 chronology acquisition and identity closure (2026-08-19)

- The NOAA ITRDB Asia directory provides three anonymous-download Ura-Dhur chronology files: `bt021-crn-noaa.txt`, `bt021-noaa.crn`, and `bt021.crn`. They were saved unchanged in an independent `raw_unopened` directory with SHA256 `f907cf78...a0228bf`, `b3ebe1d1...1cd9f59`, and `1ddc41ac...f6d6008`, respectively. No value was decoded during acquisition.
- Round 157 v1 stopped before data parsing because its expected table-header literal omitted the leading `t` in `trsgi`. It produced no scientific result. Independent v2 changed only that header recognition and its result identifier/directory.
- V2 exported 702 allowed BT021 chronology values over 1280-1920 and 1940-2000. Exactly 19 rows from 1921-1939 were skipped after year identification and before chronology/sample-count numeric conversion; 13 post-2000 rows were also skipped before value conversion.
- BT021 has zero exact affine aliases and zero near-exact standardized sequence matches across all 202 proxy6 and 297 proxy9 records. It is therefore genuinely new fitting evidence relative to the archived networks.
- BT021 correlates `0.9683` with the Bhutan 2015 JJA reconstruction over 606 allowed common years but is not exactly affine-equivalent. Scientific independence is nevertheless false because Bhutan 2015 explicitly derives from the Ura-Dhur/BT021 material. The published Bhutan curve cannot validate a BT021-fitted candidate.
- Decision: advance BT021 only as a new eastern-HMA fitting candidate to a separately frozen modern OOF test. Acquisition and identity outputs are in `climate_reconstruction_bt021_raw_chronology_acquisition_20260819_v1` and `climate_reconstruction_bt021_identity_closure_20260819_v2`.

## 293. BT021 low-frequency modern OOF closure (2026-08-19)

- Round 158 tested BT021 against Berkeley Earth and CRU JJA temperature at `27.4223 N, 90.9675 E` using the centered 11-year Round 16 filter and three fully purged 1940-2000 outer blocks. It fit only an intercept and one BT021 slope; no proxy subset, season, window, target location, or threshold was selected from results.
- The fitted BT021 slope is positive in all six product-fold fits and Berkeley/CRU OOF predictions correlate `0.8645`, but transfer is temporally unstable. First-block test correlations are `0.896/0.916`; third-block correlations reverse to `-0.443/-0.435` for Berkeley/CRU.
- Pooled Berkeley correlation/skill are `-0.115/0.0187`; CRU values are `+0.232/0.1929`. Each product has positive test correlation in only one of three blocks. Five of seven frozen gates fail.
- Decision: close BT021 as a standalone low-frequency temperature operator and do not add it to an eastern regional reconstruction, inspect its paleo trajectory, or use the same-source Bhutan 2015 curve for validation. The result reinforces temporal nonstationarity in eastern tree-ring temperature response rather than a simple missing-record explanation.
- Round 158 read only 1940-2000 modern values, accessed no locked, 1901-1920, Bhutan paleo, or Jionglacuo value, ran neither EnKF nor OGGM, changed no source, and left all tracked hashes unchanged. Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_bt021_low_frequency_modern_oof_20260819_v1/`.
"""


def main() -> int:
    root = Path(__file__).resolve()
    while root.parent != root and not (root / "LIA_PROJECT_FACTS.md").exists():
        root = root.parent
    facts = root / "LIA_PROJECT_FACTS.md"
    text = facts.read_text(encoding="utf-8")
    for heading in (
        "## 292. Public BT021 chronology acquisition and identity closure",
        "## 293. BT021 low-frequency modern OOF closure",
    ):
        if heading in text:
            raise RuntimeError(f"facts-section-already-present:{heading}")
    if "## 291. Regional JJA low-frequency temperature-operator closure" not in text:
        raise RuntimeError("expected-section-291-not-found")
    facts.write_text(text.rstrip() + SECTIONS + "\n", encoding="utf-8")
    print("ROUND157_158_FACTS_SECTIONS_292_293_APPENDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
