"""Append the verified Round 156 result to the canonical facts contract."""

from pathlib import Path


SECTION = """

## 291. Regional JJA low-frequency temperature-operator closure (2026-08-19)

- Round 156 tested the separately contracted regional low-frequency route permitted by Section 289. It retained the Round 16 source-native JJA observations, Berkeley Earth and CRU targets, centered 11-year mean, and three purged outer blocks, changing only the operator scale from one all-HMA common mode to fixed west `<80 E`, central `80-90 E`, and east `>=90 E` networks.
- Regional support is adequate: 37 west, 41 central, and 68 east formally documented JJA temperature records. The full candidate improves median RMSE by `39.513%` relative to the all-HMA common-mode control, and the first/third-block median improvements are positive (`5.729%/57.947%`). These relative gains do not establish absolute validity.
- Only central HMA has transferable low-frequency signal in both products. Its Berkeley/CRU absolute skills are `29.446%/47.370%` and pooled correlations are `0.548/0.772`; all three fitted slopes are positive for both products.
- West and east fail absolute and phase gates. West has Berkeley/CRU skill `-0.095%/-23.365%` with correlations `-0.799/+0.238`; east has skill `-21.872%/-24.004%` with correlations `-0.653/-0.631`. Across all six branches, median absolute skill is `-10.983%`, median pooled correlation is `-0.197`, only `61.111%` of block paths have positive absolute skill, and only `33.333%` of branches retain positive slopes in all folds.
- Decision: close the fixed three-region JJA common-mode operator as a full-HMA repair. The central result is a valid mechanism finding but, under the preregistered contract, is attribution only and cannot be selected post hoc to assemble a partial production field. No paleo trajectory was inspected or generated.
- Round 156 read only 1940-2000 values, accessed no 1901-1939 or Jionglacuo value, ran neither EnKF nor OGGM, changed no source, and left all tracked source hashes unchanged. Evidence is in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_temperature_regional_jja_low_frequency_oof_v1/`.
"""


def main() -> int:
    root = Path(__file__).resolve()
    while root.parent != root and not (root / "LIA_PROJECT_FACTS.md").exists():
        root = root.parent
    facts = root / "LIA_PROJECT_FACTS.md"
    text = facts.read_text(encoding="utf-8")
    heading = "## 291. Regional JJA low-frequency temperature-operator closure"
    if heading in text:
        raise RuntimeError("round156-facts-section-already-present")
    if "## 290. Direct posterior corroboration" not in text:
        raise RuntimeError("expected-section-290-not-found")
    facts.write_text(text.rstrip() + SECTION + "\n", encoding="utf-8")
    print("ROUND156_FACTS_SECTION_291_APPENDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
