"""Append verified Round 178-184 temperature evidence to the project contract."""

from pathlib import Path


SECTION = r"""

## 312. New eastern-HMA tree-ring candidate modern-OOF closure (2026-08-21)

- Round 178 anonymously recovered two CC-BY-4.0 Mendeley datasets. Dataset
  `10.17632/889n6ccx4z.1` contains five raw Hengduan tree-ring files and a
  chronology/reconstruction workbook; dataset `10.17632/s7t4zc34sr.2`
  contains a reconstruction-only workbook. All seven publisher SHA256 checks
  passed and the preserved source files remained unchanged.
- Round 179 found 168 sample identifiers across the five Hengduan raw files,
  zero pairwise sample overlap among those files, and zero exact matches to
  existing project RWL files. The reconstruction-only dataset has no raw
  chronology and therefore cannot be a new production PSM candidate.
- Frozen Round 180 modern OOF used only the published Hengduan standard
  chronology as predictor and the documented May-September target. It has zero
  proxy6/proxy9 aliases and positive OOF skill, but correlations are only
  `0.2167` for Berkeley Earth and `0.2958` for CRU, below the joint `0.30`
  gate. Close it as a production temperature operator without tuning.
- Rounds 181-182 recovered the official Springer Haizi Mountain supplement
  unchanged (SHA256
  `35302c0ba1912b015ee8d8b1a5c05df9aa82dd44d4467f7b07278e723611d3eb`)
  and established its zero-value structure: year, standard chronology,
  reconstructed mean annual temperature, and sample depth in columns A-D.
- Round 183 v1 was never run because a literal newline token remained in the
  mechanically derived driver. Independent v2 changed only that syntax. The
  Haizi chronology is non-aliased, has modern sample depth at least 42, and
  passes the positive-skill, fold-direction, slope-direction, split-sign, and
  cross-product gates. Its OOF correlations are nevertheless only `0.1150`
  for Berkeley Earth and `0.2853` for CRU, so it also fails the frozen joint
  `0.30` gate and is closed as a production operator.
- Neither failed candidate may be rescued by tuning season, lag, sign,
  location, folds, or threshold. Both results add genuinely new evidence but
  reinforce that one additional eastern tree-ring chronology does not repair
  the archived temperature family.
- Rounds 178-183 decoded no 1921-1939 numeric value, inspected no candidate
  paleo trajectory, opened no Jionglacuo value, and ran no EnKF, correction,
  production reconstruction, or OGGM. Evidence is in
  `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_178_mendeley_east_hma_temperature_candidates_v1/`,
  `round_180_hengduan_chronology_modern_oof_v2/`,
  `round_181_haizi_903y_temperature_supplement_v2/`,
  `round_182_haizi_903y_structure_v1/`, and
  `round_183_haizi_903y_chronology_modern_oof_v2/`.

## 313. Eastern Tibetan Plateau external warm-bias audit (2026-08-21)

- Round 184 used the reconstruction-only Mendeley dataset
  `10.17632/s7t4zc34sr.2` strictly as independent low-frequency comparison
  evidence. Its target is eastern Tibetan Plateau June-September mean
  temperature, while the project and four-family curves are HMA annual means;
  this spatial-seasonal mismatch precludes field validation or direct fitting.
- On the eight complete 50-year windows from 1500-1549 through 1850-1899, the
  external reconstruction is colder than the project six-member curve in all
  eight. The median project-minus-external difference is `+0.197 degC`, which
  independently supports the direction of the diagnosed project warm bias.
- The external curve is not closer to the four-family central curve: its median
  absolute distance is `0.231 degC` from the four-family central estimate and
  `0.197 degC` from the project curve. It lies within the four-family range in
  only three of eight windows; 50-year correlations are `0.258` with the
  four-family central curve and `0.491` with the project curve.
- Therefore the strict two-part confirmation gate fails. Retain the warm-bias
  direction as independently supported, but do not interpret the previous
  `+0.383 degC` project-minus-four-family value as a validated global offset.
  A uniform translation to the four-family central curve is not authorized.
- The parser decoded year first and skipped all 19 locked rows before external
  value conversion. No 1921-1939 numeric value or Jionglacuo value was opened,
  and no fitting, correction, EnKF, production reconstruction, or OGGM run
  occurred. Evidence is in
  `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_184_east_tp_external_temperature_bias_audit_v1/`.
"""


def main() -> int:
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "LIA_PROJECT_FACTS.md").exists())
    facts = root / "LIA_PROJECT_FACTS.md"
    text = facts.read_text(encoding="utf-8")
    marker = "## 312. New eastern-HMA tree-ring candidate modern-OOF closure"
    if marker in text:
        raise RuntimeError("facts-section-already-present")
    facts.write_text(text.rstrip() + SECTION + "\n", encoding="utf-8")
    print("APPENDED_FACTS_SECTIONS_312_313")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
