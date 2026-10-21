"""Append resolved Round 185-188 findings to the canonical project facts."""

from pathlib import Path


ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent and not (ROOT / "LIA_PROJECT_FACTS.md").exists():
    ROOT = ROOT.parent
FACTS = ROOT / "LIA_PROJECT_FACTS.md"

SECTION = r"""

## 314. Central-JJA historical support boundary (2026-08-21)

- Round 185 exactly matched all 41 `central_80_90` JJA identities from the
  Round 156 network to the original LMR CSV. Nineteen 1921-1939 rows were
  skipped immediately after year decoding and before any selected proxy-token
  conversion.
- Fourteen records satisfy the frozen long-core rule: at least 40 finite years
  in every 50-year window from 1550 through 1899 and at least 40 finite years
  in 1940-2000. Each of the seven paleo windows has all 14 long-core records.
- The CSV and Round 156 MAT modern values agree exactly for these records
  (maximum common-value absolute error zero). The support audit therefore
  passed, but support alone did not authorize a historical operator.
- All tracked sources remained unchanged. Evidence is in
  `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_185_central_jja_historical_support_preflight_v1/`.

## 315. Central-JJA temporal-transport failure (2026-08-21)

- Round 186 froze the 14 coverage-only long-core records and reran the exact
  Round 156 three-block modern OOF structure. The subset fails decisively:
  Berkeley Earth correlation/skill are `-0.590/-0.404` and CRU are
  `-0.664/-0.717`; both products also have a negative middle-fold slope.
- Its untuned seven-window 1550-1899 shape correlates `-0.507` with independent
  Chongce for both product mappings, compared with `-0.112` for the current
  local project curve. Only three of six adjacent changes agree. Close the
  1550 long-core operator; no annual paleo candidate was exported.
- Round 187 then used coverage-only nested cohorts to locate the temporal
  boundary of modern skill. The 1550 cohort has 14 records and fails. The 1600,
  1650, and 1700 cohorts have 30, 31, and 33 records and pass both product
  gates. The 35-record 1750 cohort fails Berkeley correlation before the 1800,
  1850, and full 41-record modern branches pass again.
- This non-monotonic ladder shows that the strong full-network modern OOF
  result depends on proxy composition and is not a stable property of the
  records available throughout 1550-1899. The earliest passing 1600 cutoff is
  diagnostic only and cannot be selected directly for production.
- Evidence is in
  `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_186_central_jja_chongce_shape_validation_v1/`
  and `round_187_central_jja_coverage_skill_ladder_v1/`.

## 316. Central-JJA 1600-1899 independent-shape closure (2026-08-21)

- Round 188 prospectively froze the earliest Round 187 passing cohort: 30
  records selected by coverage plus modern OOF only. Its modern OOF remains
  positive and transferable within 1940-2000 (Berkeley correlation/skill
  `0.438/0.071`; CRU `0.633/0.255`; all fold slopes positive).
- The independent 1600-1899 Chongce shape test nevertheless fails every
  substantive paleo gate. Both product mappings correlate `-0.515` with
  Chongce, worsening the current project correlation of `-0.157` by `-0.358`;
  only two of five adjacent 50-year changes agree. Positive full-fit slopes and
  identical Berkeley/CRU shape do not rescue the historical mismatch.
- Decision: close the current central-JJA common-mode route as a correction for
  the 1500-1899 HMA warm bias. Modern instrumental skill and historical
  low-frequency transport are separate requirements; the former does not
  establish the latter. The present evidence supports proxy-network temporal
  turnover or PSM nonstationarity as an identifiability problem, not a simple
  global offset.
- The warm-bias direction retained from Round 184 remains valid, but neither a
  uniform translation nor this central regional operator is authorized. The
  1500-1599 interval is especially underconstrained. A production correction
  now requires genuinely independent long-period regional evidence or a new
  physically defined operator that passes both modern OOF and historical
  validation; further threshold tuning on these same records is not admissible.
- Round 188 skipped all 19 locked rows before proxy conversion, exported no
  annual paleo trajectory, accessed no Jionglacuo value, and ran no EnKF,
  production correction, or OGGM. All sources remained unchanged. Evidence is
  in `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_188_central_jja_1600_chongce_shape_v1/`.
"""


def main() -> None:
    text = FACTS.read_text(encoding="utf-8")
    if "## 314. Central-JJA historical support boundary" in text:
        raise RuntimeError("round185-188-facts-already-present")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Last updated:"):
            lines[index] = (
                "Last updated: 2026-08-21 (central-JJA historical transport route closed; "
                "no production temperature correction authorized)"
            )
            break
    else:
        raise RuntimeError("last-updated-line-not-found")
    updated = "\n".join(lines).rstrip() + SECTION + "\n"
    FACTS.write_text(updated, encoding="utf-8", newline="\n")
    print("ROUND185_188_FACTS_UPDATED")


if __name__ == "__main__":
    main()

