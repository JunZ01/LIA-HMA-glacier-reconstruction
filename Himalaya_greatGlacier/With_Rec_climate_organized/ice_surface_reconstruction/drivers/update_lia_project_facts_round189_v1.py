"""Append the Round 189 Jionglacuo source correction to project facts."""

from pathlib import Path


ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent and not (ROOT / "LIA_PROJECT_FACTS.md").exists():
    ROOT = ROOT.parent
FACTS = ROOT / "LIA_PROJECT_FACTS.md"

SECTION = r"""

## 317. Jionglacuo source-identity correction and Round 184 invalidation (2026-08-21)

- This section supersedes all scientific claims in Section 313 and the sentence
  in Section 316 that retained Round 184 warm-bias support. Those claims are
  retracted.
- Round 189 established an exact source chain without reopening the workbook:
  Mendeley dataset `10.17632/s7t4zc34sr.2` links to Nature article
  `10.1038/s41612-025-01254-y`, and the official article identifies the
  79-core, 31-tree Balfour-spruce sampling site as Jionglacuo. The official
  Methods and figure-caption identities both match; the archived official HTML
  contains three Jionglacuo mentions.
- Round 184's source manifest confirms that it parsed the `s7t4zc34sr_v2`
  `Temperature reconstruction.xlsx` workbook. This violated the explicit
  project prohibition on opening Jionglacuo. Round 184 is therefore invalid,
  not merely weak or seasonally mismatched. Its values, plot, correlations,
  offsets, and claimed external support for the warm-bias direction must never
  be used.
- The source file remains unchanged for audit preservation but is permanently
  prohibited from all future parsing, hashing, plotting, validation, fitting,
  EnKF, and OGGM work. The Round 184 directory contains an explicit
  `INVALIDATED_BY_ROUND189.md` marker.
- Rounds 185-188 are unaffected because they use LMR records, Berkeley/CRU
  products, and the permitted Chongce window evidence rather than the
  Jionglacuo source. Their central-JJA transport and shape failures remain
  valid.
- After this correction, the project still has internal/source-overlapping
  evidence of a 1500-1899 temperature offset, but Round 184 no longer provides
  independent confirmation of its direction or amplitude. Production
  correction remains unauthorized.
- Round 189 read only the Mendeley metadata snapshot, Round 184 JSON metadata,
  and the official Nature article HTML. It did not open or hash the workbook,
  decode any Jionglacuo or 1921-1939 numeric value, or run EnKF or OGGM. All
  local inputs remained unchanged. Evidence is in
  `validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2/round_189_jionglacuo_source_contract_correction_v1/`.
"""


def main() -> None:
    text = FACTS.read_text(encoding="utf-8")
    if "## 317. Jionglacuo source-identity correction" in text:
        raise RuntimeError("round189-facts-already-present")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Last updated:"):
            lines[index] = (
                "Last updated: 2026-08-21 (Round 184 invalidated as prohibited "
                "Jionglacuo source; no production correction authorized)"
            )
            break
    else:
        raise RuntimeError("last-updated-line-not-found")
    FACTS.write_text("\n".join(lines).rstrip() + SECTION + "\n", encoding="utf-8", newline="\n")
    print("ROUND189_FACTS_UPDATED")


if __name__ == "__main__":
    main()

