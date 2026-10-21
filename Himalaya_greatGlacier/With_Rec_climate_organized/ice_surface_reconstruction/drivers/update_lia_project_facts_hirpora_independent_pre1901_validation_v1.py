"""Append the resolved Hirpora validation result to the canonical facts file."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
FACTS = ROOT / "LIA_PROJECT_FACTS.md"
HEADER = "## 278. Independent Hirpora arbitration of the western temperature anomaly (2026-08-19)"

SECTION = r"""

## 278. Independent Hirpora arbitration of the western temperature anomaly (2026-08-19)

- A prospectively frozen test digitized only the 1773-1900 portion of Figure 7 from Malik and Sukumar (2021), a June-July temperature reconstruction from Hirpora in the Pir Panjal/Kashmir region. The nearest archived proxy6/proxy9 site is 185.7 km away and no formal record identity overlap was found.
- The digitization passes all three preregistered masks. Pixel-column coverage is 99.25-99.63%, longest missing runs are 2-4 pixels, and maximum annual temperature sensitivity across masks is 0.0534 degC. No post-1900 curve pixel was analyzed.
- Against Hirpora over 1861-1900, the archived alpha=1 western branch has 15-year correlation -0.395 and RMSE 2.538 baseline SD for the medium mask. Alpha=0.25 changes these to -0.280 and 2.499, only a 1.55% RMSE improvement and therefore fails the frozen 5% gate.
- The already frozen alpha=0 branch changes correlation to +0.366 and RMSE to 1.865 baseline SD. It improves RMSE by 26.5%, reduces the 1881-1900 mean error by 1.264 baseline SD, and changes the branch trend to the same negative sign as Hirpora in all three masks.
- Decision: independent western evidence supports the diagnosis that the archived western temperature-operator covariance pathway is over-amplified and that the alpha=1 1861-1900 warming is not transferable. This is a mechanism result, not permission to promote alpha=0: the uniform network-wide alpha route previously failed its modern/external frozen gates and remains closed.
- Hirpora does not resolve the separate 1500-1899 HMA background offset. No production correction, EnKF run, OGGM run, or post-hoc alpha fitting is authorized. All tracked input hashes remained unchanged.
"""


def main() -> int:
    text = FACTS.read_text(encoding="utf-8")
    if HEADER in text:
        print("section-already-present")
        return 0
    FACTS.write_text(text.rstrip() + SECTION + "\n", encoding="utf-8")
    print("section-278-appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
