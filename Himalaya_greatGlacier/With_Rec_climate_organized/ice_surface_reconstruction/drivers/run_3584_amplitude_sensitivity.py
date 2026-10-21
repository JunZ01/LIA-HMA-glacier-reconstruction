"""Run the saved-chain amplitude analysis with exact candidate matching."""

from __future__ import annotations

from pathlib import Path

from . import analyze_3584_amplitude_sensitivity as analysis


_raw_score_chain = analysis._score_chain


def _score_chain(summary_path: Path) -> dict:
    row = _raw_score_chain(summary_path)
    row["selected_candidate"] = (
        Path(row["source_dir"]).name == "glen_x0p75_dt_0p7"
    )
    return row


analysis._score_chain = _score_chain


if __name__ == "__main__":
    raise SystemExit(analysis.main())
