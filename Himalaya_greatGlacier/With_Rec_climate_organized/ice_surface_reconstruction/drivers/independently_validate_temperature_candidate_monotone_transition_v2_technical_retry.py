"""Technical retry for independent monotone-transition validation.

V1 propagated NaNs outside the supported HMA mask through ordinary sums.
This adapter applies the preregistered fixed-denominator, missing-value-safe
weighted calculations and otherwise delegates the unchanged V1 validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import independently_validate_temperature_candidate_monotone_transition_v1 as core


DRIVER = Path(__file__).resolve()
OUTPUT = core.LOOP / "temperature_candidate_monotone_transition_independent_validation_v2"


def weighted_mean_missing_safe(
    values: np.ndarray, mask: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    return np.nansum(values * weights[None], axis=(1, 2)) / np.sum(weights)


def spatial_gram_missing_safe(
    correction_fields: np.ndarray, mask: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    normalized = weights / np.sum(weights)
    flattened = np.nan_to_num(
        correction_fields[:20].reshape(20, -1), nan=0.0, posinf=0.0, neginf=0.0
    )
    return (flattened * normalized.reshape(-1)[None]) @ flattened.T


def main() -> int:
    core.OUTPUT = OUTPUT
    core.DRIVER = DRIVER
    core.weighted_mean = weighted_mean_missing_safe
    core.spatial_gram = spatial_gram_missing_safe
    result = core.main()

    summary_path = OUTPUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "schema": "hma-temperature-candidate-monotone-transition-independent-validation-v2",
            "technical_retry": {
                "supersedes_failed_attempt": str(
                    core.LOOP / "temperature_candidate_monotone_transition_independent_validation_v1"
                ),
                "failure": "ordinary sums propagated unsupported-grid NaNs into HMA metrics",
                "correction": "use missing-safe numerators with the unchanged fixed HMA weight denominator",
                "scientific_contract_changed": False,
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "TECHNICAL_RETRY.md").write_text(
        "# Technical retry\n\n"
        "V1 reproduced all station metrics but propagated unsupported-grid NaNs "
        "into the HMA calculations. V2 changes only the numerical missing-value "
        "handling: unsupported cells contribute zero to the numerator while the "
        "frozen full HMA weight denominator is retained. No data, gate, path, "
        "sealed-period value, or scientific rule changes.\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
