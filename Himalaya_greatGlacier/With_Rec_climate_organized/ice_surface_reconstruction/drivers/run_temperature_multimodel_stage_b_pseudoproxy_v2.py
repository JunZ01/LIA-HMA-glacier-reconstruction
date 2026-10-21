from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_temperature_multimodel_stage_b_pseudoproxy_v1 as v1


DRIVER = Path(__file__).resolve()
V1_DRIVER = Path(v1.__file__).resolve()
v1.OUT = v1.LOOP / "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_v2"


class SharedObservationNoise:
    """Reuse one proxy6/proxy9 noise draw across the three nontruth priors."""

    def __init__(self, seed: int):
        self.generator = np.random.Generator(np.random.PCG64(seed))
        self.call_index = 0
        self.cached: dict[int, np.ndarray] = {}

    def normal(self, loc, scale, size):
        within_fold = self.call_index % 6
        network_slot = within_fold % 2
        if within_fold < 2:
            value = self.generator.normal(loc=loc, scale=scale, size=size)
            self.cached[network_slot] = value
        else:
            value = self.cached[network_slot].copy()
            if value.shape != tuple(size):
                raise RuntimeError("shared-noise-shape")
        self.call_index += 1
        return value


def main() -> None:
    v1.np.random.default_rng = lambda seed: SharedObservationNoise(seed)
    v1.__file__ = str(DRIVER)
    v1.main()

    summary_path = v1.OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["supersedes_unrun_driver"] = "run_temperature_multimodel_stage_b_pseudoproxy_v1.py"
    summary["v2_change"] = (
        "each fixed-seed pseudoproxy noise realization is shared exactly across all three nontruth priors"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest_path = v1.OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["implementation_v1"] = {
        "path": str(V1_DRIVER),
        "sha256": v1.sha256(V1_DRIVER),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
